;; Gradient health monitoring and diagnostics system

(module nanograd-diagnostics
  (
   ;; Gradient statistics
   gradient-stats compute-gradient-norm
   gradient-l2-norm gradient-inf-norm
   
   ;; Health checks
   check-gradient-health
   has-exploding-gradients?
   has-vanishing-gradients?
   
   ;; Monitoring
   make-gradient-monitor
   gradient-monitor?
   record-step!
   get-gradient-history
   diagnose-training
   print-gradient-report
   
   ;; Remediation
   clip-gradients!
   suggest-learning-rate
   )
  
  (import
   scheme
   (chicken base)
   (chicken format)
   (srfi 1)
   (srfi 4)
   (srfi 69)  ; Hash tables
   yasos
   blas
   nanograd-autograd
   )

  ;; Hygienic macro for dtype-based operation dispatch
  (include "with-dtype.scm")

  ;;; ==================================================================
  ;;; Gradient Statistics
  ;;; ==================================================================

  (define (gradient-l2-norm grad dtype)
    "Compute L2 norm (Euclidean norm) of gradient vector"
    (let ((n (vector-length-for-dtype grad dtype)))
      (sqrt (with-dtype dtype (dot n grad grad)))))

  (define (gradient-inf-norm grad dtype)
    "Compute infinity norm (max absolute value) of gradient"
    (let ((n (vector-length-for-dtype grad dtype)))
      (let loop ((i 0) (max-val 0.0))
        (if (= i n)
            max-val
            (let ((val (abs (with-dtype dtype (elt-ref grad i)))))
              (loop (+ i 1) (max max-val val)))))))

  (define (compute-gradient-norm parameters #!key (norm-type 'l2))
    "Compute gradient norm across all parameters"
    (case norm-type
      ((l2)
       ;; Global L2 norm: sqrt(sum of all squared gradients)
       (sqrt
        (fold
         (lambda (param acc)
           (let ((grad (tensor-grad param)))
             (if grad
                 (let* ((dtype (tensor-dtype param))
                        (norm (gradient-l2-norm grad dtype)))
                   (+ acc (* norm norm)))
                 acc)))
         0.0
         parameters)))
      
      ((inf)
       ;; Global infinity norm: max absolute gradient
       (fold
        (lambda (param acc)
          (let ((grad (tensor-grad param)))
            (if grad
                (let* ((dtype (tensor-dtype param))
                       (norm (gradient-inf-norm grad dtype)))
                  (max acc norm))
                acc)))
        0.0
        parameters))
      
      (else
       (error 'compute-gradient-norm "Unknown norm type" norm-type))))

  (define (gradient-stats parameters)
    "Compute gradient statistics"
    (let ((l2-norm (compute-gradient-norm parameters norm-type: 'l2))
          (inf-norm (compute-gradient-norm parameters norm-type: 'inf))
          (layer-stats '()))
      
      ;; Per-parameter statistics
      (for-each
       (lambda (param idx)
         (let ((grad (tensor-grad param)))
           (when grad
             (let* ((dtype (tensor-dtype param))
                    (n (vector-length-for-dtype grad dtype))
                    (l2 (gradient-l2-norm grad dtype))
                    (inf (gradient-inf-norm grad dtype))
                    ;; Compute mean
                    (sum (with-dtype dtype (fold + 0.0 grad)))
                    (mean (/ sum n)))
               
               (set! layer-stats
                     (cons `((layer . ,idx)
                            (l2-norm . ,l2)
                            (inf-norm . ,inf)
                            (mean-abs . ,mean)
                            (size . ,n))
                           layer-stats))))))
       parameters
       (iota (length parameters)))
      
      `((global-l2-norm . ,l2-norm)
        (global-inf-norm . ,inf-norm)
        (layer-stats . ,(reverse layer-stats)))))

  ;;; ==================================================================
  ;;; Gradient Health Checks
  ;;; ==================================================================

  (define (has-exploding-gradients? parameters #!key 
                                    (threshold 10.0)
                                    (norm-type 'l2))
    "Check if gradients are exploding (too large)"
    (let ((norm (compute-gradient-norm parameters norm-type: norm-type)))
      (or (> norm threshold)
          (not (finite? norm)))))  ; NaN or Inf

  (define (has-vanishing-gradients? parameters #!key 
                                    (threshold 1e-7)
                                    (norm-type 'l2))
    "Check if gradients are vanishing (too small)"
    (let ((norm (compute-gradient-norm parameters norm-type: norm-type)))
      (< norm threshold)))

  (define (check-gradient-health parameters #!key
                                 (exploding-threshold 10.0)
                                 (vanishing-threshold 1e-7)
                                 (warn? #t))
    "Perform gradient health check"
    (let ((stats (gradient-stats parameters))
          (issues '()))
      
      (let ((global-norm (cdr (assoc 'global-l2-norm stats)))
            (global-inf (cdr (assoc 'global-inf-norm stats))))
        
        ;; Check for exploding gradients
        (when (or (> global-norm exploding-threshold)
                  (not (finite? global-norm)))
          (set! issues (cons 'exploding-gradients issues))
          (when warn?
            (printf "WARNING: Exploding gradients detected!\n")
            (printf "  Global L2 norm: ~A (threshold: ~A)\n" 
                    global-norm exploding-threshold)))
        
        ;; Check for vanishing gradients
        (when (< global-norm vanishing-threshold)
          (set! issues (cons 'vanishing-gradients issues))
          (when warn?
            (printf "WARNING: Vanishing gradients detected!\n")
            (printf "  Global L2 norm: ~A (threshold: ~A)\n" 
                    global-norm vanishing-threshold)))
        
        ;; Check for NaN or Inf
        (when (not (finite? global-inf))
          (set! issues (cons 'nan-gradients issues))
          (when warn?
            (printf "ERROR: NaN or Inf in gradients!\n")))
        
        ;; Check layer-wise imbalance
        (let* ((layer-stats (cdr (assoc 'layer-stats stats)))
               (layer-norms (map (lambda (s) (cdr (assoc 'l2-norm s))) 
                                layer-stats)))
          (when (and (not (null? layer-norms))
                     (> (length layer-norms) 1))
            (let ((min-norm (apply min layer-norms))
                  (max-norm (apply max layer-norms)))
              (when (and (> min-norm 0) 
                        (> (/ max-norm min-norm) 1000.0))
                (set! issues (cons 'gradient-imbalance issues))
                (when warn?
                  (printf "WARNING: Large gradient imbalance across layers!\n")
                  (printf "  Ratio: ~A:1\n" (/ max-norm min-norm)))))))
        
        `((healthy? . ,(null? issues))
          (issues . ,issues)
          (stats . ,stats)))))

  ;;; ==================================================================
  ;;; Gradient Monitor (Tracks History)
  ;;; ==================================================================

  (define-predicate gradient-monitor?)
  (define-operation (record-step! monitor step-num parameters))
  (define-operation (get-gradient-history monitor))
  (define-operation (diagnose-training monitor))

  (define (make-gradient-monitor #!key 
                                (exploding-threshold 10.0)
                                (vanishing-threshold 1e-7)
                                (history-size 100))
    "Create a gradient monitoring object"
    (let ((history '())
          (warnings '())
          (step-count 0)
          (exp-thresh exploding-threshold)
          (van-thresh vanishing-threshold)
          (max-history history-size))
      
      (object
       ((gradient-monitor? self) #t)
       
       ((record-step! self step-num parameters)
        (set! step-count (+ step-count 1))
        
        ;; Compute statistics
        (let* ((stats (gradient-stats parameters))
               (global-norm (cdr (assoc 'global-l2-norm stats)))
               (global-inf (cdr (assoc 'global-inf-norm stats)))
               (health (check-gradient-health parameters
                                            exploding-threshold: exp-thresh
                                            vanishing-threshold: van-thresh
                                            warn?: #f)))
          
          ;; Record in history
          (set! history
                (cons `((step . ,step-num)
                       (global-l2-norm . ,global-norm)
                       (global-inf-norm . ,global-inf)
                       (healthy? . ,(cdr (assoc 'healthy? health)))
                       (issues . ,(cdr (assoc 'issues health))))
                      history))
          
          ;; Limit history size
          (when (> (length history) max-history)
            (set! history (take history max-history)))
          
          ;; Record warnings
          (let ((issues (cdr (assoc 'issues health))))
            (when (not (null? issues))
              (set! warnings
                    (cons `((step . ,step-num)
                           (issues . ,issues))
                          warnings))))
          
          health))
       
       ((get-gradient-history self)
        (reverse history))
       
       ((diagnose-training self)
        "Analyze training health based on gradient history"
        (if (null? history)
            '((diagnosis . "No data collected yet"))
            (let* ((recent (take history (min 10 (length history))))
                   (norms (map (lambda (h) (cdr (assoc 'global-l2-norm h))) 
                              recent))
                   (mean-norm (/ (apply + norms) (length norms)))
                   (max-norm (apply max norms))
                   (min-norm (apply min norms))
                   (unhealthy-count (length (filter
                                            (lambda (h) 
                                              (not (cdr (assoc 'healthy? h))))
                                            recent))))
              
              `((total-steps . ,step-count)
                (recent-steps . ,(length recent))
                (mean-gradient-norm . ,mean-norm)
                (max-gradient-norm . ,max-norm)
                (min-gradient-norm . ,min-norm)
                (unhealthy-steps . ,unhealthy-count)
                (warning-count . ,(length warnings))
                (gradient-stability . ,(if (> max-norm 0)
                                          (/ min-norm max-norm)
                                          0.0))
                (recommendations . ,(generate-recommendations
                                    mean-norm max-norm min-norm
                                    unhealthy-count
                                    exp-thresh van-thresh))))))
       )))

  (define (generate-recommendations mean-norm max-norm min-norm 
                                   unhealthy-count exp-thresh van-thresh)
    "Generate training recommendations based on gradient statistics"
    (let ((recommendations '()))
      
      ;; Exploding gradients
      (when (> max-norm exp-thresh)
        (set! recommendations
              (cons "Apply gradient clipping to prevent exploding gradients"
                    recommendations))
        (set! recommendations
              (cons (format #f "Reduce learning rate (current norm: ~A)" max-norm)
                    recommendations)))
      
      ;; Vanishing gradients
      (when (< mean-norm van-thresh)
        (set! recommendations
              (cons "Gradients are vanishing - consider:"
                    recommendations))
        (set! recommendations
              (cons "  - Use ReLU or Leaky ReLU instead of sigmoid/tanh"
                    recommendations))
        (set! recommendations
              (cons "  - Reduce network depth or add skip connections"
                    recommendations))
        (set! recommendations
              (cons "  - Use batch normalization or layer normalization"
                    recommendations))
        (set! recommendations
              (cons (format #f "  - Increase learning rate (current norm: ~A)" mean-norm)
                    recommendations)))
      
      ;; High instability
      (when (and (> max-norm 0) (< (/ min-norm max-norm) 0.1))
        (set! recommendations
              (cons "High gradient instability detected - consider:"
                    recommendations))
        (set! recommendations
              (cons "  - Use adaptive optimizers (Adam, RMSprop)"
                    recommendations))
        (set! recommendations
              (cons "  - Implement learning rate warmup"
                    recommendations)))
      
      ;; Many unhealthy steps
      (when (> unhealthy-count 5)
        (set! recommendations
              (cons "Training is unstable - immediate action needed"
                    recommendations)))
      
      (if (null? recommendations)
          '("Training appears healthy")
          (reverse recommendations))))

  ;;; ==================================================================
  ;;; Gradient Remediation
  ;;; ==================================================================

  (define (clip-gradients! parameters #!key 
                          (max-norm 1.0)
                          (norm-type 'l2))
    "Clip gradients to prevent exploding gradients"
    (let ((current-norm (compute-gradient-norm parameters norm-type: norm-type)))
      
      (when (> current-norm max-norm)
        (let ((scale (/ max-norm current-norm)))
          
          ;; Scale all gradients
          (for-each
           (lambda (param)
             (let ((grad (tensor-grad param)))
               (when grad
                 (let* ((dtype (tensor-dtype param))
                        (n (vector-length-for-dtype grad dtype)))
                   (with-dtype dtype (scal! n scale grad))))
               ))
           parameters)
          
          (printf "  Gradient clipping applied (scale: ~A)\n" scale)
          current-norm))  ; Return current-norm if clipping was applied
      
      #f))  ; Return #f if no clipping needed

  (define (suggest-learning-rate parameters current-lr gradient-norm)
    "Suggest learning rate adjustment based on gradient norm"
    (cond
     ;; Exploding gradients
     ((> gradient-norm 10.0)
      (let ((suggested (* current-lr 0.1)))
        (printf "  Suggested learning rate: ~A (reduce by 10x)\n" suggested)
        suggested))
     
     ((> gradient-norm 5.0)
      (let ((suggested (* current-lr 0.5)))
        (printf "  Suggested learning rate: ~A (reduce by 2x)\n" suggested)
        suggested))
     
     ;; Vanishing gradients
     ((< gradient-norm 1e-6)
      (let ((suggested (* current-lr 10.0)))
        (printf "  Suggested learning rate: ~A (increase by 10x)\n" suggested)
        suggested))
     
     ((< gradient-norm 1e-4)
      (let ((suggested (* current-lr 2.0)))
        (printf "  Suggested learning rate: ~A (increase by 2x)\n" suggested)
        suggested))
     
     ;; Healthy range
     (else
      (printf "  Learning rate appears appropriate: ~A\n" current-lr)
      current-lr)))

  ;;; ==================================================================
  ;;; Diagnostic Reports
  ;;; ==================================================================

  (define (print-gradient-report parameters #!optional (label ""))
    "Print gradient health report"
    (printf "\n")
    (printf "Gradient Health Report~A\n" 
            (if (string=? label "") "" (string-append ": " label)))
    
    (let ((health (check-gradient-health parameters warn?: #f)))
      (let ((healthy? (cdr (assoc 'healthy? health)))
            (issues (cdr (assoc 'issues health)))
            (stats (cdr (assoc 'stats health))))
        
        (printf "Status: ~A\n" 
                (if healthy? "HEALTHY" "ISSUES DETECTED"))
        
        (when (not healthy?)
          (printf "Issues: ~A\n" issues))
        
        (printf "\nGlobal Statistics:\n")
        (printf "  L2 Norm:  ~A\n" (cdr (assoc 'global-l2-norm stats)))
        (printf "  Inf Norm: ~A\n" (cdr (assoc 'global-inf-norm stats)))
        
        (printf "\nPer-Layer Statistics:\n")
        (let ((layer-stats (cdr (assoc 'layer-stats stats))))
          (for-each
           (lambda (ls)
             (printf "  Layer ~A: L2=~A, Inf=~A, Mean=~A\n"
                     (cdr (assoc 'layer ls))
                     (cdr (assoc 'l2-norm ls))
                     (cdr (assoc 'inf-norm ls))
                     (cdr (assoc 'mean-abs ls))))
           layer-stats))
        
        )))

) ;; end module

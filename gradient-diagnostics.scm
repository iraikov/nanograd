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
   (chicken fixnum)
   (chicken flonum)
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

  ;; One pass over a gradient vector: the sum of squares, the largest
  ;; absolute value and the sum of absolute values, in double precision.
  ;; A NaN element makes the largest absolute value NaN (max would
  ;; ignore it), so that non-finite gradients are detected.
  (define (gradient-vector-stats grad dtype)
    (let ((n (vector-length-for-dtype grad dtype)))
      (define-syntax scan
        (syntax-rules ()
          ((_ ref)
           (let loop ((i 0) (ss 0.0) (mx 0.0) (sa 0.0))
             (if (fx= i n)
                 (values ss (if (nan? ss) +nan.0 mx) sa n)
                 (let ((a (fpabs (ref grad i))))
                   (loop (fx+ i 1) (fp+ ss (fp* a a)) (if (fp> a mx) a mx) (fp+ sa a))))))))
      (case dtype
        ((f32) (scan f32vector-ref))
        ((f64) (scan f64vector-ref))
        (else (error 'gradient-vector-stats "Unsupported dtype" dtype)))))

  (define (nan-max a b)
    (if (or (nan? a) (nan? b)) +nan.0 (max a b)))

  (define (gradient-l2-norm grad dtype)
    "Compute L2 norm (Euclidean norm) of gradient vector"
    (let-values (((ss mx sa n) (gradient-vector-stats grad dtype)))
      (sqrt ss)))

  (define (gradient-inf-norm grad dtype)
    "Compute infinity norm (max absolute value) of gradient"
    (let-values (((ss mx sa n) (gradient-vector-stats grad dtype)))
      mx))

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
                 (let-values (((ss mx sa n) (gradient-vector-stats grad (tensor-dtype param))))
                   (+ acc ss))
                 acc)))
         0.0
         parameters)))
      
      ((inf)
       ;; Global infinity norm: max absolute gradient
       (fold
        (lambda (param acc)
          (let ((grad (tensor-grad param)))
            (if grad
                (nan-max acc (gradient-inf-norm grad (tensor-dtype param)))
                acc)))
        0.0
        parameters))
      
      (else
       (error 'compute-gradient-norm "Unknown norm type" norm-type))))

  (define (gradient-stats parameters)
    "Compute gradient statistics: the global L2 and infinity norms, and
     for each parameter with a gradient its index, L2 norm, infinity
     norm, mean absolute value and size.  Each gradient is read once."
    (let loop ((ps parameters) (idx 0) (ss-total 0.0) (inf-total 0.0) (layer-stats '()))
      (if (null? ps)
          `((global-l2-norm . ,(sqrt ss-total))
            (global-inf-norm . ,inf-total)
            (layer-stats . ,(reverse layer-stats)))
          (let* ((param (car ps))
                 (grad (tensor-grad param)))
            (if grad
                (let-values (((ss mx sa n) (gradient-vector-stats grad (tensor-dtype param))))
                  (loop (cdr ps) (+ idx 1) (+ ss-total ss) (nan-max inf-total mx)
                        (cons `((layer . ,idx)
                                (l2-norm . ,(sqrt ss))
                                (inf-norm . ,mx)
                                (mean-abs . ,(if (zero? n) 0.0 (/ sa n)))
                                (size . ,n))
                              layer-stats)))
                (loop (cdr ps) (+ idx 1) ss-total inf-total layer-stats))))))

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
                                 (imbalance-threshold 1000.0)
                                 (warn? #t))
    "Perform gradient health check.  Gradients are unbalanced when the
     largest per-parameter L2 norm exceeds IMBALANCE-THRESHOLD times the
     smallest one (this check is skipped when some gradient is zero)."
    (gradient-health-of-stats (gradient-stats parameters)
                              exploding-threshold vanishing-threshold
                              imbalance-threshold warn?))

  ;; The health check of check-gradient-health, on the result STATS of
  ;; gradient-stats.
  (define (gradient-health-of-stats stats exploding-threshold vanishing-threshold
                                    imbalance-threshold warn?)
    (let ((issues '()))
      
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
                        (> (/ max-norm min-norm) imbalance-threshold))
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
                                (imbalance-threshold 1000.0)
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
               (health (gradient-health-of-stats stats exp-thresh van-thresh
                                                 imbalance-threshold #f)))
          
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
                          (norm-type 'l2)
                          (verbose? #t))
    "Clip gradients to prevent exploding gradients.  Returns the norm
     before clipping if clipping was applied, #f otherwise."
    (let ((current-norm (compute-gradient-norm parameters norm-type: norm-type)))
      (if (> current-norm max-norm)
          (let ((scale (/ max-norm current-norm)))
            ;; Scale all gradients
            (for-each
             (lambda (param)
               (let ((grad (tensor-grad param)))
                 (when grad
                   (let* ((dtype (tensor-dtype param))
                          (n (vector-length-for-dtype grad dtype)))
                     (with-dtype dtype (scal! n scale grad))))))
             parameters)
            (when verbose?
              (printf "  Gradient clipping applied (scale: ~A)\n" scale))
            current-norm)
          #f)))

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
        
        (printf "\nPer-Layer Statistics (Mean = mean absolute value):\n")
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

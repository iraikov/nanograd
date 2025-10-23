;; Fixed optimizer tests using realistic gradient health checks

;; This replaces problematic tests in test-optimizer-units.scm

;;; ==================================================================
;;; Fixed SGD Momentum Test
;;; ==================================================================

(define (test-sgd-momentum)
  (printf "\n=== Testing SGD with Momentum ===\n")
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-sgd params 
                       learning-rate: 0.01
                       momentum: 0.9)))
    
    (let* ((state (optimizer-state opt))
           (momentum-val (cdr (assoc 'momentum state))))
      (assert-equal momentum-val 0.9 1e-10
                    "Momentum = 0.9"))
    
    (let ((x (make-tensor32 (f32vector 1.0 2.0) '(2)))
          (target (make-tensor32 (f32vector 5.0) '(1))))
      
      ;; Collect initial and final losses (more realistic than monotonic check)
      (let* ((initial-loss (compute-loss model x target))
             (losses 
              (let loop ((step 0) (acc '()))
                (if (= step 20)
                    (reverse acc)
                    (let ((loss (train-one-step model x target opt)))
                      (loop (+ step 1) (cons loss acc))))))
             (final-loss (last losses))
             ;; Average of last 5 losses
             (final-avg (/ (apply + (take-right losses 5)) 5.0))
             ;; Average of first 5 losses
             (initial-avg (/ (apply + (take losses 5)) 5.0)))
        
        ;; Test 1: Overall improvement
        (assert-less-than final-loss initial-loss
                         "Final loss < initial loss")
        
        ;; Test 2: Average improvement (accounts for oscillation)
        (assert-less-than final-avg initial-avg
                         "Average loss improves with momentum")
        
        ;; Test 3: Not diverging (no gradient explosion)
        (assert-less-than final-loss (* 10.0 initial-loss)
                         "Training doesn't diverge")))))

;;; ==================================================================
;;; Enhanced Momentum Test with Gradient Monitoring
;;; ==================================================================

(define (test-sgd-momentum-with-monitoring)
  (printf "\n=== Testing SGD Momentum with Gradient Monitoring ===\n")
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-sgd params 
                       learning-rate: 0.01
                       momentum: 0.9))
         (monitor (make-gradient-monitor exploding-threshold: 10.0
                                        vanishing-threshold: 1e-8)))
    
    (let ((x (make-tensor32 (f32vector 1.0 2.0) '(2)))
          (target (make-tensor32 (f32vector 5.0) '(1))))
      
      ;; Train with monitoring
      (do ((step 0 (+ step 1)))
          ((= step 20))
        (let* ((output (forward model x))
               (loss (mse-loss output target)))
          (backward! loss)
          
          ;; Record gradient health
          (record-step! monitor step params)
          
          (step! opt)
          (zero-grad-layer! model)))
      
      ;; Check training health
      (let ((diagnosis (diagnose-training monitor)))
        (let ((unhealthy-count (cdr (assoc 'unhealthy-steps diagnosis)))
              (mean-norm (cdr (assoc 'mean-gradient-norm diagnosis))))
          
          ;; Most steps should be healthy
          (assert-less-than unhealthy-count 5
                           "Unhealthy steps < 25%")
          
          ;; Gradients should be in reasonable range
          (assert-true (and (> mean-norm 1e-8) (< mean-norm 100.0))
                      "Mean gradient norm in healthy range"))))))

;;; ==================================================================
;;; Realistic RMSprop Momentum Test
;;; ==================================================================

(define (test-rmsprop-with-momentum)
  (printf "\n=== Testing RMSprop with Momentum ===\n")
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-rmsprop params 
                           learning-rate: 0.01
                           momentum: 0.9)))
    
    (let* ((state (optimizer-state opt))
           (momentum-val (cdr (assoc 'momentum state))))
      (assert-equal momentum-val 0.9 1e-10 "Momentum = 0.9"))
    
    (let ((x (make-tensor32 (f32vector 1.0 2.0) '(2)))
          (target (make-tensor32 (f32vector 3.0) '(1))))
      
      ;; Measure improvement over epochs
      (let* ((losses
              (let loop ((step 0) (acc '()))
                (if (= step 20)
                    (reverse acc)
                    (let ((loss (train-one-step model x target opt)))
                      (loop (+ step 1) (cons loss acc))))))
             ;; Use smoothed comparison
             (early-avg (/ (apply + (take losses 5)) 5.0))
             (late-avg (/ (apply + (take-right losses 5)) 5.0)))
        
        ;; Check average improvement (more robust than monotonic)
        (assert-less-than late-avg early-avg
                         "Average loss improves with RMSprop+momentum")
        
        ;; Ensure stability (not exploding)
        (assert-true (every (lambda (l) (and (finite? l) (< l 1000.0))) losses)
                    "Loss remains stable")))))

;;; ==================================================================
;;; Gradient Health-Aware Training Test
;;; ==================================================================

(define (test-gradient-aware-training)
  (printf "\n=== Testing Gradient-Aware Training ===\n")
  
  (let* ((model (make-sequential
                 (list 
                  (make-dense-layer 3 10 activation: (make-relu))
                  (make-dense-layer 10 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params learning-rate: 0.01))
         (monitor (make-gradient-monitor)))
    
    (let ((training-data (list
                          (cons (f32vector 1.0 0.0 0.0) 1.0)
                          (cons (f32vector 0.0 1.0 0.0) 2.0)
                          (cons (f32vector 0.0 0.0 1.0) 3.0))))
      
      ;; Train with gradient monitoring
      (do ((epoch 0 (+ epoch 1)))
          ((= epoch 30))
        
        (for-each
         (lambda (sample)
           (let* ((x (make-tensor32 (car sample) '(3)))
                  (target (make-tensor32 (f32vector (cdr sample)) '(1)))
                  (output (forward model x))
                  (loss (mse-loss output target)))
             
             (backward! loss)
             
             ;; Check and clip if needed
             (when (has-exploding-gradients? params threshold: 5.0)
               (clip-gradients! params max-norm: 5.0))
             
             (record-step! monitor epoch params)
             (step! opt)
             (zero-grad-layer! model)))
         training-data))
      
      ;; Verify training was healthy
      (let ((diagnosis (diagnose-training monitor)))
        (let ((mean-norm (cdr (assoc 'mean-gradient-norm diagnosis)))
              (warning-count (cdr (assoc 'warning-count diagnosis))))
          
          (assert-true (< warning-count 10)
                      "Few gradient warnings during training")
          
          (assert-true (and (> mean-norm 1e-7) (< mean-norm 10.0))
                      "Gradients remain in healthy range"))))))

;;; ==================================================================
;;; Convergence Test with Gradient Validation
;;; ==================================================================

(define (test-convergence-with-gradient-check)
  (printf "\n=== Testing Convergence with Gradient Validation ===\n")
  
  ;; Learn y = 2x1 + 3x2 + 1
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params learning-rate: 0.1))
         (monitor (make-gradient-monitor)))
    
    (let ((training-data (list
                          (cons (f32vector 1.0 0.0) 3.0)  ; 2*1 + 3*0 + 1
                          (cons (f32vector 0.0 1.0) 4.0)  ; 2*0 + 3*1 + 1
                          (cons (f32vector 1.0 1.0) 6.0)  ; 2*1 + 3*1 + 1
                          (cons (f32vector 2.0 1.0) 8.0)))) ; 2*2 + 3*1 + 1
      
      ;; Train with gradient monitoring
      (do ((epoch 0 (+ epoch 1)))
          ((= epoch 100))
        
        (for-each
         (lambda (sample)
           (let* ((x (make-tensor32 (car sample) '(2)))
                  (target (make-tensor32 (f32vector (cdr sample)) '(1))))
             (train-one-step model x target opt)
             
             ;; Periodic gradient check
             (when (zero? (modulo epoch 10))
               (let* ((output (forward model x))
                      (loss (mse-loss output target)))
                 (backward! loss)
                 (record-step! monitor epoch params)
                 (zero-grad-layer! model)))))
         training-data))
      
      ;; Test learned function
      (let* ((test-x (make-tensor32 (f32vector 3.0 2.0) '(2)))
             (expected 13.0)  ; 2*3 + 3*2 + 1 = 13
             (output (forward model test-x))
             (predicted (f32vector-ref (tensor-data output) 0)))
        
        (assert-equal predicted expected 0.5
                     "Model converges to correct function")
        
        ;; Verify gradients stayed healthy
        (let ((diagnosis (diagnose-training monitor)))
          (assert-true (< (cdr (assoc 'warning-count diagnosis)) 5)
                      "Few gradient issues during convergence"))))))

;;; ==================================================================
;;; Test Gradient Clipping Effectiveness
;;; ==================================================================

(define (test-gradient-clipping-prevents-explosion)
  (printf "\n=== Testing Gradient Clipping Prevents Explosion ===\n")
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         ;; Use high learning rate to induce problems
         (opt-unclipped (make-sgd params learning-rate: 1.0))
         (opt-clipped (make-sgd params learning-rate: 1.0)))
    
    ;; Test 1: Without clipping (should explode)
    (let ((x (make-tensor32 (f32vector 10.0 10.0) '(2)))
          (target (make-tensor32 (f32vector 1.0) '(1)))
          (exploded? #f))
      
      (do ((step 0 (+ step 1)))
          ((or (= step 10) exploded?))
        (let* ((output (forward model x))
               (loss (mse-loss output target)))
          (backward! loss)
          
          ;; Check for explosion
          (when (has-exploding-gradients? params threshold: 100.0)
            (set! exploded? #t))
          
          (step! opt-unclipped)
          (zero-grad-layer! model)))
      
      (assert-true exploded? 
                  "High learning rate causes gradient explosion"))
    
    ;; Reset model
    (let ((weight (car params)))
      (f32vector-set! (tensor-data weight) 0 0.0)
      (f32vector-set! (tensor-data weight) 1 0.0))
    
    ;; Test 2: With clipping (should stabilize)
    (let ((x (make-tensor32 (f32vector 10.0 10.0) '(2)))
          (target (make-tensor32 (f32vector 1.0) '(1)))
          (max-norm 0.0))
      
      (do ((step 0 (+ step 1)))
          ((= step 10))
        (let* ((output (forward model x))
               (loss (mse-loss output target)))
          (backward! loss)
          
          ;; Clip gradients
          (clip-gradients! params max-norm: 1.0)
          
          ;; Track max gradient norm
          (let ((norm (compute-gradient-norm params)))
            (set! max-norm (max max-norm norm)))
          
          (step! opt-clipped)
          (zero-grad-layer! model)))
      
      ;; Clipping should keep gradients bounded
      (assert-less-than max-norm 1.1
                       "Gradient clipping keeps norms bounded"))))

;;; ==================================================================
;;; Summary
;;; ==================================================================

;; These tests demonstrate:
;; 1. Using average losses instead of monotonic decrease
;; 2. Gradient health monitoring integration
;; 3. Realistic convergence criteria
;; 4. Proper handling of optimizer oscillations
;; 5. Validation of gradient clipping effectiveness

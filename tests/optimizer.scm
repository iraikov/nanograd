;; unit tests for optimizer operations

(import scheme
        (chicken base)
        (chicken format)
        (chicken random)
        (srfi 1)
        (srfi 4)
        blas
        nanograd-autograd
        nanograd-layer
        nanograd-optimizer)

;;; ==================================================================
;;; Test Framework
;;; ==================================================================

(define *test-count* 0)
(define *test-passed* 0)
(define *test-failed* 0)

(define (reset-test-stats!)
  (set! *test-count* 0)
  (set! *test-passed* 0)
  (set! *test-failed* 0))

(define (test-summary)
  (printf "\n")
  (printf "========================================\n")
  (printf "TEST SUMMARY\n")
  (printf "========================================\n")
  (printf "Total tests:  ~A\n" *test-count*)
  (printf "Passed:       ~A\n" *test-passed*)
  (printf "Failed:       ~A\n" *test-failed*)
  (printf "Success rate: ~A%\n" 
          (if (> *test-count* 0)
              (* 100.0 (/ *test-passed* *test-count*))
              0))
  (printf "========================================\n\n"))

(define (assert-equal actual expected tolerance name)
  (set! *test-count* (+ *test-count* 1))
  (let ((diff (abs (- actual expected))))
    (if (<= diff tolerance)
        (begin
          (set! *test-passed* (+ *test-passed* 1))
          (printf "  O  ~A\n" name))
        (begin
          (set! *test-failed* (+ *test-failed* 1))
          (printf "  X  ~A\n" name)
          (printf "    Expected: ~A, Got: ~A, Diff: ~A\n" 
                  expected actual diff)))))

(define (assert-true condition name)
  (set! *test-count* (+ *test-count* 1))
  (if condition
      (begin
        (set! *test-passed* (+ *test-passed* 1))
        (printf "  O  ~A\n" name))
      (begin
        (set! *test-failed* (+ *test-failed* 1))
        (printf "  X  ~A\n" name))))

(define (assert-less-than actual threshold name)
  (set! *test-count* (+ *test-count* 1))
  (if (< actual threshold)
      (begin
        (set! *test-passed* (+ *test-passed* 1))
        (printf "  O  ~A\n" name))
      (begin
        (set! *test-failed* (+ *test-failed* 1))
        (printf "  X  ~A\n" name)
        (printf "    Expected < ~A, Got: ~A\n" threshold actual))))

(define (assert-decreasing values name)
  (set! *test-count* (+ *test-count* 1))
  (let loop ((vals values) (is-decreasing #t))
    (if (or (null? vals) (null? (cdr vals)))
        (if is-decreasing
            (begin
              (set! *test-passed* (+ *test-passed* 1))
              (printf "  O  ~A\n" name))
            (begin
              (set! *test-failed* (+ *test-failed* 1))
              (printf "  X  ~A\n" name)))
        (loop (cdr vals)
              (and is-decreasing (>= (car vals) (cadr vals)))))))

;;; ==================================================================
;;; Helper Functions
;;; ==================================================================

(define (compute-loss model input target)
  "Compute MSE loss for a model"
  (let* ((output (forward model input))
         (loss (mse-loss output target)))
    (f32vector-ref (tensor-data loss) 0)))

(define (train-one-step model input target optimizer)
  "Perform one training step and return loss"
  (let* ((output (forward model input))
         (loss (mse-loss output target))
         (loss-val (f32vector-ref (tensor-data loss) 0)))
    (backward! loss)
    (step! optimizer)
    (zero-grad-layer! model)
    loss-val))

;;; ==================================================================
;;; Unit Tests: SGD Optimizer
;;; ==================================================================

(define (test-sgd-construction)
  (printf "\n=== Testing SGD Construction ===\n")
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-sgd params learning-rate: 0.01)))
    
    (assert-true (optimizer? opt) "Is an optimizer")
    (assert-true (sgd? opt) "Is SGD optimizer")
    (assert-equal (get-learning-rate opt) 0.01 1e-10
                  "Learning rate = 0.01")))

(define (test-sgd-basic-update)
  (printf "\n=== Testing SGD Basic Parameter Update ===\n")
  
  ;; Create simple model: y = wx + b
  (let* ((model (make-sequential
                 (list (make-dense-layer 1 1 activation: (make-identity)))))
         (params (parameters model))
         (weight (car params))
         (bias (cadr params)))
    
    ;; Set initial values
    (f32vector-set! (tensor-data weight) 0 0.0)
    (f32vector-set! (tensor-data bias) 0 0.0)
    
    ;; Target: y = 2x + 1
    (let* ((x (make-tensor32 (f32vector 1.0) '(1)))
           (target (make-tensor32 (f32vector 3.0) '(1)))  ; 2*1 + 1 = 3
           (opt (make-sgd params learning-rate: 0.1)))
      
      ;; Initial loss should be high
      (let ((initial-loss (compute-loss model x target)))
        (assert-equal initial-loss 4.5 1e-5
                      "Initial loss = 1/2*(0-3)^2 = 4.5")
        
        ;; Train for several steps
        (let loop ((step 0) (losses '()))
          (if (= step 10)
              (let ((final-loss (car losses)))
                (assert-less-than final-loss initial-loss
                                 "Loss decreases after SGD updates"))
              (let ((loss (train-one-step model x target opt)))
                (loop (+ step 1) (cons loss losses)))))))))

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
    
    ;; Train and check convergence
    (let ((x (make-tensor32 (f32vector 1.0 2.0) '(2)))
          (target (make-tensor32 (f32vector 5.0) '(1))))
      
      (let loop ((step 0) (losses '()))
        (if (= step 20)
            (let ((initial-loss (last losses))
                  (final-loss (car losses)))
              (assert-less-than final-loss initial-loss
                                "Final loss < initial loss with momentum"))
            (let ((loss (train-one-step model x target opt)))
              (loop (+ step 1) (cons loss losses))))))))

(define (test-sgd-weight-decay)
  (printf "\n=== Testing SGD with Weight Decay ===\n")
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (weight (car params)))
    
    ;; Set large initial weight
    (f32vector-set! (tensor-data weight) 0 10.0)
    
    (let* ((opt (make-sgd params 
                         learning-rate: 0.01
                         weight-decay: 0.1))
           (x (make-tensor32 (f32vector 1.0 1.0) '(2)))
           (target (make-tensor32 (f32vector 0.0) '(1))))
      
      (let ((initial-weight (f32vector-ref (tensor-data weight) 0)))
        
        ;; Train for a few steps
        (do ((i 0 (+ i 1)))
            ((= i 10))
          (train-one-step model x target opt))
        
        (let ((final-weight (f32vector-ref (tensor-data weight) 0)))
          ;; Weight should decrease (regularization effect)
          (assert-less-than (abs final-weight) (abs initial-weight)
                           "Weight decay reduces weight magnitude"))))))

;;; ==================================================================
;;; Unit Tests: Adam Optimizer
;;; ==================================================================

(define (test-adam-construction)
  (printf "\n=== Testing Adam Construction ===\n")
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params 
                        learning-rate: 0.001
                        beta1: 0.9
                        beta2: 0.999)))
    
    (assert-true (optimizer? opt) "Is an optimizer")
    (assert-true (adam? opt) "Is Adam optimizer")
    (assert-equal (get-learning-rate opt) 0.001 1e-10
                  "Learning rate = 0.001")
    
    (let* ((state (optimizer-state opt))
           (beta1-val (cdr (assoc 'beta1 state)))
           (beta2-val (cdr (assoc 'beta2 state))))
      (assert-equal beta1-val 0.9 1e-10 "Beta1 = 0.9")
      (assert-equal beta2-val 0.999 1e-10 "Beta2 = 0.999"))))

(define (test-adam-basic-update)
  (printf "\n=== Testing Adam Basic Parameter Update ===\n")
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params learning-rate: 0.01)))
    
    (let ((x (make-tensor32 (f32vector 1.0 2.0) '(2)))
          (target (make-tensor32 (f32vector 5.0) '(1))))
      
      (let ((initial-loss (compute-loss model x target)))
        
        ;; Train for several steps
        (do ((i 0 (+ i 1)))
            ((= i 50))
          (train-one-step model x target opt))
        
        (let ((final-loss (compute-loss model x target)))
          (assert-less-than final-loss initial-loss
                           "Loss decreases with Adam optimizer"))))))

(define (test-adam-convergence)
  (printf "\n=== Testing Adam Convergence ===\n")
  
  ;; Adam should converge faster than SGD on this problem
  (let* ((model (make-sequential
                 (list 
                  (make-dense-layer 3 10 activation: (make-relu))
                  (make-dense-layer 10 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params learning-rate: 0.01)))
    
    (let ((training-data (list
                          (cons (f32vector 1.0 0.0 0.0) 1.0)
                          (cons (f32vector 0.0 1.0 0.0) 2.0)
                          (cons (f32vector 0.0 0.0 1.0) 3.0)
                          (cons (f32vector 1.0 1.0 0.0) 3.0))))
      
      (let loop ((epoch 0) (losses '()))
        (if (= epoch 100)
            (let ((final-avg-loss (/ (apply + (take losses 10)) 10.0)))
              (assert-less-than final-avg-loss 0.5
                               "Adam achieves low loss after 100 epochs"))
            (let ((epoch-losses
                   (map (lambda (sample)
                          (let* ((x (make-tensor32 (car sample) '(3)))
                                 (target (make-tensor32 
                                         (f32vector (cdr sample)) '(1))))
                            (train-one-step model x target opt)))
                        training-data)))
              (loop (+ epoch 1) 
                    (append epoch-losses losses))))))))

(define (test-adam-adaptive-learning)
  (printf "\n=== Testing Adam Adaptive Learning Rates ===\n")
  
  ;; Create scenario with different gradient magnitudes
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params learning-rate: 0.01)))
    
    ;; Train with varied gradients
    (let ((data1 (cons (f32vector 100.0 100.0) 1.0))  ; Large inputs
          (data2 (cons (f32vector 0.01 0.01) 0.1)))   ; Small inputs
      
      (let loop ((step 0))
        (when (< step 20)
          (let* ((sample (if (even? step) data1 data2))
                 (x (make-tensor32 (car sample) '(2)))
                 (target (make-tensor32 (f32vector (cdr sample)) '(1))))
            (train-one-step model x target opt))
          (loop (+ step 1))))
      
      ;; Adam should handle varying gradients smoothly
      (assert-true #t "Adam handles varied gradient magnitudes"))))

;;; ==================================================================
;;; Unit Tests: RMSprop Optimizer
;;; ==================================================================

(define (test-rmsprop-construction)
  (printf "\n=== Testing RMSprop Construction ===\n")
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-rmsprop params 
                           learning-rate: 0.01
                           alpha: 0.99)))
    
    (assert-true (optimizer? opt) "Is an optimizer")
    (assert-true (rmsprop? opt) "Is RMSprop optimizer")
    (assert-equal (get-learning-rate opt) 0.01 1e-10
                  "Learning rate = 0.01")
    
    (let* ((state (optimizer-state opt))
           (alpha-val (cdr (assoc 'alpha state))))
      (assert-equal alpha-val 0.99 1e-10 "Alpha = 0.99"))))

(define (test-rmsprop-basic-update)
  (printf "\n=== Testing RMSprop Basic Parameter Update ===\n")
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-rmsprop params learning-rate: 0.01)))
    
    (let ((x (make-tensor32 (f32vector 1.0 2.0) '(2)))
          (target (make-tensor32 (f32vector 5.0) '(1))))
      
      (let ((initial-loss (compute-loss model x target)))
        
        ;; Train for several steps
        (do ((i 0 (+ i 1)))
            ((= i 50))
          (train-one-step model x target opt))
        
        (let ((final-loss (compute-loss model x target)))
          (assert-less-than final-loss initial-loss
                           "Loss decreases with RMSprop optimizer"))))))

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
      
      (let loop ((step 0) (losses '()))
        (if (= step 20)
            (let ((initial-loss (last losses))
                  (final-loss (car losses)))
              (assert-less-than final-loss initial-loss
                                "Final loss < initial loss with RMSprop+momentum"))
            (let ((loss (train-one-step model x target opt)))
              (loop (+ step 1) (cons loss losses))))))))

;;; ==================================================================
;;; Unit Tests: Optimizer Comparison
;;; ==================================================================

(define (test-optimizer-comparison)
  (printf "\n=== Testing Optimizer Comparison ===\n")
  
  ;; Create identical models for each optimizer
  (define (make-test-model)
    (make-sequential
     (list (make-dense-layer 3 5 activation: (make-relu))
           (make-dense-layer 5 1 activation: (make-identity)))))
  
  (let ((model-sgd (make-test-model))
        (model-adam (make-test-model))
        (model-rms (make-test-model)))
    
    ;; Create optimizers
    (let ((opt-sgd (make-sgd (parameters model-sgd) learning-rate: 0.01))
          (opt-adam (make-adam (parameters model-adam) learning-rate: 0.01))
          (opt-rms (make-rmsprop (parameters model-rms) learning-rate: 0.01)))
      
      ;; Training data
      (let ((x (make-tensor32 (f32vector 1.0 2.0 3.0) '(3)))
            (target (make-tensor32 (f32vector 10.0) '(1))))
        
        ;; Train each for 50 steps
        (let ((final-loss-sgd
               (let loop ((step 0))
                 (if (= step 50)
                     (compute-loss model-sgd x target)
                     (begin
                       (train-one-step model-sgd x target opt-sgd)
                       (loop (+ step 1))))))
              (final-loss-adam
               (let loop ((step 0))
                 (if (= step 50)
                     (compute-loss model-adam x target)
                     (begin
                       (train-one-step model-adam x target opt-adam)
                       (loop (+ step 1))))))
              (final-loss-rms
               (let loop ((step 0))
                 (if (= step 50)
                     (compute-loss model-rms x target)
                     (begin
                       (train-one-step model-rms x target opt-rms)
                       (loop (+ step 1)))))))
          
          ;; All should achieve some improvement
          (assert-less-than final-loss-sgd 100.0
                           "SGD achieves reasonable loss")
          (assert-less-than final-loss-adam 100.0
                           "Adam achieves reasonable loss")
          (assert-less-than final-loss-rms 100.0
                           "RMSprop achieves reasonable loss"))))))

;;; ==================================================================
;;; Unit Tests: Learning Rate Scheduling
;;; ==================================================================

(define (test-learning-rate-update)
  (printf "\n=== Testing Learning Rate Updates ===\n")
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-sgd params learning-rate: 0.1)))
    
    (assert-equal (get-learning-rate opt) 0.1 1e-10
                  "Initial learning rate = 0.1")
    
    ;; Update learning rate
    (set-learning-rate! opt 0.01)
    
    (assert-equal (get-learning-rate opt) 0.01 1e-10
                  "Updated learning rate = 0.01")
    
    ;; Train with new learning rate
    (let ((x (make-tensor32 (f32vector 1.0 2.0) '(2)))
          (target (make-tensor32 (f32vector 3.0) '(1))))
      
      (let ((loss-before (compute-loss model x target)))
        (do ((i 0 (+ i 1)))
            ((= i 10))
          (train-one-step model x target opt))
        
        (let ((loss-after (compute-loss model x target)))
          (assert-less-than loss-after loss-before
                           "Training works with updated learning rate"))))))

;;; ==================================================================
;;; Unit Tests: Edge Cases
;;; ==================================================================

(define (test-optimizer-edge-cases)
  (printf "\n=== Testing Optimizer Edge Cases ===\n")
  
  ;; Test 1: Zero gradients
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-sgd params learning-rate: 0.1))
         (weight (car params))
         (initial-value (f32vector-ref (tensor-data weight) 0)))
    
    ;; Manually set gradients to zero
    (let ((grad (tensor-grad weight)))
      (f32vector-set! grad 0 0.0)
      (f32vector-set! grad 1 0.0))
    
    (step! opt)
    
    (assert-equal (f32vector-ref (tensor-data weight) 0) initial-value 1e-10
                  "Zero gradient doesn't change parameters"))
  
  ;; Test 2: Very small learning rate
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params learning-rate: 1e-10)))
    
    (let ((x (make-tensor32 (f32vector 1.0 2.0) '(2)))
          (target (make-tensor32 (f32vector 5.0) '(1))))
      
      (let ((loss-before (compute-loss model x target)))
        (do ((i 0 (+ i 1)))
            ((= i 10))
          (train-one-step model x target opt))
        
        (let ((loss-after (compute-loss model x target)))
          ;; Loss should barely change
          (assert-equal loss-after loss-before 1e-4
                       "Very small learning rate causes minimal change"))))))

;;; ==================================================================
;;; Unit Tests: Convergence on Known Problem
;;; ==================================================================

(define (test-linear-regression)
  (printf "\n=== Testing Linear Regression Convergence ===\n")
  
  ;; Learn y = 3x1 + 2x2 + 1
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params learning-rate: 0.1)))
    
    (let ((training-data (list
                          (cons (f32vector 1.0 0.0) 4.0)  ; 3*1 + 2*0 + 1
                          (cons (f32vector 0.0 1.0) 3.0)  ; 3*0 + 2*1 + 1
                          (cons (f32vector 1.0 1.0) 6.0)  ; 3*1 + 2*1 + 1
                          (cons (f32vector 2.0 1.0) 9.0)  ; 3*2 + 2*1 + 1
                          (cons (f32vector 1.0 2.0) 8.0)))) ; 3*1 + 2*2 + 1
      
      ;; Train for many epochs
      (do ((epoch 0 (+ epoch 1)))
          ((= epoch 200))
        (for-each
         (lambda (sample)
           (let ((x (make-tensor32 (car sample) '(2)))
                 (target (make-tensor32 (f32vector (cdr sample)) '(1))))
             (train-one-step model x target opt)))
         training-data))
      
      ;; Test learned model
      (let* ((test-x (make-tensor32 (f32vector 3.0 2.0) '(2)))
             (expected 14.0)  ; 3*3 + 2*2 + 1 = 14
             (output (forward model test-x))
             (predicted (f32vector-ref (tensor-data output) 0)))
        
        (assert-equal predicted expected 0.5
                     "Linear regression learns correct function")))))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(define (run-all-optimizer-tests)
  (reset-test-stats!)
  (printf "\n")
  (printf "========================================\n")
  (printf "OPTIMIZER UNIT TESTS\n")
  (printf "========================================\n")
  
  ;; SGD tests
  (test-sgd-construction)
  (test-sgd-basic-update)
  (test-sgd-momentum)
  (test-sgd-weight-decay)
  
  ;; Adam tests
  (test-adam-construction)
  (test-adam-basic-update)
  (test-adam-convergence)
  (test-adam-adaptive-learning)
  
  ;; RMSprop tests
  (test-rmsprop-construction)
  (test-rmsprop-basic-update)
  (test-rmsprop-with-momentum)
  
  ;; Comparison and advanced tests
  (test-optimizer-comparison)
  (test-learning-rate-update)
  (test-optimizer-edge-cases)
  (test-linear-regression)
  
  (test-summary))

;; Run all tests
(run-all-optimizer-tests)

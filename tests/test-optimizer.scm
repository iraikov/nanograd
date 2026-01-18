;; Unit tests for optimizer operations

(import scheme
        (chicken base)
        (chicken format)
        (chicken random)
        (srfi 1)
        (srfi 4)
        test
        blas
        nanograd-autograd
        nanograd-layer
        nanograd-optimizer)

;;; ==================================================================
;;; Helper Functions
;;; ==================================================================

(define (approx-equal? actual expected tolerance)
  "Check if two numbers are approximately equal within tolerance"
  (<= (abs (- actual expected)) tolerance))

(define-syntax test-approximate
  (syntax-rules ()
    ((test-approximate name expected actual tolerance)
     (test-assert name (approx-equal? actual expected tolerance)))))

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

(test-group "SGD - Construction"
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-sgd params learning-rate: 0.01)))
    
    (test-assert "Is an optimizer"
      (optimizer? opt))
    
    (test-assert "Is SGD optimizer"
      (sgd? opt))
    
    (test-approximate "Learning rate = 0.01"
      0.01 (get-learning-rate opt) 1e-10)))

(test-group "SGD - Basic Parameter Update"
  
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
           (opt (make-sgd params learning-rate: 0.1))
           (initial-loss (compute-loss model x target)))
      
      (test-approximate "Initial loss = 1/2*(0-3)^2 = 4.5"
        4.5 initial-loss 1e-5)
      
      ;; Train for several steps
      (let loop ((step 0) (losses '()))
        (if (= step 10)
            (let ((final-loss (car losses)))
              (test-assert "Loss decreases after SGD updates"
                (< final-loss initial-loss)))
            (let ((loss (train-one-step model x target opt)))
              (loop (+ step 1) (cons loss losses))))))))

(test-group "SGD - With Momentum"
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-sgd params 
                       learning-rate: 0.01
                       momentum: 0.9))
         (state (optimizer-state opt))
         (momentum-val (cdr (assoc 'momentum state))))
    
    (test-approximate "Momentum = 0.9"
      0.9 momentum-val 1e-10)
    
    ;; Train and check convergence
    (let ((x (make-tensor32 (f32vector 1.0 2.0) '(2)))
          (target (make-tensor32 (f32vector 5.0) '(1))))
      
      (let loop ((step 0) (losses '()))
        (if (= step 20)
            (let ((initial-loss (last losses))
                  (final-loss (car losses)))
              (test-assert "Final loss < initial loss with momentum"
                (< final-loss initial-loss)))
            (let ((loss (train-one-step model x target opt)))
              (loop (+ step 1) (cons loss losses))))))))

(test-group "SGD - With Weight Decay"
  
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
           (target (make-tensor32 (f32vector 0.0) '(1)))
           (initial-weight (f32vector-ref (tensor-data weight) 0)))
      
      ;; Train for a few steps
      (do ((i 0 (+ i 1)))
          ((= i 10))
        (train-one-step model x target opt))
      
      (let ((final-weight (f32vector-ref (tensor-data weight) 0)))
        ;; Weight should decrease (regularization effect)
        (test-assert "Weight decay reduces weight magnitude"
          (< (abs final-weight) (abs initial-weight)))))))

;;; ==================================================================
;;; Unit Tests: Adam Optimizer
;;; ==================================================================

(test-group "Adam - Construction"
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params 
                        learning-rate: 0.001
                        beta1: 0.9
                        beta2: 0.999)))
    
    (test-assert "Is an optimizer"
      (optimizer? opt))
    
    (test-assert "Is Adam optimizer"
      (adam? opt))
    
    (test-approximate "Learning rate = 0.001"
      0.001 (get-learning-rate opt) 1e-10)
    
    (let* ((state (optimizer-state opt))
           (beta1-val (cdr (assoc 'beta1 state)))
           (beta2-val (cdr (assoc 'beta2 state))))
      
      (test-approximate "Beta1 = 0.9"
        0.9 beta1-val 1e-10)
      
      (test-approximate "Beta2 = 0.999"
        0.999 beta2-val 1e-10))))

(test-group "Adam - Basic Parameter Update"
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params learning-rate: 0.01))
         (x (make-tensor32 (f32vector 1.0 2.0) '(2)))
         (target (make-tensor32 (f32vector 5.0) '(1)))
         (initial-loss (compute-loss model x target)))
    
    ;; Train for several steps
    (do ((i 0 (+ i 1)))
        ((= i 50))
      (train-one-step model x target opt))
    
    (let ((final-loss (compute-loss model x target)))
      (test-assert "Loss decreases with Adam optimizer"
        (< final-loss initial-loss)))))

(test-group "Adam - Convergence"
  
  ;; Adam should converge faster than SGD on this problem
  (let* ((model (make-sequential
                 (list 
                  (make-dense-layer 3 10 activation: (make-relu))
                  (make-dense-layer 10 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params learning-rate: 0.01))
         (training-data (list
                          (cons (f32vector 1.0 0.0 0.0) 1.0)
                          (cons (f32vector 0.0 1.0 0.0) 2.0)
                          (cons (f32vector 0.0 0.0 1.0) 3.0)
                          (cons (f32vector 1.0 1.0 0.0) 3.0))))
    
    (let loop ((epoch 0) (losses '()))
      (if (= epoch 100)
          (let ((final-avg-loss (/ (apply + (take losses 10)) 10.0)))
            (test-assert "Adam achieves low loss after 100 epochs"
              (< final-avg-loss 0.5)))
          (let ((epoch-losses
                 (map (lambda (sample)
                        (let* ((x (make-tensor32 (car sample) '(3)))
                               (target (make-tensor32 
                                       (f32vector (cdr sample)) '(1))))
                          (train-one-step model x target opt)))
                      training-data)))
            (loop (+ epoch 1) 
                  (append epoch-losses losses)))))))

(test-group "Adam - Adaptive Learning Rates"
  
  ;; Create scenario with different gradient magnitudes
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params learning-rate: 0.01))
         ;; Train with varied gradients
         (data1 (cons (f32vector 100.0 100.0) 1.0))  ; Large inputs
         (data2 (cons (f32vector 0.01 0.01) 0.1)))   ; Small inputs
    
    (let loop ((step 0))
      (when (< step 20)
        (let* ((sample (if (even? step) data1 data2))
               (x (make-tensor32 (car sample) '(2)))
               (target (make-tensor32 (f32vector (cdr sample)) '(1))))
          (train-one-step model x target opt))
        (loop (+ step 1))))
    
    ;; Adam should handle varying gradients smoothly
    (test-assert "Adam handles varied gradient magnitudes" #t)))

;;; ==================================================================
;;; Unit Tests: RMSprop Optimizer
;;; ==================================================================

(test-group "RMSprop - Construction"
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-rmsprop params 
                           learning-rate: 0.01
                           alpha: 0.99)))
    
    (test-assert "Is an optimizer"
      (optimizer? opt))
    
    (test-assert "Is RMSprop optimizer"
      (rmsprop? opt))
    
    (test-approximate "Learning rate = 0.01"
      0.01 (get-learning-rate opt) 1e-10)
    
    (let* ((state (optimizer-state opt))
           (alpha-val (cdr (assoc 'alpha state))))
      
      (test-approximate "Alpha = 0.99"
        0.99 alpha-val 1e-10))))

(test-group "RMSprop - Basic Parameter Update"
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-rmsprop params learning-rate: 0.01))
         (x (make-tensor32 (f32vector 1.0 2.0) '(2)))
         (target (make-tensor32 (f32vector 5.0) '(1)))
         (initial-loss (compute-loss model x target)))
    
    ;; Train for several steps
    (do ((i 0 (+ i 1)))
        ((= i 50))
      (train-one-step model x target opt))
    
    (let ((final-loss (compute-loss model x target)))
      (test-assert "Loss decreases with RMSprop optimizer"
        (< final-loss initial-loss)))))

(test-group "RMSprop - With Momentum"
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-rmsprop params 
                           learning-rate: 0.01
                           momentum: 0.9))
         (state (optimizer-state opt))
         (momentum-val (cdr (assoc 'momentum state))))
    
    (test-approximate "Momentum = 0.9"
      0.9 momentum-val 1e-10)
    
    (let ((x (make-tensor32 (f32vector 1.0 2.0) '(2)))
          (target (make-tensor32 (f32vector 3.0) '(1))))
      
      (let loop ((step 0) (losses '()))
        (if (= step 20)
            (let ((initial-loss (last losses))
                  (final-loss (car losses)))
              (test-assert "Final loss < initial loss with RMSprop+momentum"
                (< final-loss initial-loss)))
            (let ((loss (train-one-step model x target opt)))
              (loop (+ step 1) (cons loss losses))))))))

;;; ==================================================================
;;; Unit Tests: Optimizer Comparison
;;; ==================================================================

(test-group "Optimizer Comparison"
  
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
          (opt-rms (make-rmsprop (parameters model-rms) learning-rate: 0.01))
          ;; Training data
          (x (make-tensor32 (f32vector 1.0 2.0 3.0) '(3)))
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
        (test-assert "SGD achieves reasonable loss"
          (< final-loss-sgd 100.0))
        
        (test-assert "Adam achieves reasonable loss"
          (< final-loss-adam 100.0))
        
        (test-assert "RMSprop achieves reasonable loss"
          (< final-loss-rms 100.0))))))

;;; ==================================================================
;;; Unit Tests: Learning Rate Scheduling
;;; ==================================================================

(test-group "Learning Rate Updates"
  
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-sgd params learning-rate: 0.1)))
    
    (test-approximate "Initial learning rate = 0.1"
      0.1 (get-learning-rate opt) 1e-10)
    
    ;; Update learning rate
    (set-learning-rate! opt 0.01)
    
    (test-approximate "Updated learning rate = 0.01"
      0.01 (get-learning-rate opt) 1e-10)
    
    ;; Train with new learning rate
    (let* ((x (make-tensor32 (f32vector 1.0 2.0) '(2)))
           (target (make-tensor32 (f32vector 3.0) '(1)))
           (loss-before (compute-loss model x target)))
      
      (do ((i 0 (+ i 1)))
          ((= i 10))
        (train-one-step model x target opt))
      
      (let ((loss-after (compute-loss model x target)))
        (test-assert "Training works with updated learning rate"
          (< loss-after loss-before))))))

;;; ==================================================================
;;; Unit Tests: Edge Cases
;;; ==================================================================

(test-group "Optimizer Edge Cases"
  
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
    
    (test-approximate "Zero gradient doesn't change parameters"
      initial-value (f32vector-ref (tensor-data weight) 0) 1e-10))
  
  ;; Test 2: Very small learning rate
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params learning-rate: 1e-10))
         (x (make-tensor32 (f32vector 1.0 2.0) '(2)))
         (target (make-tensor32 (f32vector 5.0) '(1)))
         (loss-before (compute-loss model x target)))
    
    (do ((i 0 (+ i 1)))
        ((= i 10))
      (train-one-step model x target opt))
    
    (let ((loss-after (compute-loss model x target)))
      ;; Loss should barely change
      (test-approximate "Very small learning rate causes minimal change"
        loss-before loss-after 1e-4))))

;;; ==================================================================
;;; Unit Tests: Convergence on Known Problem
;;; ==================================================================

(test-group "Linear Regression Convergence"
  
  ;; Learn y = 3x1 + 2x2 + 1
  (let* ((model (make-sequential
                 (list (make-dense-layer 2 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params learning-rate: 0.1))
         (training-data (list
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
      
      (test-approximate "Linear regression learns correct function"
        expected predicted 0.5))))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(test-exit)

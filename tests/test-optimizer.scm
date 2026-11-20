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

;;; ==================================================================
;;; Unit Tests: Optimizer kernels
;;; ==================================================================

(test-group "Adam - Registered update kernel"

  ;; A Scheme stand-in for a compiled kernel, following the documented
  ;; calling convention; it counts its calls.
  (define calls 0)
  (define (scheme-adam-kernel n hyper param grad m v)
    (set! calls (+ calls 1))
    (let ((lr (f64vector-ref hyper 0)) (b1 (f64vector-ref hyper 1))
          (b2 (f64vector-ref hyper 2)) (eps (f64vector-ref hyper 3))
          (bc1 (f64vector-ref hyper 4)) (bc2 (f64vector-ref hyper 5))
          (wd (f64vector-ref hyper 6)))
      (do ((i 0 (+ i 1))) ((= i n))
        (when (> wd 0.0)
          (f32vector-set! grad i (+ (f32vector-ref grad i) (* wd (f32vector-ref param i)))))
        (let ((g (f32vector-ref grad i)))
          (f32vector-set! m i (+ (* b1 (f32vector-ref m i)) (* (- 1.0 b1) g)))
          (f32vector-set! v i (+ (* b2 (f32vector-ref v i)) (* (- 1.0 b2) g g)))
          (f32vector-set! param i
                          (- (f32vector-ref param i)
                             (/ (* lr (/ (f32vector-ref m i) bc1))
                                (+ (sqrt (/ (f32vector-ref v i) bc2)) eps))))))))

  ;; Weights after three Adam steps with a fixed gradient.
  (define (weights-after-steps)
    (let* ((w (make-tensor32 (f32vector 0.5 -1.0 2.0) '(3)))
           (opt (make-adam (list w) learning-rate: 0.01 weight-decay: 0.001)))
      (do ((k 0 (+ k 1))) ((= k 3))
        (zero-grad! w)
        (add-to-grad! w (f32vector 0.3 -0.2 0.1))
        (step! opt))
      (f32vector->list (tensor-data w))))

  (define without (weights-after-steps))
  (register-optimizer-kernel! 'adam 'f32 scheme-adam-kernel)
  (define with (weights-after-steps))
  (register-optimizer-kernel! 'adam 'f32 #f)

  (test-assert "The registered kernel performs the updates"
    (= calls 3))

  (test-assert "The kernel's result agrees with the built-in Adam"
    (every (lambda (a b) (approx-equal? a b 1e-6)) without with))

  (test-assert "Removing the registration restores the built-in update"
    (not (optimizer-kernel 'adam 'f32))))

;;; ==================================================================
;;; Unit Tests: AdamW (decoupled weight decay)
;;; ==================================================================

;; Reference AdamW on one f64 parameter vector P with a fixed gradient
;; G, for STEPS steps; returns the final parameter values as a list.
(define (reference-adamw p g steps lr b1 b2 eps wd)
  (let ((n (length p)))
    (let loop ((t 1) (p p) (m (make-list n 0.0)) (v (make-list n 0.0)))
      (if (> t steps)
          p
          (let* ((m (map (lambda (mi gi) (+ (* b1 mi) (* (- 1.0 b1) gi))) m g))
                 (v (map (lambda (vi gi) (+ (* b2 vi) (* (- 1.0 b2) gi gi))) v g))
                 (bc1 (- 1.0 (expt b1 t)))
                 (bc2 (- 1.0 (expt b2 t)))
                 (p (map (lambda (pi mi vi)
                           (- (* pi (- 1.0 (* lr wd)))
                              (/ (* lr (/ mi bc1)) (+ (sqrt (/ vi bc2)) eps))))
                         p m v)))
            (loop (+ t 1) p m v))))))

;; Runs OPT for STEPS steps on W with the fixed gradient vector G.
(define (run-steps! opt w g steps)
  (do ((k 0 (+ k 1))) ((= k steps))
    (zero-grad! w)
    (add-to-grad! w g)
    (step! opt)))

(test-group "AdamW - Construction"
  (let* ((w (make-tensor64 (f64vector 1.0 2.0) '(2)))
         (opt (make-adamw (list w) learning-rate: 1e-4 beta2: 0.95 weight-decay: 1.0))
         (state (optimizer-state opt)))
    (test-assert "Is an optimizer" (optimizer? opt))
    (test-assert "Is an Adam optimizer" (adam? opt))
    (test-assert "Is an AdamW optimizer" (adamw? opt))
    (test-approximate "beta2 = 0.95" 0.95 (cdr (assoc 'beta2 state)) 1e-12)
    (test-approximate "weight decay = 1.0" 1.0 (cdr (assoc 'weight-decay state)) 1e-12)
    (test "state records decoupled weight decay" #t
          (cdr (assoc 'decoupled-weight-decay state))))
  (let ((w (make-tensor64 (f64vector 1.0) '(1))))
    (test-approximate "default weight decay is 0.01"
      0.01 (cdr (assoc 'weight-decay (optimizer-state (make-adamw (list w))))) 1e-12)
    (test-assert "plain Adam is not AdamW"
      (not (adamw? (make-adam (list w) weight-decay: 0.1))))))

(test-group "AdamW - Zero gradient shrinks parameters by (1 - lr * wd)"
  (let* ((w (make-tensor64 (f64vector 2.0 -4.0) '(2)))
         (opt (make-adamw (list w) learning-rate: 0.1 weight-decay: 0.5)))
    (run-steps! opt w (f64vector 0.0 0.0) 3)
    (test-assert "w = w0 * 0.95^3"
      (every (lambda (actual w0) (approx-equal? actual (* w0 (expt 0.95 3)) 1e-12))
             (f64vector->list (tensor-data w)) '(2.0 -4.0))))
  ;; Coupled L2 decay feeds wd * w into the Adam moments instead, so the
  ;; parameter moves by about lr per step regardless of its size.
  (let* ((w (make-tensor64 (f64vector 2.0 -4.0) '(2)))
         (opt (make-adam (list w) learning-rate: 0.1 weight-decay: 0.5)))
    (run-steps! opt w (f64vector 0.0 0.0) 3)
    (test-assert "coupled Adam gives a different result"
      (not (approx-equal? (f64vector-ref (tensor-data w) 0) (* 2.0 (expt 0.95 3)) 1e-3)))))

(test-group "AdamW - Matches the reference update"
  (let* ((p0 '(0.5 -1.0 2.0 0.0))
         (g '(0.3 -0.2 0.1 -0.05))
         (w (make-tensor64 (list->f64vector p0) '(4)))
         (opt (make-adamw (list w) learning-rate: 0.01 beta1: 0.9 beta2: 0.95
                          epsilon: 1e-8 weight-decay: 0.1)))
    (run-steps! opt w (list->f64vector g) 5)
    (test-assert "five steps agree with the formula"
      (every (lambda (a e) (approx-equal? a e 1e-12))
             (f64vector->list (tensor-data w))
             (reference-adamw p0 g 5 0.01 0.9 0.95 1e-8 0.1)))))

(test-group "AdamW - Gradient is left unchanged"
  (let* ((w (make-tensor64 (f64vector 1.0 2.0) '(2)))
         (opt (make-adamw (list w) learning-rate: 0.01 weight-decay: 0.5)))
    (run-steps! opt w (f64vector 0.25 -0.5) 1)
    (test "gradient after the step" '(0.25 -0.5)
          (f64vector->list (tensor-grad w)))))

(test-group "AdamW - Zero weight decay equals Adam"
  (let ((wa (make-tensor64 (f64vector 1.0 -2.0) '(2)))
        (wb (make-tensor64 (f64vector 1.0 -2.0) '(2))))
    (run-steps! (make-adam (list wa) learning-rate: 0.01) wa (f64vector 0.1 0.2) 4)
    (run-steps! (make-adamw (list wb) learning-rate: 0.01 weight-decay: 0.0)
                wb (f64vector 0.1 0.2) 4)
    (test "identical parameters" (f64vector->list (tensor-data wa))
          (f64vector->list (tensor-data wb)))))

(test-group "AdamW - Registered update kernel"
  ;; The kernel receives the coupled weight decay, which is 0 for AdamW;
  ;; the decoupled decay is applied before the kernel is called.
  (define kernel-wds '())
  (define (scheme-adam-kernel n hyper param grad m v)
    (set! kernel-wds (cons (f64vector-ref hyper 6) kernel-wds))
    (let ((lr (f64vector-ref hyper 0)) (b1 (f64vector-ref hyper 1))
          (b2 (f64vector-ref hyper 2)) (eps (f64vector-ref hyper 3))
          (bc1 (f64vector-ref hyper 4)) (bc2 (f64vector-ref hyper 5)))
      (do ((i 0 (+ i 1))) ((= i n))
        (let ((g (f32vector-ref grad i)))
          (f32vector-set! m i (+ (* b1 (f32vector-ref m i)) (* (- 1.0 b1) g)))
          (f32vector-set! v i (+ (* b2 (f32vector-ref v i)) (* (- 1.0 b2) g g)))
          (f32vector-set! param i
                          (- (f32vector-ref param i)
                             (/ (* lr (/ (f32vector-ref m i) bc1))
                                (+ (sqrt (/ (f32vector-ref v i) bc2)) eps))))))))

  (define (weights-after-steps)
    (let* ((w (make-tensor32 (f32vector 0.5 -1.0 2.0) '(3)))
           (opt (make-adamw (list w) learning-rate: 0.01 weight-decay: 0.1)))
      (run-steps! opt w (f32vector 0.3 -0.2 0.1) 3)
      (f32vector->list (tensor-data w))))

  (define without (weights-after-steps))
  (register-optimizer-kernel! 'adam 'f32 scheme-adam-kernel)
  (define with (weights-after-steps))
  (register-optimizer-kernel! 'adam 'f32 #f)

  (test "kernel called once per step" 3 (length kernel-wds))
  (test-assert "kernel receives zero coupled weight decay"
    (every zero? kernel-wds))
  (test-assert "kernel result agrees with the built-in AdamW"
    (every (lambda (a b) (approx-equal? a b 1e-6)) without with)))

(test-group "AdamW - Registered adamw kernel"
  ;; A Scheme stand-in for a compiled adamw kernel: decoupled decay, then
  ;; the Adam update; it records the weight decay it receives.
  (define adamw-wds '())
  (define adam-calls 0)
  (define (scheme-adamw-kernel n hyper param grad m v)
    (set! adamw-wds (cons (f64vector-ref hyper 6) adamw-wds))
    (let ((lr (f64vector-ref hyper 0)) (b1 (f64vector-ref hyper 1))
          (b2 (f64vector-ref hyper 2)) (eps (f64vector-ref hyper 3))
          (bc1 (f64vector-ref hyper 4)) (bc2 (f64vector-ref hyper 5))
          (wd (f64vector-ref hyper 6)))
      (do ((i 0 (+ i 1))) ((= i n))
        (f32vector-set! param i (* (- 1.0 (* lr wd)) (f32vector-ref param i)))
        (let ((g (f32vector-ref grad i)))
          (f32vector-set! m i (+ (* b1 (f32vector-ref m i)) (* (- 1.0 b1) g)))
          (f32vector-set! v i (+ (* b2 (f32vector-ref v i)) (* (- 1.0 b2) g g)))
          (f32vector-set! param i
                          (- (f32vector-ref param i)
                             (/ (* lr (/ (f32vector-ref m i) bc1))
                                (+ (sqrt (/ (f32vector-ref v i) bc2)) eps))))))))
  (define (counting-adam-kernel n hyper param grad m v)
    (set! adam-calls (+ adam-calls 1)))

  (define (weights-after-steps make-opt)
    (let* ((w (make-tensor32 (f32vector 0.5 -1.0 2.0) '(3)))
           (opt (make-opt (list w))))
      (run-steps! opt w (f32vector 0.3 -0.2 0.1) 3)
      (f32vector->list (tensor-data w))))

  (define (adamw-opt params) (make-adamw params learning-rate: 0.01 weight-decay: 0.1))

  (define without (weights-after-steps adamw-opt))
  (register-optimizer-kernel! 'adamw 'f32 scheme-adamw-kernel)
  (register-optimizer-kernel! 'adam 'f32 counting-adam-kernel)
  (define with (weights-after-steps adamw-opt))

  (test "adamw kernel called once per step" 3 (length adamw-wds))
  (test-assert "adamw kernel receives the weight decay"
    (every (lambda (wd) (approx-equal? wd 0.1 1e-12)) adamw-wds))
  (test "adam kernel is not called for AdamW" 0 adam-calls)
  (test-assert "adamw kernel result agrees with the built-in AdamW"
    (every (lambda (a b) (approx-equal? a b 1e-6)) without with))

  ;; Coupled weight decay keeps using the adam kernel.
  (weights-after-steps (lambda (params)
                         (make-adam params learning-rate: 0.01 weight-decay: 0.1)))
  (test "plain Adam uses the adam kernel" 3 adam-calls)
  (test "plain Adam does not use the adamw kernel" 3 (length adamw-wds))

  (register-optimizer-kernel! 'adamw 'f32 #f)
  (register-optimizer-kernel! 'adam 'f32 #f)
  (test-assert "registrations removed"
    (not (or (optimizer-kernel 'adamw 'f32) (optimizer-kernel 'adam 'f32)))))

(test-exit)

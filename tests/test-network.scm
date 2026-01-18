;; Unit tests for complete neural network training

(import scheme
        (chicken base)
        (chicken format)
        (chicken random)
        (srfi 1)
        (srfi 4)
        test
        nanograd-autograd
        nanograd-layer
        nanograd-optimizer
        nanograd-diagnostics)

;;; ==================================================================
;;; Helper Functions
;;; ==================================================================

(define (approx-equal? actual expected tolerance)
  "Check if two numbers are approximately equal within tolerance"
  (<= (abs (- actual expected)) tolerance))

(define (argmax vec)
  "Find index of maximum value in f32vector"
  (let loop ((i 1) (max-i 0) (max-val (f32vector-ref vec 0)))
    (if (= i (f32vector-length vec))
        max-i
        (let ((val (f32vector-ref vec i)))
          (if (> val max-val)
              (loop (+ i 1) i val)
              (loop (+ i 1) max-i max-val))))))

(define (set-random-seed! seed)
  "Set random seed for reproducibility"
  (set-pseudo-random-seed! (number->string seed)))

(define (in-range? val min-val max-val)
  "Check if value is in range [min-val, max-val]"
  (and (>= val min-val) (<= val max-val)))

(define-syntax test-approximate
  (syntax-rules ()
    ((test-approximate name expected actual tolerance)
     (test-assert name (approx-equal? actual expected tolerance)))))

;;; ==================================================================
;;; Linear Regression with SGD
;;; ==================================================================

(test-group "Linear Regression with SGD"
  
  (set-random-seed! 42)
  
  ;; Generate deterministic data: y = 3x + 2
  (define training-data
    (list
     (cons 0.0 2.0)
     (cons 1.0 5.0)
     (cons -1.0 -1.0)
     (cons 2.0 8.0)
     (cons -2.0 -4.0)
     (cons 0.5 3.5)
     (cons -0.5 0.5)))
  
  ;; Create model
  (define model
    (make-sequential
     (list
      (make-dense-layer 1 1 activation: (make-identity)))
     name: "LinearRegression"))
  
  ;; Initialize parameters
  (let ((params (parameters model)))
    (f32vector-set! (tensor-data (car params)) 0 0.5)
    (f32vector-set! (tensor-data (cadr params)) 0 0.5))
  
  ;; Create optimizer
  (define optimizer (make-sgd (parameters model) learning-rate: 0.1))
  
  ;; Train
  (do ((epoch 1 (+ epoch 1)))
      ((> epoch 50))
    
    (for-each
     (lambda (sample)
       (let* ((x (make-tensor32 (f32vector (car sample)) '(1)))
              (target (make-tensor32 (f32vector (cdr sample)) '(1)))
              (pred (forward model x))
              (loss (mse-loss pred target)))
         
         (backward! loss)
         (step! optimizer)
         (zero-grad-layer! model)))
     training-data))
  
  ;; Test learned parameters
  (let* ((params (parameters model))
         (weight (f32vector-ref (tensor-data (car params)) 0))
         (bias (f32vector-ref (tensor-data (cadr params)) 0)))
    
    (test-assert "Weight converges near 3.0"
      (in-range? weight 2.5 3.5))
    
    (test-assert "Bias converges near 2.0"
      (in-range? bias 1.5 2.5))
    
    ;; Test prediction
    (let* ((test-x (make-tensor32 (f32vector 3.0) '(1)))
           (pred (forward model test-x))
           (pred-val (f32vector-ref (tensor-data pred) 0)))
      
      (test-assert "Prediction for x=3 is near 11.0"
        (in-range? pred-val 10.0 12.0)))))

;;; ==================================================================
;;; Linear Regression Optimizer Comparison
;;; ==================================================================

(test-group "Optimizer Comparison"
  
  (define data
    (list (cons 0.0 1.0)
          (cons 1.0 3.0)
          (cons 2.0 5.0)
          (cons 3.0 7.0)))
  
  (define (test-optimizer name make-opt max-loss)
    (set-random-seed! 42)
    
    (let ((model (make-sequential
                  (list (make-dense-layer 1 1 activation: (make-identity))))))
      
      ;; Initialize
      (f32vector-set! (tensor-data (car (parameters model))) 0 0.0)
      (f32vector-set! (tensor-data (cadr (parameters model))) 0 0.0)
      
      (let ((opt (make-opt (parameters model))))
        
        ;; Train
        (do ((epoch 0 (+ epoch 1)))
            ((= epoch 30))
          (for-each
           (lambda (sample)
             (let* ((x (make-tensor32 (f32vector (car sample)) '(1)))
                    (y (make-tensor32 (f32vector (cdr sample)) '(1)))
                    (pred (forward model x))
                    (loss (mse-loss pred y)))
               (backward! loss)
               (step! opt)
               (zero-grad-layer! model)))
           data))
        
        ;; Check final loss
        (let ((total-loss 0.0))
          (for-each
           (lambda (sample)
             (let* ((x (make-tensor32 (f32vector (car sample)) '(1)))
                    (y (make-tensor32 (f32vector (cdr sample)) '(1)))
                    (pred (forward model x))
                    (loss (mse-loss pred y)))
               (set! total-loss (+ total-loss 
                                  (f32vector-ref (tensor-data loss) 0)))))
           data)
          
          (let ((avg-loss (/ total-loss (length data))))
            (test-assert (format #f "~A achieves low loss (< ~A)" name max-loss)
              (< avg-loss max-loss)))))))
  
  (test-optimizer "SGD" 
                 (lambda (p) (make-sgd p learning-rate: 0.1))
                 0.5)
  
  (test-optimizer "Adam"
                 (lambda (p) (make-adam p learning-rate: 0.1))
                 0.1)
  
  (test-optimizer "RMSprop"
                 (lambda (p) (make-rmsprop p learning-rate: 0.1))
                 0.1))

;;; ==================================================================
;;; Binary Classification
;;; ==================================================================

(test-group "Binary Classification"
  
  (set-random-seed! 123)
  
  ;; Create well-separated dataset
  (define training-data
    (append
     ;; Class 0: points at (0, 0)
     (list
      (cons (f32vector 0.0 0.0) (f32vector 1.0 0.0))
      (cons (f32vector 0.1 0.0) (f32vector 1.0 0.0))
      (cons (f32vector 0.0 0.1) (f32vector 1.0 0.0))
      (cons (f32vector -0.1 0.0) (f32vector 1.0 0.0))
      (cons (f32vector 0.0 -0.1) (f32vector 1.0 0.0)))
     ;; Class 1: points at (1, 1)
     (list
      (cons (f32vector 1.0 1.0) (f32vector 0.0 1.0))
      (cons (f32vector 0.9 1.0) (f32vector 0.0 1.0))
      (cons (f32vector 1.0 0.9) (f32vector 0.0 1.0))
      (cons (f32vector 1.1 1.0) (f32vector 0.0 1.0))
      (cons (f32vector 1.0 1.1) (f32vector 0.0 1.0)))))
  
  ;; Create model
  (define model
    (make-sequential
     (list
      (make-dense-layer 2 8 activation: (make-relu))
      (make-dense-layer 8 2 activation: (make-sigmoid)))
     name: "BinaryClassifier"))
  
  (define optimizer (make-adam (parameters model) learning-rate: 0.01))
  
  ;; Train
  (do ((epoch 1 (+ epoch 1)))
      ((> epoch 100))
    
    (for-each
     (lambda (sample)
       (let* ((x (make-tensor32 (car sample) '(2)))
              (target (make-tensor32 (cdr sample) '(2)))
              (pred (forward model x))
              (loss (mse-loss pred target)))
         
         (backward! loss)
         (step! optimizer)
         (zero-grad-layer! model)))
     training-data))
  
  ;; Test accuracy
  (let ((correct 0))
    (for-each
     (lambda (sample)
       (let* ((x (make-tensor32 (car sample) '(2)))
              (pred (forward model x))
              (pred-data (tensor-data pred))
              (target-data (cdr sample))
              (pred-class (if (> (f32vector-ref pred-data 0)
                                (f32vector-ref pred-data 1))
                              0 1))
              (true-class (if (> (f32vector-ref target-data 0)
                                (f32vector-ref target-data 1))
                              0 1)))
         (when (= pred-class true-class)
           (set! correct (+ correct 1)))))
     training-data)
    
    (let ((accuracy (* 100.0 (/ correct (length training-data)))))
      (test-assert "Binary classification achieves >80% accuracy"
        (> accuracy 80.0))
      (printf "    Final accuracy: ~A%\n" accuracy))))

;;; ==================================================================
;;; Multi-class Classification
;;; ==================================================================

(test-group "Multi-class Classification"
  
  (set-random-seed! 456)
  
  ;; Create 3-class dataset
  (define training-data
    (append
     ;; Class 0: bottom-left
     (list
      (cons (f32vector -1.0 -1.0) 0)
      (cons (f32vector -0.9 -1.0) 0)
      (cons (f32vector -1.0 -0.9) 0)
      (cons (f32vector -0.8 -0.8) 0))
     ;; Class 1: bottom-right
     (list
      (cons (f32vector 1.0 -1.0) 1)
      (cons (f32vector 0.9 -1.0) 1)
      (cons (f32vector 1.0 -0.9) 1)
      (cons (f32vector 0.8 -0.8) 1))
     ;; Class 2: top-center
     (list
      (cons (f32vector 0.0 1.0) 2)
      (cons (f32vector 0.1 1.0) 2)
      (cons (f32vector -0.1 0.9) 2)
      (cons (f32vector 0.0 0.8) 2))))
  
  ;; Create model
  (define model
    (make-sequential
     (list
      (make-dense-layer 2 16 activation: (make-relu))
      (make-dense-layer 16 8 activation: (make-relu))
      (make-dense-layer 8 3 activation: (make-identity)))
     name: "MultiClassifier"))
  
  (define optimizer (make-adam (parameters model) learning-rate: 0.01))
  
  ;; Train
  (do ((epoch 1 (+ epoch 1)))
      ((> epoch 100))
    
    (for-each
     (lambda (sample)
       (let* ((x (make-tensor32 (car sample) '(2)))
              (target-class (cdr sample))
              (target-vec (make-f32vector 3 0.0)))
         (f32vector-set! target-vec target-class 1.0)
         (let* ((target (make-tensor32 target-vec '(3)))
                (logits (forward model x))
                (probs (softmax logits))
                (loss (cross-entropy-loss probs target)))
           (backward! loss)
           (step! optimizer)
           (zero-grad-layer! model))))
     training-data))
  
  ;; Test accuracy
  (let ((correct 0))
    (for-each
     (lambda (sample)
       (let* ((x (make-tensor32 (car sample) '(2)))
              (logits (forward model x))
              (pred-class (argmax (tensor-data logits)))
              (true-class (cdr sample)))
         (when (= pred-class true-class)
           (set! correct (+ correct 1)))))
     training-data)
    
    (let ((accuracy (* 100.0 (/ correct (length training-data)))))
      (test-assert "Multi-class achieves >70% accuracy"
        (> accuracy 70.0))
      (printf "    Final accuracy: ~A%\n" accuracy))))

;;; ==================================================================
;;; Learning Rate Decay
;;; ==================================================================

(test-group "Learning Rate Decay"
  
  (set-random-seed! 789)
  
  (define data
    (list (cons 0.0 1.0)
          (cons 1.0 2.0)
          (cons 2.0 3.0)))
  
  (define model
    (make-sequential
     (list (make-dense-layer 1 1 activation: (make-identity)))))
  
  (define optimizer (make-sgd (parameters model) learning-rate: 1.0))
  
  (define initial-lr (get-learning-rate optimizer))
  
  ;; Train with decay
  (do ((epoch 1 (+ epoch 1)))
      ((> epoch 20))
    
    ;; Decay learning rate
    (let ((decay 0.1))
      (set-learning-rate! optimizer 
                         (/ 1.0 (+ 1.0 (* decay epoch)))))
    
    (for-each
     (lambda (sample)
       (let* ((x (make-tensor32 (f32vector (car sample)) '(1)))
              (target (make-tensor32 (f32vector (cdr sample)) '(1)))
              (pred (forward model x))
              (loss (mse-loss pred target)))
         
         (backward! loss)
         (step! optimizer)
         (zero-grad-layer! model)))
     data))
  
  (let ((final-lr (get-learning-rate optimizer)))
    
    (test-assert "Learning rate decreased during training"
      (< final-lr initial-lr))
    
    (let ((expected-final-lr (/ 1.0 (+ 1.0 (* 0.1 20)))))
      (test-approximate "Learning rate matches decay formula"
        expected-final-lr
        final-lr
        0.01))))

;;; ==================================================================
;;; Batch Training vs Sequential
;;; ==================================================================

(test-group "Batch Training"
  
  (set-random-seed! 101)
  
  (define data
    (list (cons (f32vector 0.0 0.0) 0.0)
          (cons (f32vector 1.0 0.0) 1.0)
          (cons (f32vector 0.0 1.0) 1.0)
          (cons (f32vector 1.0 1.0) 0.0)))
  
  (define model-batch
    (make-sequential
     (list
      (make-dense-layer 2 4 activation: (make-relu))
      (make-dense-layer 4 1 activation: (make-sigmoid)))))
  
  (define opt-batch (make-adam (parameters model-batch) learning-rate: 0.1))
  
  ;; Train with batches
  (do ((epoch 0 (+ epoch 1)))
      ((= epoch 50))
    
    ;; Accumulate gradients
    (for-each
     (lambda (sample)
       (let* ((x (make-tensor32 (car sample) '(2)))
              (target (make-tensor32 (f32vector (cdr sample)) '(1)))
              (pred (forward model-batch x))
              (loss (mse-loss pred target)))
         (backward! loss)))
     data)
    
    ;; Update once per epoch
    (step! opt-batch)
    (zero-grad-layer! model-batch))
  
  ;; Test final loss
  (let ((total-loss 0.0))
    (for-each
     (lambda (sample)
       (let* ((x (make-tensor32 (car sample) '(2)))
              (target (make-tensor32 (f32vector (cdr sample)) '(1)))
              (pred (forward model-batch x))
              (loss (mse-loss pred target)))
         (set! total-loss (+ total-loss 
                            (f32vector-ref (tensor-data loss) 0)))))
     data)
    
    (let ((avg-loss (/ total-loss (length data))))
      (test-assert "Batch training achieves low loss on XOR"
        (< avg-loss 0.3)))))

;;; ==================================================================
;;; Overfitting Detection
;;; ==================================================================

(test-group "Overfitting Detection"
  
  (set-random-seed! 202)
  
  (define train-data
    (list (cons (f32vector 0.0) 0.0)
          (cons (f32vector 1.0) 1.0)))
  
  (define test-data
    (list (cons (f32vector 0.5) 0.5)
          (cons (f32vector 1.5) 1.5)))
  
  ;; Very large model
  (define model
    (make-sequential
     (list
      (make-dense-layer 1 20 activation: (make-relu))
      (make-dense-layer 20 20 activation: (make-relu))
      (make-dense-layer 20 1 activation: (make-identity)))))
  
  (define optimizer (make-adam (parameters model) learning-rate: 0.01))
  
  ;; Train extensively
  (do ((epoch 0 (+ epoch 1)))
      ((= epoch 200))
    
    (for-each
     (lambda (sample)
       (let* ((x (make-tensor32 (car sample) '(1)))
              (target (make-tensor32 (f32vector (cdr sample)) '(1)))
              (pred (forward model x))
              (loss (mse-loss pred target)))
         (backward! loss)
         (step! optimizer)
         (zero-grad-layer! model)))
     train-data))
  
  ;; Measure loss
  (define (compute-loss dataset)
    (let ((total 0.0))
      (for-each
       (lambda (sample)
         (let* ((x (make-tensor32 (car sample) '(1)))
                (target (make-tensor32 (f32vector (cdr sample)) '(1)))
                (pred (forward model x))
                (loss (mse-loss pred target)))
           (set! total (+ total (f32vector-ref (tensor-data loss) 0)))))
       dataset)
      (/ total (length dataset))))
  
  (let ((train-loss (compute-loss train-data))
        (test-loss (compute-loss test-data)))
    
    (test-assert "Training loss very low (overfitting)"
      (< train-loss 0.01))
    
    (printf "    Train loss: ~A, Test loss: ~A\n" train-loss test-loss)
    (printf "    Gap indicates ~A\n" 
            (if (> test-loss (* 2.0 train-loss))
                "potential overfitting"
                "good generalization"))))

;;; ==================================================================
;;; Convergence Speed Comparison
;;; ==================================================================

(test-group "Optimizer Convergence Speed"
  
  (define data
    (list (cons 0.0 0.0)
          (cons 1.0 2.0)
          (cons 2.0 4.0)))
  
  (define (test-optimizer name make-opt)
    (set-random-seed! 42)
    
    (let ((model (make-sequential
                  (list (make-dense-layer 1 1 activation: (make-identity)))))
          (epochs-to-converge 0)
          (target-loss 0.01))
      
      (f32vector-set! (tensor-data (car (parameters model))) 0 0.0)
      (f32vector-set! (tensor-data (cadr (parameters model))) 0 0.0)
      
      (let ((opt (make-opt (parameters model))))
        
        (let loop ((epoch 0))
          (when (< epoch 100)
            ;; Train one epoch
            (for-each
             (lambda (sample)
               (let* ((x (make-tensor32 (f32vector (car sample)) '(1)))
                      (y (make-tensor32 (f32vector (cdr sample)) '(1)))
                      (pred (forward model x))
                      (loss (mse-loss pred y)))
                 (backward! loss)
                 (step! opt)
                 (zero-grad-layer! model)))
             data)
            
            ;; Check convergence
            (let ((total-loss 0.0))
              (for-each
               (lambda (sample)
                 (let* ((x (make-tensor32 (f32vector (car sample)) '(1)))
                        (y (make-tensor32 (f32vector (cdr sample)) '(1)))
                        (pred (forward model x))
                        (loss (mse-loss pred y)))
                   (set! total-loss (+ total-loss 
                                      (f32vector-ref (tensor-data loss) 0)))))
               data)
              
              (if (< (/ total-loss (length data)) target-loss)
                  (set! epochs-to-converge (+ epoch 1))
                  (loop (+ epoch 1))))))
        
        (if (> epochs-to-converge 0)
            (printf "    ~A converged in ~A epochs\n" name epochs-to-converge)
            (printf "    ~A did not converge in 100 epochs\n" name))
        
        epochs-to-converge)))
  
  (let ((sgd-epochs (test-optimizer "SGD" 
                                   (lambda (p) (make-sgd p learning-rate: 0.1))))
        (adam-epochs (test-optimizer "Adam"
                                    (lambda (p) (make-adam p learning-rate: 0.1)))))
    
    (when (and (> sgd-epochs 0) (> adam-epochs 0))
      (printf "    Adam is ~Ax faster than SGD\n" 
              (/ (exact->inexact sgd-epochs) adam-epochs)))))

;;; ==================================================================
;;; Gradient Clipping Integration
;;; ==================================================================

(test-group "Gradient Clipping in Training"
  
  (set-random-seed! 303)
  
  (define data
    (list (cons (f32vector 10.0) 20.0)
          (cons (f32vector 20.0) 40.0)
          (cons (f32vector 30.0) 60.0)))
  
  (define model
    (make-sequential
     (list (make-dense-layer 1 1 activation: (make-identity)))))
  
  (define optimizer (make-sgd (parameters model) learning-rate: 0.5))
  
  (let ((max-grad-norm 0.0)
        (max-clip-norm 1.0))
    (do ((epoch 0 (+ epoch 1)))
        ((= epoch 20))
      
      (for-each
       (lambda (sample)
         (let* ((x (make-tensor32 (car sample) '(1)))
                (target (make-tensor32 (f32vector (cdr sample)) '(1)))
                (pred (forward model x))
                (loss (mse-loss pred target)))
           
           (backward! loss)
           
           (let ((grad-norm (clip-gradients! (parameters model) max-norm: max-clip-norm)))
             (if grad-norm (set! max-grad-norm (max max-grad-norm grad-norm))))
           
           (step! optimizer)
           (zero-grad-layer! model)))
       data))
    
    (printf "    Max gradient norm observed: ~A\n" max-grad-norm)
    (test-assert "Gradients remain finite during training"
      (finite? max-grad-norm))))

;;; ==================================================================
;;; Model Persistence (Parameter Extraction)
;;; ==================================================================

(test-group "Parameter Extraction"
  
  (set-random-seed! 404)
  
  (define model
    (make-sequential
     (list (make-dense-layer 2 3 activation: (make-relu))
           (make-dense-layer 3 1 activation: (make-identity)))))
  
  ;; Train briefly
  (let ((opt (make-adam (parameters model) learning-rate: 0.01))
        (data (list (cons (f32vector 1.0 2.0) 3.0))))
    
    (do ((i 0 (+ i 1)))
        ((= i 10))
      (for-each
       (lambda (sample)
         (let* ((x (make-tensor32 (car sample) '(2)))
                (y (make-tensor32 (f32vector (cdr sample)) '(1)))
                (pred (forward model x))
                (loss (mse-loss pred y)))
           (backward! loss)
           (step! opt)
           (zero-grad-layer! model)))
       data)))
  
  (let ((params (parameters model)))
    
    (test "Correct number of parameters"
      4
      (length params))
    
    (test-assert "All parameters are valid tensors with data"
      (every (lambda (p)
              (and (tensor? p)
                   (> (f32vector-length (tensor-data p)) 0)))
            params))
    
    (let ((total-params
           (fold (lambda (p acc)
                  (+ acc (f32vector-length (tensor-data p))))
                0
                params)))
      
      (test "Total parameter count correct"
        13
        total-params))))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(test-exit)

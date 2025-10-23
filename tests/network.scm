;; Comprehensive unit tests for complete neural network training examples

(import scheme
        (chicken base)
        (chicken format)
        (chicken random)
        (srfi 1)
        (srfi 4)
        nanograd-autograd
        nanograd-layer
        nanograd-optimizer
        nanograd-diagnostics)

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
          (printf "  O ~A\n" name))
        (begin
          (set! *test-failed* (+ *test-failed* 1))
          (printf "  X ~A\n" name)
          (printf "    Expected: ~A, Got: ~A, Diff: ~A\n" 
                  expected actual diff)))))

(define (assert-less-than actual threshold name)
  (set! *test-count* (+ *test-count* 1))
  (if (< actual threshold)
      (begin
        (set! *test-passed* (+ *test-passed* 1))
        (printf "  O ~A\n" name))
      (begin
        (set! *test-failed* (+ *test-failed* 1))
        (printf "  X ~A\n" name)
        (printf "    Expected < ~A, Got: ~A\n" threshold actual))))

(define (assert-greater-than actual threshold name)
  (set! *test-count* (+ *test-count* 1))
  (if (> actual threshold)
      (begin
        (set! *test-passed* (+ *test-passed* 1))
        (printf "  O ~A\n" name))
      (begin
        (set! *test-failed* (+ *test-failed* 1))
        (printf "  X ~A\n" name)
        (printf "    Expected > ~A, Got: ~A\n" threshold actual))))

(define (assert-in-range actual min-val max-val name)
  (set! *test-count* (+ *test-count* 1))
  (if (and (>= actual min-val) (<= actual max-val))
      (begin
        (set! *test-passed* (+ *test-passed* 1))
        (printf "  O ~A\n" name))
      (begin
        (set! *test-failed* (+ *test-failed* 1))
        (printf "  X ~A\n" name)
        (printf "    Expected in [~A, ~A], Got: ~A\n" 
                min-val max-val actual))))

(define (assert-true condition name)
  (set! *test-count* (+ *test-count* 1))
  (if condition
      (begin
        (set! *test-passed* (+ *test-passed* 1))
        (printf "  O ~A\n" name))
      (begin
        (set! *test-failed* (+ *test-failed* 1))
        (printf "  X ~A\n" name))))

;;; ==================================================================
;;; Helper Functions
;;; ==================================================================

(define (argmax vec)
  (let loop ((i 1) (max-i 0) (max-val (f32vector-ref vec 0)))
    (if (= i (f32vector-length vec))
        max-i
        (let ((val (f32vector-ref vec i)))
          (if (> val max-val)
              (loop (+ i 1) i val)
              (loop (+ i 1) max-i max-val))))))

(define (set-random-seed! seed)
  "Set random seed for reproducibility"
  (set-pseudo-random-seed! (number->string seed))
  )

;;; ==================================================================
;;; Test 1: Linear Regression with SGD
;;; ==================================================================

(define (test-linear-regression-sgd)
  (printf "\n=== Testing Linear Regression with SGD ===\n")
  
  ;; Fixed seed for reproducibility
  (set-random-seed! 42)
  
  ;; Generate deterministic data: y = 3x + 2
  (define training-data
    (list
     (cons 0.0 2.0)    ; 3*0 + 2 = 2
     (cons 1.0 5.0)    ; 3*1 + 2 = 5
     (cons -1.0 -1.0)  ; 3*(-1) + 2 = -1
     (cons 2.0 8.0)    ; 3*2 + 2 = 8
     (cons -2.0 -4.0)  ; 3*(-2) + 2 = -4
     (cons 0.5 3.5)    ; 3*0.5 + 2 = 3.5
     (cons -0.5 0.5))) ; 3*(-0.5) + 2 = 0.5
  
  ;; Create model
  (define model
    (make-sequential
     (list
      (make-dense-layer 1 1 activation: (make-identity)))
     name: "LinearRegression"))
  
  ;; Initialize parameters to known values
  (let ((params (parameters model)))
    (f32vector-set! (tensor-data (car params)) 0 0.5)  ; weight
    (f32vector-set! (tensor-data (cadr params)) 0 0.5)) ; bias
  
  ;; Create optimizer
  (define optimizer (make-sgd (parameters model) learning-rate: 0.1))
  
  ;; Train for fixed epochs
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
    
    (assert-in-range weight 2.5 3.5
                    "Weight converges near 3.0")
    (assert-in-range bias 1.5 2.5
                    "Bias converges near 2.0")
    
    ;; Test predictions
    (let* ((test-x (make-tensor32 (f32vector 3.0) '(1)))
           (pred (forward model test-x))
           (pred-val (f32vector-ref (tensor-data pred) 0))
           (expected 11.0))  ; 3*3 + 2 = 11
      
      (assert-in-range pred-val 10.0 12.0
                      "Prediction for x=3 is near 11.0"))))

;;; ==================================================================
;;; Test 2: Linear Regression Optimizer Comparison
;;; ==================================================================

(define (test-optimizer-comparison)
  (printf "\n=== Testing Optimizer Comparison ===\n")
  
  ;; Simple dataset
  (define data
    (list (cons 0.0 1.0)
          (cons 1.0 3.0)
          (cons 2.0 5.0)
          (cons 3.0 7.0)))  ; y = 2x + 1
  
  ;; Test each optimizer
  (define (test-optimizer name make-opt expected-range)
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
            (assert-less-than avg-loss (cdr expected-range)
                             (format #f "~A achieves low loss" name)))))))
  
  ;; Test different optimizers
  (test-optimizer "SGD" 
                 (lambda (p) (make-sgd p learning-rate: 0.1))
                 (cons 0.0 0.5))
  
  (test-optimizer "Adam"
                 (lambda (p) (make-adam p learning-rate: 0.1))
                 (cons 0.0 0.1))
  
  (test-optimizer "RMSprop"
                 (lambda (p) (make-rmsprop p learning-rate: 0.1))
                 (cons 0.0 0.1)))

;;; ==================================================================
;;; Test 3: Binary Classification
;;; ==================================================================

(define (test-binary-classification)
  (printf "\n=== Testing Binary Classification ===\n")
  
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
      (assert-greater-than accuracy 80.0
                          "Binary classification achieves >80% accuracy")
      (printf "    Final accuracy: ~A%\n" accuracy))))

;;; ==================================================================
;;; Test 4: Multi-class Classification (XOR-like)
;;; ==================================================================

(define (test-multiclass-classification)
  (printf "\n=== Testing Multi-class Classification ===\n")
  
  (set-random-seed! 456)
  
  ;; Create 3-class dataset with clear separation
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
           (if (zero? (modulo epoch 10))
               (printf "Loss at epoch ~A: ~A\n" epoch (tensor-data loss)))
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
      (assert-greater-than accuracy 70.0
                          "Multi-class achieves >70% accuracy")
      (printf "    Final accuracy: ~A%\n" accuracy))))

;;; ==================================================================
;;; Test 5: Learning Rate Decay
;;; ==================================================================

(define (test-learning-rate-decay)
  (printf "\n=== Testing Learning Rate Decay ===\n")
  
  (set-random-seed! 789)
  
  ;; Simple dataset
  (define data
    (list (cons 0.0 1.0)
          (cons 1.0 2.0)
          (cons 2.0 3.0)))
  
  ;; Create model
  (define model
    (make-sequential
     (list (make-dense-layer 1 1 activation: (make-identity)))))
  
  (define optimizer (make-sgd (parameters model) 
                              learning-rate: 1.0))  ; High initial LR
  
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
    
    ;; Verify LR decreased
    (assert-less-than final-lr initial-lr
                     "Learning rate decreased during training")
    
    ;; Verify specific decay amount
    (let ((expected-final-lr (/ 1.0 (+ 1.0 (* 0.1 20)))))
      (assert-equal final-lr expected-final-lr 0.01
                   "Learning rate matches decay formula"))))

;;; ==================================================================
;;; Test 6: Batch Training vs Sequential
;;; ==================================================================

(define (test-batch-training)
  (printf "\n=== Testing Batch Training ===\n")
  
  (set-random-seed! 101)
  
  ;; Dataset
  (define data
    (list (cons (f32vector 0.0 0.0) 0.0)
          (cons (f32vector 1.0 0.0) 1.0)
          (cons (f32vector 0.0 1.0) 1.0)
          (cons (f32vector 1.0 1.0) 0.0)))  ; XOR
  
  ;; Model for batch training
  (define model-batch
    (make-sequential
     (list
      (make-dense-layer 2 4 activation: (make-relu))
      (make-dense-layer 4 1 activation: (make-sigmoid)))))
  
  (define opt-batch (make-adam (parameters model-batch) learning-rate: 0.1))
  
  ;; Train with batches (accumulate gradients)
  (do ((epoch 0 (+ epoch 1)))
      ((= epoch 50))
    
    ;; Accumulate gradients over all samples
    (for-each
     (lambda (sample)
       (let* ((x (make-tensor32 (car sample) '(2)))
              (target (make-tensor32 (f32vector (cdr sample)) '(1)))
              (pred (forward model-batch x))
              (loss (mse-loss pred target)))
         (backward! loss)))  ; Don't update yet
     data)
    
    ;; Update once per epoch (batch update)
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
      (assert-less-than avg-loss 0.3
                       "Batch training achieves low loss on XOR"))))

;;; ==================================================================
;;; Test 7: Overfitting Detection
;;; ==================================================================

(define (test-overfitting-detection)
  (printf "\n=== Testing Overfitting Detection ===\n")
  
  (set-random-seed! 202)
  
  ;; Small training set
  (define train-data
    (list (cons (f32vector 0.0) 0.0)
          (cons (f32vector 1.0) 1.0)))
  
  ;; Separate test set
  (define test-data
    (list (cons (f32vector 0.5) 0.5)
          (cons (f32vector 1.5) 1.5)))
  
  ;; Very large model (prone to overfitting)
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
  
  ;; Measure train vs test loss
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
    
    (assert-less-than train-loss 0.01
                     "Training loss very low (overfitting)")
    
    ;; Test loss should be higher (sign of overfitting)
    (printf "    Train loss: ~A, Test loss: ~A\n" train-loss test-loss)
    (printf "    Gap indicates ~A\n" 
            (if (> test-loss (* 2.0 train-loss))
                "potential overfitting"
                "good generalization"))))

;;; ==================================================================
;;; Test 8: Convergence Speed Comparison
;;; ==================================================================

(define (test-convergence-speed)
  (printf "\n=== Testing Optimizer Convergence Speed ===\n")
  
  ;; Simple problem
  (define data
    (list (cons 0.0 0.0)
          (cons 1.0 2.0)
          (cons 2.0 4.0)))  ; y = 2x
  
  ;; Test convergence speed
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
  
  ;; Compare optimizers
  (let ((sgd-epochs (test-optimizer "SGD" 
                                   (lambda (p) (make-sgd p learning-rate: 0.1))))
        (adam-epochs (test-optimizer "Adam"
                                    (lambda (p) (make-adam p learning-rate: 0.1)))))
    
    (when (and (> sgd-epochs 0) (> adam-epochs 0))
      (printf "    Adam is ~Ax faster than SGD\n" 
              (/ (exact->inexact sgd-epochs) adam-epochs)))))

;;; ==================================================================
;;; Test 9: Gradient Clipping Integration
;;; ==================================================================

(define (test-gradient-clipping-integration)
  (printf "\n=== Testing Gradient Clipping in Training ===\n")
  
  (set-random-seed! 303)
  
  ;; Dataset with large values (may cause gradient issues)
  (define data
    (list (cons (f32vector 10.0) 20.0)
          (cons (f32vector 20.0) 40.0)
          (cons (f32vector 30.0) 60.0)))
  
  ;; Create model
  (define model
    (make-sequential
     (list (make-dense-layer 1 1 activation: (make-identity)))))
  
  ;; High learning rate (would normally cause problems)
  (define optimizer (make-sgd (parameters model) learning-rate: 0.5))
  
  ;; Train with gradient clipping
  (let ((max-grad-norm 0.0)
        (max-grad-norm-after-clip 0.0)
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
           
           ;; Track max gradient AFTER clipping
           (let ((params (parameters model)))
             (let ((grad (tensor-grad (car params))))
               (let ((g (f32vector-ref grad 0)))
                 (let ((norm (sqrt (* g g))))
                   (set! max-grad-norm-after-clip 
                         (max max-grad-norm-after-clip norm))))))

           
           (step! optimizer)
           (zero-grad-layer! model)))
       data))
    
    (printf "    Max gradient norm observed: ~A\n" max-grad-norm)
    (assert-true (finite? max-grad-norm)
                "Gradients remain finite during training")))

;;; ==================================================================
;;; Test 10: Model Persistence (Parameter Extraction)
;;; ==================================================================

(define (test-parameter-extraction)
  (printf "\n=== Testing Parameter Extraction ===\n")
  
  (set-random-seed! 404)
  
  ;; Create and train model
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
  
  ;; Extract parameters
  (let ((params (parameters model)))
    
    (assert-equal (length params) 4 0
                 "Correct number of parameters (2 layers × 2)")
    
    ;; Verify parameters are tensors with data
    (let ((all-have-data? 
           (every (lambda (p)
                   (and (tensor? p)
                        (> (f32vector-length (tensor-data p)) 0)))
                 params)))
      
      (assert-true all-have-data?
                  "All parameters are valid tensors with data"))
    
    ;; Count total parameters
    (let ((total-params
           (fold (lambda (p acc)
                  (+ acc (f32vector-length (tensor-data p))))
                0
                params)))
      
      ;; Layer 1: 2*3 weights + 3 biases = 9
      ;; Layer 2: 3*1 weights + 1 bias = 4
      ;; Total = 13
      (assert-equal total-params 13 0
                   "Total parameter count correct"))))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(define (run-all-network-tests)
  (reset-test-stats!)
  (printf "\n")
  (printf "========================================\n")
  (printf "COMPLETE NETWORK TRAINING TESTS\n")
  (printf "========================================\n")
  
  (test-linear-regression-sgd)
  (test-optimizer-comparison)
  (test-binary-classification)
  (test-multiclass-classification)
  (test-learning-rate-decay)
  (test-batch-training)
  (test-overfitting-detection)
  (test-convergence-speed)
  (test-gradient-clipping-integration)
  (test-parameter-extraction)
  
  (test-summary))

;; Run all tests
(run-all-network-tests)

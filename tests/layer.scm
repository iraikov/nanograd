;; unit tests for layer operations

(import scheme
        (chicken base)
        (chicken format)
        (chicken random)
        (srfi 1)
        (srfi 4)
        blas
        nanograd-autograd
        nanograd-layer)

;;; ==================================================================
;;; Test Framework (same as autograd tests)
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

(define (assert-shape-equal tensor expected-shape name)
  (set! *test-count* (+ *test-count* 1))
  (let ((actual-shape (tensor-shape tensor)))
    (if (equal? actual-shape expected-shape)
        (begin
          (set! *test-passed* (+ *test-passed* 1))
          (printf "  O ~A\n" name))
        (begin
          (set! *test-failed* (+ *test-failed* 1))
          (printf "  X ~A\n" name)
          (printf "    Expected shape: ~A, Got: ~A\n" 
                  expected-shape actual-shape)))))

(define (assert-true condition name)
  (set! *test-count* (+ *test-count* 1))
  (if condition
      (begin
        (set! *test-passed* (+ *test-passed* 1))
        (printf "  O ~A\n" name))
      (begin
        (set! *test-failed* (+ *test-failed* 1))
        (printf "  X ~A\n" name))))

(define (assert-range actual min-val max-val name)
  (set! *test-count* (+ *test-count* 1))
  (if (and (>= actual min-val) (<= actual max-val))
      (begin
        (set! *test-passed* (+ *test-passed* 1))
        (printf "  O ~A\n" name))
      (begin
        (set! *test-failed* (+ *test-failed* 1))
        (printf "  X ~A\n" name)
        (printf "    Expected range: [~A, ~A], Got: ~A\n" 
                min-val max-val actual))))

;;; ==================================================================
;;; Unit Tests: Activation Functions as Objects
;;; ==================================================================

(define (test-activation-objects)
  (printf "\n=== Testing Activation Function Objects ===\n")
  
  ;; Test 1: ReLU activation
  (let* ((relu-act (make-relu))
         (x (make-tensor32 (f32vector -1.0 0.0 1.0) '(3)))
         (y (activation-forward relu-act x))
         (expected (f32vector 0.0 0.0 1.0)))
    (assert-true (activation? relu-act) "ReLU is activation")
    ;(assert-equal (activation-name relu-act) "ReLU" 0 "ReLU name")
    (let ((data (tensor-data y)))
      (assert-equal (f32vector-ref data 0) 0.0 1e-5 "ReLU(-1) = 0")
      (assert-equal (f32vector-ref data 2) 1.0 1e-5 "ReLU(1) = 1")))
  
  ;; Test 2: Sigmoid activation
  (let* ((sig-act (make-sigmoid))
         (x (make-tensor32 (f32vector 0.0) '(1)))
         (y (activation-forward sig-act x)))
    (assert-true (activation? sig-act) "Sigmoid is activation")
    (assert-equal (f32vector-ref (tensor-data y) 0) 0.5 1e-5
                  "Sigmoid(0) = 0.5"))
  
  ;; Test 3: Identity activation
  (let* ((id-act (make-identity))
         (x (make-tensor32 (f32vector 5.0) '(1)))
         (y (activation-forward id-act x)))
    (assert-equal (f32vector-ref (tensor-data y) 0) 5.0 1e-5
                  "Identity(5) = 5")))

;;; ==================================================================
;;; Unit Tests: Dense Layer
;;; ==================================================================

(define (test-dense-layer-construction)
  (printf "\n=== Testing Dense Layer Construction ===\n")
  
  (let ((layer (make-dense-layer 10 5 activation: (make-relu))))
    (assert-true (layer? layer) "Is a layer")
    (assert-true (dense-layer? layer) "Is a dense layer")
    (assert-equal (layer-input-size layer) 10 0 "Input size = 10")
    (assert-equal (layer-output-size layer) 5 0 "Output size = 5")
    
    ;; Check parameters
    (let ((params (parameters layer)))
      (assert-equal (length params) 2 0 "Has 2 parameters (W and b)")
      (assert-shape-equal (car params) '(5 10) "Weight shape (5x10)")
      (assert-shape-equal (cadr params) '(5) "Bias shape (5)"))))

(define (test-dense-layer-forward)
  (printf "\n=== Testing Dense Layer Forward Pass ===\n")
  
  ;; Create layer with known weights
  (let* ((layer (make-dense-layer 2 3 activation: (make-identity)))
         (params (parameters layer))
         (weights (car params))
         (biases (cadr params)))
    
    ;; Set known weight values: [[1, 2], [3, 4], [5, 6]]
    (let ((w-data (tensor-data weights)))
      (f32vector-set! w-data 0 1.0)
      (f32vector-set! w-data 1 2.0)
      (f32vector-set! w-data 2 3.0)
      (f32vector-set! w-data 3 4.0)
      (f32vector-set! w-data 4 5.0)
      (f32vector-set! w-data 5 6.0))
    
    ;; Set known bias values: [0.1, 0.2, 0.3]
    (let ((b-data (tensor-data biases)))
      (f32vector-set! b-data 0 0.1)
      (f32vector-set! b-data 1 0.2)
      (f32vector-set! b-data 2 0.3))
    
    ;; Input: [1, 2]
    (let* ((input (make-tensor32 (f32vector 1.0 2.0) '(2)))
           (output (forward layer input)))
      ;; Expected: W @ x + b = [1*1+2*2, 3*1+4*2, 5*1+6*2] + [0.1, 0.2, 0.3]
      ;;                      = [5, 11, 17] + [0.1, 0.2, 0.3]
      ;;                      = [5.1, 11.2, 17.3]
      (assert-shape-equal output '(3) "Output shape")
      (assert-equal (f32vector-ref (tensor-data output) 0) 5.1 1e-4
                    "Output[0] = 5.1")
      (assert-equal (f32vector-ref (tensor-data output) 1) 11.2 1e-4
                    "Output[1] = 11.2")
      (assert-equal (f32vector-ref (tensor-data output) 2) 17.3 1e-4
                    "Output[2] = 17.3"))))

(define (test-dense-layer-gradient)
  (printf "\n=== Testing Dense Layer Gradients ===\n")
  
  (let* ((layer (make-dense-layer 2 3 activation: (make-identity)))
         (input (make-tensor32 (f32vector 1.0 2.0) '(2)))
         (output (forward layer input))
         (target (make-tensor32 (f32vector 0.0 0.0 0.0) '(3)))
         (loss (mse-loss output target)))
    
    (backward! loss)
    
    ;; Check that gradients exist
    (let ((params (parameters layer)))
      (assert-true (not (equal? (tensor-grad (car params)) #f))
                   "Weight gradients computed")
      (assert-true (not (equal? (tensor-grad (cadr params)) #f))
                   "Bias gradients computed"))))

;;; ==================================================================
;;; Unit Tests: Sequential Container
;;; ==================================================================

(define (test-sequential)
  (printf "\n=== Testing Sequential Container ===\n")
  
  ;; Test 1: Two-layer network
  (let* ((net (make-sequential
               (list
                (make-dense-layer 4 8 activation: (make-relu))
                (make-dense-layer 8 2 activation: (make-identity)))))
         (input (make-tensor32 (f32vector 1.0 2.0 3.0 4.0) '(4)))
         (output (forward net input)))
    
    (assert-true (layer? net) "Sequential is a layer")
    (assert-true (sequential? net) "Is sequential")
    (assert-equal (layer-input-size net) 4 0 "Input size = 4")
    (assert-equal (layer-output-size net) 2 0 "Output size = 2")
    (assert-shape-equal output '(2) "Output shape correct")
    
    ;; Check all parameters accessible
    (let ((params (parameters net)))
      (assert-equal (length params) 4 0 "Has 4 parameters (2 layers * 2)")))
  
  ;; Test 2: Deep network
  (let* ((deep-net (make-sequential
                    (list
                     (make-dense-layer 5 10 activation: (make-relu))
                     (make-dense-layer 10 10 activation: (make-relu))
                     (make-dense-layer 10 3 activation: (make-identity)))))
         (input (make-tensor32 (make-f32vector 5 1.0) '(5)))
         (output (forward deep-net input)))
    
    (assert-shape-equal output '(3) "Deep net output shape")))

;;; ==================================================================
;;; Unit Tests: Conv2D Layer
;;; ==================================================================

(define (test-conv2d-layer-construction)
  (printf "\n=== Testing Conv2D Layer Construction ===\n")
  
  (let ((layer (make-conv2d-layer 3 16 3 
                                  stride: 1 
                                  padding: 1
                                  activation: (make-relu))))
    
    (assert-true (layer? layer) "Is a layer")
    (assert-true (conv2d-layer? layer) "Is a conv2d layer")
    (assert-equal (layer-input-size layer) 3 0 "Input channels = 3")
    (assert-equal (layer-output-size layer) 16 0 "Output channels = 16")
    
    ;; Check parameters
    (let ((params (parameters layer)))
      (assert-equal (length params) 2 0 "Has 2 parameters")
      (assert-shape-equal (car params) '(16 3 3 3) "Weight shape")
      (assert-shape-equal (cadr params) '(16) "Bias shape"))))

(define (test-conv2d-layer-forward)
  (printf "\n=== Testing Conv2D Layer Forward Pass ===\n")
  
  ;; Test 1: Basic forward pass
  (let* ((layer (make-conv2d-layer 1 8 3 stride: 1 padding: 1))
         (input (make-tensor32 (make-f32vector 16 0.5) '(1 4 4)))
         (output (forward layer input)))
    
    (assert-shape-equal output '(8 4 4) "Output shape with padding"))
  
  ;; Test 2: With stride
  (let* ((layer (make-conv2d-layer 1 4 3 stride: 2 padding: 0))
         (input (make-tensor32 (make-f32vector 64 1.0) '(1 8 8)))
         (output (forward layer input)))
    
    ;; 8x8 input, 3x3 kernel, stride=2, padding=0
    ;; Output: (8-3)/2 + 1 = 3
    (assert-shape-equal output '(4 3 3) "Output shape with stride=2")))

(define (test-conv2d-layer-gradient)
  (printf "\n=== Testing Conv2D Layer Gradients ===\n")
  
  (let* ((layer (make-conv2d-layer 1 4 3 stride: 1 padding: 0))
         (input (make-tensor32 (make-f32vector 16 1.0) '(1 4 4)))
         (output (forward layer input))
         (loss (dot-op (flatten-tensor output) (flatten-tensor output))))
    
    (backward! loss)
    
    (let ((params (parameters layer)))
      (assert-true (not (equal? (tensor-grad (car params)) #f))
                   "Weight gradients computed")
      (assert-true (not (equal? (tensor-grad (cadr params)) #f))
                   "Bias gradients computed"))))

;;; ==================================================================
;;; Unit Tests: Training Loop
;;; ==================================================================

(define (test-simple-training)
  (printf "\n=== Testing Simple Training Loop ===\n")
  
  ;; Create simple linear model: y = 2x
  (let* ((model (make-sequential
                 (list
                  (make-dense-layer 1 1 activation: (make-identity)))))
         (training-data (list
                         (cons (f32vector 1.0) 2.0)
                         (cons (f32vector 2.0) 4.0)
                         (cons (f32vector 3.0) 6.0))))
    
    ;; Train for a few epochs
    (let loop ((epoch 0) (prev-loss 1000.0))
      (when (< epoch 50)
        (let ((total-loss 0.0))
          (for-each
           (lambda (sample)
             (let* ((x (make-tensor32 (car sample) '(1)))
                    (target (make-tensor32 (f32vector (cdr sample)) '(1)))
                    (pred (forward model x))
                    (loss (mse-loss pred target))
                    (loss-val (f32vector-ref (tensor-data loss) 0)))
               
               (set! total-loss (+ total-loss loss-val))
               (backward! loss)
               
               ;; Simple gradient descent
               (for-each
                (lambda (param)
                  (let ((data (tensor-data param))
                        (grad (tensor-grad param))
                        (n (f32vector-length (tensor-data param))))
                    (do ((i 0 (+ i 1)))
                        ((= i n))
                      (f32vector-set! data i
                                     (- (f32vector-ref data i)
                                        (* 0.01 (f32vector-ref grad i)))))))
                (parameters model))
               
               (zero-grad-layer! model)))
           training-data)
          
          (let ((avg-loss (/ total-loss (length training-data))))
            (when (= epoch 49)
              ;; After training, loss should decrease
              (assert-true (< avg-loss prev-loss)
                           "Loss decreases during training"))
            (loop (+ epoch 1) avg-loss)))))))

(define (test-activation-comparison)
  (printf "\n=== Testing Different Activations ===\n")
  
  (let ((input (make-tensor32 (f32vector -1.0 0.0 1.0 2.0) '(4))))
    
    ;; ReLU layer
    (let* ((relu-layer (make-dense-layer 4 4 activation: (make-relu)))
           (output (forward relu-layer input))
           (data (tensor-data output)))
      (assert-true (>= (f32vector-ref data 0) 0.0)
                   "ReLU output non-negative"))
    
    ;; Sigmoid layer
    (let* ((sigmoid-layer (make-dense-layer 4 4 activation: (make-sigmoid)))
           (output (forward sigmoid-layer input))
           (data (tensor-data output)))
      (assert-true (and (>= (f32vector-ref data 0) 0.0)
                        (<= (f32vector-ref data 0) 1.0))
                   "Sigmoid output in [0,1]"))))

;;; ==================================================================
;;; Unit Tests: Parameter Count
;;; ==================================================================

(define (test-parameter-count)
  (printf "\n=== Testing Parameter Counting ===\n")
  
  ;; Dense layer: 10 -> 5
  ;; Parameters: weights (5x10=50) + biases (5) = 55
  (let* ((layer (make-dense-layer 10 5))
         (params (parameters layer))
         (total-params (fold
                        (lambda (p acc)
                          (let ((data (tensor-data p)))
                            (+ acc (f32vector-length data))))
                        0
                        params)))
    (assert-equal total-params 55 0 "Dense layer parameter count"))
  
  ;; Conv2D: 3 channels -> 8 channels, 3x3 kernel
  ;; Parameters: weights (8x3x3x3=216) + biases (8) = 224
  (let* ((layer (make-conv2d-layer 3 8 3))
         (params (parameters layer))
         (total-params (fold
                        (lambda (p acc)
                          (let ((data (tensor-data p)))
                            (+ acc (f32vector-length data))))
                        0
                        params)))
    (assert-equal total-params 224 0 "Conv2D layer parameter count")))

;;; ==================================================================
;;; Unit Tests: Mixed Operations
;;; ==================================================================

(define (test-conv-to-dense)
  (printf "\n=== Testing Conv2D to Dense Integration ===\n")
  
  (let* ((input (make-tensor32 (make-f32vector 64 1.0) '(1 8 8)))
         (conv-layer (make-conv2d-layer 1 4 3 stride: 2 padding: 1))
         (conv-out (forward conv-layer input))
         (flat (flatten-tensor conv-out))
         (dense-layer (make-dense-layer 64 10))
         (output (forward dense-layer flat)))
    
    (assert-shape-equal conv-out '(4 4 4) "Conv output shape")
    (assert-shape-equal flat '(64) "Flattened shape")
    (assert-shape-equal output '(10) "Final output shape")
    
    ;; Test gradient flow
    (let ((loss (dot-op output output)))
      (backward! loss)
      (assert-true (not (equal? (tensor-grad input) #f))
                   "Gradient flows through conv->flatten->dense"))))

(define (test-zero-grad)
  (printf "\n=== Testing Zero Gradient ===\n")
  
  (let* ((layer (make-dense-layer 3 2))
         (input (make-tensor32 (f32vector 1.0 2.0 3.0) '(3)))
         (output (forward layer input)))
    
    (backward! output)
    
    (let ((params (parameters layer)))
      ;; Check gradients exist
      (assert-true (not (equal? (tensor-grad (car params)) #f))
                   "Gradients exist before zero")
      
      ;; Zero gradients
      (zero-grad-layer! layer)
      
      ;; Check first gradient value is zero
      (let ((grad (tensor-grad (car params))))
        (assert-equal (f32vector-ref grad 0) 0.0 1e-10
                      "Gradient zeroed")))))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(define (run-all-layer-tests)
  (reset-test-stats!)
  (printf "\n")
  (printf "========================================\n")
  (printf "Layer unit tests\n")
  (printf "========================================\n")
  
  (test-activation-objects)
  (test-dense-layer-construction)
  (test-dense-layer-forward)
  (test-dense-layer-gradient)
  (test-sequential)
  (test-conv2d-layer-construction)
  (test-conv2d-layer-forward)
  (test-conv2d-layer-gradient)
  (test-simple-training)
  (test-activation-comparison)
  (test-parameter-count)
  (test-conv-to-dense)
  (test-zero-grad)
  
  (test-summary))

;; Run all tests
(run-all-layer-tests)

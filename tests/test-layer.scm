;; unit tests for layer operations

(import scheme
        (chicken base)
        (chicken format)
        (chicken random)
        (srfi 1)
        (srfi 4)
        blas
        yasos
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


(define (test-exit)
  (if (= *test-passed* *test-count*)
      (exit)
      (exit 1)))


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

(define (test-dense-layer-dimensions)
  (printf "\n=== Testing Dense Layer Dimensions ===\n")
  
  (let ((dense (make-dense-layer 512 256 dtype: 'f32)))
    
    ;; Test dimension queries
    (assert-equal (layer-input-size dense) 512 1e-5
                 "layer-input-size returns feature dimension")
    
    (assert-equal (layer-output-size dense) 256 1e-5
                 "layer-output-size returns feature dimension")
    
    ;; Test with 1D input
    (let* ((input-1d (make-tensor32 (make-f32vector 512 1.0) '(512)))
           (output-1d (forward dense input-1d)))
      (assert-shape-equal output-1d '(256)
                         "1D output shape matches layer-output-size"))
    
    ;; Test with 2D input (batch)
    (let* ((input-2d (make-tensor32 (make-f32vector (* 128 512) 1.0) '(128 512)))
           (output-2d (forward dense input-2d)))
      (assert-shape-equal output-2d '(128 256)
                         "2D output has batch dimension preserved")
      
      ;; Verify output feature dimension matches layer-output-size
      (assert-equal (cadr (tensor-shape output-2d)) 
                    (layer-output-size dense) 1e-5
                    "2D output feature dim matches layer-output-size"))))

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
;;; Batched MaxPool2D
;;; ==================================================================

(define (test-maxpool2d-batched)
  (printf "\n=== Testing Batched MaxPool2D ===\n")
  
  ;; Test 1: 3D input (single image) - basic functionality
  (let* ((input (make-tensor32 (f32vector 1.0 2.0 3.0 4.0
                                         5.0 6.0 7.0 8.0
                                         9.0 10.0 11.0 12.0
                                         13.0 14.0 15.0 16.0) '(1 4 4)))
         (output (maxpool2d input 2 stride: 2)))
    
    (assert-shape-equal output '(1 2 2) "3D MaxPool2D output shape")
    
    ;; Check max values: each 2x2 window
    ;; Window 1: [1,2,5,6] -> max = 6
    ;; Window 2: [3,4,7,8] -> max = 8
    ;; Window 3: [9,10,13,14] -> max = 14
    ;; Window 4: [11,12,15,16] -> max = 16
    (let ((data (tensor-data output)))
      (assert-equal (f32vector-ref data 0) 6.0 1e-5 "MaxPool window 1")
      (assert-equal (f32vector-ref data 1) 8.0 1e-5 "MaxPool window 2")
      (assert-equal (f32vector-ref data 2) 14.0 1e-5 "MaxPool window 3")
      (assert-equal (f32vector-ref data 3) 16.0 1e-5 "MaxPool window 4")))
  
  ;; Test 2: 4D input (batch of 2) - basic functionality
  (let* ((input (make-tensor32 (make-f32vector (* 2 1 4 4) 1.0) '(2 1 4 4)))
         (output (maxpool2d input 2 stride: 2)))
    
    (assert-shape-equal output '(2 1 2 2) "4D MaxPool2D output shape"))
  
  ;; Test 3: 4D input with multiple channels
  (let* ((input (make-tensor32 (make-f32vector (* 3 2 8 8) 2.0) '(3 2 8 8)))
         (output (maxpool2d input 2 stride: 2)))
    
    (assert-shape-equal output '(3 2 4 4) "4D MaxPool2D multi-channel shape")
    
    ;; All values should be 2.0 since input is uniform
    (let ((data (tensor-data output)))
      (assert-equal (f32vector-ref data 0) 2.0 1e-5 "MaxPool uniform input")))
  
  ;; Test 4: Gradient flow for 3D input
  (let* ((input (make-tensor32 (make-f32vector 16 1.0) '(1 4 4)))
         (output (maxpool2d input 2 stride: 2))
         (loss (dot-op (flatten-tensor output) (flatten-tensor output))))
    
    (backward! loss)
    (assert-true (not (equal? (tensor-grad input) #f))
                 "3D MaxPool2D gradient flows"))
  
  ;; Test 5: Gradient flow for 4D input
  (let* ((input (make-tensor32 (make-f32vector (* 2 1 4 4) 1.0) '(2 1 4 4)))
         (output (maxpool2d input 2 stride: 2))
         (loss (dot-op (flatten-tensor output) (flatten-tensor output))))
    
    (backward! loss)
    (assert-true (not (equal? (tensor-grad input) #f))
                 "4D MaxPool2D gradient flows"))
  
  ;; Test 6: Different strides
  (let* ((input (make-tensor32 (make-f32vector (* 1 1 8 8) 1.0) '(1 1 8 8)))
         (output (maxpool2d input 3 stride: 1)))
    
    ;; With 8x8 input, 3x3 kernel, stride 1: output = (8-3)/1 + 1 = 6
    (assert-shape-equal output '(1 1 6 6) "MaxPool2D with stride=1"))
  
  ;; Test 7: Batched with different values
  (let* ((batch-size 2)
         (C 1)
         (H 4)
         (W 4)
         ;; Create input with different values for each batch element
         (input-data (make-f32vector (* batch-size C H W) 0.0)))
    
    ;; Fill first batch with values 1-16
    (do ((i 0 (+ i 1)))
        ((= i 16))
      (f32vector-set! input-data i (exact->inexact (+ i 1))))
    
    ;; Fill second batch with values 17-32
    (do ((i 0 (+ i 1)))
        ((= i 16))
      (f32vector-set! input-data (+ i 16) (exact->inexact (+ i 17))))
    
    (let* ((input (make-tensor32 input-data (list batch-size C H W)))
           (output (maxpool2d input 2 stride: 2)))
      
      (assert-shape-equal output (list batch-size C 2 2) "Batched MaxPool shape")
      
      ;; Check first batch output (same as test 1)
      (let ((data (tensor-data output)))
        (assert-equal (f32vector-ref data 0) 6.0 1e-5 "Batch 0 window 1")
        (assert-equal (f32vector-ref data 1) 8.0 1e-5 "Batch 0 window 2")
        (assert-equal (f32vector-ref data 2) 14.0 1e-5 "Batch 0 window 3")
        (assert-equal (f32vector-ref data 3) 16.0 1e-5 "Batch 0 window 4")
        
        ;; Check second batch output (offset by +16)
        (assert-equal (f32vector-ref data 4) 22.0 1e-5 "Batch 1 window 1")
        (assert-equal (f32vector-ref data 5) 24.0 1e-5 "Batch 1 window 2")
        (assert-equal (f32vector-ref data 6) 30.0 1e-5 "Batch 1 window 3")
        (assert-equal (f32vector-ref data 7) 32.0 1e-5 "Batch 1 window 4")))))

(define (test-batched-conv-maxpool)
  (printf "\n=== Testing Batched Conv2D + MaxPool2D ===\n")
  
  (let* ((batch-size 4)
         (input (make-tensor32 (make-f32vector (* batch-size 3 32 32) 1.0) 
                              (list batch-size 3 32 32)))
         (conv-layer (make-conv2d-layer 3 16 3 stride: 1 padding: 1))
         (conv-out (forward conv-layer input)))
    
    (assert-shape-equal conv-out (list batch-size 16 32 32) "Batched Conv output")
    
    (let ((pool-out (maxpool2d conv-out 2 stride: 2)))
      (assert-shape-equal pool-out (list batch-size 16 16 16) "Batched MaxPool after Conv")
      
      ;; Test gradient flow through both layers
      (let ((loss (dot-op (flatten-tensor pool-out) (flatten-tensor pool-out))))
        (backward! loss)
        (assert-true (not (equal? (tensor-grad input) #f))
                     "Gradient flows through Conv->MaxPool with batching")))))

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
  (test-dense-layer-dimensions)
  (test-sequential)
  (test-conv2d-layer-construction)
  (test-conv2d-layer-forward)
  (test-conv2d-layer-gradient)
  (test-simple-training)
  (test-activation-comparison)
  (test-parameter-count)
  (test-conv-to-dense)
  (test-zero-grad)
  (test-maxpool2d-batched)
  (test-batched-conv-maxpool) 
  (test-summary))

;; Run all tests
(run-all-layer-tests)
(test-exit)

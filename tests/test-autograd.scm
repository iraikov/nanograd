;; Unit tests for autograd operations

(import scheme
        (chicken base)
        (chicken format)
        (srfi 1)
        (srfi 4)
        test
        blas
        nanograd-autograd)

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

(define (vector-approx-equal? vec1 vec2 tolerance)
  "Check if two f32vectors are approximately equal within tolerance"
  (let ((n1 (f32vector-length vec1))
        (n2 (f32vector-length vec2)))
    (and (= n1 n2)
         (let loop ((i 0))
           (cond
            ((= i n1) #t)
            ((> (abs (- (f32vector-ref vec1 i)
                       (f32vector-ref vec2 i)))
                tolerance)
             #f)
            (else (loop (+ i 1))))))))

(define (test-vector-equal vec1 vec2 tolerance)
  "Test helper for vector equality with tolerance"
  (test-assert (vector-approx-equal? vec1 vec2 tolerance)))

;;; ==================================================================
;;; Unit Tests: Basic Arithmetic Operations
;;; ==================================================================

(test-group "Addition"
  
  (test "Forward: [2,3,4] + [1,2,3] = [3,5,7]"
    (f32vector 3.0 5.0 7.0)
    (let* ((x (make-tensor32 (f32vector 2.0 3.0 4.0) '(3)))
           (y (make-tensor32 (f32vector 1.0 2.0 3.0) '(3)))
           (z (add x y)))
      (tensor-data z)))
  
  (test-group "Gradients"
    (let* ((x (make-tensor32 (f32vector 2.0 3.0) '(2)))
           (y (make-tensor32 (f32vector 1.0 2.0) '(2)))
           (z (add x y)))
      (backward! z)
      
      (test "dL/dx = [1,1]"
        (f32vector 1.0 1.0)
        (tensor-grad x))
      
      (test "dL/dy = [1,1]"
        (f32vector 1.0 1.0)
        (tensor-grad y)))))

(test-group "Subtraction"
  
  (test "Forward: [5,7,9] - [2,3,4] = [3,4,5]"
    (f32vector 3.0 4.0 5.0)
    (let* ((x (make-tensor32 (f32vector 5.0 7.0 9.0) '(3)))
           (y (make-tensor32 (f32vector 2.0 3.0 4.0) '(3)))
           (z (sub x y)))
      (tensor-data z)))
  
  (test-group "Gradients"
    (let* ((x (make-tensor32 (f32vector 5.0 7.0) '(2)))
           (y (make-tensor32 (f32vector 2.0 3.0) '(2)))
           (z (sub x y)))
      (backward! z)
      
      (test "dL/dx = [1,1]"
        (f32vector 1.0 1.0)
        (tensor-grad x))
      
      (test "dL/dy = [-1,-1]"
        (f32vector -1.0 -1.0)
        (tensor-grad y)))))

(test-group "Multiplication"
  
  (test "Forward: [2,3,4] * [5,6,7] = [10,18,28]"
    (f32vector 10.0 18.0 28.0)
    (let* ((x (make-tensor32 (f32vector 2.0 3.0 4.0) '(3)))
           (y (make-tensor32 (f32vector 5.0 6.0 7.0) '(3)))
           (z (mul x y)))
      (tensor-data z)))
  
  (test-group "Gradients"
    (let* ((x (make-tensor32 (f32vector 2.0 3.0) '(2)))
           (y (make-tensor32 (f32vector 4.0 5.0) '(2)))
           (z (mul x y)))
      (backward! z)
      
      (test "dL/dx = y = [4,5]"
        (f32vector 4.0 5.0)
        (tensor-grad x))
      
      (test "dL/dy = x = [2,3]"
        (f32vector 2.0 3.0)
        (tensor-grad y)))))

(test-group "Division"
  
  (test "Forward: [10,15,20] / [2,3,4] = [5,5,5]"
    (f32vector 5.0 5.0 5.0)
    (let* ((x (make-tensor32 (f32vector 10.0 15.0 20.0) '(3)))
           (y (make-tensor32 (f32vector 2.0 3.0 4.0) '(3)))
           (z (div x y)))
      (tensor-data z)))
  
  (test-group "Gradients"
    (let* ((x (make-tensor32 (f32vector 6.0 8.0) '(2)))
           (y (make-tensor32 (f32vector 2.0 4.0) '(2)))
           (z (div x y)))
      (backward! z)
      
      (test-vector-equal (tensor-grad x) (f32vector 0.5 0.25) 1e-5)
      (test-vector-equal (tensor-grad y) (f32vector -1.5 -0.5) 1e-5))))

;;; ==================================================================
;;; Unit Tests: Matrix Operations
;;; ==================================================================

(test-group "Matrix Multiplication"
  
  (test "Matrix-vector product"
    (f32vector 17.0 39.0)
    (let* ((A (make-tensor32 (f32vector 1.0 2.0
                                        3.0 4.0) '(2 2)))
           (x (make-tensor32 (f32vector 5.0 6.0) '(2)))
           (y (matmul-op A x)))
      (tensor-data y)))
  
  (test "Matrix-matrix product"
    (f32vector 19.0 22.0 43.0 50.0)
    (let* ((A (make-tensor32 (f32vector 1.0 2.0
                                        3.0 4.0) '(2 2)))
           (B (make-tensor32 (f32vector 5.0 6.0
                                        7.0 8.0) '(2 2)))
           (C (matmul-op A B)))
      (tensor-data C))))

(test-group "Dot Product"
  
  (test-approximate "Dot product: [1,2,3] · [4,5,6] = 32"
    32.0
    (let* ((x (make-tensor32 (f32vector 1.0 2.0 3.0) '(3)))
           (y (make-tensor32 (f32vector 4.0 5.0 6.0) '(3)))
           (z (dot-op x y)))
      (f32vector-ref (tensor-data z) 0))
    1e-5)
  
  (test-group "Gradients"
    (let* ((x (make-tensor32 (f32vector 2.0 3.0) '(2)))
           (y (make-tensor32 (f32vector 4.0 5.0) '(2)))
           (z (dot-op x y)))
      (backward! z)
      
      (test "dL/dx = y"
        (f32vector 4.0 5.0)
        (tensor-grad x))
      
      (test "dL/dy = x"
        (f32vector 2.0 3.0)
        (tensor-grad y)))))

;;; ==================================================================
;;; Unit Tests: Activation Functions
;;; ==================================================================

(test-group "ReLU"
  
  (test "Forward pass"
    (f32vector 0.0 0.0 0.0 1.0 2.0)
    (let* ((x (make-tensor32 (f32vector -2.0 -1.0 0.0 1.0 2.0) '(5)))
           (y (relu x)))
      (tensor-data y)))
  
  (test "Gradient"
    (f32vector 0.0 0.0 1.0 1.0)
    (let* ((x (make-tensor32 (f32vector -1.0 0.0 1.0 2.0) '(4)))
           (y (relu x)))
      (backward! y)
      (tensor-grad x))))

(test-group "Sigmoid"
  
  (test-approximate "Sigmoid(0) = 0.5"
    0.5
    (let* ((x (make-tensor32 (f32vector 0.0) '(1)))
           (y (sigmoid x)))
      (f32vector-ref (tensor-data y) 0))
    1e-5)
  
  (test-approximate "Sigmoid gradient at 0 = 0.25"
    0.25
    (let* ((x (make-tensor32 (f32vector 0.0) '(1)))
           (y (sigmoid x)))
      (backward! y)
      (f32vector-ref (tensor-grad x) 0))
    1e-5))

(test-group "Tanh"
  
  (test-approximate "Tanh(0) = 0"
    0.0
    (let* ((x (make-tensor32 (f32vector 0.0) '(1)))
           (y (tanh-op x)))
      (f32vector-ref (tensor-data y) 0))
    1e-5)
  
  (test-approximate "Tanh gradient at 0 = 1.0"
    1.0
    (let* ((x (make-tensor32 (f32vector 0.0) '(1)))
           (y (tanh-op x)))
      (backward! y)
      (f32vector-ref (tensor-grad x) 0))
    1e-5))

(test-group "Softmax"
  
  (test "Uniform input gives equal probabilities"
    #t
    (let* ((x (make-tensor32 (f32vector 0.0 0.0 0.0) '(3)))
           (y (softmax x))
           (data (tensor-data y)))
      (vector-approx-equal? data (f32vector 0.333333 0.333333 0.333333) 1e-4)))
  
  (test-approximate "Probabilities sum to 1"
    1.0
    (let* ((x (make-tensor32 (f32vector 1.0 2.0 3.0) '(3)))
           (y (softmax x))
           (sum (apply + (f32vector->list (tensor-data y)))))
      sum)
    1e-5))

;;; ==================================================================
;;; Unit Tests: Loss Functions
;;; ==================================================================

(test-group "MSE Loss"
  
  (test-approximate "Perfect prediction gives zero loss"
    0.0
    (let* ((pred (make-tensor32 (f32vector 1.0 2.0 3.0) '(3)))
           (target (make-tensor32 (f32vector 1.0 2.0 3.0) '(3)))
           (loss (mse-loss pred target)))
      (f32vector-ref (tensor-data loss) 0))
    1e-5)
  
  (test-approximate "Known loss value"
    1.25
    (let* ((pred (make-tensor32 (f32vector 1.0 2.0) '(2)))
           (target (make-tensor32 (f32vector 0.0 0.0) '(2)))
           (loss (mse-loss pred target)))
      (f32vector-ref (tensor-data loss) 0))
    1e-5))

;;; ==================================================================
;;; Unit Tests: Advanced Operations
;;; ==================================================================

(test-group "Reshape"
  
  (test "Reshape (2,2) -> (4)"
    '(4)
    (let* ((x (make-tensor32 (f32vector 1.0 2.0 3.0 4.0) '(2 2)))
           (y (reshape x '(4))))
      (tensor-shape y)))
  
  (test "Reshape (6) -> (2,3)"
    '(2 3)
    (let* ((x (make-tensor32 (f32vector 1.0 2.0 3.0 4.0 5.0 6.0) '(6)))
           (y (reshape x '(2 3))))
      (tensor-shape y))))

(test-group "RMSNorm"
  
  (test "Basic normalization"
    #t
    (let* ((x (make-tensor32 (f32vector 1.0 2.0 3.0 4.0) '(4)))
           (weight (make-tensor32 (f32vector 1.0 1.0 1.0 1.0) '(4)))
           (y (rmsnorm x weight))
           (expected (f32vector 0.365 0.730 1.095 1.460)))
      (vector-approx-equal? (tensor-data y) expected 1e-2))))

(test-group "Conv2D"
  
  (test-approximate "Identity convolution extracts center"
    5.0
    (let* ((input (make-tensor32 (f32vector 
                                  1.0 2.0 3.0
                                  4.0 5.0 6.0
                                  7.0 8.0 9.0) '(1 3 3)))
           (weight (make-tensor32 (f32vector
                                   0.0 0.0 0.0
                                   0.0 1.0 0.0
                                   0.0 0.0 0.0) '(1 1 3 3)))
           (bias #f)
           (output (conv2d input weight bias stride: 1 padding: 0)))
      (f32vector-ref (tensor-data output) 0))
    1e-5)
  
  (test "Output shape"
    '(1 1 1)
    (let* ((input (make-tensor32 (f32vector 
                                  1.0 2.0 3.0
                                  4.0 5.0 6.0
                                  7.0 8.0 9.0) '(1 3 3)))
           (weight (make-tensor32 (f32vector
                                   0.0 0.0 0.0
                                   0.0 1.0 0.0
                                   0.0 0.0 0.0) '(1 1 3 3)))
           (bias #f)
           (output (conv2d input weight bias stride: 1 padding: 0)))
      (tensor-shape output)))
  
  (test-approximate "Edge detection convolution"
    8.0
    (let* ((input (make-tensor32 (f32vector 
                                  1.0 2.0 3.0
                                  4.0 5.0 6.0
                                  7.0 8.0 9.0) '(1 3 3)))
           (weight (make-tensor32 (f32vector
                                   -1.0 0.0 1.0
                                   -2.0 0.0 2.0
                                   -1.0 0.0 1.0) '(1 1 3 3)))
           (bias #f)
           (output (conv2d input weight bias stride: 1 padding: 0)))
      (f32vector-ref (tensor-data output) 0))
    1e-5))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(test-exit)

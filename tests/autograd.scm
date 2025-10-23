;; unit tests for autograd operations

(import scheme
        (chicken base)
        (chicken format)
        (srfi 1)
        (srfi 4)
        blas
        nanograd-autograd)

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

(define (assert-vector-equal vec1 vec2 tolerance name)
  (set! *test-count* (+ *test-count* 1))
  (let ((n1 (f32vector-length vec1))
        (n2 (f32vector-length vec2)))
    (if (not (= n1 n2))
        (begin
          (set! *test-failed* (+ *test-failed* 1))
          (printf "  âœ— ~A (length mismatch: ~A vs ~A)\n" name n1 n2))
        (let loop ((i 0) (max-diff 0.0) (all-ok? #t))
          (if (= i n1)
              (if all-ok?
                  (begin
                    (set! *test-passed* (+ *test-passed* 1))
                    (printf "  O ~A\n" name))
                  (begin
                    (set! *test-failed* (+ *test-failed* 1))
                    (printf "  X ~A (max diff: ~A > ~A)\n" name max-diff tolerance)))
              (let ((diff (abs (- (f32vector-ref vec1 i)
                                 (f32vector-ref vec2 i)))))
                (loop (+ i 1) 
                      (max max-diff diff)
                      (and all-ok? (<= diff tolerance)))))))))

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

;;; ==================================================================
;;; Unit Tests: Basic Arithmetic Operations
;;; ==================================================================

(define (test-addition)
  (printf "\n=== Testing Addition ===\n")
  
  ;; Test 1: Simple addition
  (let* ((x (make-tensor32 (f32vector 2.0 3.0 4.0) '(3)))
         (y (make-tensor32 (f32vector 1.0 2.0 3.0) '(3)))
         (z (add x y))
         (expected (f32vector 3.0 5.0 7.0)))
    (assert-vector-equal (tensor-data z) expected 1e-5 
                        "Forward: [2,3,4] + [1,2,3] = [3,5,7]"))
  
  ;; Test 2: Gradient check
  (let* ((x (make-tensor32 (f32vector 2.0 3.0) '(2)))
         (y (make-tensor32 (f32vector 1.0 2.0) '(2)))
         (z (add x y)))
    (backward! z)
    (assert-vector-equal (tensor-grad x) (f32vector 1.0 1.0) 1e-5
                        "Gradient dL/dx = [1,1]")
    (assert-vector-equal (tensor-grad y) (f32vector 1.0 1.0) 1e-5
                        "Gradient dL/dy = [1,1]")))

(define (test-subtraction)
  (printf "\n=== Testing Subtraction ===\n")
  
  ;; Test 1: Simple subtraction
  (let* ((x (make-tensor32 (f32vector 5.0 7.0 9.0) '(3)))
         (y (make-tensor32 (f32vector 2.0 3.0 4.0) '(3)))
         (z (sub x y))
         (expected (f32vector 3.0 4.0 5.0)))
    (assert-vector-equal (tensor-data z) expected 1e-5
                        "Forward: [5,7,9] - [2,3,4] = [3,4,5]"))
  
  ;; Test 2: Gradient check
  (let* ((x (make-tensor32 (f32vector 5.0 7.0) '(2)))
         (y (make-tensor32 (f32vector 2.0 3.0) '(2)))
         (z (sub x y)))
    (backward! z)
    (assert-vector-equal (tensor-grad x) (f32vector 1.0 1.0) 1e-5
                        "Gradient dL/dx = [1,1]")
    (assert-vector-equal (tensor-grad y) (f32vector -1.0 -1.0) 1e-5
                        "Gradient dL/dy = [-1,-1]")))

(define (test-multiplication)
  (printf "\n=== Testing Multiplication ===\n")
  
  ;; Test 1: Element-wise multiplication
  (let* ((x (make-tensor32 (f32vector 2.0 3.0 4.0) '(3)))
         (y (make-tensor32 (f32vector 5.0 6.0 7.0) '(3)))
         (z (mul x y))
         (expected (f32vector 10.0 18.0 28.0)))
    (assert-vector-equal (tensor-data z) expected 1e-5
                        "Forward: [2,3,4] * [5,6,7] = [10,18,28]"))
  
  ;; Test 2: Gradients
  (let* ((x (make-tensor32 (f32vector 2.0 3.0) '(2)))
         (y (make-tensor32 (f32vector 4.0 5.0) '(2)))
         (z (mul x y)))
    (backward! z)
    (assert-vector-equal (tensor-grad x) (f32vector 4.0 5.0) 1e-5
                        "Gradient dL/dx = y = [4,5]")
    (assert-vector-equal (tensor-grad y) (f32vector 2.0 3.0) 1e-5
                        "Gradient dL/dy = x = [2,3]")))

(define (test-division)
  (printf "\n=== Testing Division ===\n")
  
  ;; Test 1: Element-wise division
  (let* ((x (make-tensor32 (f32vector 10.0 15.0 20.0) '(3)))
         (y (make-tensor32 (f32vector 2.0 3.0 4.0) '(3)))
         (z (div x y))
         (expected (f32vector 5.0 5.0 5.0)))
    (assert-vector-equal (tensor-data z) expected 1e-5
                        "Forward: [10,15,20] / [2,3,4] = [5,5,5]"))
  
  ;; Test 2: Gradients
  (let* ((x (make-tensor32 (f32vector 6.0 8.0) '(2)))
         (y (make-tensor32 (f32vector 2.0 4.0) '(2)))
         (z (div x y)))
    (backward! z)
    (assert-vector-equal (tensor-grad x) (f32vector 0.5 0.25) 1e-5
                        "Gradient dL/dx = 1/y = [0.5, 0.25]")
    (assert-vector-equal (tensor-grad y) (f32vector -1.5 -0.5) 1e-5
                        "Gradient dL/dy = -x/y^2 = [-1.5, -0.5]")))

;;; ==================================================================
;;; Unit Tests: Matrix Operations
;;; ==================================================================

(define (test-matmul)
  (printf "\n=== Testing Matrix Multiplication ===\n")
  
  ;; Test 1: Matrix @ Vector
  (let* ((A (make-tensor32 (f32vector 1.0 2.0
                                      3.0 4.0) '(2 2)))
         (x (make-tensor32 (f32vector 5.0 6.0) '(2)))
         (y (matmul-op A x))
         ;; [1 2] [5]   [17]
         ;; [3 4] [6] = [39]
         (expected (f32vector 17.0 39.0)))
    (assert-vector-equal (tensor-data y) expected 1e-5
                        "Matrix-vector product"))
  
  ;; Test 2: Matrix @ Matrix
  (let* ((A (make-tensor32 (f32vector 1.0 2.0
                                      3.0 4.0) '(2 2)))
         (B (make-tensor32 (f32vector 5.0 6.0
                                      7.0 8.0) '(2 2)))
         (C (matmul-op A B))
         ;; [1 2] [5 6]   [19 22]
         ;; [3 4] [7 8] = [43 50]
         (expected (f32vector 19.0 22.0 43.0 50.0)))
    (assert-vector-equal (tensor-data C) expected 1e-5
                        "Matrix-matrix product")))

(define (test-dot-product)
  (printf "\n=== Testing Dot Product ===\n")
  
  ;; Test 1: Simple dot product
  (let* ((x (make-tensor32 (f32vector 1.0 2.0 3.0) '(3)))
         (y (make-tensor32 (f32vector 4.0 5.0 6.0) '(3)))
         (z (dot-op x y))
         ;; 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
         (expected 32.0))
    (assert-equal (f32vector-ref (tensor-data z) 0) expected 1e-5
                  "Dot product: [1,2,3] · [4,5,6] = 32"))
  
  ;; Test 2: Gradient check
  (let* ((x (make-tensor32 (f32vector 2.0 3.0) '(2)))
         (y (make-tensor32 (f32vector 4.0 5.0) '(2)))
         (z (dot-op x y)))
    (backward! z)
    (assert-vector-equal (tensor-grad x) (f32vector 4.0 5.0) 1e-5
                        "Gradient dL/dx = y")
    (assert-vector-equal (tensor-grad y) (f32vector 2.0 3.0) 1e-5
                        "Gradient dL/dy = x")))

;;; ==================================================================
;;; Unit Tests: Activation Functions
;;; ==================================================================

(define (test-relu)
  (printf "\n=== Testing ReLU ===\n")
  
  ;; Test 1: Forward pass
  (let* ((x (make-tensor32 (f32vector -2.0 -1.0 0.0 1.0 2.0) '(5)))
         (y (relu x))
         (expected (f32vector 0.0 0.0 0.0 1.0 2.0)))
    (assert-vector-equal (tensor-data y) expected 1e-5
                        "ReLU([-2,-1,0,1,2]) = [0,0,0,1,2]"))
  
  ;; Test 2: Gradient
  (let* ((x (make-tensor32 (f32vector -1.0 0.0 1.0 2.0) '(4)))
         (y (relu x)))
    (backward! y)
    (assert-vector-equal (tensor-grad x) (f32vector 0.0 0.0 1.0 1.0) 1e-5
                        "ReLU gradient: [0,0,1,1]")))

(define (test-sigmoid)
  (printf "\n=== Testing Sigmoid ===\n")
  
  ;; Test 1: Forward at 0
  (let* ((x (make-tensor32 (f32vector 0.0) '(1)))
         (y (sigmoid x))
         (expected 0.5))
    (assert-equal (f32vector-ref (tensor-data y) 0) expected 1e-5
                  "Sigmoid(0) = 0.5"))
  
  ;; Test 2: Gradient at 0
  (let* ((x (make-tensor32 (f32vector 0.0) '(1)))
         (y (sigmoid x)))
    (backward! y)
    ;; σ'(0) = σ(0) * (1 - σ(0)) = 0.5 * 0.5 = 0.25
    (assert-equal (f32vector-ref (tensor-grad x) 0) 0.25 1e-5
                  "Sigmoid gradient at 0 = 0.25")))

(define (test-tanh)
  (printf "\n=== Testing Tanh ===\n")
  
  ;; Test 1: Forward at 0
  (let* ((x (make-tensor32 (f32vector 0.0) '(1)))
         (y (tanh-op x))
         (expected 0.0))
    (assert-equal (f32vector-ref (tensor-data y) 0) expected 1e-5
                  "Tanh(0) = 0"))
  
  ;; Test 2: Gradient at 0
  (let* ((x (make-tensor32 (f32vector 0.0) '(1)))
         (y (tanh-op x)))
    (backward! y)
    ;; tanh'(0) = 1 - tanh²(0) = 1 - 0 = 1
    (assert-equal (f32vector-ref (tensor-grad x) 0) 1.0 1e-5
                  "Tanh gradient at 0 = 1.0")))

(define (test-softmax)
  (printf "\n=== Testing Softmax ===\n")
  
  ;; Test 1: Uniform input
  (let* ((x (make-tensor32 (f32vector 0.0 0.0 0.0) '(3)))
         (y (softmax x))
         ;; All equal -> equal probabilities
         (expected (f32vector 0.333333 0.333333 0.333333)))
    (assert-vector-equal (tensor-data y) expected 1e-4
                        "Softmax([0,0,0]) = [0.33,0.33,0.33]"))
  
  ;; Test 2: Sum to 1
  (let* ((x (make-tensor32 (f32vector 1.0 2.0 3.0) '(3)))
         (y (softmax x))
         (sum (apply + (f32vector->list (tensor-data y)))))
    (assert-equal sum 1.0 1e-5
                  "Softmax probabilities sum to 1")))

;;; ==================================================================
;;; Unit Tests: Loss Functions
;;; ==================================================================

(define (test-mse-loss)
  (printf "\n=== Testing MSE Loss ===\n")
  
  ;; Test 1: Perfect prediction
  (let* ((pred (make-tensor32 (f32vector 1.0 2.0 3.0) '(3)))
         (target (make-tensor32 (f32vector 1.0 2.0 3.0) '(3)))
         (loss (mse-loss pred target))
         (expected 0.0))
    (assert-equal (f32vector-ref (tensor-data loss) 0) expected 1e-5
                  "MSE loss for perfect prediction = 0"))
  
  ;; Test 2: Known loss
  (let* ((pred (make-tensor32 (f32vector 1.0 2.0) '(2)))
         (target (make-tensor32 (f32vector 0.0 0.0) '(2)))
         (loss (mse-loss pred target))
         ;; MSE = (1² + 2²)/2 = 5/2 = 2.5
         (expected 2.5))
    (assert-equal (f32vector-ref (tensor-data loss) 0) expected 1e-5
                  "MSE loss = 2.5")))

;;; ==================================================================
;;; Unit Tests: Advanced Operations
;;; ==================================================================

(define (test-reshape)
  (printf "\n=== Testing Reshape ===\n")
  
  ;; Test 1: 2D to 1D
  (let* ((x (make-tensor32 (f32vector 1.0 2.0 3.0 4.0) '(2 2)))
         (y (reshape x '(4))))
    (assert-shape-equal y '(4) "Reshape (2,2) -> (4)"))
  
  ;; Test 2: 1D to 2D
  (let* ((x (make-tensor32 (f32vector 1.0 2.0 3.0 4.0 5.0 6.0) '(6)))
         (y (reshape x '(2 3))))
    (assert-shape-equal y '(2 3) "Reshape (6) -> (2,3)")))

(define (test-rmsnorm)
  (printf "\n=== Testing RMSNorm ===\n")
  
  ;; Test 1: Basic normalization
  (let* ((x (make-tensor32 (f32vector 1.0 2.0 3.0 4.0) '(4)))
         (weight (make-tensor32 (f32vector 1.0 1.0 1.0 1.0) '(4)))
         (y (rmsnorm x weight))
         ;; RMS = sqrt((1 + 4 + 9 + 16)/4) = sqrt(7.5) ≈ 2.739
         ;; Normalized: [0.365, 0.730, 1.095, 1.460]
         (expected (f32vector 0.365 0.730 1.095 1.460)))
    (assert-vector-equal (tensor-data y) expected 1e-2
                        "RMSNorm basic normalization")))

(define (test-conv2d)
  (printf "\n=== Testing Conv2D ===\n")
  
  ;; Test 1: Identity convolution
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
    (assert-shape-equal output '(1 1 1) "Conv2D output shape")
    (assert-equal (f32vector-ref (tensor-data output) 0) 5.0 1e-5
                  "Identity conv extracts center = 5.0"))
  
  ;; Test 2: Edge detection
  (let* ((input (make-tensor32 (f32vector 
                                1.0 2.0 3.0
                                4.0 5.0 6.0
                                7.0 8.0 9.0) '(1 3 3)))
         (weight (make-tensor32 (f32vector
                                 -1.0 0.0 1.0
                                 -2.0 0.0 2.0
                                 -1.0 0.0 1.0) '(1 1 3 3)))
         (bias #f)
         (output (conv2d input weight bias stride: 1 padding: 0))
         ;; sum = -1*1 + 0*2 + 1*3 + -2*4 + 0*5 + 2*6 + -1*7 + 0*8 + 1*9 = 8
         (expected 8.0))
    (assert-equal (f32vector-ref (tensor-data output) 0) expected 1e-5
                  "Edge detection conv = 8.0")))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(define (run-all-autograd-tests)
  (reset-test-stats!)
  (printf "\n")
  (printf "========================================\n")
  (printf "AUTOGRAD UNIT TESTS\n")
  (printf "========================================\n")
  
  (test-addition)
  (test-subtraction)
  (test-multiplication)
  (test-division)
  (test-matmul)
  (test-dot-product)
  (test-relu)
  (test-sigmoid)
  (test-tanh)
  (test-softmax)
  (test-mse-loss)
  (test-reshape)
  (test-rmsnorm)
  (test-conv2d)
  
  (test-summary))

;; Run all tests
(run-all-autograd-tests)

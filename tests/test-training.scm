;; Unit tests for gradient clipping by value in nanograd-training

(import scheme
        (chicken base)
        (chicken format)
        (srfi 1)
        (srfi 4)
        test
        nanograd-autograd
        nanograd-training)

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

(define tol 1e-6)

(define (grad->list tensor)
  "The gradient of TENSOR as a list."
  (case (tensor-dtype tensor)
    ((f32) (f32vector->list (tensor-grad tensor)))
    ((f64) (f64vector->list (tensor-grad tensor)))))

(define (lists-approx-equal? actual expected tolerance)
  (and (= (length actual) (length expected))
       (every (lambda (a e) (approx-equal? a e tolerance)) actual expected)))

(define (make-param32 shape grad-values)
  "An f32 parameter of SHAPE whose gradient holds GRAD-VALUES."
  (let ((p (make-tensor32 (make-f32vector (apply * shape) 0.0) shape)))
    (set-grad! p (list->f32vector grad-values))
    p))

(define (make-param64 shape grad-values)
  "An f64 parameter of SHAPE whose gradient holds GRAD-VALUES."
  (let ((p (make-tensor64 (make-f64vector (apply * shape) 0.0) shape)))
    (set-grad! p (list->f64vector grad-values))
    p))

;;; ==================================================================
;;; Unit Tests: clip-gradients-by-value!
;;; ==================================================================

(test-group "Clip by Value: f32 Gradients"
  (let* ((p (make-param32 '(3) '(-2.0 0.3 5.0)))
         (grad-before (tensor-grad p))
         (n-clipped (clip-gradients-by-value! (list p) 1.0)))
    (test "two elements clipped" 2 n-clipped)
    (test-assert "gradient is clipped to [-1, 1]"
                 (lists-approx-equal? (grad->list p) '(-1.0 0.3 1.0) tol))
    (test-assert "gradient vector is modified in place"
                 (eq? grad-before (tensor-grad p)))))

(test-group "Clip by Value: f64 Gradients"
  (let* ((p (make-param64 '(4) '(-0.75 0.25 0.6 -0.1)))
         (n-clipped (clip-gradients-by-value! (list p) 0.5)))
    (test "two elements clipped" 2 n-clipped)
    (test-assert "gradient is clipped to [-0.5, 0.5]"
                 (lists-approx-equal? (grad->list p) '(-0.5 0.25 0.5 -0.1) 1e-12))))

(test-group "Clip by Value: Multi-Dimensional Parameter"
  (let* ((p (make-param32 '(2 3) '(3.0 -3.0 0.0 1.5 -0.5 2.0)))
         (n-clipped (clip-gradients-by-value! (list p) 1.0)))
    (test "every element is visited" 4 n-clipped)
    (test-assert "all six elements are within bounds"
                 (lists-approx-equal? (grad->list p)
                                      '(1.0 -1.0 0.0 1.0 -0.5 1.0) tol))))

(test-group "Clip by Value: Several Parameters"
  (let* ((p1 (make-param32 '(2) '(4.0 0.1)))
         (p2 (make-param32 '(3) '(-4.0 -5.0 0.2)))
         (n-clipped (clip-gradients-by-value! (list p1 p2) 1.0)))
    (test "counts are summed over parameters" 3 n-clipped)
    (test-assert "first parameter clipped"
                 (lists-approx-equal? (grad->list p1) '(1.0 0.1) tol))
    (test-assert "second parameter clipped"
                 (lists-approx-equal? (grad->list p2) '(-1.0 -1.0 0.2) tol))))

(test-group "Clip by Value: Boundary Values"
  ;; Elements exactly at +/- clip-value are within bounds and not counted
  (let* ((p (make-param32 '(3) '(1.0 -1.0 0.0)))
         (n-clipped (clip-gradients-by-value! (list p) 1.0)))
    (test "nothing clipped" 0 n-clipped)
    (test-assert "gradient unchanged"
                 (lists-approx-equal? (grad->list p) '(1.0 -1.0 0.0) tol))))

(test-group "Clip by Value: Parameters Without Gradients"
  (let ((frozen (make-tensor32 (f32vector 1.0 2.0) '(2) requires-grad?: #f))
        (fresh (make-tensor32 (f32vector 1.0 2.0) '(2))))
    (test "frozen parameter is skipped" 0
          (clip-gradients-by-value! (list frozen) 1.0))
    (test-assert "frozen parameter still has no gradient"
                 (not (tensor-grad frozen)))
    (test "zero gradient needs no clipping" 0
          (clip-gradients-by-value! (list fresh) 1.0))
    (test "empty parameter list" 0
          (clip-gradients-by-value! '() 1.0))))

(test-group "Clip by Value: Gradients From Backward"
  ;; loss = sum(10 * x), so every element of the gradient is 10
  (let* ((x (make-tensor32 (f32vector 1.0 2.0 3.0) '(3)))
         (loss (sum-tensor (scale-op x 10.0))))
    (backward! loss)
    (test-assert "backward gradient is 10"
                 (lists-approx-equal? (grad->list x) '(10.0 10.0 10.0) tol))
    (test "all elements clipped" 3 (clip-gradients-by-value! (list x) 2.0))
    (test-assert "gradient clipped to 2"
                 (lists-approx-equal? (grad->list x) '(2.0 2.0 2.0) tol))
    (test-approximate "global norm after clipping" (* 2.0 (sqrt 3.0))
                      (compute-gradient-norm (list x)) 1e-5)))

(test-exit)

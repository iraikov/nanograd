;; Tests for tensor slice operation

(import scheme
        (chicken base)
        (chicken format)
        (srfi 1)
        (srfi 4)
        test
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
;;; Basic Slicing - Shape and Data
;;; ==================================================================

(test-group "Basic Slicing"
  
  (test-assert "Slice shape [0:1] is (1, 2)"
    (let* ((input (make-tensor32 
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)))
           (slice1 (slice-tensor input 0 1)))
      (equal? (tensor-shape slice1) '(1 2))))
  
  (test-assert "Slice [0:1] first element is 1.0"
    (let* ((input (make-tensor32 
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)))
           (slice1 (slice-tensor input 0 1)))
      (approx-equal? (f32vector-ref (tensor-data slice1) 0) 1.0 1e-6)))
  
  (test-assert "Slice [0:1] second element is 2.0"
    (let* ((input (make-tensor32 
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)))
           (slice1 (slice-tensor input 0 1)))
      (approx-equal? (f32vector-ref (tensor-data slice1) 1) 2.0 1e-6)))
  
  (test-assert "Slice shape [1:2] is (1, 2)"
    (let* ((input (make-tensor32 
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)))
           (slice2 (slice-tensor input 1 1)))
      (equal? (tensor-shape slice2) '(1 2))))
  
  (test-assert "Slice [1:2] first element is 3.0"
    (let* ((input (make-tensor32 
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)))
           (slice2 (slice-tensor input 1 1)))
      (approx-equal? (f32vector-ref (tensor-data slice2) 0) 3.0 1e-6)))
  
  (test-assert "Slice [1:2] second element is 4.0"
    (let* ((input (make-tensor32 
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)))
           (slice2 (slice-tensor input 1 1)))
      (approx-equal? (f32vector-ref (tensor-data slice2) 1) 4.0 1e-6)))
  
  (test-assert "Slice shape [2:3] is (1, 2)"
    (let* ((input (make-tensor32 
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)))
           (slice3 (slice-tensor input 2 1)))
      (equal? (tensor-shape slice3) '(1 2))))
  
  (test-assert "Slice [2:3] first element is 5.0"
    (let* ((input (make-tensor32 
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)))
           (slice3 (slice-tensor input 2 1)))
      (approx-equal? (f32vector-ref (tensor-data slice3) 0) 5.0 1e-6)))
  
  (test-assert "Slice [2:3] second element is 6.0"
    (let* ((input (make-tensor32 
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)))
           (slice3 (slice-tensor input 2 1)))
      (approx-equal? (f32vector-ref (tensor-data slice3) 1) 6.0 1e-6))))

;;; ==================================================================
;;; Multi-row Slicing
;;; ==================================================================

(test-group "Multi-row Slicing"
  
  (test-assert "Slice [0:2] shape is (2, 2)"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)))
           (slice (slice-tensor input 0 2)))
      (equal? (tensor-shape slice) '(2 2))))
  
  (test-assert "Slice [0:2] element [0,0] is 1.0"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)))
           (slice (slice-tensor input 0 2)))
      (approx-equal? (f32vector-ref (tensor-data slice) 0) 1.0 1e-6)))
  
  (test-assert "Slice [0:2] element [0,1] is 2.0"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)))
           (slice (slice-tensor input 0 2)))
      (approx-equal? (f32vector-ref (tensor-data slice) 1) 2.0 1e-6)))
  
  (test-assert "Slice [0:2] element [1,0] is 3.0"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)))
           (slice (slice-tensor input 0 2)))
      (approx-equal? (f32vector-ref (tensor-data slice) 2) 3.0 1e-6)))
  
  (test-assert "Slice [0:2] element [1,1] is 4.0"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)))
           (slice (slice-tensor input 0 2)))
      (approx-equal? (f32vector-ref (tensor-data slice) 3) 4.0 1e-6)))
  
  (test-assert "Slice [1:3] shape is (2, 2)"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)))
           (slice (slice-tensor input 1 2)))
      (equal? (tensor-shape slice) '(2 2))))
  
  (test-assert "Slice [1:3] element [0,0] is 3.0"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)))
           (slice (slice-tensor input 1 2)))
      (approx-equal? (f32vector-ref (tensor-data slice) 0) 3.0 1e-6)))
  
  (test-assert "Slice [1:3] element [1,1] is 6.0"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)))
           (slice (slice-tensor input 1 2)))
      (approx-equal? (f32vector-ref (tensor-data slice) 3) 6.0 1e-6)))
  
  (test-assert "Slice [2:4] shape is (2, 2)"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)))
           (slice (slice-tensor input 2 2)))
      (equal? (tensor-shape slice) '(2 2))))
  
  (test-assert "Slice [2:4] element [0,0] is 5.0"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)))
           (slice (slice-tensor input 2 2)))
      (approx-equal? (f32vector-ref (tensor-data slice) 0) 5.0 1e-6)))
  
  (test-assert "Slice [2:4] element [1,1] is 8.0"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)))
           (slice (slice-tensor input 2 2)))
      (approx-equal? (f32vector-ref (tensor-data slice) 3) 8.0 1e-6))))

;;; ==================================================================
;;; Gradient Flow - Simple Case
;;; ==================================================================

(test-group "Simple Gradient Flow"
  
  (test-assert "Input has gradient after backward"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice (squeeze (slice-tensor input 1 1)))
           (target (make-tensor32 (f32vector 10.0 20.0) '(2)))
           (loss (mse-loss slice target)))
      (backward! loss)
      (tensor-grad input)))
  
  (test-assert "Gradient at position [0,0] is 0 (not sliced)"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice (squeeze (slice-tensor input 1 1)))
           (target (make-tensor32 (f32vector 10.0 20.0) '(2)))
           (loss (mse-loss slice target)))
      (backward! loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 0) 0.0 1e-6)))
  
  (test-assert "Gradient at position [0,1] is 0 (not sliced)"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice (squeeze (slice-tensor input 1 1)))
           (target (make-tensor32 (f32vector 10.0 20.0) '(2)))
           (loss (mse-loss slice target)))
      (backward! loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 1) 0.0 1e-6)))
  
  (test-assert "Gradient at position [1,0] is -3.5 (sliced)"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice (squeeze (slice-tensor input 1 1)))
           (target (make-tensor32 (f32vector 10.0 20.0) '(2)))
           (loss (mse-loss slice target)))
      (backward! loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 2) -3.5 1e-5)))
  
  (test-assert "Gradient at position [1,1] is -8.0 (sliced)"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice (squeeze (slice-tensor input 1 1)))
           (target (make-tensor32 (f32vector 10.0 20.0) '(2)))
           (loss (mse-loss slice target)))
      (backward! loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 3) -8.0 1e-5)))
  
  (test-assert "Gradient at position [2,0] is 0 (not sliced)"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice (squeeze (slice-tensor input 1 1)))
           (target (make-tensor32 (f32vector 10.0 20.0) '(2)))
           (loss (mse-loss slice target)))
      (backward! loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 4) 0.0 1e-6)))
  
  (test-assert "Gradient at position [2,1] is 0 (not sliced)"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice (squeeze (slice-tensor input 1 1)))
           (target (make-tensor32 (f32vector 10.0 20.0) '(2)))
           (loss (mse-loss slice target)))
      (backward! loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 5) 0.0 1e-6))))

;;; ==================================================================
;;; Gradient Accumulation
;;; ==================================================================

(test-group "Gradient Accumulation"
  
  (test-assert "Gradient at [0,0] from first slice"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice1 (squeeze (slice-tensor input 0 1)))
           (slice2 (squeeze (slice-tensor input 2 1)))
           (target1 (make-tensor32 (f32vector 2.0 4.0) '(2)))
           (target2 (make-tensor32 (f32vector 6.0 8.0) '(2)))
           (loss1 (mse-loss slice1 target1))
           (loss2 (mse-loss slice2 target2))
           (total-loss (add loss1 loss2)))
      (backward! total-loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 0) -0.5 1e-6)))
  
  (test-assert "Gradient at [0,1] from first slice"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice1 (squeeze (slice-tensor input 0 1)))
           (slice2 (squeeze (slice-tensor input 2 1)))
           (target1 (make-tensor32 (f32vector 2.0 4.0) '(2)))
           (target2 (make-tensor32 (f32vector 6.0 8.0) '(2)))
           (loss1 (mse-loss slice1 target1))
           (loss2 (mse-loss slice2 target2))
           (total-loss (add loss1 loss2)))
      (backward! total-loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 1) -1.0 1e-6)))
  
  (test-assert "Gradient at [1,0] is 0 (not sliced)"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice1 (squeeze (slice-tensor input 0 1)))
           (slice2 (squeeze (slice-tensor input 2 1)))
           (target1 (make-tensor32 (f32vector 2.0 4.0) '(2)))
           (target2 (make-tensor32 (f32vector 6.0 8.0) '(2)))
           (loss1 (mse-loss slice1 target1))
           (loss2 (mse-loss slice2 target2))
           (total-loss (add loss1 loss2)))
      (backward! total-loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 2) 0.0 1e-6)))
  
  (test-assert "Gradient at [1,1] is 0 (not sliced)"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice1 (squeeze (slice-tensor input 0 1)))
           (slice2 (squeeze (slice-tensor input 2 1)))
           (target1 (make-tensor32 (f32vector 2.0 4.0) '(2)))
           (target2 (make-tensor32 (f32vector 6.0 8.0) '(2)))
           (loss1 (mse-loss slice1 target1))
           (loss2 (mse-loss slice2 target2))
           (total-loss (add loss1 loss2)))
      (backward! total-loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 3) 0.0 1e-6)))
  
  (test-assert "Gradient at [2,0] from second slice"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice1 (squeeze (slice-tensor input 0 1)))
           (slice2 (squeeze (slice-tensor input 2 1)))
           (target1 (make-tensor32 (f32vector 2.0 4.0) '(2)))
           (target2 (make-tensor32 (f32vector 6.0 8.0) '(2)))
           (loss1 (mse-loss slice1 target1))
           (loss2 (mse-loss slice2 target2))
           (total-loss (add loss1 loss2)))
      (backward! total-loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 4) -0.5 1e-6)))
  
  (test-assert "Gradient at [2,1] from second slice"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice1 (squeeze (slice-tensor input 0 1)))
           (slice2 (squeeze (slice-tensor input 2 1)))
           (target1 (make-tensor32 (f32vector 2.0 4.0) '(2)))
           (target2 (make-tensor32 (f32vector 6.0 8.0) '(2)))
           (loss1 (mse-loss slice1 target1))
           (loss2 (mse-loss slice2 target2))
           (total-loss (add loss1 loss2)))
      (backward! total-loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 5) -1.0 1e-6))))

;;; ==================================================================
;;; Multiple Operations on Slice
;;; ==================================================================

(test-group "Operations on Slices"
  
  (test-assert "Scaled slice [0] = 6.0 * 2 = 12.0"
    (let* ((input (make-tensor32
                   (f32vector 2.0 4.0 6.0 8.0 10.0 12.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice (squeeze (slice-tensor input 1 1)))
           (scaled (scale-op slice 2.0)))
      (approx-equal? (f32vector-ref (tensor-data scaled) 0) 12.0 1e-6)))
  
  (test-assert "Scaled slice [1] = 8.0 * 2 = 16.0"
    (let* ((input (make-tensor32
                   (f32vector 2.0 4.0 6.0 8.0 10.0 12.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice (squeeze (slice-tensor input 1 1)))
           (scaled (scale-op slice 2.0)))
      (approx-equal? (f32vector-ref (tensor-data scaled) 1) 16.0 1e-6)))
  
  (test-assert "Added result [0] = 12.0 + 1.0 = 13.0"
    (let* ((input (make-tensor32
                   (f32vector 2.0 4.0 6.0 8.0 10.0 12.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice (squeeze (slice-tensor input 1 1)))
           (scaled (scale-op slice 2.0))
           (const (make-tensor32 (f32vector 1.0 2.0) '(2)))
           (added (add scaled const)))
      (approx-equal? (f32vector-ref (tensor-data added) 0) 13.0 1e-6)))
  
  (test-assert "Added result [1] = 16.0 + 2.0 = 18.0"
    (let* ((input (make-tensor32
                   (f32vector 2.0 4.0 6.0 8.0 10.0 12.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice (squeeze (slice-tensor input 1 1)))
           (scaled (scale-op slice 2.0))
           (const (make-tensor32 (f32vector 1.0 2.0) '(2)))
           (added (add scaled const)))
      (approx-equal? (f32vector-ref (tensor-data added) 1) 18.0 1e-6)))
  
  (test-assert "Gradient exists after complex operations"
    (let* ((input (make-tensor32
                   (f32vector 2.0 4.0 6.0 8.0 10.0 12.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice (squeeze (slice-tensor input 1 1)))
           (scaled (scale-op slice 2.0))
           (const (make-tensor32 (f32vector 1.0 2.0) '(2)))
           (added (add scaled const))
           (target (make-tensor32 (f32vector 20.0 30.0) '(2)))
           (loss (mse-loss added target)))
      (backward! loss)
      (tensor-grad input)))
  
  (test-assert "Row 0 not affected"
    (let* ((input (make-tensor32
                   (f32vector 2.0 4.0 6.0 8.0 10.0 12.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice (squeeze (slice-tensor input 1 1)))
           (scaled (scale-op slice 2.0))
           (const (make-tensor32 (f32vector 1.0 2.0) '(2)))
           (added (add scaled const))
           (target (make-tensor32 (f32vector 20.0 30.0) '(2)))
           (loss (mse-loss added target)))
      (backward! loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 0) 0.0 1e-6)))
  
  (test-assert "Row 1 gradient through chain"
    (let* ((input (make-tensor32
                   (f32vector 2.0 4.0 6.0 8.0 10.0 12.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice (squeeze (slice-tensor input 1 1)))
           (scaled (scale-op slice 2.0))
           (const (make-tensor32 (f32vector 1.0 2.0) '(2)))
           (added (add scaled const))
           (target (make-tensor32 (f32vector 20.0 30.0) '(2)))
           (loss (mse-loss added target)))
      (backward! loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 2) -7.0 1e-5)))
  
  (test-assert "Row 2 not affected"
    (let* ((input (make-tensor32
                   (f32vector 2.0 4.0 6.0 8.0 10.0 12.0)
                   '(3 2)
                   requires-grad?: #t))
           (slice (squeeze (slice-tensor input 1 1)))
           (scaled (scale-op slice 2.0))
           (const (make-tensor32 (f32vector 1.0 2.0) '(2)))
           (added (add scaled const))
           (target (make-tensor32 (f32vector 20.0 30.0) '(2)))
           (loss (mse-loss added target)))
      (backward! loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 4) 0.0 1e-6))))

;;; ==================================================================
;;; 3D Tensor Slicing
;;; ==================================================================

(test-group "3D Tensor Slicing"
  
  (test-assert "Slice channel 0 shape is (1, 2, 2)"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(2 2 2)
                   requires-grad?: #t))
           (slice (slice-tensor input 0 1)))
      (equal? (tensor-shape slice) '(1 2 2))))
  
  (test-assert "Channel 0, element [0,0] is 1.0"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(2 2 2)
                   requires-grad?: #t))
           (slice (slice-tensor input 0 1)))
      (approx-equal? (f32vector-ref (tensor-data slice) 0) 1.0 1e-6)))
  
  (test-assert "Channel 0, element [1,1] is 4.0"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(2 2 2)
                   requires-grad?: #t))
           (slice (slice-tensor input 0 1)))
      (approx-equal? (f32vector-ref (tensor-data slice) 3) 4.0 1e-6)))
  
  (test-assert "Slice channel 1 shape is (1, 2, 2)"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(2 2 2)
                   requires-grad?: #t))
           (slice (slice-tensor input 1 1)))
      (equal? (tensor-shape slice) '(1 2 2))))
  
  (test-assert "Channel 1, element [0,0] is 5.0"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(2 2 2)
                   requires-grad?: #t))
           (slice (slice-tensor input 1 1)))
      (approx-equal? (f32vector-ref (tensor-data slice) 0) 5.0 1e-6)))
  
  (test-assert "Channel 1, element [1,1] is 8.0"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(2 2 2)
                   requires-grad?: #t))
           (slice (slice-tensor input 1 1)))
      (approx-equal? (f32vector-ref (tensor-data slice) 3) 8.0 1e-6)))
  
  (test-assert "Channel 0 has gradient"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(2 2 2)
                   requires-grad?: #t))
           (slice (slice-tensor input 0 1))
           (target (make-tensor32 (f32vector 10.0 11.0 12.0 13.0) '(1 2 2)))
           (loss (mse-loss slice target)))
      (backward! loss)
      (not (= (f32vector-ref (tensor-grad input) 0) 0.0))))
  
  (test-assert "Channel 1 has zero gradient"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(2 2 2)
                   requires-grad?: #t))
           (slice (slice-tensor input 0 1))
           (target (make-tensor32 (f32vector 10.0 11.0 12.0 13.0) '(1 2 2)))
           (loss (mse-loss slice target)))
      (backward! loss)
      (approx-equal? (f32vector-ref (tensor-grad input) 4) 0.0 1e-6))))

;;; ==================================================================
;;; Edge Cases
;;; ==================================================================

(test-group "Edge Cases"
  
  (test-assert "Full slice preserves shape"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0)
                   '(2 2)))
           (full (slice-tensor input 0 2)))
      (equal? (tensor-shape full) '(2 2))))
  
  (test-assert "Full slice data correct [0]"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0)
                   '(2 2)))
           (full (slice-tensor input 0 2)))
      (approx-equal? (f32vector-ref (tensor-data full) 0) 1.0 1e-6)))
  
  (test-assert "Full slice data correct [3]"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0)
                   '(2 2)))
           (full (slice-tensor input 0 2)))
      (approx-equal? (f32vector-ref (tensor-data full) 3) 4.0 1e-6)))
  
  (test-assert "Single row slice shape"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0)
                   '(2 2)))
           (single (slice-tensor input 0 1)))
      (equal? (tensor-shape single) '(1 2))))
  
  (test-assert "Last row slice correct"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0)
                   '(2 2)))
           (last (slice-tensor input 1 1)))
      (approx-equal? (f32vector-ref (tensor-data last) 0) 3.0 1e-6))))

;;; ==================================================================
;;; Chain of Slices
;;; ==================================================================

(test-group "Chain of Slices"
  
  (test-assert "First slice shape"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)
                   requires-grad?: #t))
           (slice1 (slice-tensor input 1 2)))
      (equal? (tensor-shape slice1) '(2 2))))
  
  (test-assert "Sum [0] = 1.0 + 3.0 = 4.0"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)
                   requires-grad?: #t))
           (slice1 (slice-tensor input 1 2))
           (slice2 (slice-tensor input 0 1))
           (sum1 (add (reshape slice2 '(2)) 
                     (reshape (slice-tensor slice1 0 1) '(2)))))
      (approx-equal? (f32vector-ref (tensor-data sum1) 0) 4.0 1e-6)))
  
  (test-assert "Sum [1] = 2.0 + 4.0 = 6.0"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)
                   requires-grad?: #t))
           (slice1 (slice-tensor input 1 2))
           (slice2 (slice-tensor input 0 1))
           (sum1 (add (reshape slice2 '(2)) 
                     (reshape (slice-tensor slice1 0 1) '(2)))))
      (approx-equal? (f32vector-ref (tensor-data sum1) 1) 6.0 1e-6)))
  
  (test-assert "Row 0 has gradient from sum"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)
                   requires-grad?: #t))
           (slice1 (slice-tensor input 1 2))
           (slice2 (slice-tensor input 0 1))
           (sum1 (add (reshape slice2 '(2)) 
                     (reshape (slice-tensor slice1 0 1) '(2))))
           (target (make-tensor32 (f32vector 10.0 20.0) '(2)))
           (loss (mse-loss sum1 target)))
      (backward! loss)
      (not (= (f32vector-ref (tensor-grad input) 0) 0.0))))
  
  (test-assert "Row 1 has gradient from sum"
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                   '(4 2)
                   requires-grad?: #t))
           (slice1 (slice-tensor input 1 2))
           (slice2 (slice-tensor input 0 1))
           (sum1 (add (reshape slice2 '(2)) 
                     (reshape (slice-tensor slice1 0 1) '(2))))
           (target (make-tensor32 (f32vector 10.0 20.0) '(2)))
           (loss (mse-loss sum1 target)))
      (backward! loss)
      (not (= (f32vector-ref (tensor-grad input) 2) 0.0)))))

;;; ==================================================================
;;; Numerical Gradient Check
;;; ==================================================================

(test-group "Numerical Gradient Check"
  
  (let ((epsilon 1e-3))
    
    ;; Function: take middle row and compute squared sum
    (define (compute-loss x)
      (let* ((slice (slice-tensor x 1 1))
             (slice-flat (reshape slice '(2)))
             (squared (mul slice-flat slice-flat)))
        (sum-tensor squared)))
    
    (let* ((input (make-tensor32
                   (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                   '(3 2)
                   requires-grad?: #t))
           (loss (compute-loss input)))
      
      (backward! loss)
      (let ((analytical-grad (tensor-grad input)))
        
        ;; Compute numerical gradient for each position
        (do ((i 0 (+ i 1)))
            ((= i 6))
          (let ((input-plus (make-tensor32
                             (scopy (tensor-data input))
                             '(3 2)))
                (input-minus (make-tensor32
                              (scopy (tensor-data input))
                              '(3 2))))
            
            ;; Perturb +epsilon
            (f32vector-set! (tensor-data input-plus) i
                            (+ (f32vector-ref (tensor-data input) i) epsilon))
            
            ;; Perturb -epsilon
            (f32vector-set! (tensor-data input-minus) i
                            (- (f32vector-ref (tensor-data input) i) epsilon))
            
            ;; Compute numerical gradient
            (let* ((loss-plus (f32vector-ref 
                               (tensor-data (compute-loss input-plus)) 0))
                   (loss-minus (f32vector-ref
                                (tensor-data (compute-loss input-minus)) 0))
                   (numerical (/ (- loss-plus loss-minus) (* 2.0 epsilon)))
                   (analytical (f32vector-ref analytical-grad i)))
              
              (test-assert (sprintf "Gradient check position ~A" i)
                           (approx-equal? analytical numerical 1e-3)))))))))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(test-exit)

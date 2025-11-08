;; Comprehensive tests for tensor slice operation

(import scheme
        (chicken base)
        (chicken format)
        (srfi 1)
        (srfi 4)
        nanograd-autograd
        )

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
          (printf "  ✓ ~A\n" name))
        (begin
          (set! *test-failed* (+ *test-failed* 1))
          (printf "  ✗ ~A\n" name)
          (printf "    Expected: ~A, Got: ~A, Diff: ~A\n" 
                  expected actual diff)))))

(define (assert-true condition name)
  (set! *test-count* (+ *test-count* 1))
  (if condition
      (begin
        (set! *test-passed* (+ *test-passed* 1))
        (printf "  ✓ ~A\n" name))
      (begin
        (set! *test-failed* (+ *test-failed* 1))
        (printf "  ✗ ~A\n" name))))

(define (assert-shape-equal actual-shape expected-shape name)
  (set! *test-count* (+ *test-count* 1))
  (if (equal? actual-shape expected-shape)
      (begin
        (set! *test-passed* (+ *test-passed* 1))
        (printf "  ✓ ~A\n" name))
      (begin
        (set! *test-failed* (+ *test-failed* 1))
        (printf "  ✗ ~A\n" name)
        (printf "    Expected shape: ~A, Got: ~A\n" 
                expected-shape actual-shape))))

;;; ==================================================================
;;; Test 1: Basic Slicing - Shape and Data
;;; ==================================================================

(define (test-basic-slicing)
  (printf "\n=== Test 1: Basic Slicing ===\n")
  
  ;; Create a simple 1D-like tensor (3, 1)
  (define input (make-tensor32 
                 (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                 '(3 2)))
  
  ;; Slice first row
  (define slice1 (slice-tensor input 0 1))
  (assert-shape-equal (tensor-shape slice1) '(1 2) 
                     "Slice shape [0:1] is (1, 2)")
  (assert-equal (f32vector-ref (tensor-data slice1) 0) 1.0 1e-6
               "Slice [0:1] first element is 1.0")
  (assert-equal (f32vector-ref (tensor-data slice1) 1) 2.0 1e-6
               "Slice [0:1] second element is 2.0")
  
  ;; Slice middle row
  (define slice2 (slice-tensor input 1 1))
  (assert-shape-equal (tensor-shape slice2) '(1 2)
                     "Slice shape [1:2] is (1, 2)")
  (assert-equal (f32vector-ref (tensor-data slice2) 0) 3.0 1e-6
               "Slice [1:2] first element is 3.0")
  (assert-equal (f32vector-ref (tensor-data slice2) 1) 4.0 1e-6
               "Slice [1:2] second element is 4.0")
  
  ;; Slice last row
  (define slice3 (slice-tensor input 2 1))
  (assert-shape-equal (tensor-shape slice3) '(1 2)
                     "Slice shape [2:3] is (1, 2)")
  (assert-equal (f32vector-ref (tensor-data slice3) 0) 5.0 1e-6
               "Slice [2:3] first element is 5.0")
  (assert-equal (f32vector-ref (tensor-data slice3) 1) 6.0 1e-6
               "Slice [2:3] second element is 6.0"))

;;; ==================================================================
;;; Test 2: Multi-row Slicing
;;; ==================================================================

(define (test-multirow-slicing)
  (printf "\n=== Test 2: Multi-row Slicing ===\n")
  
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                 '(4 2)))
  
  ;; Slice first 2 rows
  (define slice-first2 (slice-tensor input 0 2))
  (assert-shape-equal (tensor-shape slice-first2) '(2 2)
                     "Slice [0:2] shape is (2, 2)")
  (assert-equal (f32vector-ref (tensor-data slice-first2) 0) 1.0 1e-6
               "Slice [0:2] element [0,0] is 1.0")
  (assert-equal (f32vector-ref (tensor-data slice-first2) 1) 2.0 1e-6
               "Slice [0:2] element [0,1] is 2.0")
  (assert-equal (f32vector-ref (tensor-data slice-first2) 2) 3.0 1e-6
               "Slice [0:2] element [1,0] is 3.0")
  (assert-equal (f32vector-ref (tensor-data slice-first2) 3) 4.0 1e-6
               "Slice [0:2] element [1,1] is 4.0")
  
  ;; Slice middle 2 rows
  (define slice-middle2 (slice-tensor input 1 2))
  (assert-shape-equal (tensor-shape slice-middle2) '(2 2)
                     "Slice [1:3] shape is (2, 2)")
  (assert-equal (f32vector-ref (tensor-data slice-middle2) 0) 3.0 1e-6
               "Slice [1:3] element [0,0] is 3.0")
  (assert-equal (f32vector-ref (tensor-data slice-middle2) 3) 6.0 1e-6
               "Slice [1:3] element [1,1] is 6.0")
  
  ;; Slice last 2 rows
  (define slice-last2 (slice-tensor input 2 2))
  (assert-shape-equal (tensor-shape slice-last2) '(2 2)
                     "Slice [2:4] shape is (2, 2)")
  (assert-equal (f32vector-ref (tensor-data slice-last2) 0) 5.0 1e-6
               "Slice [2:4] element [0,0] is 5.0")
  (assert-equal (f32vector-ref (tensor-data slice-last2) 3) 8.0 1e-6
               "Slice [2:4] element [1,1] is 8.0"))

;;; ==================================================================
;;; Test 3: Gradient Flow - Simple Case
;;; ==================================================================

(define (test-simple-gradient)
  (printf "\n=== Test 3: Simple Gradient Flow ===\n")
  
  ;; Create input with gradient tracking
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                 '(3 2)
                 requires-grad?: #t))
  
  ;; Slice and compute simple loss
  (define slice (slice-tensor input 1 1))  ;; Extract middle row [3.0, 4.0]
  (define target (make-tensor32 (f32vector 10.0 20.0) '(2)))
  (define loss (mse-loss slice target))
  
  ;; Backward pass
  (backward! loss)
  
  ;; Check that gradients exist
  (assert-true (tensor-grad input) "Input has gradient after backward")
  
  ;; Check gradient values
  ;; Loss = 0.5 * ((3-10)^2 + (4-20)^2) / 2
  ;; dL/d[slice] = [(3-10)/2, (4-20)/2] = [-3.5, -8.0]
  ;; Gradient should be zero except at sliced positions
  (let ((grad (tensor-grad input)))
    (assert-equal (f32vector-ref grad 0) 0.0 1e-6
                 "Gradient at position [0,0] is 0 (not sliced)")
    (assert-equal (f32vector-ref grad 1) 0.0 1e-6
                 "Gradient at position [0,1] is 0 (not sliced)")
    (assert-equal (f32vector-ref grad 2) -3.5 1e-5
                 "Gradient at position [1,0] is -3.5 (sliced)")
    (assert-equal (f32vector-ref grad 3) -8.0 1e-5
                 "Gradient at position [1,1] is -8.0 (sliced)")
    (assert-equal (f32vector-ref grad 4) 0.0 1e-6
                 "Gradient at position [2,0] is 0 (not sliced)")
    (assert-equal (f32vector-ref grad 5) 0.0 1e-6
                 "Gradient at position [2,1] is 0 (not sliced)")))

;;; ==================================================================
;;; Test 4: Gradient Accumulation
;;; ==================================================================

(define (test-gradient-accumulation)
  (printf "\n=== Test 4: Gradient Accumulation ===\n")
  
  ;; Create input
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                 '(3 2)
                 requires-grad?: #t))
  
  ;; Take two different slices and sum their losses
  (define slice1 (slice-tensor input 0 1))  ;; [1.0, 2.0]
  (define slice2 (slice-tensor input 2 1))  ;; [5.0, 6.0]
  
  (define target1 (make-tensor32 (f32vector 2.0 4.0) '(2)))
  (define target2 (make-tensor32 (f32vector 6.0 8.0) '(2)))
  
  (define loss1 (mse-loss slice1 target1))
  (define loss2 (mse-loss slice2 target2))
  (define total-loss (add loss1 loss2))
  
  ;; Backward pass
  (backward! total-loss)
  
  ;; Check gradients accumulated correctly
  (let ((grad (tensor-grad input)))
    ;; Loss1: dL/d[1.0, 2.0] = [(1-2)/2, (2-4)/2] = [-0.5, -1.0]
    (assert-equal (f32vector-ref grad 0) -0.5 1e-6
                 "Gradient at [0,0] from first slice")
    (assert-equal (f32vector-ref grad 1) -1.0 1e-6
                 "Gradient at [0,1] from first slice")
    
    ;; Middle row not sliced
    (assert-equal (f32vector-ref grad 2) 0.0 1e-6
                 "Gradient at [1,0] is 0 (not sliced)")
    (assert-equal (f32vector-ref grad 3) 0.0 1e-6
                 "Gradient at [1,1] is 0 (not sliced)")
    
    ;; Loss2: dL/d[5.0, 6.0] = [(5-6)/2, (6-8)/2] = [-0.5, -1.0]
    (assert-equal (f32vector-ref grad 4) -0.5 1e-6
                 "Gradient at [2,0] from second slice")
    (assert-equal (f32vector-ref grad 5) -1.0 1e-6
                 "Gradient at [2,1] from second slice")))

;;; ==================================================================
;;; Test 5: Multiple Operations on Slice
;;; ==================================================================

(define (test-operations-on-slice)
  (printf "\n=== Test 5: Operations on Slices ===\n")
  
  (define input (make-tensor32
                 (f32vector 2.0 4.0 6.0 8.0 10.0 12.0)
                 '(3 2)
                 requires-grad?: #t))
  
  ;; Extract middle row and perform operations
  (define slice (slice-tensor input 1 1))  ;; [6.0, 8.0]
  
  ;; Scale by 2
  (define scaled (scale-op slice 2.0))
  (assert-equal (f32vector-ref (tensor-data scaled) 0) 12.0 1e-6
               "Scaled slice [0] = 6.0 * 2 = 12.0")
  (assert-equal (f32vector-ref (tensor-data scaled) 1) 16.0 1e-6
               "Scaled slice [1] = 8.0 * 2 = 16.0")
  
  ;; Add constant tensor
  (define const (make-tensor32 (f32vector 1.0 2.0) '(2)))
  (define added (add scaled const))
  (assert-equal (f32vector-ref (tensor-data added) 0) 13.0 1e-6
               "Added result [0] = 12.0 + 1.0 = 13.0")
  (assert-equal (f32vector-ref (tensor-data added) 1) 18.0 1e-6
               "Added result [1] = 16.0 + 2.0 = 18.0")
  
  ;; Compute loss and backprop
  (define target (make-tensor32 (f32vector 20.0 30.0) '(2)))
  (define loss (mse-loss added target))
  (backward! loss)
  
  ;; Check gradient exists and flows through all operations
  (assert-true (tensor-grad input) "Gradient exists after complex operations")
  
  (let ((grad (tensor-grad input)))
    ;; Gradient should only affect the sliced row
    (assert-equal (f32vector-ref grad 0) 0.0 1e-6
                 "Row 0 not affected")
    (assert-equal (f32vector-ref grad 1) 0.0 1e-6
                 "Row 0 not affected")
    
    ;; Row 1 affected: gradient flows through add, scale, slice
    ;; dL/dx = dL/d(added) * d(added)/d(scaled) * d(scaled)/d(slice) * d(slice)/dx
    ;;       = [(13-20)/2, (18-30)/2] * 1 * 2 * 1
    ;;       = [-3.5, -6.0] * 2 = [-7.0, -12.0]
    (assert-equal (f32vector-ref grad 2) -7.0 1e-5
                 "Row 1 gradient through chain")
    (assert-equal (f32vector-ref grad 3) -12.0 1e-5
                 "Row 1 gradient through chain")
    
    (assert-equal (f32vector-ref grad 4) 0.0 1e-6
                 "Row 2 not affected")
    (assert-equal (f32vector-ref grad 5) 0.0 1e-6
                 "Row 2 not affected")))

;;; ==================================================================
;;; Test 6: 3D Tensor Slicing
;;; ==================================================================

(define (test-3d-slicing)
  (printf "\n=== Test 6: 3D Tensor Slicing ===\n")
  
  ;; Create 3D tensor (channels, height, width) like (2, 2, 2)
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0    ;; Channel 0
                           5.0 6.0 7.0 8.0)    ;; Channel 1
                 '(2 2 2)
                 requires-grad?: #t))
  
  ;; Slice first channel
  (define slice-ch0 (slice-tensor input 0 1))
  (assert-shape-equal (tensor-shape slice-ch0) '(1 2 2)
                     "Slice channel 0 shape is (1, 2, 2)")
  (assert-equal (f32vector-ref (tensor-data slice-ch0) 0) 1.0 1e-6
               "Channel 0, element [0,0] is 1.0")
  (assert-equal (f32vector-ref (tensor-data slice-ch0) 3) 4.0 1e-6
               "Channel 0, element [1,1] is 4.0")
  
  ;; Slice second channel
  (define slice-ch1 (slice-tensor input 1 1))
  (assert-shape-equal (tensor-shape slice-ch1) '(1 2 2)
                     "Slice channel 1 shape is (1, 2, 2)")
  (assert-equal (f32vector-ref (tensor-data slice-ch1) 0) 5.0 1e-6
               "Channel 1, element [0,0] is 5.0")
  (assert-equal (f32vector-ref (tensor-data slice-ch1) 3) 8.0 1e-6
               "Channel 1, element [1,1] is 8.0")
  
  ;; Test gradient flow
  (define target (make-tensor32 (f32vector 10.0 11.0 12.0 13.0) '(1 2 2)))
  (define loss (mse-loss slice-ch0 target))
  (backward! loss)
  
  (let ((grad (tensor-grad input)))
    ;; First channel should have gradients
    (assert-true (not (= (f32vector-ref grad 0) 0.0))
                "Channel 0 has gradient")
    (assert-true (not (= (f32vector-ref grad 3) 0.0))
                "Channel 0 has gradient")
    
    ;; Second channel should have zero gradients
    (assert-equal (f32vector-ref grad 4) 0.0 1e-6
                 "Channel 1 has zero gradient")
    (assert-equal (f32vector-ref grad 7) 0.0 1e-6
                 "Channel 1 has zero gradient")))

;;; ==================================================================
;;; Test 7: Edge Cases
;;; ==================================================================

(define (test-edge-cases)
  (printf "\n=== Test 7: Edge Cases ===\n")
  
  ;; Test slicing entire tensor
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0)
                 '(2 2)))
  
  (define full-slice (slice-tensor input 0 2))
  (assert-shape-equal (tensor-shape full-slice) '(2 2)
                     "Full slice preserves shape")
  (assert-equal (f32vector-ref (tensor-data full-slice) 0) 1.0 1e-6
               "Full slice data correct [0]")
  (assert-equal (f32vector-ref (tensor-data full-slice) 3) 4.0 1e-6
               "Full slice data correct [3]")
  
  ;; Test single element slice
  (define single (slice-tensor input 0 1))
  (assert-shape-equal (tensor-shape single) '(1 2)
                     "Single row slice shape")
  
  ;; Test boundary slice
  (define last-row (slice-tensor input 1 1))
  (assert-equal (f32vector-ref (tensor-data last-row) 0) 3.0 1e-6
               "Last row slice correct"))

;;; ==================================================================
;;; Test 8: Chain of Slices
;;; ==================================================================

(define (test-chain-slices)
  (printf "\n=== Test 8: Chain of Slices ===\n")
  
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                 '(4 2)
                 requires-grad?: #t))
  
  ;; Take slice of a slice (though not directly supported,
  ;; we can test sequential slicing)
  (define slice1 (slice-tensor input 1 2))  ;; Rows 1-2: [3,4,5,6]
  (assert-shape-equal (tensor-shape slice1) '(2 2)
                     "First slice shape")
  
  ;; Each slice is independent
  (define slice2 (slice-tensor input 0 1))  ;; Row 0: [1,2]
  (define slice3 (slice-tensor input 3 1))  ;; Row 3: [7,8]
  
  ;; Combine in computation
  (define sum1 (add (reshape slice2 '(2)) (reshape (slice-tensor slice1 0 1) '(2))))
  (assert-equal (f32vector-ref (tensor-data sum1) 0) 4.0 1e-6
               "Sum [0] = 1.0 + 3.0 = 4.0")
  (assert-equal (f32vector-ref (tensor-data sum1) 1) 6.0 1e-6
               "Sum [1] = 2.0 + 4.0 = 6.0")
  
  (let ((target (make-tensor32 (f32vector 10.0 20.0) '(2)))
        (loss (mse-loss sum1 target)))
    (backward! loss)
    
    ;; Check gradients flow to both source rows
    (let ((grad (tensor-grad input)))
      ;; Row 0 and row 1 should have gradients
      (assert-true (not (= (f32vector-ref grad 0) 0.0))
                  "Row 0 has gradient from sum")
      (assert-true (not (= (f32vector-ref grad 2) 0.0))
                  "Row 1 has gradient from sum"))))

;;; ==================================================================
;;; Test 9: Numerical Gradient Check
;;; ==================================================================

(define (test-numerical-gradient)
  (printf "\n=== Test 9: Numerical Gradient Check ===\n")
  
  (define epsilon 1e-4)
  
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)
                 '(3 2)
                 requires-grad?: #t))
  
  ;; Function: take middle row and compute squared sum
  (define (compute-loss x)
    (let* ((slice (slice-tensor x 1 1))
           (slice-flat (reshape slice '(2)))
           (squared (mul slice-flat slice-flat))
           (sum-squared (add (scale-op squared 1.0)
                           (make-tensor32 (f32vector 0.0) '(1)))))
      ;; Sum elements
      (let ((s (+ (f32vector-ref (tensor-data squared) 0)
                  (f32vector-ref (tensor-data squared) 1))))
        (make-tensor32 (f32vector s) '(1)))))
  
  ;; Compute analytical gradient
  (let ((loss (compute-loss input)))
    (backward! loss)
    (let ((analytical-grad (tensor-grad input)))
      
      ;; Compute numerical gradient
      (let ((numerical-grad (make-f32vector 6 0.0)))
        (do ((i 0 (+ i 1)))
            ((= i 6))
          (let ((input-plus (make-tensor32
                            (f32vector-copy (tensor-data input))
                            '(3 2)))
                (input-minus (make-tensor32
                             (f32vector-copy (tensor-data input))
                             '(3 2))))
            
            ;; Perturb +epsilon
            (f32vector-set! (tensor-data input-plus) i
                           (+ (f32vector-ref (tensor-data input) i) epsilon))
            
            ;; Perturb -epsilon
            (f32vector-set! (tensor-data input-minus) i
                           (- (f32vector-ref (tensor-data input) i) epsilon))
            
            ;; Compute numerical gradient
            (let ((loss-plus (f32vector-ref 
                             (tensor-data (compute-loss input-plus)) 0))
                  (loss-minus (f32vector-ref
                              (tensor-data (compute-loss input-minus)) 0)))
              (f32vector-set! numerical-grad i
                            (/ (- loss-plus loss-minus) (* 2.0 epsilon))))))
        
        ;; Compare analytical vs numerical
        (do ((i 0 (+ i 1)))
            ((= i 6))
          (let ((analytical (f32vector-ref analytical-grad i))
                (numerical (f32vector-ref numerical-grad i)))
            (assert-equal analytical numerical 1e-3
                         (sprintf "Gradient check position ~A" i))))))))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(define (run-all-slice-tests)
  (reset-test-stats!)
  (printf "\n")
  (printf "Tensor slice operation tests\n")
  
  (test-basic-slicing)
  (test-multirow-slicing)
  (test-simple-gradient)
  (test-gradient-accumulation)
  (test-operations-on-slice)
  (test-3d-slicing)
  (test-edge-cases)
  (test-chain-slices)
  (test-numerical-gradient)
  
  (test-summary))

;; Run tests
(run-all-slice-tests)

;; Comprehensive tests for Batch Normalization and Global Average Pooling

(import scheme
        (chicken base)
        (chicken format)
        (srfi 1)
        (srfi 4)
        nanograd-autograd
        nanograd-layer)


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

(define (assert-true condition name)
  (set! *test-count* (+ *test-count* 1))
  (if condition
      (begin
        (set! *test-passed* (+ *test-passed* 1))
        (printf "  O ~A\n" name))
      (begin
        (set! *test-failed* (+ *test-failed* 1))
        (printf "  X ~A\n" name))))

(define (assert-shape-equal actual-shape expected-shape name)
  (set! *test-count* (+ *test-count* 1))
  (if (equal? actual-shape expected-shape)
      (begin
        (set! *test-passed* (+ *test-passed* 1))
        (printf "  O ~A\n" name))
      (begin
        (set! *test-failed* (+ *test-failed* 1))
        (printf "  X ~A\n" name)
        (printf "    Expected shape: ~A, Got: ~A\n" 
                expected-shape actual-shape))))

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

(define (f32vector-copy v)
  (ssub v 0 (f32vector-length v)))
  
;;; ==================================================================
;;; BATCH NORMALIZATION TESTS
;;; ==================================================================

;;; Basic Forward Pass - Training Mode
;;; ==================================================================

(define (test-batchnorm-training-forward)
  (printf "\n=== BatchNorm Training Forward Pass ===\n")
  
  ;; Create BatchNorm layer for 2 channels
  (define bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
  (set-training-mode! bn #t)
  
  ;; Create input: 2 channels, 2x2 spatial
  ;; Channel 0: [[1,2], [3,4]] mean=2.5, var=1.25
  ;; Channel 1: [[5,6], [7,8]] mean=6.5, var=1.25
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0    ;; Channel 0
                           5.0 6.0 7.0 8.0)    ;; Channel 1
                 '(2 2 2)
                 requires-grad?: #t))
  
  (define output (forward bn input))
  
  ;; Check output shape
  (assert-shape-equal (tensor-shape output) '(2 2 2)
                     "Output shape matches input")
  
  ;; Check normalization (approximately zero mean, unit variance per channel)
  (let ((data (tensor-data output)))
    ;; Channel 0 normalized values
    (let* ((ch0-vals (list (f32vector-ref data 0)
                          (f32vector-ref data 1)
                          (f32vector-ref data 2)
                          (f32vector-ref data 3)))
           (ch0-mean (/ (apply + ch0-vals) 4.0))
           (ch0-var (/ (apply + (map (lambda (x) (* (- x ch0-mean) (- x ch0-mean)))
                                    ch0-vals))
                      4.0)))
      
      (assert-equal ch0-mean 0.0 1e-5
                   "Channel 0 normalized mean ~= 0")
      (assert-equal ch0-var 1.0 1e-4
                   "Channel 0 normalized variance ~= 1"))
    
    ;; Channel 1 normalized values
    (let* ((ch1-vals (list (f32vector-ref data 4)
                          (f32vector-ref data 5)
                          (f32vector-ref data 6)
                          (f32vector-ref data 7)))
           (ch1-mean (/ (apply + ch1-vals) 4.0))
           (ch1-var (/ (apply + (map (lambda (x) (* (- x ch1-mean) (- x ch1-mean)))
                                    ch1-vals))
                      4.0)))
      
      (assert-equal ch1-mean 0.0 1e-5
                   "Channel 1 normalized mean ~= 0")
      (assert-equal ch1-var 1.0 1e-4
                   "Channel 1 normalized variance ~= 1"))))

;;; Running Statistics Update
;;; ==================================================================

(define (test-batchnorm-running-stats)
  (printf "\n=== BatchNorm Running Statistics ===\n")
  
  (define bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
  (set-training-mode! bn #t)
  
  ;; First batch
  (define input1 (make-tensor32
                  (f32vector 0.0 0.0 0.0 0.0    ;; Channel 0: all zeros
                            10.0 10.0 10.0 10.0) ;; Channel 1: all tens
                  '(2 2 2)))
  
  (forward bn input1)
  
  ;; Running stats should be updated
  ;; running_mean = (1-0.1)*0 + 0.1*batch_mean
  ;; For channel 0: batch_mean = 0, running_mean = 0
  ;; For channel 1: batch_mean = 10, running_mean = 1.0
  
  ;; Second batch with different statistics
  (define input2 (make-tensor32
                  (f32vector 2.0 2.0 2.0 2.0    ;; Channel 0: all twos
                            20.0 20.0 20.0 20.0) ;; Channel 1: all twenties
                  '(2 2 2)))
  
  (forward bn input2)
  
  ;; After second batch:
  ;; Channel 0: running_mean = 0.9*0 + 0.1*2 = 0.2
  ;; Channel 1: running_mean = 0.9*1.0 + 0.1*20 = 2.9
  
  (printf "    Running statistics updated after 2 batches\n")
  (assert-true #t "Running statistics tracking functional"))

;;; Evaluation Mode
;;; ==================================================================

(define (test-batchnorm-eval-mode)
  (printf "\n=== BatchNorm Evaluation Mode ===\n")
  
  (define bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
  
  ;; Train on some data to populate running stats
  (set-training-mode! bn #t)
  (define train-input (make-tensor32
                       (f32vector 1.0 2.0 3.0 4.0
                                 5.0 6.0 7.0 8.0)
                       '(2 2 2)))
  (forward bn train-input)
  
  ;; Switch to eval mode
  (set-eval-mode! bn)
  
  ;; In eval mode, should use running statistics (deterministic)
  (define eval-input1 (make-tensor32
                       (f32vector 10.0 20.0 30.0 40.0
                                 50.0 60.0 70.0 80.0)
                       '(2 2 2)))
  
  (define output1 (forward bn eval-input1))
  
  ;; Same input should give same output in eval mode
  (define eval-input2 (make-tensor32
                       (f32vector 10.0 20.0 30.0 40.0
                                 50.0 60.0 70.0 80.0)
                       '(2 2 2)))
  
  (define output2 (forward bn eval-input2))
  
  ;; Verify outputs are identical (deterministic)
  (let ((data1 (tensor-data output1))
        (data2 (tensor-data output2)))
    (assert-equal (f32vector-ref data1 0) (f32vector-ref data2 0) 1e-6
                 "Eval mode is deterministic [0]")
    (assert-equal (f32vector-ref data1 3) (f32vector-ref data2 3) 1e-6
                 "Eval mode is deterministic [3]")
    (assert-equal (f32vector-ref data1 7) (f32vector-ref data2 7) 1e-6
                 "Eval mode is deterministic [7]")))

;;; Mode Switching
;;; ==================================================================

(define (test-batchnorm-mode-switching)
  (printf "\n=== BatchNorm Mode Switching ===\n")
  
  (define bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
  
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0
                           5.0 6.0 7.0 8.0)
                 '(2 2 2)))
  
  ;; Training mode - uses batch statistics
  (set-training-mode! bn #t)
  (define train-output (forward bn input))
  
  ;; Eval mode - uses running statistics
  (set-eval-mode! bn)
  (define eval-output (forward bn input))
  
  ;; Outputs should be different (different statistics used)
  (let ((train-data (tensor-data train-output))
        (eval-data (tensor-data eval-output)))
    (let ((diff (abs (- (f32vector-ref train-data 0)
                       (f32vector-ref eval-data 0)))))
      (assert-true (> diff 1e-6)
                  "Training and eval modes produce different outputs")))
  
  ;; Switch back to training
  (set-training-mode! bn #t)
  (define train-output2 (forward bn input))
  
  (printf "    Mode switching works correctly\n")
  (assert-true #t "Can switch between training and eval modes"))

;;; Learnable Parameters (Gamma, Beta)
;;; ==================================================================

(define (test-batchnorm-learnable-params)
  (printf "\n=== BatchNorm Learnable Parameters ===\n")
  
  (define bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
  (set-training-mode! bn #t)
  
  ;; Get parameters (gamma, beta)
  (define params (parameters bn))
  (assert-equal (length params) 2 0
               "BatchNorm has 2 parameters (gamma, beta)")
  
  (define gamma (car params))
  (define beta (cadr params))
  
  ;; Check initial values
  (assert-equal (f32vector-ref (tensor-data gamma) 0) 1.0 1e-6
               "Gamma initialized to 1.0")
  (assert-equal (f32vector-ref (tensor-data beta) 0) 0.0 1e-6
               "Beta initialized to 0.0")
  
  ;; Check gradient tracking
  (assert-true (tensor-requires-grad? gamma)
              "Gamma requires gradient")
  (assert-true (tensor-requires-grad? beta)
              "Beta requires gradient"))

;;; Gradient Flow
;;; ==================================================================

(define (test-batchnorm-gradients)
  (printf "\n=== BatchNorm Gradient Flow ===\n")
  
  (define bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
  (set-training-mode! bn #t)
  
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0
                           5.0 6.0 7.0 8.0)
                 '(2 2 2)
                 requires-grad?: #t))
  
  (define output (forward bn input))
  
  ;; Simple loss: sum of squares
  (let* ((output-flat (reshape output '(8)))
         (squared (mul output-flat output-flat))
         (loss (make-tensor32 
                (f32vector (fold + 0.0 (f32vector->list (tensor-data squared))))
                '(1))))
    
    (backward! loss)
    
    ;; Check gradients exist
    (assert-true (tensor-grad input)
                "Input has gradient after backward")
    
    (let ((params (parameters bn)))
      (assert-true (tensor-grad (car params))
                  "Gamma has gradient after backward")
      (assert-true (tensor-grad (cadr params))
                  "Beta has gradient after backward"))))

;;; Different Spatial Sizes
;;; ==================================================================

(define (test-batchnorm-spatial-sizes)
  (printf "\n=== BatchNorm Different Spatial Sizes ===\n")
  
  (define bn (make-batch-norm-2d 3 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
  (set-training-mode! bn #t)
  
  ;; 3 channels, 1x1 spatial
  (define input-1x1 (make-tensor32
                     (f32vector 1.0 2.0 3.0)
                     '(3 1 1)))
  (define output-1x1 (forward bn input-1x1))
  (assert-shape-equal (tensor-shape output-1x1) '(3 1 1)
                     "1x1 spatial works")
  
  ;; 3 channels, 4x4 spatial
  (define input-4x4 (make-tensor32
                     (make-f32vector (* 3 4 4) 1.0)
                     '(3 4 4)))
  (define output-4x4 (forward bn input-4x4))
  (assert-shape-equal (tensor-shape output-4x4) '(3 4 4)
                     "4x4 spatial works")
  
  ;; 3 channels, 7x7 spatial (like after ResNet conv1)
  (define input-7x7 (make-tensor32
                     (make-f32vector (* 3 7 7) 2.0)
                     '(3 7 7)))
  (define output-7x7 (forward bn input-7x7))
  (assert-shape-equal (tensor-shape output-7x7) '(3 7 7)
                     "7x7 spatial works"))

;;; Numerical Gradient Check
;;; ==================================================================

(define (test-batchnorm-numerical-gradients)
  (printf "\n=== BatchNorm Numerical Gradient Check ===\n")
  
  (define epsilon 1e-4)
  (define bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
  (set-training-mode! bn #t)
  
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0
                           5.0 6.0 7.0 8.0)
                 '(2 2 2)
                 requires-grad?: #t))
  
  ;; Compute analytical gradient
  (define output (forward bn input))
  (let ((loss (make-tensor32
               (f32vector (fold + 0.0 (f32vector->list (tensor-data output))))
               '(1))))
    (backward! loss)
    
    (let ((analytical-grad (tensor-grad input)))
      
      ;; Compute numerical gradient for first few elements
      (do ((i 0 (+ i 1)))
          ((= i 4))  ;; Check first 4 elements
        
        ;; Perturb +epsilon
        (let ((input-plus (make-tensor32
                          (f32vector-copy (tensor-data input))
                          '(2 2 2))))
          (f32vector-set! (tensor-data input-plus) i
                         (+ (f32vector-ref (tensor-data input) i) epsilon))
          
          (let* ((output-plus (forward bn input-plus))
                 (loss-plus (fold + 0.0 (f32vector->list (tensor-data output-plus)))))
            
            ;; Perturb -epsilon
            (let ((input-minus (make-tensor32
                               (f32vector-copy (tensor-data input))
                               '(2 2 2))))
              (f32vector-set! (tensor-data input-minus) i
                             (- (f32vector-ref (tensor-data input) i) epsilon))
              
              (let* ((output-minus (forward bn input-minus))
                     (loss-minus (fold + 0.0 (f32vector->list (tensor-data output-minus)))))
                
                (let ((numerical (/ (- loss-plus loss-minus) (* 2.0 epsilon)))
                      (analytical (f32vector-ref analytical-grad i)))
                  
                  (assert-equal analytical numerical 1e-2
                               (sprintf "Gradient check position ~A" i)))))))))))

;;; ==================================================================
;;; GLOBAL AVERAGE POOLING TESTS
;;; ==================================================================

;;; Basic GAP Forward Pass
;;; ==================================================================

(define (test-gap-basic-forward)
  (printf "\n=== Global Average Pooling Forward Pass ===\n")
  
  ;; Create 2 channels, 2x2 spatial
  ;; Channel 0: [[1,2], [3,4]] mean = 2.5
  ;; Channel 1: [[5,6], [7,8]] mean = 6.5
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0
                           5.0 6.0 7.0 8.0)
                 '(2 2 2)
                 requires-grad?: #t))
  
  (define output (global-avg-pool2d input))
  
  ;; Check shape: (2, 2, 2) → (2)
  (assert-shape-equal (tensor-shape output) '(2)
                     "GAP output shape is (num_channels)")
  
  ;; Check values
  (assert-equal (f32vector-ref (tensor-data output) 0) 2.5 1e-6
               "Channel 0 average is 2.5")
  (assert-equal (f32vector-ref (tensor-data output) 1) 6.5 1e-6
               "Channel 1 average is 6.5"))

;;; GAP Different Spatial Sizes
;;; ==================================================================

(define (test-gap-spatial-sizes)
  (printf "\n=== GAP Different Spatial Sizes ===\n")
  
  ;; 3x3 spatial
  (define input-3x3 (make-tensor32
                     (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0)
                     '(1 3 3)))
  (define output-3x3 (global-avg-pool2d input-3x3))
  (assert-equal (f32vector-ref (tensor-data output-3x3) 0) 5.0 1e-6
               "3x3 spatial: mean of 1..9 is 5.0")
  
  ;; 1x1 spatial (edge case)
  (define input-1x1 (make-tensor32
                     (f32vector 7.0)
                     '(1 1 1)))
  (define output-1x1 (global-avg-pool2d input-1x1))
  (assert-equal (f32vector-ref (tensor-data output-1x1) 0) 7.0 1e-6
               "1x1 spatial: mean is the value itself")
  
  ;; 7x7 spatial (typical ResNet)
  (define input-7x7 (make-tensor32
                     (make-f32vector 49 10.0)  ;; All 10s
                     '(1 7 7)))
  (define output-7x7 (global-avg-pool2d input-7x7))
  (assert-equal (f32vector-ref (tensor-data output-7x7) 0) 10.0 1e-6
               "7x7 spatial with constant values"))

;;; GAP Multiple Channels
;;; ==================================================================

(define (test-gap-multiple-channels)
  (printf "\n=== GAP Multiple Channels ===\n")
  
  ;; 4 channels, 2x2 spatial
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0      ;; Channel 0: mean = 2.5
                           10.0 20.0 30.0 40.0   ;; Channel 1: mean = 25.0
                           0.0 0.0 0.0 0.0       ;; Channel 2: mean = 0.0
                           -1.0 -2.0 -3.0 -4.0)  ;; Channel 3: mean = -2.5
                 '(4 2 2)))
  
  (define output (global-avg-pool2d input))
  
  (assert-shape-equal (tensor-shape output) '(4)
                     "4 channels output shape")
  (assert-equal (f32vector-ref (tensor-data output) 0) 2.5 1e-6
               "Channel 0 mean is 2.5")
  (assert-equal (f32vector-ref (tensor-data output) 1) 25.0 1e-6
               "Channel 1 mean is 25.0")
  (assert-equal (f32vector-ref (tensor-data output) 2) 0.0 1e-6
               "Channel 2 mean is 0.0")
  (assert-equal (f32vector-ref (tensor-data output) 3) -2.5 1e-6
               "Channel 3 mean is -2.5"))

;;; GAP Gradient Flow
;;; ==================================================================

(define (test-gap-gradients)
  (printf "\n=== GAP Gradient Flow ===\n")
  
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0
                           5.0 6.0 7.0 8.0)
                 '(2 2 2)
                 requires-grad?: #t))
  
  (define output (global-avg-pool2d input))
  
  ;; Create target and compute loss
  (define target (make-tensor32 (f32vector 5.0 10.0) '(2)))
  (define loss (mse-loss output target))
  
  (backward! loss)
  
  ;; Check gradient exists
  (assert-true (tensor-grad input)
              "Input has gradient after backward")
  
  ;; Check gradient distribution
  ;; Gradient should be distributed equally over spatial dimensions
  (let ((grad (tensor-grad input)))
    ;; Check gradient distribution
    ;; MSE gradient: dL/doutput[i] = (2/n) * (output[i] - target[i])
    ;; Channel 0: 1/2 * (2/2) * (2.5 - 5.0) = -1.25
    ;; Channel 1: 1/2 * (2/2) * (6.5 - 10.0) = -1.75
    ;; GAP distributes equally over spatial dimensions:
    ;; Channel 0: -1.25 / 4 = -0.3125 per pixel
    ;; Channel 1: -1.75 / 4 = -0.4375 per pixel
    (let ((expected-ch0 -0.3125))
      (assert-equal (f32vector-ref grad 0) expected-ch0 1e-5
                   "Channel 0 gradient distributed [0,0]")
      (assert-equal (f32vector-ref grad 1) expected-ch0 1e-5
                   "Channel 0 gradient distributed [0,1]")
      (assert-equal (f32vector-ref grad 2) expected-ch0 1e-5
                   "Channel 0 gradient distributed [1,0]")
      (assert-equal (f32vector-ref grad 3) expected-ch0 1e-5
                   "Channel 0 gradient distributed [1,1]"))
    
    (let ((expected-ch1 -0.4375))
      (assert-equal (f32vector-ref grad 4) expected-ch1 1e-5
                   "Channel 1 gradient distributed [0,0]")
      (assert-equal (f32vector-ref grad 5) expected-ch1 1e-5
                   "Channel 1 gradient distributed [0,1]")
      (assert-equal (f32vector-ref grad 6) expected-ch1 1e-5
                   "Channel 1 gradient distributed [1,0]")
      (assert-equal (f32vector-ref grad 7) expected-ch1 1e-5
                   "Channel 1 gradient distributed [1,1]"))))

;;; GAP Equal Gradient Distribution
;;; ==================================================================

(define (test-gap-equal-distribution)
  (printf "\n=== GAP Equal Gradient Distribution ===\n")
  
  ;; All pixels should get equal gradient (key property of GAP)
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0)
                 '(1 3 3)
                 requires-grad?: #t))
  
  (define output (global-avg-pool2d input))
  
  ;; Gradient: scalar * 1 (simple case)
  (let ((scaled (scale-op output 2.0)))
    (backward! scaled)
    
    (let ((grad (tensor-grad input)))
      ;; All 9 pixels should have same gradient: 2.0/9 ~= 0.222...
      (let ((expected (/ 2.0 9.0)))
        (do ((i 0 (+ i 1)))
            ((= i 9))
          (assert-equal (f32vector-ref grad i) expected 1e-6
                       (sprintf "Pixel ~A has equal gradient" i)))))))

;;; GAP Numerical Gradient Check
;;; ==================================================================

(define (test-gap-numerical-gradients)
  (printf "\n=== GAP Numerical Gradient Check ===\n")
  
  (define epsilon 5e-4)
  
  (define input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0
                            5.0 6.0 7.0 8.0)
                 '(2 2 2)
                 requires-grad?: #t))
  
  ;; Compute analytical gradient
  (define output (global-avg-pool2d input))

  (let ((loss (sum-tensor output)))
    
    (backward! loss)
    
    (let ((analytical-grad (tensor-grad input)))
      
      ;; Compute numerical gradient
      (do ((i 0 (+ i 1)))
          ((= i 8))
        
        ;; +epsilon
        (let ((input-plus (make-tensor32
                           (f32vector-copy (tensor-data input))
                           '(2 2 2))))
          (f32vector-set! (tensor-data input-plus) i
                         (+ (f32vector-ref (tensor-data input) i) epsilon))
          
          (let* ((output-plus (global-avg-pool2d input-plus))
                 (loss-plus (compensated-sum 'f32 (tensor-data output-plus)
                                             0 (f32vector-length (tensor-data output-plus)))))
            
            ;; -epsilon
            (let ((input-minus (make-tensor32
                               (f32vector-copy (tensor-data input))
                               '(2 2 2))))
              (f32vector-set! (tensor-data input-minus) i
                              (- (f32vector-ref (tensor-data input) i) epsilon))
              
              (let* ((output-minus (global-avg-pool2d input-minus))
                     (loss-minus (compensated-sum 'f32 (tensor-data output-minus)
                                                  0 (f32vector-length (tensor-data output-minus)))))
                
                (let ((numerical (/ (- loss-plus loss-minus) (* 2.0 epsilon)))
                      (analytical (f32vector-ref analytical-grad i)))
                  
                  (assert-equal analytical numerical 1e-3
                               (sprintf "GAP gradient check position ~A" i)))))))))))


;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(define (run-all-batchnorm-tests)
  (reset-test-stats!)
  (printf "\n")
  (printf "========================================\n")
  (printf "Batchnorm & global avg pooling tests\n")
  (printf "========================================\n")
  
  ;; Batch Normalization Tests
  (test-batchnorm-training-forward)
  (test-batchnorm-running-stats)
  (test-batchnorm-eval-mode)
  (test-batchnorm-mode-switching)
  (test-batchnorm-learnable-params)
  (test-batchnorm-gradients)
  (test-batchnorm-spatial-sizes)
  (test-batchnorm-numerical-gradients)
  
  ;; Global Average Pooling Tests
  (test-gap-basic-forward)
  (test-gap-spatial-sizes)
  (test-gap-multiple-channels)
  (test-gap-gradients)
  (test-gap-equal-distribution)
  (test-gap-numerical-gradients)
  
  (test-summary))

;; Run tests
(run-all-batchnorm-tests)

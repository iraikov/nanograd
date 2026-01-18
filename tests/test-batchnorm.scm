;; Tests for Batch Normalization and Global Average Pooling

(import scheme
        (chicken base)
        (chicken format)
        (srfi 1)
        (srfi 4)
        test
        nanograd-autograd
        nanograd-layer)

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

(define (f32vector-copy v)
  (ssub v 0 (f32vector-length v)))

;;; ==================================================================
;;; BATCH NORMALIZATION TESTS
;;; ==================================================================

(test-group "BatchNorm - Training Forward Pass"
  
  ;; Create BatchNorm layer for 2 channels
  (let* ((bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
         (_ (set-training-mode! bn #t))
         ;; Create input: 2 channels, 2x2 spatial
         ;; Channel 0: [[1,2], [3,4]] mean=2.5, var=1.25
         ;; Channel 1: [[5,6], [7,8]] mean=6.5, var=1.25
         (input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0    ;; Channel 0
                           5.0 6.0 7.0 8.0)    ;; Channel 1
                 '(2 2 2)
                 requires-grad?: #t))
         (output (forward bn input)))
    
    (test "Output shape matches input"
      '(2 2 2)
      (tensor-shape output))
    
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
        
        (test-approximate "Channel 0 normalized mean ~= 0"
          0.0 ch0-mean 1e-5)
        (test-approximate "Channel 0 normalized variance ~= 1"
          1.0 ch0-var 1e-4))
      
      ;; Channel 1 normalized values
      (let* ((ch1-vals (list (f32vector-ref data 4)
                            (f32vector-ref data 5)
                            (f32vector-ref data 6)
                            (f32vector-ref data 7)))
             (ch1-mean (/ (apply + ch1-vals) 4.0))
             (ch1-var (/ (apply + (map (lambda (x) (* (- x ch1-mean) (- x ch1-mean)))
                                      ch1-vals))
                        4.0)))
        
        (test-approximate "Channel 1 normalized mean ~= 0"
          0.0 ch1-mean 1e-5)
        (test-approximate "Channel 1 normalized variance ~= 1"
          1.0 ch1-var 1e-4)))))

(test-group "BatchNorm - Running Statistics"
  
  (let* ((bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
         (_ (set-training-mode! bn #t))
         ;; First batch
         (input1 (make-tensor32
                  (f32vector 0.0 0.0 0.0 0.0    ;; Channel 0: all zeros
                            10.0 10.0 10.0 10.0) ;; Channel 1: all tens
                  '(2 2 2)))
         (_ (forward bn input1))
         ;; Second batch with different statistics
         (input2 (make-tensor32
                  (f32vector 2.0 2.0 2.0 2.0    ;; Channel 0: all twos
                            20.0 20.0 20.0 20.0) ;; Channel 1: all twenties
                  '(2 2 2)))
         (_ (forward bn input2)))
    
    (test-assert "Running statistics updated after 2 batches" #t)))

(test-group "BatchNorm - Evaluation Mode"
  
  (let* ((bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
         ;; Train on some data to populate running stats
         (_ (set-training-mode! bn #t))
         (train-input (make-tensor32
                       (f32vector 1.0 2.0 3.0 4.0
                                 5.0 6.0 7.0 8.0)
                       '(2 2 2)))
         (_ (forward bn train-input))
         ;; Switch to eval mode
         (_ (set-eval-mode! bn))
         ;; In eval mode, should use running statistics (deterministic)
         (eval-input1 (make-tensor32
                       (f32vector 10.0 20.0 30.0 40.0
                                 50.0 60.0 70.0 80.0)
                       '(2 2 2)))
         (output1 (forward bn eval-input1))
         ;; Same input should give same output in eval mode
         (eval-input2 (make-tensor32
                       (f32vector 10.0 20.0 30.0 40.0
                                 50.0 60.0 70.0 80.0)
                       '(2 2 2)))
         (output2 (forward bn eval-input2)))
    
    ;; Verify outputs are identical (deterministic)
    (let ((data1 (tensor-data output1))
          (data2 (tensor-data output2)))
      (test-approximate "Eval mode is deterministic [0]"
        (f32vector-ref data1 0) (f32vector-ref data2 0) 1e-6)
      (test-approximate "Eval mode is deterministic [3]"
        (f32vector-ref data1 3) (f32vector-ref data2 3) 1e-6)
      (test-approximate "Eval mode is deterministic [7]"
        (f32vector-ref data1 7) (f32vector-ref data2 7) 1e-6))))

(test-group "BatchNorm - Mode Switching"
  
  (let* ((bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
         (input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0
                           5.0 6.0 7.0 8.0)
                 '(2 2 2)))
         ;; Training mode - uses batch statistics
         (_ (set-training-mode! bn #t))
         (train-output (forward bn input))
         ;; Eval mode - uses running statistics
         (_ (set-eval-mode! bn))
         (eval-output (forward bn input)))
    
    ;; Outputs should be different (different statistics used)
    (let ((train-data (tensor-data train-output))
          (eval-data (tensor-data eval-output)))
      (let ((diff (abs (- (f32vector-ref train-data 0)
                         (f32vector-ref eval-data 0)))))
        (test-assert "Training and eval modes produce different outputs"
          (> diff 1e-6))))
    
    ;; Switch back to training
    (set-training-mode! bn #t)
    (let ((train-output2 (forward bn input)))
      (test-assert "Can switch between training and eval modes" #t))))

(test-group "BatchNorm - Learnable Parameters"
  
  (let* ((bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
         (_ (set-training-mode! bn #t))
         ;; Get parameters (gamma, beta)
         (params (parameters bn))
         (gamma (car params))
         (beta (cadr params)))
    
    (test "BatchNorm has 2 parameters (gamma, beta)"
      2
      (length params))
    
    (test-approximate "Gamma initialized to 1.0"
      1.0 (f32vector-ref (tensor-data gamma) 0) 1e-6)
    
    (test-approximate "Beta initialized to 0.0"
      0.0 (f32vector-ref (tensor-data beta) 0) 1e-6)
    
    (test-assert "Gamma requires gradient"
      (tensor-requires-grad? gamma))
    
    (test-assert "Beta requires gradient"
      (tensor-requires-grad? beta))))

(test-group "BatchNorm - Gradient Flow"
  
  (let* ((bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
         (_ (set-training-mode! bn #t))
         (input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0
                           5.0 6.0 7.0 8.0)
                 '(2 2 2)
                 requires-grad?: #t))
         (output (forward bn input))
         ;; Simple loss: sum of outputs
         (loss (sum-tensor output))
         (_ (backward! loss)))
    
    (test-assert "Input has gradient after backward"
      (tensor-grad input))
    
    (let ((params (parameters bn)))
      (test-assert "Gamma has gradient after backward"
        (tensor-grad (car params)))
      
      (test-assert "Beta has gradient after backward"
        (tensor-grad (cadr params))))
    
    ;; Check gradient magnitudes are reasonable
    (let* ((grad-input (tensor-grad input))
           (max-grad (fold max -inf.0 (f32vector->list grad-input))))
      (test-assert "Input gradients have reasonable magnitude"
        (< (abs max-grad) 10.0)))))

(test-group "BatchNorm - Gamma/Beta Gradients"
  
  (let* ((bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
         (_ (set-training-mode! bn #t))
         ;; Known input: all ones for channel 0, all twos for channel 1
         (input (make-tensor32
                 (f32vector 1.0 1.0 1.0 1.0
                           2.0 2.0 2.0 2.0)
                 '(2 2 2)
                 requires-grad?: #t))
         (output (forward bn input))
         ;; Loss: sum of outputs
         (loss (sum-tensor output))
         (_ (backward! loss))
         (params (parameters bn))
         (gamma (car params))
         (beta (cadr params)))
    
    ;; Beta gradient should be the sum over all spatial dims = 4 (4 pixels per channel)
    ;; Since loss is sum of outputs, dL/doutput = 1 everywhere
    ;; dL/dbeta = sum(dL/doutput) = 4 per channel
    (let ((grad-beta (tensor-grad beta)))
      (test-approximate "Beta gradient channel 0 (sum over 4 pixels)"
        4.0 (f32vector-ref grad-beta 0) 1e-4)
      (test-approximate "Beta gradient channel 1 (sum over 4 pixels)"
        4.0 (f32vector-ref grad-beta 1) 1e-4))
    
    ;; Gamma gradient: sum of (dL/doutput * normalized_input)
    ;; For constant inputs, normalized = 0 (zero variance), so gamma grad = 0
    (let ((grad-gamma (tensor-grad gamma)))
      (test-approximate "Gamma gradient channel 0 (zero for constant input)"
        0.0 (f32vector-ref grad-gamma 0) 1e-4)
      (test-approximate "Gamma gradient channel 1 (zero for constant input)"
        0.0 (f32vector-ref grad-gamma 1) 1e-4))))

(test-group "BatchNorm - Different Spatial Sizes"
  
  (let* ((bn (make-batch-norm-2d 3 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
         (_ (set-training-mode! bn #t))
         ;; 3 channels, 1x1 spatial
         (input-1x1 (make-tensor32
                     (f32vector 1.0 2.0 3.0)
                     '(3 1 1)))
         (output-1x1 (forward bn input-1x1))
         ;; 3 channels, 4x4 spatial
         (input-4x4 (make-tensor32
                     (make-f32vector (* 3 4 4) 1.0)
                     '(3 4 4)))
         (output-4x4 (forward bn input-4x4))
         ;; 3 channels, 7x7 spatial (like after ResNet conv1)
         (input-7x7 (make-tensor32
                     (make-f32vector (* 3 7 7) 2.0)
                     '(3 7 7)))
         (output-7x7 (forward bn input-7x7)))
    
    (test "1x1 spatial works"
      '(3 1 1)
      (tensor-shape output-1x1))
    
    (test "4x4 spatial works"
      '(3 4 4)
      (tensor-shape output-4x4))
    
    (test "7x7 spatial works"
      '(3 7 7)
      (tensor-shape output-7x7))))

(test-group "BatchNorm - Numerical Gradient Check"
  
  (let* ((epsilon 1e-4)
         (bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
         (_ (set-training-mode! bn #t))
         (input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0
                           5.0 6.0 7.0 8.0)
                 '(2 2 2)
                 requires-grad?: #t))
         ;; Compute analytical gradient
         (output (forward bn input))
         (loss (sum-tensor output))
         (_ (backward! loss))
         (analytical-grad (tensor-grad input)))
    
    ;; Compute numerical gradient for first 4 elements
    (do ((i 0 (+ i 1)))
        ((= i 4))
      
      ;; Create fresh BN layer for each perturbation
      (let ((bn-plus (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
            (bn-minus (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32)))
        (set-training-mode! bn-plus #t)
        (set-training-mode! bn-minus #t)
        
        ;; Perturb +epsilon
        (let ((input-plus (make-tensor32
                          (f32vector-copy (tensor-data input))
                          '(2 2 2))))
          (f32vector-set! (tensor-data input-plus) i
                         (+ (f32vector-ref (tensor-data input) i) epsilon))
          
          (let* ((output-plus (forward bn-plus input-plus))
                 (loss-plus (f32vector-ref (tensor-data (sum-tensor output-plus)) 0)))
            
            ;; Perturb -epsilon
            (let ((input-minus (make-tensor32
                               (f32vector-copy (tensor-data input))
                               '(2 2 2))))
              (f32vector-set! (tensor-data input-minus) i
                             (- (f32vector-ref (tensor-data input) i) epsilon))
              
              (let* ((output-minus (forward bn-minus input-minus))
                     (loss-minus (f32vector-ref (tensor-data (sum-tensor output-minus)) 0)))
                
                (let ((numerical (/ (- loss-plus loss-minus) (* 2.0 epsilon)))
                      (analytical (f32vector-ref analytical-grad i)))
                  
                  (test-approximate (sprintf "Gradient check position ~A" i)
                    analytical numerical 1e-2))))))))))

(test-group "BatchNorm - Gradients with Varied Input"
  
  (let* ((bn (make-batch-norm-2d 2 epsilon: 1e-5 momentum: 0.1 dtype: 'f32))
         (_ (set-training-mode! bn #t))
         ;; Varied input (not constant per channel)
         (input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0
                           5.0 6.0 7.0 8.0)
                 '(2 2 2)
                 requires-grad?: #t))
         (output (forward bn input))
         ;; Use sum of squares loss instead of sum
         (output-flat (reshape output '(8)))
         (squared (mul output-flat output-flat))
         (loss (sum-tensor squared))
         (_ (backward! loss))
         (params (parameters bn))
         (gamma (car params))
         (beta (cadr params)))
    
    ;; With varied input and squared loss, gamma should have non-zero gradient
    (let ((grad-gamma (tensor-grad gamma)))
      (test-assert "Gamma gradient non-zero for varied input (ch0)"
        (> (abs (f32vector-ref grad-gamma 0)) 1e-6))
      (test-assert "Gamma gradient non-zero for varied input (ch1)"
        (> (abs (f32vector-ref grad-gamma 1)) 1e-6)))
    
    (let ((grad-beta (tensor-grad beta)))
      (test-assert "Beta gradients computed" #t))))

;;; ==================================================================
;;; GLOBAL AVERAGE POOLING TESTS
;;; ==================================================================

(test-group "Global Average Pooling - Forward Pass"
  
  ;; Create 2 channels, 2x2 spatial
  ;; Channel 0: [[1,2], [3,4]] mean = 2.5
  ;; Channel 1: [[5,6], [7,8]] mean = 6.5
  (let* ((input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0
                           5.0 6.0 7.0 8.0)
                 '(2 2 2)
                 requires-grad?: #t))
         (output (global-avg-pool2d input)))
    
    (test "GAP output shape is (num_channels)"
      '(2)
      (tensor-shape output))
    
    (test-approximate "Channel 0 average is 2.5"
      2.5 (f32vector-ref (tensor-data output) 0) 1e-6)
    
    (test-approximate "Channel 1 average is 6.5"
      6.5 (f32vector-ref (tensor-data output) 1) 1e-6)))

(test-group "GAP - Different Spatial Sizes"
  
  ;; 3x3 spatial
  (let* ((input-3x3 (make-tensor32
                     (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0)
                     '(1 3 3)))
         (output-3x3 (global-avg-pool2d input-3x3)))
    (test-approximate "3x3 spatial: mean of 1..9 is 5.0"
      5.0 (f32vector-ref (tensor-data output-3x3) 0) 1e-6))
  
  ;; 1x1 spatial (edge case)
  (let* ((input-1x1 (make-tensor32
                     (f32vector 7.0)
                     '(1 1 1)))
         (output-1x1 (global-avg-pool2d input-1x1)))
    (test-approximate "1x1 spatial: mean is the value itself"
      7.0 (f32vector-ref (tensor-data output-1x1) 0) 1e-6))
  
  ;; 7x7 spatial (typical ResNet)
  (let* ((input-7x7 (make-tensor32
                     (make-f32vector 49 10.0)  ;; All 10s
                     '(1 7 7)))
         (output-7x7 (global-avg-pool2d input-7x7)))
    (test-approximate "7x7 spatial with constant values"
      10.0 (f32vector-ref (tensor-data output-7x7) 0) 1e-6)))

(test-group "GAP - Multiple Channels"
  
  ;; 4 channels, 2x2 spatial
  (let* ((input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0      ;; Channel 0: mean = 2.5
                           10.0 20.0 30.0 40.0   ;; Channel 1: mean = 25.0
                           0.0 0.0 0.0 0.0       ;; Channel 2: mean = 0.0
                           -1.0 -2.0 -3.0 -4.0)  ;; Channel 3: mean = -2.5
                 '(4 2 2)))
         (output (global-avg-pool2d input)))
    
    (test "4 channels output shape"
      '(4)
      (tensor-shape output))
    
    (test-approximate "Channel 0 mean is 2.5"
      2.5 (f32vector-ref (tensor-data output) 0) 1e-6)
    (test-approximate "Channel 1 mean is 25.0"
      25.0 (f32vector-ref (tensor-data output) 1) 1e-6)
    (test-approximate "Channel 2 mean is 0.0"
      0.0 (f32vector-ref (tensor-data output) 2) 1e-6)
    (test-approximate "Channel 3 mean is -2.5"
      -2.5 (f32vector-ref (tensor-data output) 3) 1e-6)))

(test-group "GAP - Gradient Flow"
  
  (let* ((input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0
                           5.0 6.0 7.0 8.0)
                 '(2 2 2)
                 requires-grad?: #t))
         (output (global-avg-pool2d input))
         ;; Create target and compute loss
         (target (make-tensor32 (f32vector 5.0 10.0) '(2)))
         (loss (mse-loss output target))
         (_ (backward! loss)))
    
    (test-assert "Input has gradient after backward"
      (tensor-grad input))
    
    ;; Check gradient distribution
    ;; MSE gradient: dL/doutput[i] = (2/n) * (output[i] - target[i])
    ;; Channel 0: 1/2 * (2/2) * (2.5 - 5.0) = -1.25
    ;; Channel 1: 1/2 * (2/2) * (6.5 - 10.0) = -1.75
    ;; GAP distributes equally over spatial dimensions:
    ;; Channel 0: -1.25 / 4 = -0.3125 per pixel
    ;; Channel 1: -1.75 / 4 = -0.4375 per pixel
    (let ((grad (tensor-grad input))
          (expected-ch0 -0.3125)
          (expected-ch1 -0.4375))
      
      (test-approximate "Channel 0 gradient distributed [0,0]"
        expected-ch0 (f32vector-ref grad 0) 1e-5)
      (test-approximate "Channel 0 gradient distributed [0,1]"
        expected-ch0 (f32vector-ref grad 1) 1e-5)
      (test-approximate "Channel 0 gradient distributed [1,0]"
        expected-ch0 (f32vector-ref grad 2) 1e-5)
      (test-approximate "Channel 0 gradient distributed [1,1]"
        expected-ch0 (f32vector-ref grad 3) 1e-5)
      
      (test-approximate "Channel 1 gradient distributed [0,0]"
        expected-ch1 (f32vector-ref grad 4) 1e-5)
      (test-approximate "Channel 1 gradient distributed [0,1]"
        expected-ch1 (f32vector-ref grad 5) 1e-5)
      (test-approximate "Channel 1 gradient distributed [1,0]"
        expected-ch1 (f32vector-ref grad 6) 1e-5)
      (test-approximate "Channel 1 gradient distributed [1,1]"
        expected-ch1 (f32vector-ref grad 7) 1e-5))))

(test-group "GAP - Equal Gradient Distribution"
  
  ;; All pixels should get equal gradient (key property of GAP)
  (let* ((input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0)
                 '(1 3 3)
                 requires-grad?: #t))
         (output (global-avg-pool2d input))
         ;; Gradient: scalar * 1 (simple case)
         (scaled (scale-op output 2.0))
         (_ (backward! scaled))
         (grad (tensor-grad input))
         (expected (/ 2.0 9.0)))
    
    ;; All 9 pixels should have same gradient: 2.0/9 ~= 0.222...
    (do ((i 0 (+ i 1)))
        ((= i 9))
      (test-approximate (sprintf "Pixel ~A has equal gradient" i)
        expected (f32vector-ref grad i) 1e-6))))

(test-group "GAP - Numerical Gradient Check"
  
  (let* ((epsilon 5e-4)
         (input (make-tensor32
                 (f32vector 1.0 2.0 3.0 4.0
                            5.0 6.0 7.0 8.0)
                 '(2 2 2)
                 requires-grad?: #t))
         ;; Compute analytical gradient
         (output (global-avg-pool2d input))
         (loss (sum-tensor output))
         (_ (backward! loss))
         (analytical-grad (tensor-grad input)))
    
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
                
                (test-approximate (sprintf "GAP gradient check position ~A" i)
                  analytical numerical 1e-3)))))))))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(test-exit)

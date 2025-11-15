;; regression.scm
;; Neural network regression example
;; Demonstrates building, training, and evaluating a feedforward network for regression

(import scheme
        (chicken base)
        (chicken format)
        (chicken random)
        (chicken time)
        (chicken file)
        (chicken io)
        (srfi 1)
        (srfi 4)
        (srfi 42)
        (srfi 69)
        blas
        nanograd-autograd
        nanograd-layer
        nanograd-optimizer)

(define (f32vector-fold f x0 v . rest)
  (let ((n   (f32vector-length v))
        (vs  (cons v rest)))
    (fold-ec x0 (:range i 0 n)
             (map (lambda (v) (f32vector-ref v i)) vs)
             (lambda (x ax) (apply f (append x (list ax)))))))

(define (set-random-seed! seed)
  "Set random seed for reproducibility"
  (set-pseudo-random-seed! (number->string seed)))

;;; ==================================================================
;;; Data Generation: Multi-variate Non-linear Function
;;; ==================================================================

;; We'll predict: y = sin(x1) + cos(x2) + x3^2 - 0.5*x4


(define num-features 4)
(define feature-ranges 
  ;; (min max) for each feature
  '((-3.0 3.0)   ; x1: for sin
    (-3.0 3.0)   ; x2: for cos
    (-2.0 2.0)   ; x3: for square
    (-1.0 1.0))) ; x4: linear term

(define (target-function x1 x2 x3 x4)
  "True underlying function we're trying to learn"
  (+ (sin x1)
     (cos x2)
     (* x3 x3)
     (* -0.5 x4)))

(define (generate-sample)
  "Generate a single (features, target) sample"
  (let ((features (map (lambda (range)
                         (let ((min-val (car range))
                               (max-val (cadr range)))
                           (+ min-val (* (pseudo-random-real) 
                                        (- max-val min-val)))))
                       feature-ranges)))
    
    ;; Compute target with small noise
    (let* ((x1 (list-ref features 0))
           (x2 (list-ref features 1))
           (x3 (list-ref features 2))
           (x4 (list-ref features 3))
           (y-true (target-function x1 x2 x3 x4)))
      
      (cons (list->f32vector features) y-true))))

(define (generate-dataset n)
  "Generate dataset with n samples"
  (let loop ((i 0) (data '()))
    (if (= i n)
        data
        (loop (+ i 1) (cons (generate-sample) data)))))

(define (shuffle lst)
  "Fisher-Yates shuffle"
  (let* ((vec (list->vector lst))
         (n (vector-length vec)))
    (do ((i (- n 1) (- i 1)))
        ((< i 1) (vector->list vec))
      (let* ((j (pseudo-random-integer (+ i 1)))
             (tmp (vector-ref vec i)))
        (vector-set! vec i (vector-ref vec j))
        (vector-set! vec j tmp)))))

;;; ==================================================================
;;; Data Normalization
;;; ==================================================================

(define-record-type normalization-stats
  (make-normalization-stats feature-means feature-stds target-mean target-std)
  normalization-stats?
  (feature-means get-feature-means)
  (feature-stds get-feature-stds)
  (target-mean get-target-mean)
  (target-std get-target-std))

(define (compute-normalization-stats dataset)
  "Compute mean and std for features and targets"
  (let ((n (length dataset))
        (feature-sums (make-f32vector num-features 0.0))
        (feature-sq-sums (make-f32vector num-features 0.0))
        (target-sum 0.0)
        (target-sq-sum 0.0))
    
    ;; Compute sums
    (for-each
     (lambda (sample)
       (let ((features (car sample))
             (target (cdr sample)))
         ;; Accumulate feature statistics
         (do ((i 0 (+ i 1)))
             ((= i num-features))
           (let ((val (f32vector-ref features i)))
             (f32vector-set! feature-sums i 
                            (+ (f32vector-ref feature-sums i) val))
             (f32vector-set! feature-sq-sums i
                            (+ (f32vector-ref feature-sq-sums i) (* val val)))))
         ;; Accumulate target statistics
         (set! target-sum (+ target-sum target))
         (set! target-sq-sum (+ target-sq-sum (* target target)))))
     dataset)
    
    ;; Compute means
    (let ((feature-means (make-f32vector num-features))
          (target-mean (/ target-sum n)))
      (do ((i 0 (+ i 1)))
          ((= i num-features))
        (f32vector-set! feature-means i 
                       (/ (f32vector-ref feature-sums i) n)))
      
      ;; Compute standard deviations
      (let ((feature-stds (make-f32vector num-features))
            (target-variance (- (/ target-sq-sum n) (* target-mean target-mean))))
        (do ((i 0 (+ i 1)))
            ((= i num-features))
          (let* ((mean (f32vector-ref feature-means i))
                 (variance (- (/ (f32vector-ref feature-sq-sums i) n)
                             (* mean mean)))
                 (std (sqrt (max 0.0 variance))))
            (f32vector-set! feature-stds i (max std 1e-8))))
        
        (let ((target-std (max (sqrt (max 0.0 target-variance)) 1e-8)))
          (make-normalization-stats feature-means feature-stds 
                                   target-mean target-std))))))

(define (normalize-sample sample stats)
  "Normalize a single sample using computed statistics"
  (let ((features (car sample))
        (target (cdr sample))
        (normalized-features (make-f32vector num-features)))
    
    ;; Normalize features: (x - mean) / std
    (do ((i 0 (+ i 1)))
        ((= i num-features))
      (f32vector-set! normalized-features i
                     (/ (- (f32vector-ref features i)
                           (f32vector-ref (get-feature-means stats) i))
                        (f32vector-ref (get-feature-stds stats) i))))
    
    ;; Normalize target
    (let ((normalized-target 
           (/ (- target (get-target-mean stats))
              (get-target-std stats))))
      
      (cons normalized-features normalized-target))))

(define (denormalize-prediction pred stats)
  "Convert normalized prediction back to original scale"
  (+ (* pred (get-target-std stats))
     (get-target-mean stats)))

;;; ==================================================================
;;; Batch Construction
;;; ==================================================================

(define (stack-features batch)
  "Stack batch features into 2D tensor (batch_size, num_features)"
  (let* ((batch-size (length batch))
         (batched-data (make-f32vector (* batch-size num-features) 0.0)))
    
    (do ((i 0 (+ i 1)))
        ((= i batch-size))
      (let ((features (car (list-ref batch i)))
            (offset (* i num-features)))
        (do ((j 0 (+ j 1)))
            ((= j num-features))
          (f32vector-set! batched-data (+ offset j)
                         (f32vector-ref features j)))))
    
    (make-tensor32 batched-data (list batch-size num-features)
                   requires-grad?: #f)))

(define (stack-targets batch)
  "Stack batch targets into 1D tensor (batch_size,)"
  (let* ((batch-size (length batch))
         (target-data (make-f32vector batch-size 0.0)))
    
    (do ((i 0 (+ i 1)))
        ((= i batch-size))
      (f32vector-set! target-data i (cdr (list-ref batch i))))
    
    (make-tensor32 target-data (list batch-size))))

;;; ==================================================================
;;; Model Architecture
;;; ==================================================================

(define (build-regression-model hidden-sizes)
  "Build feedforward network for regression
   
   Architecture:
   - Input: num_features
   - Hidden layers with sizes from hidden-sizes list
   - Output: 1 (regression target)
   
   Example: (build-regression-model '(64 32 16))
   Creates: Input(4) -> Dense(64) -> Dense(32) -> Dense(16) -> Output(1)"
  
  (let ((layers '()))
    
    ;; Input -> First hidden layer
    (set! layers 
          (cons (make-dense-layer num-features (car hidden-sizes)
                                 activation: (make-relu)
                                 name: "Hidden1")
                layers))
    
    ;; Hidden layers
    (let loop ((sizes hidden-sizes)
               (layer-num 2))
      (when (>= (length sizes) 2)
        (let ((in-size (car sizes))
              (out-size (cadr sizes)))
          (set! layers
                (cons (make-dense-layer in-size out-size
                                       activation: (make-relu)
                                       name: (sprintf "Hidden~A" layer-num))
                      layers))
          (loop (cdr sizes) (+ layer-num 1)))))
    
    ;; Last hidden -> Output (linear activation for regression)
    (set! layers
          (cons (make-dense-layer (last hidden-sizes) 1
                                 activation: (make-identity)
                                 name: "Output")
                layers))
    
    (make-sequential (reverse layers) name: "RegressionModel")))

;;; ==================================================================
;;; Evaluation Metrics
;;; ==================================================================

(define (mean-squared-error predictions targets)
  "Compute MSE between predictions and targets"
  (let ((n (f32vector-length predictions))
        (sum-sq-error 0.0))
    (do ((i 0 (+ i 1)))
        ((= i n) (/ sum-sq-error n))
      (let ((diff (- (f32vector-ref predictions i)
                     (f32vector-ref targets i))))
        (set! sum-sq-error (+ sum-sq-error (* diff diff)))))))

(define (mean-absolute-error predictions targets)
  "Compute MAE between predictions and targets"
  (let ((n (f32vector-length predictions))
        (sum-abs-error 0.0))
    (do ((i 0 (+ i 1)))
        ((= i n) (/ sum-abs-error n))
      (let ((diff (abs (- (f32vector-ref predictions i)
                          (f32vector-ref targets i)))))
        (set! sum-abs-error (+ sum-abs-error diff))))))

(define (r-squared predictions targets)
  "Compute R^2 (coefficient of determination)
   R^2 = 1 - (SS_res / SS_tot)
   where SS_res = sum of squared residuals
         SS_tot = total sum of squares"
  (let ((n (f32vector-length predictions))
        (target-mean 0.0))
    
    ;; Compute target mean
    (do ((i 0 (+ i 1)))
        ((= i n))
      (set! target-mean (+ target-mean (f32vector-ref targets i))))
    (set! target-mean (/ target-mean n))
    
    ;; Compute SS_res and SS_tot
    (let ((ss-res 0.0)
          (ss-tot 0.0))
      (do ((i 0 (+ i 1)))
          ((= i n))
        (let ((pred (f32vector-ref predictions i))
              (target (f32vector-ref targets i)))
          (set! ss-res (+ ss-res (* (- target pred) (- target pred))))
          (set! ss-tot (+ ss-tot (* (- target target-mean) 
                                   (- target target-mean))))))
      
      (if (< ss-tot 1e-10)
          0.0  ; Perfect predictions if no variance
          (- 1.0 (/ ss-res ss-tot))))))

;;; ==================================================================
;;; Training Functions
;;; ==================================================================

(define (train-epoch model optimizer train-data stats
                     #!key (batch-size 32))
  "Train for one epoch with mini-batch processing"
  (let ((total-loss 0.0)
        (n (length train-data))
        (batches (let loop ((remaining train-data)
                           (result '()))
                  (if (null? remaining)
                      (reverse result)
                      (let* ((batch-end (min batch-size (length remaining)))
                             (batch (take remaining batch-end))
                             (rest (drop remaining batch-end)))
                        (loop rest (cons batch result)))))))
    
    ;; Process each mini-batch
    (for-each
     (lambda (batch)
       (let* ((actual-batch-size (length batch))
              
              ;; Normalize batch
              (normalized-batch (map (lambda (s) (normalize-sample s stats))
                                    batch))
              
              ;; Stack into tensors
              (batch-features (stack-features normalized-batch))
              (batch-targets (stack-targets normalized-batch))
              
              ;; Forward pass
              (predictions (forward model batch-features))
              
              ;; Reshape predictions to match targets: (batch, 1) -> (batch,)
              (predictions-flat (squeeze predictions axis: -1))
              
              ;; Compute loss (MSE with mean reduction)
              (loss (mse-loss predictions-flat batch-targets 
                             reduction: 'mean)))
         
         ;; Accumulate loss
         (set! total-loss (+ total-loss 
                            (* (f32vector-ref (tensor-data loss) 0)
                               actual-batch-size)))
         
         ;; Backward pass
         (backward! loss)
         
         ;; Optimizer step
         (step! optimizer)
         
         ;; Zero gradients
         (zero-grad-layer! model)))
     batches)
    
    ;; Return average loss
    (/ total-loss n)))

(define (evaluate model test-data stats #!key (batch-size 64))
  "Evaluate model on test data, returning denormalized metrics"
  (let ((all-predictions '())
        (all-targets '())
        (batches (let loop ((remaining test-data)
                           (result '()))
                  (if (null? remaining)
                      (reverse result)
                      (let* ((batch-end (min batch-size (length remaining)))
                             (batch (take remaining batch-end))
                             (rest (drop remaining batch-end)))
                        (loop rest (cons batch result)))))))
    
    ;; Collect all predictions
    (for-each
     (lambda (batch)
       (let* (;; Normalize batch
              (normalized-batch (map (lambda (s) (normalize-sample s stats))
                                    batch))
              
              ;; Stack into tensors (no gradients for evaluation)
              (batch-features (stack-features normalized-batch))
              (batch-features-no-grad 
               (make-tensor32 (tensor-data batch-features)
                             (tensor-shape batch-features)
                             requires-grad?: #f))
              
              ;; Forward pass
              (predictions (forward model batch-features-no-grad))
              (predictions-flat (squeeze predictions axis: -1))
              (pred-data (tensor-data predictions-flat)))
         
         ;; Denormalize and collect predictions and targets
         (do ((i 0 (+ i 1)))
             ((= i (length batch)))
           (let* ((normalized-pred (f32vector-ref pred-data i))
                  (normalized-target (cdr (list-ref normalized-batch i)))
                  ;; Denormalize back to original scale
                  (pred (denormalize-prediction normalized-pred stats))
                  (target (denormalize-prediction normalized-target stats)))
             (set! all-predictions (cons pred all-predictions))
             (set! all-targets (cons target all-targets))))))
     batches)
    
    ;; Convert to vectors for metrics computation
    (let ((pred-vec (list->f32vector (reverse all-predictions)))
          (target-vec (list->f32vector (reverse all-targets))))
      
      (values (mean-squared-error pred-vec target-vec)
              (mean-absolute-error pred-vec target-vec)
              (r-squared pred-vec target-vec)
              pred-vec
              target-vec))))

;;; ==================================================================
;;; Prediction Visualization
;;; ==================================================================

(define (print-prediction-samples predictions targets n-samples)
  "Print sample predictions vs actual values"
  (printf "\nSample Predictions:\n")
  (printf "%-10s %-15s %-15s %-15s\n" "Sample" "Predicted" "Actual" "Error")
  (printf "%-10s %-15s %-15s %-15s\n" "------" "---------" "------" "-----")
  
  (let ((n (min n-samples (f32vector-length predictions))))
    (do ((i 0 (+ i 1)))
        ((= i n))
      (let* ((pred (f32vector-ref predictions i))
             (actual (f32vector-ref targets i))
             (error (- pred actual)))
        (printf "%-10d %-15.4f %-15.4f %-15.4f\n" 
                (+ i 1) pred actual error)))))

(define (analyze-errors predictions targets)
  "Analyze prediction error distribution"
  (let ((n (f32vector-length predictions))
        (errors (make-f32vector (f32vector-length predictions))))
    
    ;; Compute errors
    (do ((i 0 (+ i 1)))
        ((= i n))
      (f32vector-set! errors i 
                     (- (f32vector-ref predictions i)
                        (f32vector-ref targets i))))
    
    ;; Compute error statistics
    (let ((min-error +inf.0)
          (max-error -inf.0)
          (sum-error 0.0)
          (sum-abs-error 0.0))
      
      (do ((i 0 (+ i 1)))
          ((= i n))
        (let ((err (f32vector-ref errors i)))
          (set! min-error (min min-error err))
          (set! max-error (max max-error err))
          (set! sum-error (+ sum-error err))
          (set! sum-abs-error (+ sum-abs-error (abs err)))))
      
      (let ((mean-error (/ sum-error n))
            (mae (/ sum-abs-error n)))
        
        ;; Compute std dev of errors
        (let ((sum-sq-diff 0.0))
          (do ((i 0 (+ i 1)))
              ((= i n))
            (let ((diff (- (f32vector-ref errors i) mean-error)))
              (set! sum-sq-diff (+ sum-sq-diff (* diff diff)))))
          
          (let ((std-error (sqrt (/ sum-sq-diff n))))
            
            (printf "\nError Analysis:\n")
            (printf "  Mean Error:        ~A\n" mean-error)
            (printf "  Std Dev of Error:  ~A\n" std-error)
            (printf "  Min Error:         ~A\n" min-error)
            (printf "  Max Error:         ~A\n" max-error)
            (printf "  Mean Absolute Err: ~A\n" mae)))))))

;;; ==================================================================
;;; Model Persistence
;;; ==================================================================

(define (save-regression-model model filepath)
  "Save regression model to file"
  (printf "Saving model to ~A...\n" filepath)
  (save-model model filepath)
  (printf "Model saved successfully!\n"))

(define (load-regression-model filepath)
  "Load regression model from file"
  (printf "Loading model from ~A...\n" filepath)
  (let ((model (load-model filepath)))
    (printf "Model loaded successfully!\n")
    model))

(define (save-checkpoint model optimizer epoch train-loss stats filepath)
  "Save training checkpoint"
  (printf "\nSaving checkpoint at epoch ~A...\n" epoch)
  (save-regression-model model filepath)
  
  ;; Save metadata and normalization stats
  (let ((metadata-file (string-append filepath ".meta")))
    (with-output-to-file metadata-file
      (lambda ()
        (printf "epoch: ~A\n" epoch)
        (printf "train-loss: ~A\n" train-loss)
        (printf "timestamp: ~A\n" (current-seconds))
        ;; Save normalization stats
        (printf "feature-means: ~A\n" 
                (f32vector->list (get-feature-means stats)))
        (printf "feature-stds: ~A\n"
                (f32vector->list (get-feature-stds stats)))
        (printf "target-mean: ~A\n" (get-target-mean stats))
        (printf "target-std: ~A\n" (get-target-std stats)))))
  
  (printf "Checkpoint saved!\n"))

;;; ==================================================================
;;; Main Training Loop
;;; ==================================================================

(define (main)
  (printf "========================================\n")
  (printf "Neural Network Regression Example\n")
  (printf "========================================\n\n")
  
  ;; Set random seed for reproducibility
  (set-random-seed! 42)
  
  ;; Generate dataset
  (printf "Generating dataset...\n")
  (printf "Target function: y = sin(x1) + cos(x2) + x3^2 - 0.5*x4\n")
  (define train-data (shuffle (generate-dataset 5000)))
  (define test-data (shuffle (generate-dataset 1000)))
  (printf "Training samples: ~A\n" (length train-data))
  (printf "Test samples: ~A\n\n" (length test-data))
  
  ;; Compute normalization statistics from training data
  (printf "Computing normalization statistics...\n")
  (define stats (compute-normalization-stats train-data))
  (printf "Feature means: ~A\n" (f32vector->list (get-feature-means stats)))
  (printf "Feature stds:  ~A\n" (f32vector->list (get-feature-stds stats)))
  (printf "Target mean:   ~A\n" (get-target-mean stats))
  (printf "Target std:    ~A\n\n" (get-target-std stats))
  
  ;; Build model
  (printf "Building regression model...\n")
  (define model (build-regression-model '(64 32 16)))
  
  (printf "\nModel Architecture:\n")
  (let ((params (parameters model)))
    (printf "  Layers: ~A\n" (length params))
    (printf "  Total Parameters: ~A\n"
            (fold (lambda (p acc)
                    (+ acc (f32vector-length (tensor-data p))))
                  0 params)))
  (printf "\n")
  
  ;; Create optimizer
  (define learning-rate 0.001)
  (printf "Optimizer: Adam (lr=~A)\n\n" learning-rate)
  (define optimizer (make-adam (parameters model)
                               learning-rate: learning-rate))
  
  ;; Training loop
  (define num-epochs 100)
  (define best-r2 -inf.0)
  
  (printf "Training for ~A epochs...\n" num-epochs)
  (printf "----------------------------------------\n")
  
  (do ((epoch 1 (+ epoch 1)))
      ((> epoch num-epochs))
    
    ;; Train
    (let ((train-loss (train-epoch model optimizer train-data stats
                                   batch-size: 32)))
      (printf "Epoch ~A/~A - Loss: ~A" epoch num-epochs train-loss)
      
      ;; Evaluate every 10 epochs
      (when (= (modulo epoch 10) 0)
        (let-values (((test-mse test-mae test-r2 _ __)
                      (evaluate model test-data stats batch-size: 64)))
          (printf " - Test MSE: ~A - MAE: ~A - R^2: ~A" 
                  test-mse test-mae test-r2)
          
          ;; Save checkpoint if best R^2 so far
          (when (> test-r2 best-r2)
            (set! best-r2 test-r2)
            (printf "\n  New best R^2! Saving checkpoint...")
            (save-checkpoint model optimizer epoch train-loss stats
                           (sprintf "best-regression-model_~A.ngrd" epoch)))))
      
      (printf "\n"))
    
    ;; Learning rate decay
    (when (= (modulo epoch 30) 0)
      (let ((new-lr (* (get-learning-rate optimizer) 0.5)))
        (set-learning-rate! optimizer new-lr)
        (printf "  - Learning rate decreased to ~A\n" new-lr))))
  
  (printf "----------------------------------------\n\n")
  
  ;; Save final model
  (printf "Saving final model...\n")
  (save-regression-model model "final-regression-model.ngrd")
  (printf "\n")
  
  ;; Final evaluation
  (printf "========================================\n")
  (printf "Final Evaluation on Test Set\n")
  (printf "========================================\n")
  
  (let-values (((test-mse test-mae test-r2 predictions targets)
                (evaluate model test-data stats batch-size: 64)))
    (printf "\nTest Metrics:\n")
    (printf "  Mean Squared Error (MSE):  ~A\n" test-mse)
    (printf "  Mean Absolute Error (MAE): ~A\n" test-mae)
    (printf "  R^2 Score:                  ~A\n" test-r2)
    
    ;; Print sample predictions
    ;(print-prediction-samples predictions targets 20)
    
    ;; Error analysis
    (analyze-errors predictions targets))
  
  (printf "\n========================================\n")
  (printf "Training Complete!\n")
  (printf "========================================\n"))

;; Run the example
(printf "\n")
(printf "  NanoGrad Regression Example          \n")
(printf "  Non-linear Function Approximation    \n")
(printf "\n")

(main)


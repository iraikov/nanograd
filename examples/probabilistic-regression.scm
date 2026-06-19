;; probabilistic-regression.scm
;; Neural network with uncertainty estimation
;; Outputs both mean prediction and aleatoric uncertainty

(import scheme
        (chicken base)
        (chicken format)
        (chicken random)
        (chicken time)
        (chicken sort)
        (chicken file)
        (srfi 1)
        (srfi 4)
        (srfi 42)
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
  (set-pseudo-random-seed! (number->string seed)))

;;; ==================================================================
;;; Heteroscedastic Target Function
;;; ==================================================================
;;; We'll use a function where uncertainty varies with input
;;; This demonstrates the model learning input-dependent noise

(define num-features 1)  ; Single input for visualization

(define (target-function x)
  "True underlying function: y = x * sin(x)"
  (* x (sin x)))

(define (noise-level x)
  "Input-dependent noise level: more noise at extremes"
  (* 0.3 (+ 1.0 (* 0.5 (abs x)))))

(define (generate-sample)
  "Generate sample with heteroscedastic noise"
  (let* ((x (- (* (pseudo-random-real) 16.0) 8.0))  ; Range: [-8, 8]
         (y-true (target-function x))
         (sigma (noise-level x))
         ;; Box-Muller transform for Gaussian noise
         (u1 (pseudo-random-real))
         (u2 (pseudo-random-real))
         (z (* (sqrt (* -2.0 (log u1))) (cos (* 2.0 3.14159265359 u2))))
         (noise (* sigma z))
         (y (+ y-true noise)))
    
    (cons (list->f32vector (list x)) y)))

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
;;; Normalization
;;; ==================================================================

(define-record-type normalization-stats
  (make-normalization-stats feature-means feature-stds target-mean target-std)
  normalization-stats?
  (feature-means get-feature-means)
  (feature-stds get-feature-stds)
  (target-mean get-target-mean)
  (target-std get-target-std))

(define (compute-normalization-stats dataset)
  (let ((n (length dataset))
        (feature-sums (make-f32vector num-features 0.0))
        (feature-sq-sums (make-f32vector num-features 0.0))
        (target-sum 0.0)
        (target-sq-sum 0.0))
    
    (for-each
     (lambda (sample)
       (let ((features (car sample))
             (target (cdr sample)))
         (do ((i 0 (+ i 1)))
             ((= i num-features))
           (let ((val (f32vector-ref features i)))
             (f32vector-set! feature-sums i 
                            (+ (f32vector-ref feature-sums i) val))
             (f32vector-set! feature-sq-sums i
                            (+ (f32vector-ref feature-sq-sums i) (* val val)))))
         (set! target-sum (+ target-sum target))
         (set! target-sq-sum (+ target-sq-sum (* target target)))))
     dataset)
    
    (let ((feature-means (make-f32vector num-features))
          (target-mean (/ target-sum n)))
      (do ((i 0 (+ i 1)))
          ((= i num-features))
        (f32vector-set! feature-means i 
                       (/ (f32vector-ref feature-sums i) n)))
      
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
  (let ((features (car sample))
        (target (cdr sample))
        (normalized-features (make-f32vector num-features)))
    
    (do ((i 0 (+ i 1)))
        ((= i num-features))
      (f32vector-set! normalized-features i
                     (/ (- (f32vector-ref features i)
                           (f32vector-ref (get-feature-means stats) i))
                        (f32vector-ref (get-feature-stds stats) i))))
    
    (let ((normalized-target 
           (/ (- target (get-target-mean stats))
              (get-target-std stats))))
      (cons normalized-features normalized-target))))

(define (denormalize-prediction pred stats)
  (+ (* pred (get-target-std stats))
     (get-target-mean stats)))

(define (denormalize-variance var stats)
  "Denormalize variance (scales by std squared)"
  (* var (expt (get-target-std stats) 2)))

;;; ==================================================================
;;; Batch Construction
;;; ==================================================================

(define (stack-features batch)
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
  (let* ((batch-size (length batch))
         (target-data (make-f32vector batch-size 0.0)))
    (do ((i 0 (+ i 1)))
        ((= i batch-size))
      (f32vector-set! target-data i (cdr (list-ref batch i))))
    (make-tensor32 target-data (list batch-size))))

;;; ==================================================================
;;; Probabilistic Model Architecture
;;; ==================================================================
;;; Network outputs 2 values: mean (mu) and log-variance (log sigma^2)
;;; Using log-variance ensures variance is always positive after exp()

(define (build-probabilistic-model hidden-sizes)
  "Build network that outputs mean and log-variance
   
   Output layer has 2 neurons:
   - output[0]: predicted mean
   - output[1]: predicted log-variance (log sigma^2)
   
   We use log-variance because:
   1. Ensures positive variance after exp()
   2. Numerically stable gradients
   3. Common practice in probabilistic deep learning"
  
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
    
    ;; Output layer: 2 outputs (mean and log-variance)
    (set! layers
          (cons (make-dense-layer (last hidden-sizes) 2
                                 activation: (make-identity)
                                 name: "Output")
                layers))
    
    (make-sequential (reverse layers) name: "ProbabilisticModel")))

;;; ==================================================================
;;; Gaussian Negative Log-Likelihood Loss
;;; ==================================================================

(define (gaussian-nll-loss predictions targets #!key (reduction 'mean))
  "Gaussian Negative Log-Likelihood Loss
   
   For Gaussian distribution: p(y|x) = N(mu, sigma^2)
   NLL = -log p(y|x) = 0.5 * log(2 * pi * sigma^2) + (y - mu)^2 / (2 sigma^2)
   
   Simplified (dropping constants): NLL = 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2)
   
   Args:
     predictions: Tensor of shape (batch, 2) where:
                  predictions[:, 0] = mean (mu)
                  predictions[:, 1] = log-variance (log sigma^2)
     targets: Tensor of shape (batch,) with true values
     reduction: 'mean' or 'sum'
   
   Returns:
     Scalar loss tensor
   
   Note: We add a small constant to prevent log(0) and division by zero"
  
  (let* ((dtype (tensor-dtype predictions))
         (pred-shape (tensor-shape predictions))
         (batch-size (car pred-shape))
         (pred-data (tensor-data predictions))
         (target-data (tensor-data targets))
         (requires-grad? (tensor-requires-grad? predictions))
         (epsilon 1e-6))  ; For numerical stability
    
    ;; Check shapes
    (unless (and (= (length pred-shape) 2)
                 (= (cadr pred-shape) 2))
      (error 'gaussian-nll-loss 
             (format #f "Predictions must have shape (batch, 2), got ~A" pred-shape)))
    
    (unless (= (car (tensor-shape targets)) batch-size)
      (error 'gaussian-nll-loss "Batch size mismatch"))
    
    ;; Compute loss for each sample
    (let ((total-loss 0.0))
      (do ((i 0 (+ i 1)))
          ((= i batch-size))
        (let* ((mean (f32vector-ref pred-data (* i 2)))
               (log-var (f32vector-ref pred-data (+ (* i 2) 1)))
               (target (f32vector-ref target-data i))
               
               ;; Ensure log-var is not too negative (prevents numerical issues)
               (log-var-clamped (max log-var -10.0))
               (var (exp log-var-clamped))
               
               ;; NLL = 0.5 * (log(var) + (target - mean)^2 / var)
               (squared-error (* (- target mean) (- target mean)))
               (nll (+ (* 0.5 log-var-clamped)
                       (* 0.5 (/ squared-error (+ var epsilon))))))
          
          (set! total-loss (+ total-loss nll))))
      
      ;; Create loss tensor
      (let ((loss-value (case reduction
                         ((mean) (/ total-loss batch-size))
                         ((sum) total-loss)
                         (else (error 'gaussian-nll-loss 
                                     "reduction must be 'mean or 'sum"))))
            (loss-data (make-f32vector 1)))
        
        (f32vector-set! loss-data 0 loss-value)
        
        (let ((loss-tensor (make-tensor32 loss-data '(1) 
                                         requires-grad?: requires-grad?)))
          
          ;; Set up backward pass
          (when requires-grad?
            (set-backward-fn!
             loss-tensor
             (lambda ()
               (let ((grad-out (f32vector-ref (tensor-grad loss-tensor) 0))
                     (grad-pred (make-f32vector (* batch-size 2) 0.0)))
                 
                 ;; Compute gradients w.r.t. mean and log-var
                 (do ((i 0 (+ i 1)))
                     ((= i batch-size))
                   (let* ((mean (f32vector-ref pred-data (* i 2)))
                          (log-var (f32vector-ref pred-data (+ (* i 2) 1)))
                          (target (f32vector-ref target-data i))
                          
                          (log-var-clamped (max log-var -10.0))
                          (var (exp log-var-clamped))
                          
                          (squared-error (* (- target mean) (- target mean)))
                          
                          ;; Gradient w.r.t. mean: d(NLL)/d mu = -(y - mu) / sigma^2
                          (grad-mean (/ (- mean target) (+ var epsilon)))
                          
                          ;; Gradient w.r.t. log-var: d(NLL)/d(log sigma^2) 
                          ;;   = 0.5 - 0.5 * (y - mu)^2 / sigma^2
                          (grad-log-var (- 0.5 (* 0.5 (/ squared-error 
                                                         (+ var epsilon)))))
                          
                          ;; Scale by loss gradient and reduction
                          (scale (case reduction
                                  ((mean) (/ grad-out batch-size))
                                  ((sum) grad-out))))
                     
                     (f32vector-set! grad-pred (* i 2) 
                                    (* grad-mean scale))
                     (f32vector-set! grad-pred (+ (* i 2) 1)
                                    (* grad-log-var scale))))
                 
                 (add-to-grad! predictions grad-pred)))
             (list predictions)))
          
          loss-tensor)))))

;;; ==================================================================
;;; Training Functions
;;; ==================================================================

(define (train-epoch model optimizer train-data stats
                     #!key (batch-size 32))
  "Train for one epoch with Gaussian NLL loss"
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
    
    (for-each
     (lambda (batch)
       (let* ((actual-batch-size (length batch))
              
              ;; Normalize batch
              (normalized-batch (map (lambda (s) (normalize-sample s stats))
                                    batch))
              
              ;; Stack into tensors
              (batch-features (stack-features normalized-batch))
              (batch-targets (stack-targets normalized-batch))
              
              ;; Forward pass: output is (batch, 2) with [mean, log-var]
              (predictions (forward model batch-features))
              
              ;; Compute Gaussian NLL loss
              (loss (gaussian-nll-loss predictions batch-targets 
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
    
    (/ total-loss n)))

;;; ==================================================================
;;; Evaluation with Uncertainty Quantification
;;; ==================================================================

(define (evaluate-probabilistic model test-data stats 
                                #!key (batch-size 64))
  "Evaluate model, returning predictions with uncertainty estimates"
  (let ((all-means '())
        (all-vars '())
        (all-targets '())
        (batches (let loop ((remaining test-data)
                           (result '()))
                  (if (null? remaining)
                      (reverse result)
                      (let* ((batch-end (min batch-size (length remaining)))
                             (batch (take remaining batch-end))
                             (rest (drop remaining batch-end)))
                        (loop rest (cons batch result)))))))
    
    (for-each
     (lambda (batch)
       (let* ((normalized-batch (map (lambda (s) (normalize-sample s stats))
                                    batch))
              (batch-features (stack-features normalized-batch))
              (batch-features-no-grad 
               (make-tensor32 (tensor-data batch-features)
                             (tensor-shape batch-features)
                             requires-grad?: #f))
              
              ;; Forward pass: get [mean, log-var]
              (predictions (forward model batch-features-no-grad))
              (pred-data (tensor-data predictions)))
         
         ;; Extract means and variances, denormalize
         (do ((i 0 (+ i 1)))
             ((= i (length batch)))
           (let* ((normalized-mean (f32vector-ref pred-data (* i 2)))
                  (log-var (f32vector-ref pred-data (+ (* i 2) 1)))
                  (normalized-var (exp log-var))
                  (normalized-target (cdr (list-ref normalized-batch i)))
                  
                  ;; Denormalize back to original scale
                  (mean (denormalize-prediction normalized-mean stats))
                  (var (denormalize-variance normalized-var stats))
                  (target (denormalize-prediction normalized-target stats)))
             
             (set! all-means (cons mean all-means))
             (set! all-vars (cons var all-vars))
             (set! all-targets (cons target all-targets))))))
     batches)
    
    (values (list->f32vector (reverse all-means))
            (list->f32vector (reverse all-vars))
            (list->f32vector (reverse all-targets)))))

;;; ==================================================================
;;; Uncertainty Metrics
;;; ==================================================================

(define (compute-metrics means vars targets)
  "Compute various metrics including calibration"
  (let ((n (f32vector-length means))
        (mse 0.0)
        (mae 0.0)
        (nll 0.0)
        (in-1sigma 0)
        (in-2sigma 0))
    
    ;; Compute metrics
    (do ((i 0 (+ i 1)))
        ((= i n))
      (let* ((mean (f32vector-ref means i))
             (var (f32vector-ref vars i))
             (sigma (sqrt var))
             (target (f32vector-ref targets i))
             (error (- mean target))
             (abs-error (abs error)))
        
        ;; MSE and MAE
        (set! mse (+ mse (* error error)))
        (set! mae (+ mae abs-error))
        
        ;; Negative log-likelihood
        (set! nll (+ nll (+ (* 0.5 (log (* 2.0 3.14159265359 var)))
                           (/ (* error error) (* 2.0 var)))))
        
        ;; Calibration: count points within 1σ and 2σ
        (when (<= abs-error sigma)
          (set! in-1sigma (+ in-1sigma 1)))
        (when (<= abs-error (* 2.0 sigma))
          (set! in-2sigma (+ in-2sigma 1)))))
    
    (values (/ mse n)                    ; MSE
            (/ mae n)                    ; MAE
            (/ nll n)                    ; Average NLL
            (/ in-1sigma n)              ; Fraction in 1σ (should be ~68%)
            (/ in-2sigma n))))           ; Fraction in 2σ (should be ~95%)

;;; ==================================================================
;;; Visualization
;;; ==================================================================

(define (print-predictions-with-uncertainty means vars targets n-samples)
  "Print predictions with confidence intervals"
  (printf "\nPredictions with Uncertainty:\n")
  (printf "Sample  Mean  Std Dev   Actual  In 1sigma?\n")
  (printf "------------------------------------------\n")
  
  (let ((n (min n-samples (f32vector-length means))))
    (do ((i 0 (+ i 1)))
        ((= i n))
      (let* ((mean (f32vector-ref means i))
             (var (f32vector-ref vars i))
             (sigma (sqrt var))
             (actual (f32vector-ref targets i))
             (error (abs (- mean actual)))
             (in-1sigma? (<= error sigma)))
        (printf "~A ~A ~A ~A ~A\n" 
                (+ i 1) mean sigma actual 
                (if in-1sigma? "Yes" "No"))))))

(define (analyze-uncertainty-calibration means vars targets)
  "Analyze how well uncertainty estimates are calibrated"
  (printf "\nUncertainty Calibration Analysis:\n")
  (printf "========================================\n")
  
  (let-values (((mse mae nll in-1sigma in-2sigma)
                (compute-metrics means vars targets)))
    
    (printf "Prediction Quality:\n")
    (printf "  MSE: ~A\n" mse)
    (printf "  MAE: ~A\n" mae)
    (printf "  Avg NLL: ~A\n" nll)
    (printf "\n")
    
    (printf "Calibration (how well uncertainties match errors):\n")
    (printf "  Points within 1 sigma: ~A%% (expected: 68%%)\n" 
            (* 100.0 in-1sigma))
    (printf "  Points within 2 sigma: ~A%% (expected: 95%%)\n" 
            (* 100.0 in-2sigma))
    (printf "\n")
    
    ;; Interpretation
    (cond
     ((> in-1sigma 0.75)
      (printf "Interpretation: Model is over-confident (uncertainties too small)\n"))
     ((< in-1sigma 0.60)
      (printf "Interpretation: Model is under-confident (uncertainties too large)\n"))
     (else
      (printf "Interpretation: Model uncertainties are well-calibrated\n")))
    
    (printf "========================================\n")))

;;; ==================================================================
;;; Main Training Loop
;;; ==================================================================

(define (main)
  (printf "========================================\n")
  (printf "Probabilistic Neural Network Regression\n")
  (printf "With Uncertainty Quantification\n")
  (printf "========================================\n\n")
  
  (set-random-seed! 42)
  
  ;; Generate dataset
  (printf "Generating heteroscedastic dataset...\n")
  (printf "Target: y = x * sin(x) with input-dependent noise\n")
  (printf "Noise level: sigma(x) = 0.3 * (1 + 0.5|x|)\n\n")
  
  (define train-data (shuffle (generate-dataset 2000)))
  (define test-data (shuffle (generate-dataset 400)))
  (printf "Training samples: ~A\n" (length train-data))
  (printf "Test samples: ~A\n\n" (length test-data))
  
  ;; Compute normalization
  (printf "Computing normalization statistics...\n")
  (define stats (compute-normalization-stats train-data))
  (printf "Feature mean: ~A\n" (f32vector->list (get-feature-means stats)))
  (printf "Feature std:  ~A\n" (f32vector->list (get-feature-stds stats)))
  (printf "Target mean:  ~A\n" (get-target-mean stats))
  (printf "Target std:   ~A\n\n" (get-target-std stats))
  
  ;; Build model
  (printf "Building probabilistic model...\n")
  (printf "Architecture: Input(1) -> Hidden(128,64,32) -> Output(2)\n")
  (printf "Output: [mean, log-variance]\n\n")
  (define model (build-probabilistic-model '(128 64 32)))
  
  (let ((params (parameters model)))
    (printf "Total Parameters: ~A\n\n"
            (fold (lambda (p acc)
                    (+ acc (f32vector-length (tensor-data p))))
                  0 params)))
  
  ;; Create optimizer
  (define learning-rate 0.001)
  (printf "Optimizer: Adam (lr=~A)\n\n" learning-rate)
  (define optimizer (make-adam (parameters model)
                               learning-rate: learning-rate))
  
  ;; Training loop
  (define num-epochs 100)
  (printf "Training for ~A epochs...\n" num-epochs)
  (printf "----------------------------------------\n")
  
  (do ((epoch 1 (+ epoch 1)))
      ((> epoch num-epochs))
    
    (let ((train-loss (train-epoch model optimizer train-data stats
                                   batch-size: 32)))
      (printf "Epoch ~A/~A - NLL: ~A" epoch num-epochs train-loss)
      
      ;; Evaluate every 10 epochs
      (when (= (modulo epoch 10) 0)
        (let-values (((means vars targets)
                      (evaluate-probabilistic model test-data stats)))
          (let-values (((mse mae nll in-1sigma in-2sigma)
                        (compute-metrics means vars targets)))
            (printf " - Test MSE: ~A - 1 * sigma: ~A%%" 
                    mse (* 100.0 in-1sigma)))))
      
      (printf "\n"))
    
    ;; Learning rate decay
    (when (= (modulo epoch 30) 0)
      (let ((new-lr (* (get-learning-rate optimizer) 0.5)))
        (set-learning-rate! optimizer new-lr)
        (printf "  - Learning rate decreased to ~A\n" new-lr))))
  
  (printf "----------------------------------------\n\n")
  
  ;; Final evaluation
  (printf "========================================\n")
  (printf "Final Evaluation\n")
  (printf "========================================\n")
  
  (let-values (((means vars targets)
                (evaluate-probabilistic model test-data stats)))
    
    ;; Print metrics
    (analyze-uncertainty-calibration means vars targets)
    
    ;; Print sample predictions
    (print-predictions-with-uncertainty means vars targets 20)
    )
    
  
  (printf "\n========================================\n")
  (printf "Training Complete!\n")
  (printf "========================================\n"))

;; Run the example
(printf "\n")
(printf "  NanoGrad Probabilistic Regression    \n")
(printf "  Uncertainty Quantification            \n")
(printf "\n")


(main)

;; Iris dataset classification example
;; Demonstrates building, training, and evaluating a neural network for tabular data

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
        nanograd-optimizer
        nanograd-diagnostics)

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
;;; Iris Dataset
;;; ==================================================================

;; Iris dataset: 150 samples, 4 features, 3 classes
;; Features: sepal length, sepal width, petal length, petal width
;; Classes: 0=Setosa, 1=Versicolor, 2=Virginica

(define num-features 4)
(define num-classes 3)

(define iris-data
  '(;; Setosa (0-49)
    ((5.1 3.5 1.4 0.2) 0) ((4.9 3.0 1.4 0.2) 0) ((4.7 3.2 1.3 0.2) 0)
    ((4.6 3.1 1.5 0.2) 0) ((5.0 3.6 1.4 0.2) 0) ((5.4 3.9 1.7 0.4) 0)
    ((4.6 3.4 1.4 0.3) 0) ((5.0 3.4 1.5 0.2) 0) ((4.4 2.9 1.4 0.2) 0)
    ((4.9 3.1 1.5 0.1) 0) ((5.4 3.7 1.5 0.2) 0) ((4.8 3.4 1.6 0.2) 0)
    ((4.8 3.0 1.4 0.1) 0) ((4.3 3.0 1.1 0.1) 0) ((5.8 4.0 1.2 0.2) 0)
    ((5.7 4.4 1.5 0.4) 0) ((5.4 3.9 1.3 0.4) 0) ((5.1 3.5 1.4 0.3) 0)
    ((5.7 3.8 1.7 0.3) 0) ((5.1 3.8 1.5 0.3) 0) ((5.4 3.4 1.7 0.2) 0)
    ((5.1 3.7 1.5 0.4) 0) ((4.6 3.6 1.0 0.2) 0) ((5.1 3.3 1.7 0.5) 0)
    ((4.8 3.4 1.9 0.2) 0) ((5.0 3.0 1.6 0.2) 0) ((5.0 3.4 1.6 0.4) 0)
    ((5.2 3.5 1.5 0.2) 0) ((5.2 3.4 1.4 0.2) 0) ((4.7 3.2 1.6 0.2) 0)
    ((4.8 3.1 1.6 0.2) 0) ((5.4 3.4 1.5 0.4) 0) ((5.2 4.1 1.5 0.1) 0)
    ((5.5 4.2 1.4 0.2) 0) ((4.9 3.1 1.5 0.2) 0) ((5.0 3.2 1.2 0.2) 0)
    ((5.5 3.5 1.3 0.2) 0) ((4.9 3.6 1.4 0.1) 0) ((4.4 3.0 1.3 0.2) 0)
    ((5.1 3.4 1.5 0.2) 0) ((5.0 3.5 1.3 0.3) 0) ((4.5 2.3 1.3 0.3) 0)
    ((4.4 3.2 1.3 0.2) 0) ((5.0 3.5 1.6 0.6) 0) ((5.1 3.8 1.9 0.4) 0)
    ((4.8 3.0 1.4 0.3) 0) ((5.1 3.8 1.6 0.2) 0) ((4.6 3.2 1.4 0.2) 0)
    ((5.3 3.7 1.5 0.2) 0) ((5.0 3.3 1.4 0.2) 0)
    
    ;; Versicolor (50-99)
    ((7.0 3.2 4.7 1.4) 1) ((6.4 3.2 4.5 1.5) 1) ((6.9 3.1 4.9 1.5) 1)
    ((5.5 2.3 4.0 1.3) 1) ((6.5 2.8 4.6 1.5) 1) ((5.7 2.8 4.5 1.3) 1)
    ((6.3 3.3 4.7 1.6) 1) ((4.9 2.4 3.3 1.0) 1) ((6.6 2.9 4.6 1.3) 1)
    ((5.2 2.7 3.9 1.4) 1) ((5.0 2.0 3.5 1.0) 1) ((5.9 3.0 4.2 1.5) 1)
    ((6.0 2.2 4.0 1.0) 1) ((6.1 2.9 4.7 1.4) 1) ((5.6 2.9 3.6 1.3) 1)
    ((6.7 3.1 4.4 1.4) 1) ((5.6 3.0 4.5 1.5) 1) ((5.8 2.7 4.1 1.0) 1)
    ((6.2 2.2 4.5 1.5) 1) ((5.6 2.5 3.9 1.1) 1) ((5.9 3.2 4.8 1.8) 1)
    ((6.1 2.8 4.0 1.3) 1) ((6.3 2.5 4.9 1.5) 1) ((6.1 2.8 4.7 1.2) 1)
    ((6.4 2.9 4.3 1.3) 1) ((6.6 3.0 4.4 1.4) 1) ((6.8 2.8 4.8 1.4) 1)
    ((6.7 3.0 5.0 1.7) 1) ((6.0 2.9 4.5 1.5) 1) ((5.7 2.6 3.5 1.0) 1)
    ((5.5 2.4 3.8 1.1) 1) ((5.5 2.4 3.7 1.0) 1) ((5.8 2.7 3.9 1.2) 1)
    ((6.0 2.7 5.1 1.6) 1) ((5.4 3.0 4.5 1.5) 1) ((6.0 3.4 4.5 1.6) 1)
    ((6.7 3.1 4.7 1.5) 1) ((6.3 2.3 4.4 1.3) 1) ((5.6 3.0 4.1 1.3) 1)
    ((5.5 2.5 4.0 1.3) 1) ((5.5 2.6 4.4 1.2) 1) ((6.1 3.0 4.6 1.4) 1)
    ((5.8 2.6 4.0 1.2) 1) ((5.0 2.3 3.3 1.0) 1) ((5.6 2.7 4.2 1.3) 1)
    ((5.7 3.0 4.2 1.2) 1) ((5.7 2.9 4.2 1.3) 1) ((6.2 2.9 4.3 1.3) 1)
    ((5.1 2.5 3.0 1.1) 1) ((5.7 2.8 4.1 1.3) 1)
    
    ;; Virginica (100-149)
    ((6.3 3.3 6.0 2.5) 2) ((5.8 2.7 5.1 1.9) 2) ((7.1 3.0 5.9 2.1) 2)
    ((6.3 2.9 5.6 1.8) 2) ((6.5 3.0 5.8 2.2) 2) ((7.6 3.0 6.6 2.1) 2)
    ((4.9 2.5 4.5 1.7) 2) ((7.3 2.9 6.3 1.8) 2) ((6.7 2.5 5.8 1.8) 2)
    ((7.2 3.6 6.1 2.5) 2) ((6.5 3.2 5.1 2.0) 2) ((6.4 2.7 5.3 1.9) 2)
    ((6.8 3.0 5.5 2.1) 2) ((5.7 2.5 5.0 2.0) 2) ((5.8 2.8 5.1 2.4) 2)
    ((6.4 3.2 5.3 2.3) 2) ((6.5 3.0 5.5 1.8) 2) ((7.7 3.8 6.7 2.2) 2)
    ((7.7 2.6 6.9 2.3) 2) ((6.0 2.2 5.0 1.5) 2) ((6.9 3.2 5.7 2.3) 2)
    ((5.6 2.8 4.9 2.0) 2) ((7.7 2.8 6.7 2.0) 2) ((6.3 2.7 4.9 1.8) 2)
    ((6.7 3.3 5.7 2.1) 2) ((7.2 3.2 6.0 1.8) 2) ((6.2 2.8 4.8 1.8) 2)
    ((6.1 3.0 4.9 1.8) 2) ((6.4 2.8 5.6 2.1) 2) ((7.2 3.0 5.8 1.6) 2)
    ((7.4 2.8 6.1 1.9) 2) ((7.9 3.8 6.4 2.0) 2) ((6.4 2.8 5.6 2.2) 2)
    ((6.3 2.8 5.1 1.5) 2) ((6.1 2.6 5.6 1.4) 2) ((7.7 3.0 6.1 2.3) 2)
    ((6.3 3.4 5.6 2.4) 2) ((6.4 3.1 5.5 1.8) 2) ((6.0 3.0 4.8 1.8) 2)
    ((6.9 3.1 5.4 2.1) 2) ((6.7 3.1 5.6 2.4) 2) ((6.9 3.1 5.1 2.3) 2)
    ((5.8 2.7 5.1 1.9) 2) ((6.8 3.2 5.9 2.3) 2) ((6.7 3.3 5.7 2.5) 2)
    ((6.7 3.0 5.2 2.3) 2) ((6.3 2.5 5.0 1.9) 2) ((6.5 3.0 5.2 2.0) 2)
    ((6.2 3.4 5.4 2.3) 2) ((5.9 3.0 5.1 1.8) 2)))

(define class-names '#("Setosa" "Versicolor" "Virginica"))

;;; ==================================================================
;;; Data Preprocessing
;;; ==================================================================

(define (normalize-features data)
  "Standardize features to zero mean and unit variance"
  (let* ((n (length data))
         ;; Compute means
         (means (make-f32vector num-features 0.0))
         ;; Compute standard deviations
         (stds (make-f32vector num-features 0.0)))
    
    ;; Calculate means
    (for-each
     (lambda (sample)
       (let ((features (car sample)))
         (do ((i 0 (+ i 1)))
             ((= i num-features))
           (let ((val (list-ref features i)))
             (f32vector-set! means i 
                           (+ (f32vector-ref means i) val))))))
     data)
    
    (do ((i 0 (+ i 1)))
        ((= i num-features))
      (f32vector-set! means i (/ (f32vector-ref means i) n)))
    
    ;; Calculate standard deviations
    (for-each
     (lambda (sample)
       (let ((features (car sample)))
         (do ((i 0 (+ i 1)))
             ((= i num-features))
           (let* ((val (list-ref features i))
                  (diff (- val (f32vector-ref means i))))
             (f32vector-set! stds i 
                           (+ (f32vector-ref stds i) (* diff diff)))))))
     data)
    
    (do ((i 0 (+ i 1)))
        ((= i num-features))
      (let ((std (sqrt (/ (f32vector-ref stds i) n))))
        (f32vector-set! stds i (if (< std 1e-8) 1.0 std))))
    
    ;; Normalize data
    (let ((normalized-data
           (map
            (lambda (sample)
              (let* ((features (car sample))
                     (label (cadr sample))
                     (norm-features
                      (list-tabulate
                       num-features
                       (lambda (i)
                         (/ (- (list-ref features i) 
                              (f32vector-ref means i))
                            (f32vector-ref stds i))))))
                (list norm-features label)))
            data)))
      (values normalized-data means stds))))

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

(define (split-train-test data test-ratio)
  "Split data into training and test sets"
  (let* ((n (length data))
         (n-test (inexact->exact (floor (* n test-ratio))))
         (n-train (- n n-test))
         (shuffled (shuffle data)))
    (values (take shuffled n-train)
            (drop shuffled n-train))))

;;; ==================================================================
;;; Batch Construction
;;; ==================================================================

(define (features->f32vector features)
  "Convert feature list to f32vector"
  (list->f32vector (map exact->inexact features)))

(define (stack-features batch)
  "Stack a batch of feature vectors into a 2D tensor (N, features)"
  (let* ((batch-size (length batch))
         (batched-data (make-f32vector (* batch-size num-features) 0.0)))
    
    (do ((i 0 (+ i 1)))
        ((= i batch-size))
      (let* ((sample (list-ref batch i))
             (features (car sample))
             (offset (* i num-features)))
        (do ((j 0 (+ j 1)))
            ((= j num-features))
          (f32vector-set! batched-data (+ offset j)
                         (exact->inexact (list-ref features j))))))
    
    (make-tensor32 batched-data 
                   (list batch-size num-features)
                   requires-grad?: #f)))

(define (stack-labels batch)
  "Stack batch labels into 1D tensor (N,) for class indices"
  (let* ((batch-size (length batch))
         (label-data (make-f32vector batch-size 0.0)))
    
    (do ((i 0 (+ i 1)))
        ((= i batch-size))
      (let ((label (cadr (list-ref batch i))))
        (f32vector-set! label-data i (exact->inexact label))))
    
    (make-tensor32 label-data (list batch-size))))

;;; ==================================================================
;;; MLP Architecture
;;; ==================================================================

(define (build-mlp)
  "Build a simple MLP for Iris classification
   Architecture:
   - Dense: 4 -> 16
   - ReLU
   - Dense: 16 -> 8
   - ReLU
   - Dense: 8 -> 3 (num classes)"
  
  (let* ((layers-list
          (list
           (make-dense-layer num-features 16
                            activation: (make-relu)
                            name: "Hidden1")
           
           (make-dense-layer 16 8
                            activation: (make-relu)
                            name: "Hidden2")
           
           (make-dense-layer 8 num-classes
                            activation: (make-identity)
                            name: "Output")))
         
         (model (make-sequential layers-list name: "IrisClassifier")))
    
    (list model layers-list)))

;;; ==================================================================
;;; Helper Functions
;;; ==================================================================

(define (one-hot class num-classes)
  "Convert class index to one-hot vector"
  (let ((vec (make-f32vector num-classes 0.0)))
    (f32vector-set! vec class 1.0)
    vec))

(define (argmax vec)
  "Return index of maximum value"
  (let loop ((i 1) (max-i 0) (max-val (f32vector-ref vec 0)))
    (if (= i (f32vector-length vec))
        max-i
        (let ((val (f32vector-ref vec i)))
          (if (> val max-val)
              (loop (+ i 1) i val)
              (loop (+ i 1) max-i max-val))))))

(define (argmax-offset vec offset length)
  "Find argmax in a slice of a vector starting at offset"
  (let loop ((i 1) (max-i 0) (max-val (f32vector-ref vec offset)))
    (if (= i length)
        max-i
        (let ((val (f32vector-ref vec (+ offset i))))
          (if (> val max-val)
              (loop (+ i 1) i val)
              (loop (+ i 1) max-i max-val))))))

;;; ==================================================================
;;; Training Functions
;;; ==================================================================

(define (train-epoch-batched model optimizer train-data 
                             #!key (batch-size 16) (gradient-diagnostics #f))
  "Train for one epoch with batched processing"
  (let* ((total-loss 0.0)
         (correct 0)
         (n (length train-data))
         (model-layer (car model))
         (params (parameters model-layer))
         (monitor (make-gradient-monitor 
                   exploding-threshold: 10.0
                   vanishing-threshold: 1e-7)))
    
    ;; Split data into mini-batches
    (let ((batches (let loop ((remaining train-data)
                             (i 0)
                             (result '()))
                    (if (null? remaining)
                        (reverse result)
                        (let* ((batch-end (min batch-size (length remaining)))
                               (batch (take remaining batch-end))
                               (rest (drop remaining batch-end)))
                          (loop rest (+ i 1) (cons (cons i batch) result)))))))
      
      ;; Process each mini-batch
      (for-each
       (lambda (step+batch)
         (let* ((step (car step+batch))
                (batch (cdr step+batch))
                (actual-batch-size (length batch))
                
                ;; Stack batch
                (batch-features (stack-features batch))
                (batch-labels (stack-labels batch))
                
                ;; Forward pass
                (logits (forward model-layer batch-features))
                (probs (softmax logits axis: -1))
                (loss (cross-entropy-loss probs batch-labels 
                                         reduction: 'mean
                                         from-logits: #f)))
           
           ;; Accumulate loss
           (set! total-loss (+ total-loss 
                              (* (f32vector-ref (tensor-data loss) 0)
                                 actual-batch-size)))
           
           ;; Count correct predictions
           (let ((logits-data (tensor-data logits)))
             (do ((i 0 (+ i 1)))
                 ((= i actual-batch-size))
               (let* ((offset (* i num-classes))
                      (pred-class (argmax-offset logits-data offset num-classes))
                      (true-class (inexact->exact 
                                  (f32vector-ref (tensor-data batch-labels) i))))
                 (when (= pred-class true-class)
                   (set! correct (+ correct 1))))))
           
           ;; Backward pass
           (backward! loss)
           
           ;; Check gradient health
           (if gradient-diagnostics
               (record-step! monitor step params))
           
           ;; Optimizer step
           (step! optimizer)
           
           ;; Zero gradients
           (zero-grad-layer! model-layer)))
       batches))
    
    (if gradient-diagnostics
        (begin
          (printf "\nTraining Diagnostics:\n")
          (let ((diagnosis (diagnose-training monitor)))
            (printf "Total steps: ~A\n" (cdr (assoc 'total-steps diagnosis)))
            (printf "Mean gradient norm: ~A\n" (cdr (assoc 'mean-gradient-norm diagnosis)))
            (printf "Unhealthy steps: ~A\n" (cdr (assoc 'unhealthy-steps diagnosis))))))
    
    (values (/ total-loss n) (/ correct n))))

;;; ==================================================================
;;; Evaluation
;;; ==================================================================

(define (evaluate-batched model test-data #!key (batch-size 32))
  "Evaluate model on test data using batched forward passes"
  (let ((correct 0)
        (total (length test-data))
        (confusion (make-vector (* num-classes num-classes) 0))
        (model-layer (car model)))
    
    ;; Split test data into batches
    (let ((batches (let loop ((remaining test-data)
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
                
                ;; Stack batch
                (batch-features (stack-features batch))
                (batch-labels (stack-labels batch))
                
                ;; Disable gradients
                (batch-features (make-tensor32 (tensor-data batch-features)
                                              (tensor-shape batch-features)
                                              requires-grad?: #f))
                
                ;; Forward pass
                (logits (forward model-layer batch-features))
                (logits-data (tensor-data logits)))
           
           ;; Process predictions
           (do ((i 0 (+ i 1)))
               ((= i actual-batch-size))
             (let* ((offset (* i num-classes))
                    (pred-class (argmax-offset logits-data offset num-classes))
                    (true-class (inexact->exact 
                                (f32vector-ref (tensor-data batch-labels) i))))
               
               (when (= pred-class true-class)
                 (set! correct (+ correct 1)))
               
               ;; Update confusion matrix
               (let ((idx (+ (* true-class num-classes) pred-class)))
                 (vector-set! confusion idx 
                            (+ 1 (vector-ref confusion idx))))))))
       batches))
    
    (values (/ correct total) confusion)))

(define (print-confusion-matrix confusion)
  "Pretty print confusion matrix"
  (printf "\nConfusion Matrix:\n")
  (printf "           ")
  (do ((i 0 (+ i 1)))
      ((= i num-classes))
    (printf "~A  " (vector-ref class-names i)))
  (printf "\n")
  
  (do ((true-class 0 (+ true-class 1)))
      ((= true-class num-classes))
    (printf "~A  " (vector-ref class-names true-class))
    (do ((pred-class 0 (+ pred-class 1)))
        ((= pred-class num-classes))
      (let ((idx (+ (* true-class num-classes) pred-class)))
        (printf "~A~A  " 
                (make-string (max 0 (- 11 (string-length 
                                          (vector-ref class-names pred-class)))) 
                           #\space)
                (vector-ref confusion idx))))
    (printf "\n")))

;;; ==================================================================
;;; Model Persistence
;;; ==================================================================

(define (save-mlp-model model filepath)
  "Save MLP model to file"
  (let ((model-layer (car model)))
    (printf "Saving model to ~A...\n" filepath)
    (save-model model-layer filepath)
    (printf "Model saved successfully!\n")))

(define (load-mlp-model filepath)
  "Load MLP model from file"
  (printf "Loading model from ~A...\n" filepath)
  (let* ((model-layer (load-model filepath))
         (ser (layer->serializable model-layer))
         (layer-type (assq 'type ser))
         (layer-list (if (eq? (and layer-type (cdr layer-type)) 'sequential)
                         (assq 'layers ser) '(layers)))
         (layer-objects (map serializable->layer (cdr layer-list))))
    (printf "Model loaded successfully!\n")
    (list model-layer layer-objects)))

(define (save-checkpoint model optimizer epoch train-loss train-acc filepath)
  "Save training checkpoint"
  (printf "\nSaving checkpoint at epoch ~A...\n" epoch)
  (save-mlp-model model filepath)
  
  (let ((metadata-file (string-append filepath ".meta")))
    (with-output-to-file metadata-file
      (lambda ()
        (printf "epoch: ~A\n" epoch)
        (printf "train-loss: ~A\n" train-loss)
        (printf "train-acc: ~A\n" train-acc)
        (printf "timestamp: ~A\n" (current-seconds)))))
  
  (printf "Checkpoint saved!\n"))

;;; ==================================================================
;;; Main Training Loop
;;; ==================================================================

(define (main)
  (printf "========================================\n")
  (printf "Iris Dataset Classification Example\n")
  (printf "========================================\n\n")
  
  ;; Set random seed
  (set-random-seed! 42)
  
  ;; Load and preprocess data
  (printf "Loading Iris dataset...\n")
  (printf "Total samples: ~A\n" (length iris-data))
  
  ;; Normalize features
  (let-values (((normalized-data means stds) (normalize-features iris-data)))
    
    (printf "\nFeature normalization:\n")
    (printf "  Means: ")
    (do ((i 0 (+ i 1)))
        ((= i num-features))
      (printf "~A " (f32vector-ref means i)))
    (printf "\n  Stds:  ")
    (do ((i 0 (+ i 1)))
        ((= i num-features))
      (printf "~A " (f32vector-ref stds i)))
    (printf "\n\n")
    
    ;; Split into train/test
    (let-values (((train-data test-data) 
                  (split-train-test normalized-data 0.2)))
      
      (printf "Training samples: ~A\n" (length train-data))
      (printf "Test samples: ~A\n\n" (length test-data))
      
      ;; Class distribution
      (printf "Training set class distribution:\n")
      (let ((counts (make-vector num-classes 0)))
        (for-each
         (lambda (sample)
           (let ((label (cadr sample)))
             (vector-set! counts label (+ 1 (vector-ref counts label)))))
         train-data)
        (do ((i 0 (+ i 1)))
            ((= i num-classes))
          (printf "  ~A: ~A samples\n" 
                  (vector-ref class-names i) 
                  (vector-ref counts i))))
      (printf "\n")
      
      ;; Build model
      (printf "Building MLP model...\n")
      (let* ((model (build-mlp))
             (model-layer (car model))
             (layers (cadr model)))
        
        (printf "\nModel Architecture:\n")
        (for-each
         (lambda (layer)
           (let ((params (parameters layer)))
             (printf "  ~A: ~A parameters\n"
                     (layer-name layer)
                     (fold (lambda (p acc)
                             (+ acc (apply * (tensor-shape p))))
                           0 params))))
         layers)
        
        (let ((total-params
               (fold (lambda (p acc) 
                       (+ acc (apply * (tensor-shape p))))
                     0 (parameters model-layer))))
          (printf "  Total Parameters: ~A\n\n" total-params))
        
        ;; Create optimizer
        (let ((learning-rate 0.01))
          (printf "Optimizer: Adam (lr=~A)\n\n" learning-rate)
          (let ((optimizer (make-adam (parameters model-layer)
                                      learning-rate: learning-rate
                                      weight-decay: 0.0001)))
            
            ;; Training loop
            (let ((num-epochs 150)
                  (best-acc 0.0))
              
              (printf "Training for ~A epochs...\n" num-epochs)
              (printf "----------------------------------------\n")
              
              (do ((epoch 1 (+ epoch 1)))
                  ((> epoch num-epochs))
                
                ;; Train
                (let-values (((avg-loss accuracy)
                              (train-epoch-batched model optimizer train-data
                                                  batch-size: 16
                                                  gradient-diagnostics: (= epoch 1))))
                  (printf "Epoch ~A/~A - Loss: ~A - Acc: ~A%"
                          epoch num-epochs avg-loss (* 100.0 accuracy))
                  
                  ;; Evaluate every 10 epochs
                  (when (= (modulo epoch 10) 0)
                    (let-values (((test-acc confusion) 
                                  (evaluate-batched model test-data batch-size: 32)))
                      (printf " - Test Acc: ~A%" (* 100.0 test-acc))
                      ;; Save best model
                      (when (> test-acc best-acc)
                        (set! best-acc test-acc)
                        (printf "\n  New best accuracy! Saving checkpoint...")
                        (save-checkpoint model optimizer epoch avg-loss accuracy
                                       (sprintf "best-iris-model_~A.ngrd" epoch)))))
                  
                  (printf "\n")))
              
              (printf "----------------------------------------\n\n")
              
              ;; Save final model
              (printf "Saving final model...\n")
              (save-mlp-model model "final-iris-model.ngrd")
              (printf "\n")
              
              ;; Final evaluation
              (printf "Final Evaluation on Test Set:\n")
              (let-values (((test-acc confusion) 
                            (evaluate-batched model test-data batch-size: 32)))
                (printf "Test Accuracy: ~A%\n" (* 100.0 test-acc))
                (print-confusion-matrix confusion))
              
              (printf "\n")
              
              ;; Per-class accuracy
              (printf "\nPer-Class Accuracy:\n")
              (let-values (((test-acc confusion) 
                            (evaluate-batched model test-data)))
                (do ((class 0 (+ class 1)))
                    ((= class num-classes))
                  (let ((total 0)
                        (correct 0))
                    (do ((pred 0 (+ pred 1)))
                        ((= pred num-classes))
                      (let ((idx (+ (* class num-classes) pred)))
                        (set! total (+ total (vector-ref confusion idx)))
                        (when (= class pred)
                          (set! correct (vector-ref confusion idx)))))
                    (printf "  ~A: ~A% (~A/~A)\n" 
                            (vector-ref class-names class)
                            (* 100 (/ correct total))
                            correct
                            total))))
              
              (printf "\n")
              
              ;; Sample predictions
              (printf "Sample Predictions:\n")
              (let ((n-samples (min 10 (length test-data))))
                (do ((i 0 (+ i 1)))
                    ((= i n-samples))
                  (let* ((sample (list-ref test-data i))
                         (features (car sample))
                         (true-class (cadr sample))
                         (feat-tensor (make-tensor32 
                                       (features->f32vector features)
                                       (list num-features)
                                       requires-grad?: #f))
                         (logits (forward model-layer feat-tensor))
                         (probs (softmax logits))
                         (pred-data (tensor-data probs))
                         (pred-class (argmax pred-data)))
                    
                    (printf "  Sample ~A: True=~A, Pred=~A " 
                            (+ i 1) 
                            (vector-ref class-names true-class)
                            (vector-ref class-names pred-class))
                    (if (= pred-class true-class)
                        (printf "O")
                        (printf "X"))
                    (printf " (confidence: ~A%)\n" 
                            (* 100 (f32vector-ref pred-data pred-class))))))
              
              (printf "\n========================================\n")
              (printf "Training Complete!\n")
              (printf "========================================\n"))))))))

;; Run the example
(printf "\n")
(printf "  NanoGrad Iris Classification Example  \n")
(printf "  Multi-Layer Perceptron for Tabular Data\n")
(printf "\n")

(main)

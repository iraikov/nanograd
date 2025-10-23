;; Convolutional neural network example
;; Demonstrates building, training, and evaluating a CNN for image classification

(import scheme
        (chicken base)
        (chicken format)
        (chicken random)
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
  (set-pseudo-random-seed! (number->string seed))
  )

;;; ==================================================================
;;; Data Generation: Synthetic Image Dataset
;;; ==================================================================

;; Generate synthetic 28x28 grayscale images with simple patterns
;; Class 0: Vertical lines
;; Class 1: Horizontal lines
;; Class 2: Diagonal lines (top-left to bottom-right)
;; Class 3: Diagonal lines (top-right to bottom-left)

(define image-size 28)
(define num-channels 1)
(define num-classes 4)

(define (make-blank-image)
  (make-f32vector (* num-channels image-size image-size) 0.0))

(define (add-noise! img noise-level)
  "Add random noise to image"
  (let ((n (f32vector-length img)))
    (do ((i 0 (+ i 1)))
        ((= i n))
      (f32vector-set! img i
                      (+ (f32vector-ref img i)
                         (* noise-level (- (pseudo-random-real) 0.5)))))))

(define (generate-vertical-lines)
  "Generate image with vertical lines (Class 0)"
  (let ((img (make-blank-image)))
    ;; Add 3-5 vertical lines
    (let ((num-lines (+ 3 (pseudo-random-integer 3))))
      (do ((line 0 (+ line 1)))
          ((= line num-lines))
        (let ((x (pseudo-random-integer image-size)))
          ;; Draw vertical line with some width
          (do ((y 0 (+ y 1)))
              ((= y image-size))
            (do ((dx -1 (+ dx 1)))
                ((> dx 1))
              (let ((xx (+ x dx)))
                (when (and (>= xx 0) (< xx image-size))
                  (let ((idx (+ (* y image-size) xx)))
                    (f32vector-set! img idx 1.0)))))))))
    (add-noise! img 0.1)
    img))

(define (generate-horizontal-lines)
  "Generate image with horizontal lines (Class 1)"
  (let ((img (make-blank-image)))
    (let ((num-lines (+ 3 (pseudo-random-integer 3))))
      (do ((line 0 (+ line 1)))
          ((= line num-lines))
        (let ((y (pseudo-random-integer image-size)))
          (do ((x 0 (+ x 1)))
              ((= x image-size))
            (do ((dy -1 (+ dy 1)))
                ((> dy 1))
              (let ((yy (+ y dy)))
                (when (and (>= yy 0) (< yy image-size))
                  (let ((idx (+ (* yy image-size) x)))
                    (f32vector-set! img idx 1.0)))))))))
    (add-noise! img 0.1)
    img))

(define (generate-diagonal-lr)
  "Generate image with diagonal lines top-left to bottom-right (Class 2)"
  (let ((img (make-blank-image)))
    (let ((num-lines (+ 2 (pseudo-random-integer 2))))
      (do ((line 0 (+ line 1)))
          ((= line num-lines))
        (let ((offset (- (pseudo-random-integer (* 2 image-size)) image-size)))
          (do ((i 0 (+ i 1)))
              ((= i image-size))
            (let ((x i)
                  (y (+ i offset)))
              (when (and (>= y 0) (< y image-size))
                (do ((d -1 (+ d 1)))
                    ((> d 1))
                  (let ((xx (+ x d))
                        (yy (+ y d)))
                    (when (and (>= xx 0) (< xx image-size)
                              (>= yy 0) (< yy image-size))
                      (let ((idx (+ (* yy image-size) xx)))
                        (f32vector-set! img idx 1.0)))))))))))
    (add-noise! img 0.1)
    img))

(define (generate-diagonal-rl)
  "Generate image with diagonal lines top-right to bottom-left (Class 3)"
  (let ((img (make-blank-image)))
    (let ((num-lines (+ 2 (pseudo-random-integer 2))))
      (do ((line 0 (+ line 1)))
          ((= line num-lines))
        (let ((offset (pseudo-random-integer (* 2 image-size))))
          (do ((i 0 (+ i 1)))
              ((= i image-size))
            (let ((x (- image-size 1 i))
                  (y (- (+ i offset) image-size)))
              (when (and (>= y 0) (< y image-size))
                (do ((d -1 (+ d 1)))
                    ((> d 1))
                  (let ((xx (+ x d))
                        (yy (+ y d)))
                    (when (and (>= xx 0) (< xx image-size)
                              (>= yy 0) (< yy image-size))
                      (let ((idx (+ (* yy image-size) xx)))
                        (f32vector-set! img idx 1.0)))))))))))
    (add-noise! img 0.1)
    img))

(define (generate-sample class)
  "Generate a sample image for given class"
  (case class
    ((0) (generate-vertical-lines))
    ((1) (generate-horizontal-lines))
    ((2) (generate-diagonal-lr))
    ((3) (generate-diagonal-rl))
    (else (error "Invalid class"))))

(define (generate-dataset n-per-class)
  "Generate balanced dataset with n samples per class"
  (let ((dataset '()))
    (do ((class 0 (+ class 1)))
        ((= class num-classes) (reverse (shuffle dataset)))
      (do ((i 0 (+ i 1)))
          ((= i n-per-class))
        (let ((img (generate-sample class)))
          (set! dataset (cons (cons img class) dataset)))))
    dataset))

(define (shuffle lst)
  "Fisher-Yates shuffle"
  (let* ((vec (list->vector lst))
         (n (vector-length vec)))
    (do ((i (- n 1) (- i 1)))
        ((< i 1) (vector->list vec))
      (let* ((j (pseudo-random-integer (+ i 1)))
             (tmp (vector-ref vec i)))
        (vector-set! vec i (vector-ref vec j))
        (vector-set! vec j tmp)))
    ))

(define (fill-ones! vec dtype)
  (let ((n (vector-length-for-dtype vec dtype)))
    (case dtype
      ((f32) (do ((i 0 (+ i 1)))
                 ((= i n))
               (f32vector-set! vec i 1.0)))
      ((f64) (do ((i 0 (+ i 1)))
                 ((= i n))
               (f64vector-set! vec i 1.0))))))

;;; ==================================================================
;;; CNN Architecture
;;; ==================================================================

(define (build-cnn)
  "Build a simple CNN for 28x28 grayscale image classification
   Architecture:
   - Conv2D: 1->16 channels, 3x3 kernel, stride=1, padding=1
   - ReLU
   - Conv2D: 16->32 channels, 3x3 kernel, stride=2, padding=1  (14x14)
   - ReLU
   - Conv2D: 32->64 channels, 3x3 kernel, stride=2, padding=1  (7x7)
   - ReLU
   - Flatten: 64*7*7 = 3136
   - Dense: 3136 -> 128
   - ReLU
   - Dense: 128 -> 4 (num classes)"
  ;; Create the conv and dense parts separately since we need to
  ;; explicitly flatten between them
  (let* (
         (conv-layers-list
          (list
           (make-conv2d-layer 1 16 3 
                              stride: 1 
                              padding: 1
                              activation: (make-relu)
                              name: "Conv1")
           
           (make-conv2d-layer 16 32 3
                              stride: 2
                              padding: 1
                              activation: (make-relu)
                              name: "Conv2")
           
           (make-conv2d-layer 32 64 3
                              stride: 2
                              padding: 1
                              activation: (make-relu)
                              name: "Conv3")))
          
        (conv-layers
         (make-sequential
          conv-layers-list
          name: "ConvLayers"))

        (dense-layers
         (make-sequential
          (list
           (make-dense-layer (* 64 7 7) 128
                             activation: (make-relu)
                             name: "FC1")
           
           (make-dense-layer 128 num-classes
                             activation: (make-identity)
                             name: "Output"))
          name: "DenseLayers"))
        )
    (list conv-layers dense-layers
          conv-layers-list
          )
  ))

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

(define (flatten-tensor tensor)
  "Flatten a multi-dimensional tensor to 1D"
  (let* ((shape (tensor-shape tensor))
         (total-size (apply * shape)))
    (reshape tensor (list total-size))))

(define (forward-cnn model x)
  "Forward pass through CNN with explicit flattening"
  ;; Forward through convolutional layers
  (let ((conv-layers (car model))
        (dense-layers (cadr model)))
    (let* (
           (conv-out (forward conv-layers x))
           (flat (flatten-tensor conv-out))
           (logits (forward dense-layers flat))
           )
      logits)))

;;; ==================================================================
;;; Training Functions
;;; ==================================================================

(define (train-epoch model optimizer train-data #!key (batch-size 32))
  "Train for one epoch with mini-batch gradient accumulation"
  (let ((total-loss 0.0)
        (correct 0)
        (n (length train-data))
        (conv-layers (car model))
        (dense-layers (cadr model))
        (conv-layers-internal (caddr model)))
    
    ;; Split data into mini-batches
    (let ((batches (let loop ((remaining train-data)
                             (result '()))
                    (if (null? remaining)
                        (reverse result)
                        (let* ((batch-end (min batch-size (length remaining)))
                               (batch (take remaining batch-end))
                               (rest (drop remaining batch-end)))
                          (loop rest (cons batch result))))))
          
          (batch-num 1))

      ;; Process each mini-batch
      (for-each
       (lambda (batch)
         (let ((actual-batch-size (length batch)))  ; Might be smaller for last batch

           
           ;; Accumulate gradients over the batch
           
           (for-each
            (lambda (sample)
              (let* ((img-data (car sample))
                     (true-class (cdr sample))
                     (img (make-tensor32 img-data 
                                         (list num-channels image-size image-size)))
                     (target (make-tensor32 (one-hot true-class num-classes)
                                            (list num-classes)))
                     
                     ;; Forward pass with manual flattening
                     (conv-out (forward conv-layers img))
                     ;(d (printf "Batch ~A: conv-out = ~A\n" batch-num (tensor-data conv-out)))
                     (flat (flatten-tensor conv-out))
                     (logits (forward dense-layers flat))
                     (probs (softmax logits))
                     (loss (cross-entropy-loss probs target)))

                ;(printf "Batch ~A: logits = ~A probs = ~A\n"
                ;        batch-num (tensor-data logits) (tensor-data probs))
                ;(printf "Batch ~A: true-class = ~A target = ~A loss = ~A\n"
                ;        batch-num true-class (tensor-data target) (tensor-data loss))

                ;; Accumulate metrics
                (set! total-loss (+ total-loss 
                                   (f32vector-ref (tensor-data loss) 0)))
                
                (let* ((pred-data (tensor-data logits))
                       (pred-class (argmax pred-data)))
                  (when (= pred-class true-class)
                    (set! correct (+ correct 1))))
                
                ;; Reset loss gradient before backward 
                ;(fill-ones! (tensor-grad loss) (tensor-dtype loss))

                ;; Backward pass - gradients accumulate automatically
                (backward! loss)

                ;(printf "Loss grad after backward: ~A\n" (tensor-grad loss))
                #;(printf "First param grad: ~A\n" 
                        (let ((p (car (parameters conv-layers))))
                          (tensor-grad p)))
                
                ))
            batch)

           ;; Scale gradients by 1/batch_size to get average
           (let ((scale-factor (/ 1.0 actual-batch-size)))
             (for-each
              (lambda (param)
                (let ((grad (tensor-grad param)))
                  (when grad
                    (let ((n (f32vector-length grad)))
                      (case (tensor-dtype param)
                        ((f32) (sscal! n scale-factor grad))
                        ((f64) (dscal! n scale-factor grad)))
                      ))))
              (append (parameters conv-layers)
                      (parameters dense-layers))))

           ;; Update parameters once per batch (with averaged gradients)
           (step! optimizer)

           ;; Zero gradients for next batch
           (zero-grad-layer! conv-layers)
           (zero-grad-layer! dense-layers)

           (set! batch-num (+ 1 batch-num))
         ))
       batches))
    
    (values (/ total-loss n) (/ correct n))))

(define (evaluate model test-data)
  "Evaluate model on test data"
  (let ((correct 0)
        (total (length test-data))
        (confusion (make-vector (* num-classes num-classes) 0))
        ;; Unpack model parts
        (conv-layers (car model))
        (dense-layers (cadr model)))
    
    (for-each
     (lambda (sample)
       (let* ((img-data (car sample))
              (true-class (cdr sample))
              (img (make-tensor32 img-data 
                                  (list num-channels image-size image-size)
                                  requires-grad?: #f))
              (logits (forward-cnn model img))
              (pred-data (tensor-data logits))
              (pred-class (argmax pred-data)))
         
         (when (= pred-class true-class)
           (set! correct (+ correct 1)))
         
         ;; Update confusion matrix
         (let ((idx (+ (* true-class num-classes) pred-class)))
           (vector-set! confusion idx (+ 1 (vector-ref confusion idx))))))
     test-data)
    
    (values (/ correct total) confusion)))

(define (print-confusion-matrix confusion)
  "Pretty print confusion matrix"
  (printf "\nConfusion Matrix:\n")
  (printf "         ")
  (do ((i 0 (+ i 1)))
      ((= i num-classes))
    (printf "Pred-~A  " i))
  (printf "\n")
  
  (do ((true-class 0 (+ true-class 1)))
      ((= true-class num-classes))
    (printf "True-~A  " true-class)
    (do ((pred-class 0 (+ pred-class 1)))
        ((= pred-class num-classes))
      (let ((idx (+ (* true-class num-classes) pred-class)))
        (printf "~A  " (vector-ref confusion idx))))
    (printf "\n")))

;;; ==================================================================
;;; Main Training Loop
;;; ==================================================================

(define (main)
  (printf "========================================\n")
  (printf "Convolutional Neural Network Example\n")
  (printf "========================================\n\n")
  
  ;; Set random seed for reproducibility
  (set-random-seed! 42)
  
  ;; Generate dataset
  (printf "Generating dataset...\n")
  (define train-data (generate-dataset 250))  ; 250 samples per class = 1000 total
  (define test-data (generate-dataset 50))     ; 50 samples per class = 200 total
  (printf "Training samples: ~A\n" (length train-data))
  (printf "Test samples: ~A\n\n" (length test-data))
  
  ;; Print class distribution
  (printf "Training set class distribution:\n")
  (let ((counts (make-vector num-classes 0)))
    (for-each
     (lambda (sample)
       (let ((class (cdr sample)))
         (vector-set! counts class (+ 1 (vector-ref counts class)))))
     train-data)
    (do ((i 0 (+ i 1)))
        ((= i num-classes))
      (printf "  Class ~A: ~A samples\n" i (vector-ref counts i))))
  (printf "\n")
  
  ;; Build model
  (printf "Building CNN model...\n")
  (define model (build-cnn))
  (define conv-layers (car model))
  (define dense-layers (cadr model))
  
  (printf "\nModel Architecture:\n")
  (printf "  Convolutional Layers:\n")
  (let ((conv-params (parameters conv-layers)))
    (printf "    Parameters: ~A\n" 
            (fold (lambda (p acc)
                    (+ acc (f32vector-length (tensor-data p))))
                  0 conv-params)))
  (printf "  Dense Layers:\n")
  (let ((dense-params (parameters dense-layers)))
    (printf "    Parameters: ~A\n"
            (fold (lambda (p acc)
                    (+ acc (f32vector-length (tensor-data p))))
                  0 dense-params)))
  (printf "  Total Parameters: ~A\n"
          (+ (fold (lambda (p acc) (+ acc (f32vector-length (tensor-data p))))
                   0 (parameters conv-layers))
             (fold (lambda (p acc) (+ acc (f32vector-length (tensor-data p))))
                   0 (parameters dense-layers))))
  (printf "\n")
  
  ;; Create optimizer
  (define learning-rate 0.001)
  (printf "Optimizer: Adam (lr=~A)\n\n" learning-rate)
  (define optimizer (make-adam (append (parameters conv-layers)
                                       (parameters dense-layers))
                               learning-rate: learning-rate
                               weight-decay: 0.0001))
  
  ;; Training loop
  (define num-epochs 20)
  (printf "Training for ~A epochs...\n" num-epochs)
  (printf "----------------------------------------\n")
  
  (do ((epoch 1 (+ epoch 1)))
      ((> epoch num-epochs))
    
    ;; Train
    (let-values (((avg-loss accuracy) (train-epoch model optimizer train-data batch-size: 64)))
      (printf "Epoch ~A/~A - Loss: ~A - Acc: ~A"
              epoch num-epochs avg-loss (* 100.0 accuracy))
      
      ;; Evaluate every 5 epochs
      (when (= (modulo epoch 5) 0)
        (let-values (((test-acc confusion) (evaluate model test-data)))
          (printf " - Test Acc: ~A" (* 100.0 test-acc))))
      
      (printf "\n"))
    
    ;; Learning rate decay
    (when (= (modulo epoch 10) 0)
      (let ((new-lr (* (get-learning-rate optimizer) 0.5)))
        (set-learning-rate! optimizer new-lr)
        (printf "  - Learning rate decreased to ~A\n" new-lr))))
  
  (printf "----------------------------------------\n\n")
  
  ;; Final evaluation
  (printf "Final Evaluation on Test Set:\n")
  (let-values (((test-acc confusion) (evaluate model test-data)))
    (printf "Test Accuracy: ~A\n" (* 100.0 test-acc))
    (print-confusion-matrix confusion))
  
  (printf "\n")
  
  ;; Per-class accuracy
  (printf "\nPer-Class Accuracy:\n")
  (let-values (((test-acc confusion) (evaluate model test-data)))
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
        (printf "  Class ~A: ~A (~A/~A)\n" 
                class
                (* 100 (/ correct total))
                correct
                total))))
  
  (printf "\n")
  
  ;; Test on individual samples
  (printf "Sample Predictions:\n")
  (do ((i 0 (+ i 1)))
      ((= i 200))
    (let* ((sample (list-ref test-data i))
           (img-data (car sample))
           (true-class (cdr sample))
           (img (make-tensor32 img-data 
                              (list num-channels image-size image-size)
                              requires-grad?: #f))
           (logits (forward-cnn model img))
           (probs (softmax logits))
           (pred-data (tensor-data probs))
           (pred-class (argmax pred-data)))
      
      (printf "  Sample ~A: True=~A, Pred=~A " (+ i 1) true-class pred-class)
      (if (= pred-class true-class)
          (printf "O")
          (printf "X"))
      (printf " (confidence: ~A)\n" 
              (* 100 (f32vector-ref pred-data pred-class)))))
  
  (printf "\n========================================\n")
  (printf "Training Complete!\n")
  (printf "========================================\n"))

;;; ==================================================================
;;; Visualization Helpers
;;; ==================================================================

(define (print-image img-data)
  "Print ASCII representation of image"
  (printf "\n")
  (do ((y 0 (+ y 1)))
      ((= y image-size))
    (do ((x 0 (+ x 1)))
        ((= x image-size))
      (let* ((idx (+ (* y image-size) x))
             (val (f32vector-ref img-data idx)))
        (cond
         ((< val 0.3) (printf " "))
         ((< val 0.6) (printf "."))
         (else (printf "#")))))
    (printf "\n")))

(define (visualize-predictions model test-data n)
  "Visualize n predictions with images"
  (printf "\n========================================\n")
  (printf "Prediction Visualizations\n")
  (printf "========================================\n")
  
  (do ((i 0 (+ i 1)))
      ((= i n))
    (let* ((sample (list-ref test-data i))
           (img-data (car sample))
           (true-class (cdr sample))
           (img (make-tensor32 img-data 
                              (list num-channels image-size image-size)
                              requires-grad?: #f))
           (logits (forward model img))
           (probs (softmax logits))
           (pred-data (tensor-data probs))
           (pred-class (argmax pred-data)))
      
      (printf "\n--- Sample ~A ---\n" (+ i 1))
      (printf "True Label: ~A " true-class)
      (printf "(~A)\n" 
              (case true-class
                ((0) "Vertical")
                ((1) "Horizontal")
                ((2) "Diagonal /")
                ((3) "Diagonal \\")))
      (printf "Predicted:  ~A " pred-class)
      (printf "(~A) " 
              (case pred-class
                ((0) "Vertical")
                ((1) "Horizontal")
                ((2) "Diagonal /")
                ((3) "Diagonal \\")))
      (if (= pred-class true-class)
          (printf "O\n")
          (printf "X\n"))
      
      (printf "\nClass Probabilities:\n")
      (do ((c 0 (+ c 1)))
          ((= c num-classes))
        (printf "  Class ~A: " c)
        (let ((conf (f32vector-ref pred-data c)))
          (printf "~A " (* 100 conf))
          ;; Bar chart
          (let ((bar-len (inexact->exact (floor (* conf 30)))))
            (do ((j 0 (+ j 1)))
                ((= j bar-len))
              (printf "█")))
          (printf "\n")))
      
      (printf "\nImage:\n")
      (print-image img-data))))

;;; ==================================================================
;;; Feature Map Visualization
;;; ==================================================================

(define (extract-conv-features model img-tensor layer-idx)
  "Extract feature maps from a specific convolutional layer"
  (let ((x img-tensor))
    ;; Forward through layers up to layer-idx
    (do ((i 0 (+ i 1)))
        ((= i layer-idx) x)
      (set! x (forward (list-ref (parameters model) i) x)))))

(define (print-feature-map feature-map channel)
  "Print ASCII representation of a single feature map channel"
  (let* ((shape (tensor-shape feature-map))
         (C (car shape))
         (H (cadr shape))
         (W (caddr shape))
         (data (tensor-data feature-map)))
    
    (when (< channel C)
      (printf "\nFeature Map (Channel ~A):\n" channel)
      
      ;; Find min and max for normalization
      (let ((min-val +inf.0)
            (max-val -inf.0))
        (do ((y 0 (+ y 1)))
            ((= y H))
          (do ((x 0 (+ x 1)))
              ((= x W))
            (let* ((idx (+ (* channel H W) (* y W) x))
                   (val (f32vector-ref data idx)))
              (set! min-val (min min-val val))
              (set! max-val (max max-val val)))))
        
        ;; Print normalized feature map
        (do ((y 0 (+ y 1)))
            ((= y H))
          (do ((x 0 (+ x 1)))
              ((= x W))
            (let* ((idx (+ (* channel H W) (* y W) x))
                   (val (f32vector-ref data idx))
                   (norm (if (= max-val min-val)
                             0.5
                             (/ (- val min-val) (- max-val min-val)))))
              (cond
               ((< norm 0.2) (printf " "))
               ((< norm 0.4) (printf "."))
               ((< norm 0.6) (printf ":"))
               ((< norm 0.8) (printf "o"))
               (else (printf "#")))))
          (printf "\n"))))))

;;; ==================================================================
;;; Model Analysis
;;; ==================================================================

(define (analyze-model-weights model)
  "Analyze weight statistics for each layer"
  (printf "\n========================================\n")
  (printf "Model Weight Analysis\n")
  (printf "========================================\n\n")
  
  (let ((params (parameters model))
        (layer-names '("Conv1-W" "Conv1-b" "Conv2-W" "Conv2-b" 
                      "Conv3-W" "Conv3-b" "FC1-W" "FC1-b" 
                      "FC2-W" "FC2-b")))
    
    (for-each
     (lambda (param name)
       (let* ((data (tensor-data param))
              (dtype (tensor-dtype param))
              (n (vector-length-for-dtype data dtype))
              (sum 0.0)
              (sum-sq 0.0)
              (min-val +inf.0)
              (max-val -inf.0))
         
         ;; Compute statistics
         (do ((i 0 (+ i 1)))
             ((= i n))
           (let ((val (case dtype
                       ((f32) (f32vector-ref data i))
                       ((f64) (f64vector-ref data i)))))
             (set! sum (+ sum val))
             (set! sum-sq (+ sum-sq (* val val)))
             (set! min-val (min min-val val))
             (set! max-val (max max-val val))))
         
         (let* ((mean (/ sum n))
                (variance (- (/ sum-sq n) (* mean mean)))
                (stddev (sqrt (max 0.0 variance))))
           
           (printf "~A:\n" name)
           (printf "  Shape: ~A\n" (tensor-shape param))
           (printf "  Mean:  ~8,5f\n" mean)
           (printf "  Std:   ~8,5f\n" stddev)
           (printf "  Min:   ~8,5f\n" min-val)
           (printf "  Max:   ~8,5f\n" max-val)
           (printf "\n"))))
     params
     layer-names)))

(define (count-dead-neurons model)
  "Count neurons that never activate (always output 0)"
  (printf "\n========================================\n")
  (printf "Dead Neuron Analysis\n")
  (printf "========================================\n\n")
  (printf "Note: This requires activation statistics from training data\n")
  (printf "In a full implementation, track activations during training\n"))

;;; ==================================================================
;;; Advanced Training: Data Augmentation
;;; ==================================================================

(define (augment-image img-data)
  "Apply random augmentations to image"
  (let ((aug-img (make-f32vector (f32vector-length img-data))))
    
    ;; Copy original
    (do ((i 0 (+ i 1)))
        ((= i (f32vector-length img-data)))
      (f32vector-set! aug-img i (f32vector-ref img-data i)))
    
    ;; Random horizontal flip (50% chance)
    (when (< (pseudo-random-real) 0.5)
      (do ((y 0 (+ y 1)))
          ((= y image-size))
        (do ((x 0 (+ x 1)))
            ((< x (quotient image-size 2)))
          (let ((idx1 (+ (* y image-size) x))
                (idx2 (+ (* y image-size) (- image-size 1 x)))
                (tmp (f32vector-ref aug-img idx1)))
            (f32vector-set! aug-img idx1 (f32vector-ref aug-img idx2))
            (f32vector-set! aug-img idx2 tmp)))))
    
    ;; Random brightness adjustment
    (let ((brightness-delta (* 0.2 (- (pseudo-random-real) 0.5))))
      (do ((i 0 (+ i 1)))
          ((= i (f32vector-length aug-img)))
        (f32vector-set! aug-img i
                        (max 0.0 
                             (min 1.0 
                                  (+ (f32vector-ref aug-img i) 
                                     brightness-delta))))))
    
    aug-img))

(define (train-epoch-with-augmentation model optimizer train-data)
  "Train epoch with data augmentation"
  (let ((total-loss 0.0)
        (correct 0)
        (n (length train-data)))
    
    (for-each
     (lambda (sample)
       (let* ((orig-img-data (car sample))
              (true-class (cdr sample))
              (aug-img-data (augment-image orig-img-data))
              (img (make-tensor32 aug-img-data 
                                  (list num-channels image-size image-size)))
              (target (make-tensor32 (one-hot true-class num-classes)
                                    (list num-classes)))
              
              (logits (forward model img))
              (probs (softmax logits))
              (loss (cross-entropy-loss probs target)))
         
         (set! total-loss (+ total-loss 
                            (f32vector-ref (tensor-data loss) 0)))
         
         (let* ((pred-data (tensor-data logits))
                (pred-class (argmax pred-data)))
           (when (= pred-class true-class)
             (set! correct (+ correct 1))))

         ;; Reset loss gradient before backward 
         (let ((loss-grad (tensor-grad loss)))
           (f32vector-set! loss-grad 0 1.0))
         
         (backward! loss)

         
         (step! optimizer)
         (zero-grad-layer! model)))
     train-data)
    
    (values (/ total-loss n) (/ correct n))))

;;; ==================================================================
;;; Enhanced Main with Visualizations
;;; ==================================================================

(define (main-visualization)
  (printf "========================================\n")
  (printf "CNN Training with Visualization Example\n")
  (printf "========================================\n\n")
  
  (set-random-seed! 42)
  
  ;; Generate dataset
  (printf "Generating dataset...\n")
  (define train-data (generate-dataset 250))
  (define test-data (generate-dataset 50))
  (printf "Training samples: ~A\n" (length train-data))
  (printf "Test samples: ~A\n\n" (length test-data))
  
  ;; Show sample images
  (printf "Sample Training Images:\n")
  (do ((i 0 (+ i 1)))
      ((= i 4))
    (let ((sample (list-ref train-data i)))
      (printf "\n--- Class ~A (~A) ---" 
              (cdr sample)
              (case (cdr sample)
                ((0) "Vertical")
                ((1) "Horizontal")
                ((2) "Diagonal /")
                ((3) "Diagonal \\")))
      (print-image (car sample))))
  
  (printf "\n")
  
  ;; Build and train model
  (printf "Building CNN model...\n")
  (define model (build-cnn))
  
  (printf "\nInitial weight statistics:\n")
  (analyze-model-weights model)
  
  ;; Train
  (define learning-rate 0.001)
  (printf "Training with Adam optimizer (lr=~A)...\n\n" learning-rate)
  (define optimizer (make-adam (parameters model) 
                               learning-rate: learning-rate
                               weight-decay: 0.0001))
  
  (define num-epochs 25)
  (printf "Training for ~A epochs with data augmentation...\n" num-epochs)
  (printf "----------------------------------------\n")
  
  (do ((epoch 1 (+ epoch 1)))
      ((> epoch num-epochs))
    
    (let-values (((avg-loss accuracy) 
                  (train-epoch-with-augmentation model optimizer train-data)))
      (printf "Epoch ~A/~A - Loss: ~A - Train Acc: ~A"
              epoch num-epochs avg-loss (* 100 accuracy))
      
      (when (= (modulo epoch 5) 0)
        (let-values (((test-acc confusion) (evaluate model test-data)))
          (printf " - Test Acc: ~A" (* 100 test-acc))))
      
      (printf "\n"))
    
    (when (= (modulo epoch 10) 0)
      (let ((new-lr (* (get-learning-rate optimizer) 0.5)))
        (set-learning-rate! optimizer new-lr)
        (printf "  → Learning rate decreased to ~A\n" new-lr))))
  
  (printf "----------------------------------------\n\n")
  
  ;; Final evaluation
  (printf "Final Evaluation:\n")
  (let-values (((test-acc confusion) (evaluate model test-data)))
    (printf "Test Accuracy: ~A\n" (* 100 test-acc))
    (print-confusion-matrix confusion))
  
  ;; Per-class accuracy
  (printf "\nPer-Class Metrics:\n")
  (let-values (((test-acc confusion) (evaluate model test-data)))
    (do ((class 0 (+ class 1)))
        ((= class num-classes))
      (let ((tp 0)      ; true positives
            (fp 0)      ; false positives
            (fn 0)      ; false negatives
            (total 0))
        
        (do ((true-c 0 (+ true-c 1)))
            ((= true-c num-classes))
          (do ((pred-c 0 (+ pred-c 1)))
              ((= pred-c num-classes))
            (let ((count (vector-ref confusion 
                                    (+ (* true-c num-classes) pred-c))))
              (cond
               ((and (= true-c class) (= pred-c class))
                (set! tp (+ tp count)))
               ((and (= true-c class) (not (= pred-c class)))
                (set! fn (+ fn count)))
               ((and (not (= true-c class)) (= pred-c class))
                (set! fp (+ fp count)))))))
        
        (let* ((precision (if (= (+ tp fp) 0) 
                              0.0 
                              (/ tp (+ tp fp))))
               (recall (if (= (+ tp fn) 0) 
                          0.0 
                          (/ tp (+ tp fn))))
               (f1 (if (= (+ precision recall) 0)
                       0.0
                       (* 2 (/ (* precision recall) 
                              (+ precision recall))))))
          
          (printf "  Class ~A: Precision=~A Recall=~A F1=~A\n"
                  class
                  (* 100 precision)
                  (* 100 recall)
                  (* 100 f1))))))
  
  ;; Visualize predictions
  (visualize-predictions model test-data 5)
  
  ;; Weight analysis
  (printf "\nFinal weight statistics:\n")
  (analyze-model-weights model)
  
  ;; Feature visualization (first test sample)
  (printf "\n========================================\n")
  (printf "Feature Map Visualization\n")
  (printf "========================================\n")
  (printf "\nShowing feature maps from Conv1 (first 4 channels):\n")
  
  (let* ((sample (car test-data))
         (img-data (car sample))
         (img (make-tensor32 img-data 
                            (list num-channels image-size image-size)
                            requires-grad?: #f)))
    
    (printf "\nOriginal Image:\n")
    (print-image img-data)
    
    ;; This would require modifying the forward pass to extract intermediate features
    ;; For demonstration, we note where feature extraction would occur
    (printf "\n(Feature map extraction would require additional implementation)\n"))
  
  (printf "\n========================================\n")
  (printf "Training Complete!\n")
  (printf "========================================\n"))

;; Run the basic example
(printf "\n")
(printf "  NanoGrad CNN Example                  \n")
(printf "  Image Classification with Conv Layers \n")
(printf "\n")

;; Uncomment one of these to run:
(main)                ; Basic training
; (main-visualization)     ; Training with visualizations

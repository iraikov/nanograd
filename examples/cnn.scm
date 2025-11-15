;; Convolutional neural network example
;; Demonstrates building, training, and evaluating a CNN for image classification

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
;;; Batch Construction
;;; ==================================================================

(define (stack-images batch)
  "Stack a batch of images into a single 4D tensor (N, C, H, W)"
  (let* ((batch-size (length batch))
         (sample-img (caar batch))
         (img-size (f32vector-length sample-img))
         ;; Create batched tensor data
         (batched-data (make-f32vector (* batch-size img-size) 0.0)))
    
    ;; Copy each image into the batched tensor
    (do ((i 0 (+ i 1)))
        ((= i batch-size))
      (let ((img-data (car (list-ref batch i)))
            (offset (* i img-size)))
        (do ((j 0 (+ j 1)))
            ((= j img-size))
          (f32vector-set! batched-data (+ offset j)
                         (f32vector-ref img-data j)))))
    
    ;; Return as 4D tensor: (batch_size, channels, height, width)
    (make-tensor32 batched-data 
                   (list batch-size num-channels image-size image-size)
                   requires-grad?: #f)))

(define (stack-targets batch)
  "Stack batch targets into 2D tensor (N, num_classes) for one-hot
   or 1D tensor (N,) for class indices"
  (let* ((batch-size (length batch))
         ;; Use class indices format for efficiency
         (target-data (make-f32vector batch-size 0.0)))
    
    (do ((i 0 (+ i 1)))
        ((= i batch-size))
      (let ((class (cdr (list-ref batch i))))
        (f32vector-set! target-data i (exact->inexact class))))
    
    ;; Return as 1D tensor of class indices
    (make-tensor32 target-data (list batch-size))))



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
         (dense-layers-list
          (list
           (make-dense-layer (* 64 7 7) 128
                             activation: (make-relu)
                             name: "FC1")
           
           (make-dense-layer 128 num-classes
                             activation: (make-identity)
                             name: "Output")))
         
         
        (model
         (make-sequential
          `(,@conv-layers-list
            ,(make-flatten name: "Flatten")
            ,@dense-layers-list)
          name: "CNN"))
        )
    
    (list model
          conv-layers-list dense-layers-list)
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
  (let* ((model-layer (car model))
         (logits (forward model-layer x)))
      logits))

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


(define (train-epoch-batched model optimizer train-data 
                             #!key (batch-size 32) (gradient-diagnostics #f))
  "Train for one epoch with true batch processing"
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
                
                ;; Stack entire batch into single tensors
                (batch-images (stack-images batch))
                (batch-targets (stack-targets batch))

                ;; Single forward pass for entire batch
                (logits (forward model-layer batch-images))
                
                ;; Flatten: (N, C, H, W) -> (N, C*H*W)
                ;(batch-shape (tensor-shape conv-out))
                ;(N (car batch-shape))
                ;(flattened-size (apply * (cdr batch-shape)))
                ;(flat (reshape conv-out (list N flattened-size)))
                
                ;; Dense layers: (N, features_in) -> (N, num_classes)
                ;(logits (forward dense-layers flat))
                
                ;; Softmax over classes: (N, num_classes)
                (probs (softmax logits axis: -1))
                
                ;; Cross-entropy loss with mean reduction
                ;; Returns scalar loss averaged over batch
                (loss (cross-entropy-loss probs batch-targets 
                                          reduction: 'mean
                                          from-logits: #f)))

           ;(printf "mini-batch logits: ~A\n" (tensor-data logits))
           ;(printf "mini-batch probs: ~A\n" (tensor-data probs))
           ;(printf "mini-batch batch-targets: ~A\n" (tensor-data batch-targets))
           ;(printf "mini-batch loss: ~A\n" (tensor-data loss))
           
           ;; Accumulate loss for reporting
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
                                  (f32vector-ref (tensor-data batch-targets) i))))
                 (when (= pred-class true-class)
                   (set! correct (+ correct 1))))))
           
           ;; Single backward pass for entire batch
           ;; Loss already computed with reduction='mean'
           (backward! loss)

           ;; Check gradient health
           (if gradient-diagnostics
               (record-step! monitor step params))

                  
           ;; Single optimizer step
           (step! optimizer)
           
           ;; Zero gradients for next batch
           (zero-grad-layer! model-layer)
           ))
       batches))

    (if gradient-diagnostics
        (begin
          (printf "\nTraining Diagnostics:\n")
          (let ((diagnosis (diagnose-training monitor)))
            (printf "Total steps: ~A\n" (cdr (assoc 'total-steps diagnosis)))
            (printf "Mean gradient norm: ~A\n" (cdr (assoc 'mean-gradient-norm diagnosis)))
            (printf "Unhealthy steps: ~A\n" (cdr (assoc 'unhealthy-steps diagnosis))))))
    
    (values (/ total-loss n) (/ correct n))))


(define (evaluate model test-data)
  "Evaluate model on test data"
  (let ((correct 0)
        (total (length test-data))
        (confusion (make-vector (* num-classes num-classes) 0)))
    
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
;;; Batched Evaluation
;;; ==================================================================

(define (evaluate-batched model test-data #!key (batch-size 64))
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
                
                ;; Stack batch (no gradients needed for eval)
                (batch-images (stack-images batch))
                (batch-targets (stack-targets batch))
                
                ;; Disable gradient tracking
                (batch-images (make-tensor32 (tensor-data batch-images)
                                             (tensor-shape batch-images)
                                             requires-grad?: #f))
                
                ;; Forward pass
                ;(conv-out (forward conv-layers batch-images))
                ;(batch-shape (tensor-shape conv-out))
                ;(N (car batch-shape))
                ;(flattened-size (apply * (cdr batch-shape)))
                ;(flat (reshape conv-out (list N flattened-size)))
                (logits (forward model-layer batch-images))
                (logits-data (tensor-data logits)))
           
           ;; Process predictions
           (do ((i 0 (+ i 1)))
               ((= i actual-batch-size))
             (let* ((offset (* i num-classes))
                    (pred-class (argmax-offset logits-data offset num-classes))
                    (true-class (inexact->exact 
                                (f32vector-ref (tensor-data batch-targets) i))))
               
               (when (= pred-class true-class)
                 (set! correct (+ correct 1)))
               
               ;; Update confusion matrix
               (let ((idx (+ (* true-class num-classes) pred-class)))
                 (vector-set! confusion idx 
                            (+ 1 (vector-ref confusion idx))))))))
       batches))
    
    (values (/ correct total) confusion)))

(define (argmax-offset vec offset length)
  "Find argmax in a slice of a vector starting at offset"
  (let loop ((i 1) (max-i 0) (max-val (f32vector-ref vec offset)))
    (if (= i length)
        max-i
        (let ((val (f32vector-ref vec (+ offset i))))
          (if (> val max-val)
              (loop (+ i 1) i val)
              (loop (+ i 1) max-i max-val))))))

;; Add this debugging version to your cnn.scm file
;; This will help identify where gradients are lost

(define (debug-forward-pass model batch-images batch-targets)
  "Forward pass with gradient flow debugging"
  (let ((conv-layers (cadr model))
        (dense-layers (caddr model)))
    
    (printf "\n=== Forward Pass Debug ===\n")
    
    ;; Step 1: Conv layers
    (printf "Input shape: ~A\n" (tensor-shape batch-images))
    (printf "Input requires-grad?: ~A\n" (tensor-requires-grad? batch-images))
    
    (let ((conv-out (forward conv-layers batch-images)))
      (printf "Conv output shape: ~A\n" (tensor-shape conv-out))
      (printf "Conv output requires-grad?: ~A\n" (tensor-requires-grad? conv-out))
      (printf "Conv output children count: ~A\n" 
              (length (tensor-children conv-out)))
      
      ;; Step 2: Reshape
      (let* ((batch-shape (tensor-shape conv-out))
             (N (car batch-shape))
             (flattened-size (apply * (cdr batch-shape)))
             (flat (reshape conv-out (list N flattened-size))))
        
        (printf "\nReshape:\n")
        (printf "  Flat shape: ~A\n" (tensor-shape flat))
        (printf "  Flat requires-grad?: ~A\n" (tensor-requires-grad? flat))
        ;(printf "  Flat has backward-fn?: ~A\n" 
        ;        (if (tensor-backward-fn flat) "YES" "NO"))
        (printf "  Flat children count: ~A\n" 
                (length (tensor-children flat)))
        (printf "  Flat children[0] == conv-out?: ~A\n"
                (if (null? (tensor-children flat))
                    "NO CHILDREN!"
                    (eq? (car (tensor-children flat)) conv-out)))
        
        ;; Step 3: Dense layers
        (let ((logits (forward dense-layers flat)))
          (printf "\nDense output:\n")
          (printf "  Logits shape: ~A\n" (tensor-shape logits))
          (printf "  Logits requires-grad?: ~A\n" (tensor-requires-grad? logits))
          ;(printf "  Logits has backward-fn?: ~A\n" 
          ;       (if (tensor-backward-fn logits) "YES" "NO"))
          
          ;; Step 4: Loss
          (let* ((probs (softmax logits axis: -1))
                 (loss (cross-entropy-loss probs batch-targets 
                                           reduction: 'mean
                                           from-logits: #f)))
            
            (printf "\nLoss:\n")
            (printf "  Loss value: ~A\n" (tensor-data loss))
            (printf "  Loss requires-grad?: ~A\n" (tensor-requires-grad? loss))
            
            ;; Step 5: Backward pass
            (printf "\n=== Backward Pass Debug ===\n")
            (backward! loss)
            
            (printf "After backward:\n")
            (printf "  Loss grad: ~A\n" (tensor-grad loss))
            (printf "  Logits grad sum: ~A\n" 
                    (if (tensor-grad logits)
                        (f32vector-fold + 0.0 (tensor-grad logits))
                        "NO GRAD"))
            (printf "  Flat grad sum: ~A\n" 
                    (if (tensor-grad flat)
                        (f32vector-fold + 0.0 (tensor-grad flat))
                        "NO GRAD"))
            (printf "  Conv-out grad sum: ~A\n" 
                    (if (tensor-grad conv-out)
                        (f32vector-fold + 0.0 (tensor-grad conv-out))
                        "NO GRAD"))
            
            ;; Check conv layer parameter gradients
            (printf "\nConv layer parameter gradients:\n")
            (let ((conv-params (parameters conv-layers)))
              (for-each
               (lambda (param idx)
                 (let ((grad (tensor-grad param)))
                   (printf "  Param ~A grad sum: ~A\n" 
                           idx
                           (if grad
                               (f32vector-fold + 0.0 grad)
                               "NO GRAD"))))
               conv-params
               (iota (length conv-params))))
            
            loss))))))

;; Use this to debug a single batch
(define (debug-single-batch)
  "Debug gradient flow on a single small batch"
  (set-random-seed! 42)
  
  (printf "Generating small test batch...\n")
  (let* ((test-batch (generate-dataset 2))  ; Just 2 samples per class = 8 total
         (small-batch (take test-batch 4))  ; Take 4 samples
         (model (build-cnn)))
    
    (printf "Building batch tensors...\n")
    (let ((batch-images (stack-images small-batch))
          (batch-targets (stack-targets small-batch)))
      
      (printf "Batch images shape: ~A\n" (tensor-shape batch-images))
      (printf "Batch targets shape: ~A\n" (tensor-shape batch-targets))
      
      ;; Run debug forward pass
      (debug-forward-pass model batch-images batch-targets))))

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
  (define model-layer (car model))
  (define conv-layers (make-sequential (cadr model) name: "Conv-Layers"))
  (define dense-layers (make-sequential (caddr model) name: "Dense-Layers"))
  
  (printf "\nModel Architecture:\n")
  (printf "  Convolutional Layers:\n")
  (let ((conv-params (parameters conv-layers )))
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
  (define optimizer (make-adam (parameters model-layer)
                               learning-rate: learning-rate
                               weight-decay: 0.0001))
  
  ;; Training loop
  (define num-epochs 20)
  (define best-acc 0.0)

  (printf "Training for ~A epochs...\n" num-epochs)
  (printf "----------------------------------------\n")
  
  (do ((epoch 1 (+ epoch 1)))
      ((> epoch num-epochs))
    
    ;; Train
    (let-values (((avg-loss accuracy)
                  ;(train-epoch model optimizer train-data
                  ;                     )))
                  (train-epoch-batched model optimizer train-data
                                       batch-size: 64 gradient-diagnostics: (= epoch 1))))
      (printf "Epoch ~A/~A - Loss: ~A - Acc: ~A"
              epoch num-epochs avg-loss (* 100.0 accuracy))
      
      ;; Evaluate every 5 epochs
      (when (= (modulo epoch 5) 0)
        (let-values (((test-acc confusion) (evaluate-batched model test-data batch-size: 64)))
          (printf " - Test Acc: ~A" (* 100.0 test-acc))
          ;; Save checkpoint if best so far
          (when (> test-acc best-acc)
            (set! best-acc test-acc)
            (printf "\n  New best accuracy! Saving checkpoint...")
            (save-checkpoint model optimizer epoch avg-loss accuracy
                             (sprintf "best-cnn-model_~A.ngrd" epoch)))))
          
      
      (printf "\n"))
    
    ;; Learning rate decay
    (when (= (modulo epoch 10) 0)
      (let ((new-lr (* (get-learning-rate optimizer) 0.5)))
        (set-learning-rate! optimizer new-lr)
        (printf "  - Learning rate decreased to ~A\n" new-lr))))
  
  (printf "----------------------------------------\n\n")

  ;; Save final model
  (printf "Saving final model...\n")
  (save-cnn-model model "final-cnn-model.ngrd")
  (printf "\n")
  
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

;;; ==================================================================
;;; Model Persistence
;;; ==================================================================

(define (save-cnn-model model filepath)
  "Save CNN model to file
   
   Args:
     model: The model tuple (conv-layers, dense-layers, conv-layers-internal)
     filepath: Path where to save the model
   
   The model is saved as a two-layer sequential (conv + dense)"
  
  (let ((model-layer (car model))
        (conv-layers (make-sequential (cadr model) name: "Conv-Layers"))
        (dense-layers (make-sequential (caddr model) name: "Dense-Layers")))
    
    (printf "Saving model to ~A...\n" filepath)
    
      ;; Save using the built-in serialization
      (save-model model-layer filepath)
      
      (printf "Model saved successfully!\n")
      (printf "  Conv layers parameters: ~A\n" 
              (length (parameters conv-layers)))
      (printf "  Dense layers parameters: ~A\n" 
              (length (parameters dense-layers)))
      ))

(define (load-cnn-model filepath)
  "Load CNN model from file
   
   Args:
     filepath: Path to the saved model
   
   Returns:
     Model tuple compatible with forward-cnn and train functions"
  
  (printf "Loading model from ~A...\n" filepath)
  
  ;; Load the sequential model
  (let* ((full-model (load-model filepath))
         (ser (layer->serializable full-model))
         (layer-type (assq 'type ser))
         (layer-list (if (eq? (and layer-type (cdr layer-type)) 'sequential)
                         (assq 'layers ser) '(layers)))
         (layer-objects (map serializable->layer (cdr layer-list)))
         (conv-layers (filter conv2d-layer? layer-objects))
         (dense-layers (filter dense-layer? layer-objects)))

    (printf "Model loaded successfully!\n")
    
    ;; The loaded model is a sequential containing conv and dense layers
    ;; We need to extract them to match the expected model structure
    
    (list full-model
          conv-layers
          dense-layers)))

;;; ==================================================================
;;; Model Checkpointing
;;; ==================================================================

(define (save-checkpoint model optimizer epoch train-loss train-acc filepath)
  "Save training checkpoint including model, optimizer state, and metrics"
   
  (printf "\nSaving checkpoint at epoch ~A...\n" epoch)
  (save-cnn-model model filepath)
  
  (let ((metadata-file (string-append filepath ".meta")))
    (with-output-to-file metadata-file
      (lambda ()
        (printf "epoch: ~A\n" epoch)
        (printf "train-loss: ~A\n" train-loss)
        (printf "train-acc: ~A\n" train-acc)
        (printf "timestamp: ~A\n" (current-seconds)))))
  
  (printf "Checkpoint saved!\n"))

(define (load-checkpoint filepath)
  "Load a training checkpoint
   
   Returns: (model metadata)"
  
  (printf "Loading checkpoint from ~A...\n" filepath)
  
  (let ((model (load-cnn-model filepath))
        (metadata-file (string-append filepath ".meta")))
    
    ;; Load metadata if it exists
    (let ((metadata 
           (if (file-exists? metadata-file)
               (with-input-from-file metadata-file
                 (lambda ()
                   (let ((lines '()))
                     (let loop ((line (read-line)))
                       (unless (eof-object? line)
                         (set! lines (cons line lines))
                         (loop (read-line))))
                     (reverse lines))))
               '())))
      
      (when (not (null? metadata))
        (printf "\nCheckpoint metadata:\n")
        (for-each (lambda (line) (printf "  ~A\n" line)) metadata))
      
      (values model metadata))))

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

;; Run the basic example
(printf "\n")
(printf "  NanoGrad CNN Example                  \n")
(printf "  Image Classification with Conv Layers \n")
(printf "\n")

(main)

(define loaded-model (load-cnn-model "final-cnn-model.ngrd"))

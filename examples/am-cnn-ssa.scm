;;; nanograd/examples/am-cnn-ssa.scm
;;;
;;; Convolutional Neural Network example using array-morphisms SSA backend.
;;; Demonstrates image classification with Conv2D layers using fused
;;; forward+backward execution.
;;;
;;; Differences from cnn.scm:
;;;   - Uses AM-backed layers (make-am-conv2d-layer, make-am-dense-layer)
;;;   - Uses lazy tensors instead of strict tensors
;;;   - Uses am-training-step/ssa for training (single SSA context)
;;;   - Targets passed as extra-input to am-training-step/ssa
;;;   - Automatic buffer reuse through morphism context

(import scheme (chicken base) (chicken format) (chicken random) (chicken time))
(import (only srfi-1 map fold take drop filter iota last))
(import srfi-4)
(import array-morphisms-core array-morphisms-context)
(import array-morphisms-blas-exec)
(import array-morphisms-blas-egg-backend)
(import (prefix array-morphisms-grad am:))
(import nanograd-autograd nanograd-layer nanograd-optimizer)
(import nanograd-array-morphisms)

(register-blas-backend! (make-blas-egg-backend))


;;; ============================================================
;;; Data Generation: Synthetic Image Dataset
;;; ============================================================

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
          (set! dataset (cons (cons img class) dataset)))))))

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


;;; ============================================================
;;; Batch Construction for Lazy Tensors
;;; ============================================================

(define (one-hot class num-classes)
  "Convert class index to one-hot vector"
  (let ((vec (make-f32vector num-classes 0.0)))
    (f32vector-set! vec class 1.0)
    vec))

(define (list->f32vector lst)
  "Convert a list of floats to an f32vector"
  (let* ((n (length lst))
         (vec (make-f32vector n 0.0)))
    (let loop ((i 0) (l lst))
      (if (or (>= i n) (null? l))
          vec
          (begin
            (f32vector-set! vec i (exact->inexact (car l)))
            (loop (+ i 1) (cdr l)))))))

(define (stack-images batch)
  "Stack a batch of images into a 4D tensor (N, C, H, W) as lazy tensor"
  (let* ((batch-size (length batch))
         (sample-img (caar batch))
         (img-size (f32vector-length sample-img))
         (total-size (* batch-size img-size))
         (batched-data (make-f32vector total-size 0.0)))
    ;; Copy each image into the batched tensor
    (do ((i 0 (+ i 1)))
        ((= i batch-size))
      (let ((img-data (car (list-ref batch i)))
            (offset (* i img-size)))
        (do ((j 0 (+ j 1)))
            ((= j img-size))
          (f32vector-set! batched-data (+ offset j)
                          (f32vector-ref img-data j)))))
    ;; Return as lazy tensor wrapping a morph-variable
    ;; Shape: (N, C, H, W) = (batch_size, 1, 28, 28)
    (let ((mv (am:make-var (morph-from-list (f32vector->list batched-data)
                                            (list batch-size num-channels
                                                  image-size image-size)
                                            'f32)
                           #f)))  ; requires-grad? = #f for input
      (get-or-make-lazy mv))))

(define (stack-targets batch)
  "Stack batch targets into one-hot encoded lazy tensor (N, num_classes)"
  (let* ((batch-size (length batch))
         (target-data '()))
    ;; Build flattened one-hot targets
    (do ((i 0 (+ i 1)))
        ((= i batch-size))
      (let ((class (cdr (list-ref batch i))))
        (set! target-data
              (append target-data
                      (f32vector->list (one-hot class num-classes))))))
    ;; Return as lazy tensor
    (let ((mv (am:make-var (morph-from-list target-data
                                            (list batch-size num-classes)
                                            'f32)
                           #f)))  ; requires-grad? = #f for targets
      (get-or-make-lazy mv))))


;;; ============================================================
;;; CNN Architecture (AM-backed)
;;; ============================================================

(define (build-am-cnn)
  "Build AM-backed CNN for 28x28 grayscale image classification
   
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
  
  (make-am-sequential
   (list
    ;; Conv block 1: 1 -> 16 channels, 28x28 -> 28x28
    (make-am-conv2d-layer 1 16 3
                          stride: 1
                          padding: 1
                          activation: (make-relu)
                          dtype: 'f32)
    
    ;; Conv block 2: 16 -> 32 channels, 28x28 -> 14x14
    (make-am-conv2d-layer 16 32 3
                          stride: 2
                          padding: 1
                          activation: (make-relu)
                          dtype: 'f32)
    
    ;; Conv block 3: 32 -> 64 channels, 14x14 -> 7x7
    (make-am-conv2d-layer 32 64 3
                          stride: 2
                          padding: 1
                          activation: (make-relu)
                          dtype: 'f32)
    
    ;; Flatten: (N, 64, 7, 7) -> (N, 3136)
    (make-am-flatten name: "Flatten")
    
    ;; Dense block 1: 3136 -> 128
    (make-am-dense-layer (* 64 7 7) 128
                         activation: (make-relu)
                         dtype: 'f32)
    
    ;; Output layer: 128 -> 4
    (make-am-dense-layer 128 num-classes
                         activation: (make-identity)
                         dtype: 'f32))))


;;; ============================================================
;;; Evaluation Functions
;;; ============================================================

(define (argmax vec)
  "Return index of maximum value in vector"
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

(define (evaluate model test-data batch-size)
  "Evaluate model on test data using batched forward passes"
  (let ((correct 0)
        (total (length test-data))
        (confusion (make-vector (* num-classes num-classes) 0)))
    
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
                (batch-images (stack-images batch))
                (batch-targets (stack-targets batch)))
           
           ;; Forward pass (no gradients needed for eval)
           (let ((logits (forward model batch-images)))
             ;; logits is a lazy tensor, need to extract data
             (let ((logits-data (tensor-data logits)))
               ;; Process each sample in batch
               (do ((i 0 (+ i 1)))
                   ((= i actual-batch-size))
                 (let* ((offset (* i num-classes))
                        (pred-class (argmax-offset logits-data offset num-classes))
                        (true-class (inexact->exact
                                     (argmax-offset (tensor-data batch-targets)
                                                    offset num-classes))))
                   
                   (when (= pred-class true-class)
                     (set! correct (+ correct 1)))
                   
                   ;; Update confusion matrix
                   (let ((idx (+ (* true-class num-classes) pred-class)))
                     (vector-set! confusion idx
                                  (+ 1 (vector-ref confusion idx))))))))))
       batches))
    
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


;;; ============================================================
;;; Main Training Loop
;;; ============================================================

(define (main)
  (printf "========================================\n")
  (printf "AM CNN Example (SSA Backend)\n")
  (printf "Image Classification with Array Morphisms\n")
  (printf "========================================\n\n")
  
  ;; Set random seed
  (set-pseudo-random-seed! (number->string 42))
  
  ;; Generate dataset
  (printf "Generating dataset...\n")
  (define train-data (generate-dataset 250))  ; 250 per class = 1000 total
  (define test-data (generate-dataset 50))    ; 50 per class = 200 total
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
  (printf "Building AM CNN model...\n")
  (define model (build-am-cnn))
  (define batch-size 32)
  
  ;; Count parameters
  (let ((n-params (fold (lambda (p acc)
                          (+ acc (f32vector-length (tensor-data p))))
                        0 (am-parameters model))))
    (printf "Total parameters: ~A\n" n-params))
  (printf "\n")
  
  ;; Create optimizer and SSA context
  (define learning-rate 0.001)
  (printf "Optimizer: Adam (lr=~A)\n" learning-rate)
  (define optimizer (make-adam (am-parameters model)
                               learning-rate: learning-rate
                               weight-decay: 0.0001))
  
  ;; Single SSA context for training
  (define ctx (make-morphism-context))
  (printf "\n")
  
  ;; Training loop
  (define num-epochs 20)
  (define best-acc 0.0)
  
  (printf "Training for ~A epochs (batch-size=~A)...\n" num-epochs batch-size)
  (printf "----------------------------------------\n")
  
  (define t0 (cpu-time))
  
  (let epoch-loop ((epoch 1))
    (when (<= epoch num-epochs)
      (let* ((shuffled (shuffle train-data))
             (total-loss 0.0)
             (n-batches 0))
        
        ;; Mini-batch loop
        (let batch-loop ((rem shuffled))
          (when (>= (length rem) batch-size)
            (let* ((batch (take rem batch-size))
                   ;; Stack images and targets
                   (x-lt (stack-images batch))
                   (tgt-lt (stack-targets batch))
                   ;; Loss function: cross-entropy
                   (loss-fn (lambda (logits-lt)
                              (am-cross-entropy-loss logits-lt tgt-lt)))
                   ;; Training step with SSA
                   (loss-lt (am-training-step/ssa ctx optimizer model
                                                  loss-fn x-lt tgt-lt))
                   (loss-val (f32vector-ref (tensor-data loss-lt) 0)))
              
              (set! total-loss (+ total-loss loss-val))
              (set! n-batches (+ n-batches 1))
              (batch-loop (drop rem batch-size)))))
        
        ;; Report metrics
        (let ((avg-loss (/ total-loss (max n-batches 1))))
          (printf "Epoch ~A/~A - Train Loss: ~A" epoch num-epochs avg-loss)
          
          ;; Evaluate every 5 epochs
          (when (= (modulo epoch 5) 0)
            (let-values (((test-acc confusion) (evaluate model test-data batch-size)))
              (printf " - Test Acc: ~A%" (round (* 100.0 test-acc)))
              (when (> test-acc best-acc)
                (set! best-acc test-acc)
                (printf " (NEW BEST)"))))
          
          (printf "\n")
          
          ;; Learning rate decay
          (when (= (modulo epoch 10) 0)
            (let ((new-lr (* (get-learning-rate optimizer) 0.5)))
              (set-learning-rate! optimizer new-lr)
              (printf "  - Learning rate decreased to ~A\n" new-lr))))
        
        (epoch-loop (+ epoch 1)))))
  
  (define elapsed (- (cpu-time) t0))
  (printf "----------------------------------------\n")
  (printf "Training time: ~Ams\n\n" elapsed)
  
  ;; Final evaluation
  (printf "Final Evaluation on Test Set:\n")
  (let-values (((test-acc confusion) (evaluate model test-data batch-size)))
    (printf "Test Accuracy: ~A%\n" (round (* 100.0 test-acc)))
    (print-confusion-matrix confusion))
  
  ;; Per-class accuracy
  (printf "\nPer-Class Accuracy:\n")
  (let-values (((test-acc confusion) (evaluate model test-data batch-size)))
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
        (printf "  Class ~A: ~A% (~A/~A)\n"
                class
                (if (> total 0) (round (* 100.0 (/ correct total))) 0)
                correct
                total))))
  
  ;; Context stats
  (let* ((total-steps (* num-epochs (quotient (length train-data) batch-size)))
         (s (context-stats ctx))
         (na (cdr (assq 'allocations s)))
         (nb (cdr (assq 'buffers s))))
    (printf "\nSSA context buffer reuse (~A total steps):\n" total-steps)
    (printf "  ~A allocs -> ~A buffers (~A% reduction)\n"
            na nb (round (* 100 (- 1 (/ nb na))))))
  
  (printf "\n========================================\n")
  (printf "Training Complete!\n")
  (printf "========================================\n"))


;;; Run the example
(printf "\n")
(printf "  NanoGrad AM CNN Example (SSA Backend)\n")
(printf "  Convolutional Neural Network with Fused Execution\n")
(printf "\n")

(main)

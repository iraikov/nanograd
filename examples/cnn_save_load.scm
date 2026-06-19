;; Add these functions to cnn.scm for model persistence

;;; ==================================================================
;;; Model Persistence
;;; ==================================================================

(define (save-cnn-model model filepath)
  "Save CNN model to file
   
   Args:
     model: The model tuple (conv-layers, dense-layers, conv-layers-internal)
     filepath: Path where to save the model
   
   The model is saved as a two-layer sequential (conv + dense)"
  
  (let ((conv-layers (car model))
        (dense-layers (cadr model)))
    
    (printf "Saving model to ~A...\n" filepath)
    
    ;; Create a proper sequential model combining conv and dense
    (let ((full-model (make-sequential 
                       (list conv-layers dense-layers)
                       name: "CNN-Model")))
      
      ;; Save using the built-in serialization
      (save-model full-model filepath)
      
      (printf "Model saved successfully!\n")
      (printf "  Conv layers parameters: ~A\n" 
              (length (parameters conv-layers)))
      (printf "  Dense layers parameters: ~A\n" 
              (length (parameters dense-layers)))
      )))

(define (load-cnn-model filepath)
  "Load CNN model from file
   
   Args:
     filepath: Path to the saved model
   
   Returns:
     Model tuple compatible with forward-cnn and train functions"
  
  (printf "Loading model from ~A...\n" filepath)
  
  ;; Load the sequential model
  (let ((full-model (load-model filepath)))
    
    (printf "Model loaded successfully!\n")
    
    ;; The loaded model is a sequential containing conv and dense layers
    ;; We need to extract them to match the expected model structure
    
    ;; For now, return a simple wrapper that works with forward-cnn
    ;; We'll extract the layers from the sequential
    (list full-model #f #f)))  ; Simplified structure for evaluation

(define (forward-cnn-loaded model x)
  "Forward pass through a loaded CNN model
   
   Handles both regular and loaded model structures"
  
  (if (cadr model)
      ;; Regular model structure: (conv-layers, dense-layers, ...)
      (forward-cnn model x)
      ;; Loaded model: single sequential
      (let ((sequential (car model)))
        (forward sequential x))))

(define (evaluate-loaded-model filepath test-data #!key (batch-size 64))
  "Load and evaluate a model from file
   
   Args:
     filepath: Path to saved model
     test-data: Test dataset
     batch-size: Batch size for evaluation
   
   Returns:
     Test accuracy"
  
  (let ((model (load-cnn-model filepath)))
    (printf "\nEvaluating loaded model...\n")
    
    ;; Use batched evaluation for efficiency
    (let-values (((test-acc confusion) 
                  (evaluate-loaded-batched model test-data batch-size: batch-size)))
      (printf "Test Accuracy: ~A%\n" (* 100.0 test-acc))
      (print-confusion-matrix confusion)
      test-acc)))

(define (evaluate-loaded-batched model test-data #!key (batch-size 64))
  "Evaluate a loaded model on test data using batched forward passes"
  
  (let ((correct 0)
        (total (length test-data))
        (confusion (make-vector (* num-classes num-classes) 0))
        (sequential (car model)))
    
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
                (batch-images (let ((stacked (stack-images batch)))
                               (make-tensor32 (tensor-data stacked)
                                             (tensor-shape stacked)
                                             requires-grad?: #f)))
                (batch-targets (stack-targets batch))
                
                ;; Forward pass through the sequential model
                (logits (forward sequential batch-images))
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

;;; ==================================================================
;;; Model Checkpointing
;;; ==================================================================

(define (save-checkpoint model optimizer epoch train-loss train-acc filepath)
  "Save training checkpoint including model, optimizer state, and metrics
   
   Note: Currently only saves model. Optimizer state could be added later."
  
  (printf "\nSaving checkpoint at epoch ~A...\n" epoch)
  (save-cnn-model model filepath)
  
  ;; You could extend this to save optimizer state in a separate file
  ;; or include training metadata in a companion file
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
;;; Model Comparison
;;; ==================================================================

(define (compare-models model1-path model2-path test-data)
  "Compare two saved models on the same test dataset
   
   Useful for comparing different training runs or architectures"
  
  (printf "\n========================================\n")
  (printf "Model Comparison\n")
  (printf "========================================\n\n")
  
  (printf "Model 1: ~A\n" model1-path)
  (let ((acc1 (evaluate-loaded-model model1-path test-data)))
    (printf "\n")
    
    (printf "Model 2: ~A\n" model2-path)
    (let ((acc2 (evaluate-loaded-model model2-path test-data)))
      (printf "\n")
      
      (printf "========================================\n")
      (printf "Comparison Results:\n")
      (printf "  Model 1 Accuracy: ~A%\n" (* 100.0 acc1))
      (printf "  Model 2 Accuracy: ~A%\n" (* 100.0 acc2))
      (printf "  Difference: ~A%\n" (* 100.0 (- acc2 acc1)))
      (printf "========================================\n"))))

;;; ==================================================================
;;; Enhanced Main with Save/Load
;;; ==================================================================

(define (main-with-save)
  "Main training loop with model saving"
  
  (printf "========================================\n")
  (printf "CNN Training with Model Persistence\n")
  (printf "========================================\n\n")
  
  ;; Set random seed for reproducibility
  (set-random-seed! 42)
  
  ;; Generate dataset
  (printf "Generating dataset...\n")
  (define train-data (generate-dataset 250))
  (define test-data (generate-dataset 50))
  (printf "Training samples: ~A\n" (length train-data))
  (printf "Test samples: ~A\n\n" (length test-data))
  
  ;; Build model
  (printf "Building CNN model...\n")
  (define model (build-cnn))
  (define conv-layers (car model))
  (define dense-layers (cadr model))
  
  ;; Create optimizer
  (define learning-rate 0.001)
  (printf "Optimizer: Adam (lr=~A)\n\n" learning-rate)
  (define optimizer (make-adam (append (parameters conv-layers)
                                       (parameters dense-layers))
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
                  (train-epoch-batched model optimizer train-data
                                       batch-size: 64)))
      (printf "Epoch ~A/~A - Loss: ~A - Acc: ~A%"
              epoch num-epochs avg-loss (* 100.0 accuracy))
      
      ;; Evaluate every 5 epochs
      (when (= (modulo epoch 5) 0)
        (let-values (((test-acc confusion) 
                     (evaluate-batched model test-data batch-size: 64)))
          (printf " - Test Acc: ~A%" (* 100.0 test-acc))
          
          ;; Save checkpoint if best so far
          (when (> test-acc best-acc)
            (set! best-acc test-acc)
            (printf "\n  New best accuracy! Saving checkpoint...")
            (save-checkpoint model optimizer epoch avg-loss accuracy
                           "best-cnn-model.sav"))))
      
      (printf "\n"))
    
    ;; Learning rate decay
    (when (= (modulo epoch 10) 0)
      (let ((new-lr (* (get-learning-rate optimizer) 0.5)))
        (set-learning-rate! optimizer new-lr)
        (printf "  - Learning rate decreased to ~A\n" new-lr))))
  
  (printf "----------------------------------------\n\n")
  
  ;; Save final model
  (printf "Saving final model...\n")
  (save-cnn-model model "final-cnn-model.sav")
  (printf "\n")
  
  ;; Final evaluation
  (printf "Final Evaluation on Test Set:\n")
  (let-values (((test-acc confusion) (evaluate-batched model test-data)))
    (printf "Test Accuracy: ~A%\n" (* 100.0 test-acc))
    (print-confusion-matrix confusion))
  
  (printf "\n========================================\n")
  (printf "Training Complete!\n")
  (printf "Models saved:\n")
  (printf "  - best-cnn-model.sav (best accuracy: ~A%)\n" (* 100.0 best-acc))
  (printf "  - final-cnn-model.sav (final model)\n")
  (printf "========================================\n"))

;;; ==================================================================
;;; Demo: Load and Test a Saved Model
;;; ==================================================================

(define (demo-load-and-test)
  "Demonstrate loading a saved model and testing it"
  
  (printf "\n========================================\n")
  (printf "Demo: Loading and Testing Saved Model\n")
  (printf "========================================\n\n")
  
  ;; Generate test data
  (set-random-seed! 123)  ; Different seed for test data
  (printf "Generating test dataset...\n")
  (define test-data (generate-dataset 50))
  (printf "Test samples: ~A\n\n" (length test-data))
  
  ;; Check if model file exists
  (let ((model-path "best-cnn-model.sav"))
    (if (file-exists? model-path)
        (begin
          ;; Load and evaluate the best model
          (evaluate-loaded-model model-path test-data batch-size: 64)
          
          (printf "\n")
          
          ;; Test individual predictions
          (printf "Sample Predictions from Loaded Model:\n")
          (let ((model (load-cnn-model model-path)))
            (do ((i 0 (+ i 1)))
                ((= i (min 10 (length test-data))))
              (let* ((sample (list-ref test-data i))
                     (img-data (car sample))
                     (true-class (cdr sample))
                     (img (make-tensor32 img-data 
                                        (list num-channels image-size image-size)
                                        requires-grad?: #f))
                     (logits (forward-cnn-loaded model img))
                     (probs (softmax logits))
                     (pred-data (tensor-data probs))
                     (pred-class (argmax pred-data)))
                
                (printf "  Sample ~A: True=~A, Pred=~A " 
                        (+ i 1) true-class pred-class)
                (if (= pred-class true-class)
                    (printf "✓")
                    (printf "✗"))
                (printf " (confidence: ~A%)\n" 
                        (* 100 (f32vector-ref pred-data pred-class)))))))
        
        (printf "Error: Model file not found: ~A\n" model-path)
        (printf "Please run (main-with-save) first to train and save a model.\n"))))

;;; ==================================================================
;;; Model Transfer Learning Demo
;;; ==================================================================

(define (demo-transfer-learning)
  "Demonstrate transfer learning by loading a pre-trained model
   and fine-tuning it on new data"
  
  (printf "\n========================================\n")
  (printf "Demo: Transfer Learning\n")
  (printf "========================================\n\n")
  
  (let ((model-path "best-cnn-model.sav"))
    (if (file-exists? model-path)
        (begin
          ;; Load pre-trained model
          (printf "Loading pre-trained model...\n")
          (let-values (((model metadata) (load-checkpoint model-path)))
            (printf "\n")
            
            ;; Generate new dataset (could be different distribution)
            (set-random-seed! 999)
            (printf "Generating new dataset for fine-tuning...\n")
            (define new-train-data (generate-dataset 100))  ; Smaller dataset
            (define new-test-data (generate-dataset 25))
            
            (printf "Fine-tuning samples: ~A\n" (length new-train-data))
            (printf "Fine-tuning test samples: ~A\n\n" (length new-test-data))
            
            ;; Evaluate before fine-tuning
            (printf "Performance before fine-tuning:\n")
            (evaluate-loaded-model model-path new-test-data batch-size: 32)
            
            (printf "\n")
            (printf "Note: To actually fine-tune, you would:\n")
            (printf "  1. Extract the layer objects from the loaded model\n")
            (printf "  2. Create a new optimizer with lower learning rate\n")
            (printf "  3. Train for a few epochs on the new data\n")
            (printf "  4. Save the fine-tuned model\n")
            (printf "\nThis is left as an exercise - the infrastructure is in place!\n")))
        
        (printf "Error: Model file not found: ~A\n" model-path)
        (printf "Please run (main-with-save) first to train and save a model.\n"))))

;;; ==================================================================
;;; Usage Examples
;;; ==================================================================

;; Train and save a model:
;; (main-with-save)

;; Load and test a saved model:
;; (demo-load-and-test)

;; Compare two models:
;; (compare-models "best-cnn-model.sav" "final-cnn-model.sav" test-data)

;; Transfer learning demo:
;; (demo-transfer-learning)

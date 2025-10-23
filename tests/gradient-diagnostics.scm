;; Examples of gradient health monitoring in training loops

(import scheme
        (chicken base)
        (chicken format)
        (srfi 1)
        (srfi 4)
        nanograd-autograd
        nanograd-layer
        nanograd-optimizer
        nanograd-diagnostics)

;;; ==================================================================
;;; Example 1: Basic Gradient Monitoring
;;; ==================================================================

(define (example-basic-monitoring)
  (printf "\n=== Basic Gradient Monitoring ===\n\n")
  
  (let* ((model (make-sequential
                 (list
                  (make-dense-layer 10 20 activation: (make-relu))
                  (make-dense-layer 20 10 activation: (make-relu))
                  (make-dense-layer 10 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params learning-rate: 0.01))
         (monitor (make-gradient-monitor 
                  exploding-threshold: 10.0
                  vanishing-threshold: 1e-7)))
    
    ;; Training data
    (let ((x (make-tensor32 (make-f32vector 10 1.0) '(10)))
          (target (make-tensor32 (f32vector 5.0) '(1))))
      
      ;; Training loop with monitoring
      (do ((step 0 (+ step 1)))
          ((= step 20))
        
        ;; Forward and backward
        (let* ((output (forward model x))
               (loss (mse-loss output target)))
          (backward! loss)
          
          ;; Check gradient health
          (record-step! monitor step params)
          
          ;; Update parameters
          (step! opt)
          (zero-grad-layer! model)))
      
      ;; Print diagnosis
      (printf "\nTraining Diagnosis:\n")
      (let ((diagnosis (diagnose-training monitor)))
        (printf "Total steps: ~A\n" (cdr (assoc 'total-steps diagnosis)))
        (printf "Mean gradient norm: ~A\n" (cdr (assoc 'mean-gradient-norm diagnosis)))
        (printf "Unhealthy steps: ~A\n" (cdr (assoc 'unhealthy-steps diagnosis)))
        (printf "\nRecommendations:\n")
        (for-each
         (lambda (rec) (printf "  - ~A\n" rec))
         (cdr (assoc 'recommendations diagnosis)))))))

;;; ==================================================================
;;; Training with Gradient Clipping
;;; ==================================================================

(define (example-gradient-clipping)
  (printf "\n=== Training with Gradient Clipping ===\n\n")
  
  (let* ((model (make-sequential
                 (list
                  (make-dense-layer 5 10 activation: (make-relu))
                  (make-dense-layer 10 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-sgd params learning-rate: 0.1))  ; High LR to cause issues
         (monitor (make-gradient-monitor)))
    
    (let ((x (make-tensor32 (f32vector 1.0 2.0 3.0 4.0 5.0) '(5)))
          (target (make-tensor32 (f32vector 10.0) '(1))))
      
      (printf "Training with gradient clipping...\n\n")
      
      (do ((step 0 (+ step 1)))
          ((= step 30))
        
        ;; Forward and backward
        (let* ((output (forward model x))
               (loss (mse-loss output target))
               (loss-val (f32vector-ref (tensor-data loss) 0)))
          
          (backward! loss)
          
          ;; Check for exploding gradients
          (let ((health (check-gradient-health params warn?: #f)))
            (when (not (cdr (assoc 'healthy? health)))
              (printf "Step ~A: Issues detected - ~A\n" 
                      step (cdr (assoc 'issues health)))))
          
          ;; Apply gradient clipping
          (clip-gradients! params max-norm: 1.0)
          
          ;; Record for monitoring
          (record-step! monitor step params)
          
          ;; Update
          (step! opt)
          (zero-grad-layer! model)
          
          (when (zero? (modulo step 10))
            (printf "Step ~A: Loss = ~A\n" step loss-val))))
      
      ;; Print final diagnosis
      (printf "\n")
      (let ((diagnosis (diagnose-training monitor)))
        (printf "Final gradient statistics:\n")
        (printf "  Mean norm: ~A\n" (cdr (assoc 'mean-gradient-norm diagnosis)))
        (printf "  Max norm: ~A\n" (cdr (assoc 'max-gradient-norm diagnosis)))
        (printf "  Warnings: ~A\n" (cdr (assoc 'warning-count diagnosis)))))))

;;; ==================================================================
;;; Adaptive Learning Rate Based on Gradients
;;; ==================================================================

(define (example-adaptive-learning-rate)
  (printf "\n=== Adaptive Learning Rate ===\n\n")
  
  (let* ((model (make-sequential
                 (list
                  (make-dense-layer 3 10 activation: (make-relu))
                  (make-dense-layer 10 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-sgd params learning-rate: 1.0))  ; Start with high LR
         (monitor (make-gradient-monitor)))
    
    (let ((training-data (list
                          (cons (f32vector 1.0 0.0 0.0) 1.0)
                          (cons (f32vector 0.0 1.0 0.0) 2.0)
                          (cons (f32vector 0.0 0.0 1.0) 3.0))))
      
      (printf "Training with adaptive learning rate adjustment...\n\n")
      
      (do ((epoch 0 (+ epoch 1)))
          ((= epoch 10))
        
        (printf "Epoch ~A (LR: ~A):\n" epoch (get-learning-rate opt))
        
        (let ((epoch-losses '()))
          (for-each
           (lambda (sample)
             (let* ((x (make-tensor32 (car sample) '(3)))
                    (target (make-tensor32 (f32vector (cdr sample)) '(1)))
                    (output (forward model x))
                    (loss (mse-loss output target))
                    (loss-val (f32vector-ref (tensor-data loss) 0)))
               
               (set! epoch-losses (cons loss-val epoch-losses))
               
               (backward! loss)
               
               ;; Check gradient health
               (let* ((grad-norm (compute-gradient-norm params))
                      (health (check-gradient-health params warn?: #f)))
                 
                 ;; Adjust learning rate if needed
                 (cond
                  ((has-exploding-gradients? params threshold: 5.0)
                   (printf "  Exploding gradients (norm: ~A)\n" grad-norm)
                   (let ((new-lr (* (get-learning-rate opt) 0.5)))
                     (set-learning-rate! opt new-lr)
                     (printf "  Reducing learning rate to ~A\n" new-lr)))
                  
                  ((has-vanishing-gradients? params threshold: 1e-5)
                   (printf "  Vanishing gradients (norm: ~A)\n" grad-norm)
                   (let ((new-lr (* (get-learning-rate opt) 1.5)))
                     (set-learning-rate! opt new-lr)
                     (printf "  Increasing learning rate to ~A\n" new-lr)))))
               
               ;; Clip gradients as safety measure
               (clip-gradients! params max-norm: 5.0)
               
               (step! opt)
               (zero-grad-layer! model)))
           training-data)
          
          (printf "  Average loss: ~A\n\n" 
                  (/ (apply + epoch-losses) (length epoch-losses)))))))
  )

;;; ==================================================================
;;; Layer-wise Gradient Analysis
;;; ==================================================================

(define (example-layerwise-analysis)
  (printf "\n=== Layer-wise Gradient Analysis ===\n\n")
  
  (let* ((model (make-sequential
                 (list
                  (make-dense-layer 10 20 activation: (make-relu) name: "L1")
                  (make-dense-layer 20 20 activation: (make-relu) name: "L2")
                  (make-dense-layer 20 10 activation: (make-relu) name: "L3")
                  (make-dense-layer 10 1 activation: (make-identity) name: "Output"))))
         (params (parameters model))
         (opt (make-adam params learning-rate: 0.01)))
    
    (let ((x (make-tensor32 (make-f32vector 10 0.5) '(10)))
          (target (make-tensor32 (f32vector 5.0) '(1))))
      
      ;; Train for a few steps
      (do ((step 0 (+ step 1)))
          ((= step 5))
        (let* ((output (forward model x))
               (loss (mse-loss output target)))
          (backward! loss)
          (step! opt)
          (zero-grad-layer! model)))
      
      ;; Analyze gradients layer by layer
      (printf "Layer-wise gradient analysis:\n\n")
      (let* ((output (forward model x))
             (loss (mse-loss output target)))
        (backward! loss)
        
        (let ((stats (gradient-stats params)))
          (printf "Global gradient norm: ~A\n\n" 
                  (cdr (assoc 'global-l2-norm stats)))
          
          (printf "Per-layer breakdown:\n")
          (for-each
           (lambda (layer-stat)
             (printf "  Layer ~A:\n" (cdr (assoc 'layer layer-stat)))
             (printf "    L2 norm:    ~A\n" (cdr (assoc 'l2-norm layer-stat)))
             (printf "    Inf norm:   ~A\n" (cdr (assoc 'inf-norm layer-stat)))
             (printf "    Mean |grad|: ~A\n" (cdr (assoc 'mean-abs layer-stat))))
           (cdr (assoc 'layer-stats stats)))
          
          ;; Identify problematic layers
          (printf "\n")
          (let* ((layer-stats (cdr (assoc 'layer-stats stats)))
                 (norms (map (lambda (s) (cdr (assoc 'l2-norm s))) layer-stats))
                 (max-norm (apply max norms))
                 (min-norm (apply min norms)))
            
            (when (and (> min-norm 0) (> (/ max-norm min-norm) 10.0))
              (printf "WARNING: Large gradient imbalance detected!\n")
              (printf "  Largest layer gradient: ~A\n" max-norm)
              (printf "  Smallest layer gradient: ~A\n" min-norm)
              (printf "  Ratio: ~A:1\n" (/ max-norm min-norm))
              (printf "\n  Consider:\n")
              (printf "    - Batch normalization between layers\n")
              (printf "    - Skip connections (ResNet-style)\n")
              (printf "    - Layer normalization\n"))))))))

;;; ==================================================================
;;; Complete Training with Full Monitoring
;;; ==================================================================

(define (example-complete-training)
  (printf "\n=== Complete Training with Monitoring ===\n\n")
  
  (let* ((model (make-sequential
                 (list
                  (make-dense-layer 4 16 activation: (make-relu))
                  (make-dense-layer 16 8 activation: (make-relu))
                  (make-dense-layer 8 1 activation: (make-identity)))))
         (params (parameters model))
         (opt (make-adam params learning-rate: 0.01))
         (monitor (make-gradient-monitor 
                  exploding-threshold: 5.0
                  vanishing-threshold: 1e-6
                  history-size: 50)))
    
    ;; Training data: learn XOR-like function
    (let ((training-data (list
                          (cons (f32vector 0.0 0.0 0.0 0.0) 0.0)
                          (cons (f32vector 1.0 0.0 0.0 0.0) 1.0)
                          (cons (f32vector 0.0 1.0 0.0 0.0) 1.0)
                          (cons (f32vector 1.0 1.0 0.0 0.0) 0.0)
                          (cons (f32vector 0.0 0.0 1.0 0.0) 2.0)
                          (cons (f32vector 0.0 0.0 0.0 1.0) 3.0))))
      
      (printf "Starting training with full gradient monitoring...\n\n")
      
      (do ((epoch 0 (+ epoch 1)))
          ((= epoch 50))
        
        (let ((epoch-losses '())
              (gradient-norms '()))
          
          (for-each
           (lambda (sample)
             (let* ((x (make-tensor32 (car sample) '(4)))
                    (target (make-tensor32 (f32vector (cdr sample)) '(1)))
                    (output (forward model x))
                    (loss (mse-loss output target))
                    (loss-val (f32vector-ref (tensor-data loss) 0)))
               
               (set! epoch-losses (cons loss-val epoch-losses))
               
               (backward! loss)
               
               ;; Monitor gradients
               (let* ((health (record-step! monitor epoch params))
                      (grad-norm (compute-gradient-norm params)))
                 
                 (set! gradient-norms (cons grad-norm gradient-norms))
                 
                 ;; Apply remediation if needed
                 (when (not (cdr (assoc 'healthy? health)))
                   (clip-gradients! params max-norm: 5.0)))
               
               (step! opt)
               (zero-grad-layer! model)))
           training-data)
          
          ;; Print epoch summary
          (when (zero? (modulo epoch 10))
            (printf "Epoch ~A:\n" epoch)
            (printf "  Loss: ~A\n" 
                    (/ (apply + epoch-losses) (length epoch-losses)))
            (printf "  Gradient norm: ~A\n" 
                    (/ (apply + gradient-norms) (length gradient-norms))))))
      
      ;; Final report
      
      (let ((diagnosis (diagnose-training monitor)))
        (printf "Training Statistics:\n")
        (printf "  Total steps: ~A\n" (cdr (assoc 'total-steps diagnosis)))
        (printf "  Mean gradient norm: ~A\n" 
                (cdr (assoc 'mean-gradient-norm diagnosis)))
        (printf "  Max gradient norm: ~A\n" 
                (cdr (assoc 'max-gradient-norm diagnosis)))
        (printf "  Gradient stability: ~A\n" 
                (cdr (assoc 'gradient-stability diagnosis)))
        (printf "  Unhealthy steps: ~A\n" 
                (cdr (assoc 'unhealthy-steps diagnosis)))
        (printf "  Warning count: ~A\n" 
                (cdr (assoc 'warning-count diagnosis)))
        
        (printf "\nRecommendations:\n")
        (for-each
         (lambda (rec) (printf "  ~A\n" rec))
         (cdr (assoc 'recommendations diagnosis))))
      
      ;; Print final gradient report
      (let* ((x (make-tensor32 (f32vector 1.0 0.0 0.0 0.0) '(4)))
             (target (make-tensor32 (f32vector 1.0) '(1)))
             (output (forward model x))
             (loss (mse-loss output target)))
        (backward! loss)
        (print-gradient-report params "Final State")))))

;;; ==================================================================
;;; All examples
;;; ==================================================================

(define (run-all-gradient-examples)
  
  (example-basic-monitoring)
  (example-gradient-clipping)
  (example-adaptive-learning-rate)
  (example-layerwise-analysis)
  (example-complete-training)
  
  (printf "\n")
  (printf "All gradient monitoring examples completed.\n")
  (printf "\n"))

;; Run examples
(run-all-gradient-examples)

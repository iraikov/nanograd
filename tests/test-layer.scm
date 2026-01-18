;; Unit tests for layer operations

(import scheme
        (chicken base)
        (chicken format)
        (chicken random)
        (srfi 1)
        (srfi 4)
        test
        blas
        yasos
        nanograd-autograd
        nanograd-layer)

;;; ==================================================================
;;; Helper Functions
;;; ==================================================================

(define (approx-equal? actual expected tolerance)
  "Check if two numbers are approximately equal within tolerance"
  (<= (abs (- actual expected)) tolerance))

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

(define (in-range? val min-val max-val)
  "Check if value is in range [min-val, max-val]"
  (and (>= val min-val) (<= val max-val)))

(define-syntax test-approximate
  (syntax-rules ()
    ((test-approximate name expected actual tolerance)
     (test-assert name (approx-equal? actual expected tolerance)))))

;;; ==================================================================
;;; Unit Tests: Activation Functions as Objects
;;; ==================================================================

(test-group "Activation Function Objects"
  
  (test-group "ReLU"
    (let* ((relu-act (make-relu))
           (x (make-tensor32 (f32vector -1.0 0.0 1.0) '(3)))
           (y (activation-forward relu-act x)))
      
      (test-assert "ReLU is activation" (activation? relu-act))
      
      (test-approximate "ReLU(-1) = 0"
        0.0
        (f32vector-ref (tensor-data y) 0)
        1e-5)
      
      (test-approximate "ReLU(1) = 1"
        1.0
        (f32vector-ref (tensor-data y) 2)
        1e-5)))
  
  (test-group "Sigmoid"
    (let* ((sig-act (make-sigmoid))
           (x (make-tensor32 (f32vector 0.0) '(1)))
           (y (activation-forward sig-act x)))
      
      (test-assert "Sigmoid is activation" (activation? sig-act))
      
      (test-approximate "Sigmoid(0) = 0.5"
        0.5
        (f32vector-ref (tensor-data y) 0)
        1e-5)))
  
  (test-group "Identity"
    (let* ((id-act (make-identity))
           (x (make-tensor32 (f32vector 5.0) '(1)))
           (y (activation-forward id-act x)))
      
      (test-approximate "Identity(5) = 5"
        5.0
        (f32vector-ref (tensor-data y) 0)
        1e-5))))

;;; ==================================================================
;;; Unit Tests: Dense Layer
;;; ==================================================================

(test-group "Dense Layer Construction"
  
  (let ((layer (make-dense-layer 10 5 activation: (make-relu))))
    
    (test-assert "Is a layer" (layer? layer))
    (test-assert "Is a dense layer" (dense-layer? layer))
    
    (test "Input size = 10"
      10
      (layer-input-size layer))
    
    (test "Output size = 5"
      5
      (layer-output-size layer))
    
    (test-group "Parameters"
      (let ((params (parameters layer)))
        
        (test "Has 2 parameters (W and b)"
          2
          (length params))
        
        (test "Weight shape (5x10)"
          '(5 10)
          (tensor-shape (car params)))
        
        (test "Bias shape (5)"
          '(5)
          (tensor-shape (cadr params)))))))

(test-group "Dense Layer Forward Pass"
  
  (let* ((layer (make-dense-layer 2 3 activation: (make-identity)))
         (params (parameters layer))
         (weights (car params))
         (biases (cadr params)))
    
    ;; Set known weight values
    (let ((w-data (tensor-data weights)))
      (f32vector-set! w-data 0 1.0)
      (f32vector-set! w-data 1 2.0)
      (f32vector-set! w-data 2 3.0)
      (f32vector-set! w-data 3 4.0)
      (f32vector-set! w-data 4 5.0)
      (f32vector-set! w-data 5 6.0))
    
    ;; Set known bias values
    (let ((b-data (tensor-data biases)))
      (f32vector-set! b-data 0 0.1)
      (f32vector-set! b-data 1 0.2)
      (f32vector-set! b-data 2 0.3))
    
    (let* ((input (make-tensor32 (f32vector 1.0 2.0) '(2)))
           (output (forward layer input)))
      
      (test "Output shape"
        '(3)
        (tensor-shape output))
      
      (test-approximate "Output[0] = 5.1"
        5.1
        (f32vector-ref (tensor-data output) 0)
        1e-4)
      
      (test-approximate "Output[1] = 11.2"
        11.2
        (f32vector-ref (tensor-data output) 1)
        1e-4)
      
      (test-approximate "Output[2] = 17.3"
        17.3
        (f32vector-ref (tensor-data output) 2)
        1e-4))))

(test-group "Dense Layer Gradients"
  
  (let* ((layer (make-dense-layer 2 3 activation: (make-identity)))
         (input (make-tensor32 (f32vector 1.0 2.0) '(2)))
         (output (forward layer input))
         (target (make-tensor32 (f32vector 0.0 0.0 0.0) '(3)))
         (loss (mse-loss output target)))
    
    (backward! loss)
    
    (let ((params (parameters layer)))
      (test-assert "Weight gradients computed"
        (not (equal? (tensor-grad (car params)) #f)))
      
      (test-assert "Bias gradients computed"
        (not (equal? (tensor-grad (cadr params)) #f))))))

(test-group "Dense Layer Dimensions"
  
  (let ((dense (make-dense-layer 512 256 dtype: 'f32)))
    
    (test "layer-input-size returns feature dimension"
      512
      (layer-input-size dense))
    
    (test "layer-output-size returns feature dimension"
      256
      (layer-output-size dense))
    
    (test-group "1D input"
      (let* ((input-1d (make-tensor32 (make-f32vector 512 1.0) '(512)))
             (output-1d (forward dense input-1d)))
        
        (test "Output shape matches layer-output-size"
          '(256)
          (tensor-shape output-1d))))
    
    (test-group "2D input (batch)"
      (let* ((input-2d (make-tensor32 (make-f32vector (* 128 512) 1.0) '(128 512)))
             (output-2d (forward dense input-2d)))
        
        (test "Batch dimension preserved"
          '(128 256)
          (tensor-shape output-2d))
        
        (test "Output feature dim matches layer-output-size"
          256
          (cadr (tensor-shape output-2d)))))))

;;; ==================================================================
;;; Unit Tests: Sequential Container
;;; ==================================================================

(test-group "Sequential Container"
  
  (test-group "Two-layer network"
    (let* ((net (make-sequential
                 (list
                  (make-dense-layer 4 8 activation: (make-relu))
                  (make-dense-layer 8 2 activation: (make-identity)))))
           (input (make-tensor32 (f32vector 1.0 2.0 3.0 4.0) '(4)))
           (output (forward net input)))
      
      (test-assert "Sequential is a layer" (layer? net))
      (test-assert "Is sequential" (sequential? net))
      
      (test "Input size = 4"
        4
        (layer-input-size net))
      
      (test "Output size = 2"
        2
        (layer-output-size net))
      
      (test "Output shape correct"
        '(2)
        (tensor-shape output))
      
      (test "Has 4 parameters (2 layers × 2)"
        4
        (length (parameters net)))))
  
  (test-group "Deep network"
    (let* ((deep-net (make-sequential
                      (list
                       (make-dense-layer 5 10 activation: (make-relu))
                       (make-dense-layer 10 10 activation: (make-relu))
                       (make-dense-layer 10 3 activation: (make-identity)))))
           (input (make-tensor32 (make-f32vector 5 1.0) '(5)))
           (output (forward deep-net input)))
      
      (test "Deep net output shape"
        '(3)
        (tensor-shape output)))))

;;; ==================================================================
;;; Unit Tests: Conv2D Layer
;;; ==================================================================

(test-group "Conv2D Layer Construction"
  
  (let ((layer (make-conv2d-layer 3 16 3 
                                  stride: 1 
                                  padding: 1
                                  activation: (make-relu))))
    
    (test-assert "Is a layer" (layer? layer))
    (test-assert "Is a conv2d layer" (conv2d-layer? layer))
    
    (test "Input channels = 3"
      3
      (layer-input-size layer))
    
    (test "Output channels = 16"
      16
      (layer-output-size layer))
    
    (test-group "Parameters"
      (let ((params (parameters layer)))
        
        (test "Has 2 parameters"
          2
          (length params))
        
        (test "Weight shape"
          '(16 3 3 3)
          (tensor-shape (car params)))
        
        (test "Bias shape"
          '(16)
          (tensor-shape (cadr params)))))))

(test-group "Conv2D Layer Forward Pass"
  
  (test "Basic forward pass with padding"
    '(8 4 4)
    (let* ((layer (make-conv2d-layer 1 8 3 stride: 1 padding: 1))
           (input (make-tensor32 (make-f32vector 16 0.5) '(1 4 4)))
           (output (forward layer input)))
      (tensor-shape output)))
  
  (test "Forward pass with stride"
    '(4 3 3)
    (let* ((layer (make-conv2d-layer 1 4 3 stride: 2 padding: 0))
           (input (make-tensor32 (make-f32vector 64 1.0) '(1 8 8)))
           (output (forward layer input)))
      (tensor-shape output))))

(test-group "Conv2D Layer Gradients"
  
  (let* ((layer (make-conv2d-layer 1 4 3 stride: 1 padding: 0))
         (input (make-tensor32 (make-f32vector 16 1.0) '(1 4 4)))
         (output (forward layer input))
         (loss (dot-op (flatten-tensor output) (flatten-tensor output))))
    
    (backward! loss)
    
    (let ((params (parameters layer)))
      (test-assert "Weight gradients computed"
        (not (equal? (tensor-grad (car params)) #f)))
      
      (test-assert "Bias gradients computed"
        (not (equal? (tensor-grad (cadr params)) #f))))))

;;; ==================================================================
;;; Unit Tests: Training Loop
;;; ==================================================================

(test-group "Simple Training Loop"
  
  ;; Create simple linear model: y = 2x
  (let* ((model (make-sequential
                 (list
                  (make-dense-layer 1 1 activation: (make-identity)))))
         (training-data (list
                         (cons (f32vector 1.0) 2.0)
                         (cons (f32vector 2.0) 4.0)
                         (cons (f32vector 3.0) 6.0))))
    
    ;; Store initial loss
    (let ((initial-loss 1000.0))
      
      ;; Train for epochs
      (let loop ((epoch 0) (prev-loss initial-loss))
        (when (< epoch 50)
          (let ((total-loss 0.0))
            (for-each
             (lambda (sample)
               (let* ((x (make-tensor32 (car sample) '(1)))
                      (target (make-tensor32 (f32vector (cdr sample)) '(1)))
                      (pred (forward model x))
                      (loss (mse-loss pred target))
                      (loss-val (f32vector-ref (tensor-data loss) 0)))
                 
                 (set! total-loss (+ total-loss loss-val))
                 (backward! loss)
                 
                 ;; Simple gradient descent
                 (for-each
                  (lambda (param)
                    (let ((data (tensor-data param))
                          (grad (tensor-grad param))
                          (n (f32vector-length (tensor-data param))))
                      (do ((i 0 (+ i 1)))
                          ((= i n))
                        (f32vector-set! data i
                                       (- (f32vector-ref data i)
                                          (* 0.01 (f32vector-ref grad i)))))))
                  (parameters model))
                 
                 (zero-grad-layer! model)))
             training-data)
            
            (let ((avg-loss (/ total-loss (length training-data))))
              (when (= epoch 49)
                (test-assert "Loss decreases during training"
                  (< avg-loss prev-loss)))
              (loop (+ epoch 1) avg-loss))))))))

(test-group "Activation Comparison"
  
  (let ((input (make-tensor32 (f32vector -1.0 0.0 1.0 2.0) '(4))))
    
    (test-group "ReLU layer"
      (let* ((relu-layer (make-dense-layer 4 4 activation: (make-relu)))
             (output (forward relu-layer input))
             (data (tensor-data output)))
        
        (test-assert "ReLU output non-negative"
          (>= (f32vector-ref data 0) 0.0))))
    
    (test-group "Sigmoid layer"
      (let* ((sigmoid-layer (make-dense-layer 4 4 activation: (make-sigmoid)))
             (output (forward sigmoid-layer input))
             (data (tensor-data output)))
        
        (test-assert "Sigmoid output in [0,1]"
          (and (>= (f32vector-ref data 0) 0.0)
               (<= (f32vector-ref data 0) 1.0)))))))

;;; ==================================================================
;;; Unit Tests: Parameter Count
;;; ==================================================================

(test-group "Parameter Counting"
  
  (test "Dense layer parameter count"
    55
    (let* ((layer (make-dense-layer 10 5))
           (params (parameters layer))
           (total-params (fold
                          (lambda (p acc)
                            (let ((data (tensor-data p)))
                              (+ acc (f32vector-length data))))
                          0
                          params)))
      total-params))
  
  (test "Conv2D layer parameter count"
    224
    (let* ((layer (make-conv2d-layer 3 8 3))
           (params (parameters layer))
           (total-params (fold
                          (lambda (p acc)
                            (let ((data (tensor-data p)))
                              (+ acc (f32vector-length data))))
                          0
                          params)))
      total-params)))

;;; ==================================================================
;;; Unit Tests: Mixed Operations
;;; ==================================================================

(test-group "Conv2D to Dense Integration"
  
  (let* ((input (make-tensor32 (make-f32vector 64 1.0) '(1 8 8)))
         (conv-layer (make-conv2d-layer 1 4 3 stride: 2 padding: 1))
         (conv-out (forward conv-layer input))
         (flat (flatten-tensor conv-out))
         (dense-layer (make-dense-layer 64 10))
         (output (forward dense-layer flat)))
    
    (test "Conv output shape"
      '(4 4 4)
      (tensor-shape conv-out))
    
    (test "Flattened shape"
      '(64)
      (tensor-shape flat))
    
    (test "Final output shape"
      '(10)
      (tensor-shape output))
    
    (test-group "Gradient flow"
      (let ((loss (dot-op output output)))
        (backward! loss)
        
        (test-assert "Gradient flows through conv->flatten->dense"
          (not (equal? (tensor-grad input) #f)))))))

(test-group "Zero Gradient"
  
  (let* ((layer (make-dense-layer 3 2))
         (input (make-tensor32 (f32vector 1.0 2.0 3.0) '(3)))
         (output (forward layer input)))
    
    (backward! output)
    
    (let ((params (parameters layer)))
      
      (test-assert "Gradients exist before zero"
        (not (equal? (tensor-grad (car params)) #f)))
      
      (zero-grad-layer! layer)
      
      (test-approximate "Gradient zeroed"
        0.0
        (f32vector-ref (tensor-grad (car params)) 0)
        1e-10))))

;;; ==================================================================
;;; Batched MaxPool2D
;;; ==================================================================

(test-group "Batched MaxPool2D"
  
  (test-group "3D input (single image)"
    (let* ((input (make-tensor32 (f32vector 1.0 2.0 3.0 4.0
                                           5.0 6.0 7.0 8.0
                                           9.0 10.0 11.0 12.0
                                           13.0 14.0 15.0 16.0) '(1 4 4)))
           (output (maxpool2d input 2 stride: 2)))
      
      (test "Output shape"
        '(1 2 2)
        (tensor-shape output))
      
      (let ((data (tensor-data output)))
        (test-approximate "MaxPool window 1" 6.0 (f32vector-ref data 0) 1e-5)
        (test-approximate "MaxPool window 2" 8.0 (f32vector-ref data 1) 1e-5)
        (test-approximate "MaxPool window 3" 14.0 (f32vector-ref data 2) 1e-5)
        (test-approximate "MaxPool window 4" 16.0 (f32vector-ref data 3) 1e-5))))
  
  (test "4D input (batch of 2)"
    '(2 1 2 2)
    (let* ((input (make-tensor32 (make-f32vector (* 2 1 4 4) 1.0) '(2 1 4 4)))
           (output (maxpool2d input 2 stride: 2)))
      (tensor-shape output)))
  
  (test "4D multi-channel"
    '(3 2 4 4)
    (let* ((input (make-tensor32 (make-f32vector (* 3 2 8 8) 2.0) '(3 2 8 8)))
           (output (maxpool2d input 2 stride: 2)))
      (tensor-shape output)))
  
  (test-group "Gradient flow"
    (test-assert "3D MaxPool2D gradient flows"
      (let* ((input (make-tensor32 (make-f32vector 16 1.0) '(1 4 4)))
             (output (maxpool2d input 2 stride: 2))
             (loss (dot-op (flatten-tensor output) (flatten-tensor output))))
        (backward! loss)
        (not (equal? (tensor-grad input) #f))))
    
    (test-assert "4D MaxPool2D gradient flows"
      (let* ((input (make-tensor32 (make-f32vector (* 2 1 4 4) 1.0) '(2 1 4 4)))
             (output (maxpool2d input 2 stride: 2))
             (loss (dot-op (flatten-tensor output) (flatten-tensor output))))
        (backward! loss)
        (not (equal? (tensor-grad input) #f)))))
  
  (test "Different strides"
    '(1 1 6 6)
    (let* ((input (make-tensor32 (make-f32vector (* 1 1 8 8) 1.0) '(1 1 8 8)))
           (output (maxpool2d input 3 stride: 1)))
      (tensor-shape output))))

(test-group "Batched Conv2D + MaxPool2D"
  
  (let* ((batch-size 4)
         (input (make-tensor32 (make-f32vector (* batch-size 3 32 32) 1.0) 
                              (list batch-size 3 32 32)))
         (conv-layer (make-conv2d-layer 3 16 3 stride: 1 padding: 1))
         (conv-out (forward conv-layer input)))
    
    (test "Batched Conv output"
      (list batch-size 16 32 32)
      (tensor-shape conv-out))
    
    (let ((pool-out (maxpool2d conv-out 2 stride: 2)))
      (test "Batched MaxPool after Conv"
        (list batch-size 16 16 16)
        (tensor-shape pool-out))
      
      (test-assert "Gradient flows through Conv->MaxPool with batching"
        (let ((loss (dot-op (flatten-tensor pool-out) (flatten-tensor pool-out))))
          (backward! loss)
          (not (equal? (tensor-grad input) #f)))))))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(test-exit)

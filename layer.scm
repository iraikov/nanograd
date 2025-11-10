;; YASOS-based layer system for artificial neural networks

(module nanograd-layer
  (
   ;; Layer predicates and operations
   layer? dense-layer? sequential?
   
   ;; Layer construction
   make-dense-layer make-sequential
   make-conv2d-layer conv2d-layer?
   
   ;; Layer operations
   forward parameters zero-grad-layer!
   layer-input-size layer-output-size
   layer-activation layer-name layer-norm
   set-training-mode!
   set-eval-mode!
   
   ;; Batch Normalization
   make-batch-norm-2d
   batch-norm-2d?
   
   ;; Global Average Pooling
   global-avg-pool2d

   ;; Serialization operations
   layer->serializable serializable->layer
   save-layer load-layer
   save-model load-model
   
   ;; Activation functions (as objects)
   make-relu make-tanh make-sigmoid make-identity
   make-gelu make-silu
   activation? activation-name activation-forward

   ;; Network utilities
   print-layer summary
   )
  
  (import
   scheme
   (chicken base)
   (chicken format)
   (chicken random)
   (srfi 1)
   (srfi 4)
   yasos
   blas
   s11n
   nanograd-autograd
   )


  ;; Hygienic macro for dtype-based operation dispatch
  (include "with-dtype.scm")

  ;;; ==================================================================
  ;;; Activation Functions as YASOS Objects
  ;;; ==================================================================

  (define-predicate activation?)
  (define-operation (activation-forward act x))
  (define-operation (activation-name act))

  ;; ReLU Activation
  (define (make-relu)
    (object
     ((activation? self) #t)
     ((activation-name self) "ReLU")
     ((activation-forward self x) (relu x))))

  (define (make-tanh)
  (object
   ((activation? self) #t)
   ((activation-name self) "Tanh")
   ((activation-forward self x) (tanh-op x))))

  ;; Sigmoid activation
  (define (make-sigmoid)
    (object
     ((activation? self) #t)
     ((activation-name self) "Sigmoid")
     ((activation-forward self x) (sigmoid x))))

  ;; Identity Activation (no activation)
  (define (make-identity)
    (object
     ((activation? self) #t)
     ((activation-name self) "Identity")
     ((activation-forward self x) x)))

  ;; GeLU Activation
  (define (make-gelu)
    (object
     ((activation? self) #t)
     ((activation-name self) "GeLU")
     ((activation-forward self x) (gelu x))))

  ;; SiLU / Swish Activation
  (define (make-silu)
    (object
     ((activation? self) #t)
     ((activation-name self) "SiLU")
     ((activation-forward self x) (silu x))))

  ;;; ==================================================================
  ;;; Layer Base Operations
  ;;; ==================================================================

  (define-predicate layer?)
  (define-predicate dense-layer?)
  (define-predicate sequential?)
  
  (define-operation (forward layer input))
  (define-operation (parameters layer))
  (define-operation (zero-grad-layer! layer))
  (define-operation (layer-input-size layer))
  (define-operation (layer-output-size layer))
  (define-operation (layer-activation layer))
  (define-operation (layer-name layer))

  (define-operation (set-training-mode! layer training?))
  (define-operation (set-eval-mode! layer))

  ;; operations for layer serialization
  (define-operation (save-layer layer filepath))
  (define-operation (layer->serializable layer))

  ;;; ==================================================================
  ;;; Serialization/Deserialization Helpers
  ;;; ==================================================================

  ;; Tensor serialization (uses s11n for efficient storage)
  (define (tensor->serializable tensor)
    "Convert a tensor to a serializable representation"
    (let ((data (tensor-data tensor))
          (shape (tensor-shape tensor))
          (dtype (tensor-dtype tensor))
          (requires-grad (tensor-requires-grad? tensor)))
      
      `((dtype . ,dtype)
        (shape . ,shape)
        (requires-grad . ,requires-grad)
        (data . ,data))  ; handle SRFI-4 vectors with s11n
      ))

  (define (serializable->tensor serializable-tensor)
    "Reconstruct a tensor from serializable representation"
    (let* ((dtype (cdr (assq 'dtype serializable-tensor)))
           (shape (cdr (assq 'shape serializable-tensor)))
           (requires-grad (cdr (assq 'requires-grad serializable-tensor)))
           (data (cdr (assq 'data serializable-tensor))))
      (case dtype
        ((f32) (make-tensor32 data shape requires-grad: requires-grad))
        ((f64) (make-tensor64 data shape requires-grad: requires-grad))
        (else (error 'serializable->tensor 
                     (format #f "Unknown dtype: ~A" dtype))))))

  ;; Activation Function Serialization
  (define (activation->serializable act)
    "Convert an activation function to serializable representation"
    (let ((name (activation-name act)))
      `((type . activation)
        (name . ,name))))
  
  (define (serializable->activation serializable-act)
    "Reconstruct an activation function from serializable representation"
    (let ((name (cdr (assq 'name serializable-act))))
      (cond
       ((string=? name "ReLU") (make-relu))
       ((string=? name "Tanh") (make-tanh))
       ((string=? name "Sigmoid") (make-sigmoid))
       ((string=? name "Identity") (make-identity))
       ((string=? name "GeLU") (make-gelu))
       ((string=? name "SiLU") (make-silu))
       (else (error 'serializable->activation 
                    (format #f "Unknown activation function: ~A" name))))))

  ;;; ==================================================================
  ;;; Layer Deserialization with Dimension Checking
  ;;; ==================================================================

  (define (check-dimension-match expected actual context)
    "Verify that dimensions match, error if not"
    (unless (= expected actual)
      (error 'dimension-mismatch
             (format #f "~A: expected ~A but got ~A" 
                     context expected actual))))

  (define (serializable->layer serializable-repr)
    "Reconstruct a layer from its serializable representation with dimension checking"
    (let ((layer-type (cdr (assq 'type serializable-repr))))
      (cond
       ;; Dense Layer Deserialization
       ((eq? layer-type 'dense-layer)
        (let* ((name (cdr (assq 'name serializable-repr)))
               (input-size (cdr (assq 'input-size serializable-repr)))
               (output-size (cdr (assq 'output-size serializable-repr)))
               (dtype (cdr (assq 'dtype serializable-repr)))
               (weights-ser (cdr (assq 'weights serializable-repr)))
               (biases-ser (cdr (assq 'biases serializable-repr)))
               (activation-ser (cdr (assq 'activation serializable-repr)))
               
               ;; Deserialize tensors
               (weights (serializable->tensor weights-ser))
               (biases (serializable->tensor biases-ser))
               (activation (serializable->activation activation-ser))
               
               ;; Check dimensions
               (weight-shape (tensor-shape weights))
               (bias-shape (tensor-shape biases)))
          
          ;; Validate weight dimensions
          (check-dimension-match output-size (car weight-shape)
                                "Dense layer weight rows")
          (check-dimension-match input-size (cadr weight-shape)
                                "Dense layer weight columns")
          
          ;; Validate bias dimensions
          (check-dimension-match output-size (car bias-shape)
                                "Dense layer bias size")
          
          ;; Create layer with deserialized tensors
          (object
           ;; Type predicates
           ((layer? self) #t)
           ((dense-layer? self) #t)
           
           ;; Layer info
           ((layer-name self) name)
           ((layer-input-size self) input-size)
           ((layer-output-size self) output-size)
           ((layer-activation self) activation)
           
           ;; Forward pass
           ((forward self input)
            (let ((input-shape (tensor-shape input)))
              (unless (= (car input-shape) input-size)
                (error 'forward 
                       (format #f "Input size mismatch: expected ~A, got ~A"
                               input-size (car input-shape)))))
            
            (let* ((linear-output (matmul-op weights input))
                   (output-with-bias (add linear-output biases)))
              (activation-forward activation output-with-bias)))
           
           ;; Get all parameters
           ((parameters self)
            (list weights biases))
           
           ;; Zero gradients
           ((zero-grad-layer! self)
            (zero-grad! weights)
            (zero-grad! biases))

           ((layer->serializable self)
            `((type . dense-layer)
              (name . ,name)
              (input-size . ,input-size)
              (output-size . ,output-size)
              (dtype . ,dtype)
              (weights . ,(tensor->serializable weights))
              (biases . ,(tensor->serializable biases))
              (activation . ,(activation->serializable activation))))
           
           ((save-layer self filepath)
            (save-layer-to-file self filepath)))))
       
       ;; Conv2D Layer Deserialization
       ((eq? layer-type 'conv2d-layer)
        (let* ((name (cdr (assq 'name serializable-repr)))
               (in-channels (cdr (assq 'in-channels serializable-repr)))
               (out-channels (cdr (assq 'out-channels serializable-repr)))
               (kernel-size (cdr (assq 'kernel-size serializable-repr)))
               (dtype (cdr (assq 'dtype serializable-repr)))
               (weights-ser (cdr (assq 'weights serializable-repr)))
               (biases-ser (cdr (assq 'biases serializable-repr)))
               (activation-ser (cdr (assq 'activation serializable-repr)))
               
               ;; Deserialize tensors
               (weights (serializable->tensor weights-ser))
               (biases (serializable->tensor biases-ser))
               (activation (serializable->activation activation-ser))
               
               ;; Check dimensions
               (weight-shape (tensor-shape weights))
               (bias-shape (tensor-shape biases)))
          
          ;; Validate weight dimensions (out_channels, in_channels, KH, KW)
          (check-dimension-match out-channels (car weight-shape)
                                "Conv2D output channels")
          (check-dimension-match in-channels (cadr weight-shape)
                                "Conv2D input channels")
          (check-dimension-match kernel-size (caddr weight-shape)
                                "Conv2D kernel height")
          (check-dimension-match kernel-size (cadddr weight-shape)
                                "Conv2D kernel width")
          
          ;; Validate bias dimensions
          (check-dimension-match out-channels (car bias-shape)
                                "Conv2D bias size")
          
          ;; Create layer with deserialized tensors
          (object
           ;; Type predicates
           ((layer? self) #t)
           ((conv2d-layer? self) #t)
           
           ;; Layer info
           ((layer-name self) name)
           ((layer-input-size self) in-channels)
           ((layer-output-size self) out-channels)
           ((layer-activation self) activation)
           
           ;; Forward pass
           ((forward self input)
            (let ((ishape (tensor-shape input)))
              (unless (= (car ishape) in-channels)
                (error 'forward 
                       (format #f "Input channel mismatch: expected ~A, got ~A"
                               in-channels (car ishape)))))
            
            (let* ((conv-output (conv2d input weights biases
                                        stride: 1
                                        padding: 0)))
              (activation-forward activation conv-output)))
           
           ;; Get parameters
           ((parameters self)
            (list weights biases))
           
           ;; Zero gradients
           ((zero-grad-layer! self)
            (zero-grad! weights)
            (zero-grad! biases))

           ((layer->serializable self)
            `((type . conv2d-layer)
              (name . ,name)
              (in-channels . ,in-channels)
              (out-channels . ,out-channels)
              (kernel-size . ,kernel-size)
              (dtype . ,dtype)
              (weights . ,(tensor->serializable weights))
              (biases . ,(tensor->serializable biases))
              (activation . ,(activation->serializable activation))))

           ((save-layer self filepath)
            (save-layer-to-file self filepath)))))
       
       ;; Sequential Layer Deserialization
       ((eq? layer-type 'sequential)
        (let* ((name (cdr (assq 'name serializable-repr)))
               (layers-ser (cdr (assq 'layers serializable-repr)))
               
               ;; Recursively deserialize all layers
               (layers (map serializable->layer layers-ser)))
          
          ;; Verify layer connectivity (output of layer i matches input of layer i+1)
          (let loop ((remaining-layers layers))
            (when (>= (length remaining-layers) 2)
              (let ((curr-layer (car remaining-layers))
                    (next-layer (cadr remaining-layers)))
                (check-dimension-match 
                 (layer-output-size curr-layer)
                 (layer-input-size next-layer)
                 (format #f "Sequential layer connectivity between ~A and ~A"
                         (layer-name curr-layer)
                         (layer-name next-layer))))
              (loop (cdr remaining-layers))))
          
          ;; Create sequential layer
          (object
           ;; Type predicates
           ((layer? self) #t)
           ((sequential? self) #t)
           
           ;; Layer info
           ((layer-name self) name)
           ((layer-input-size self) 
            (if (null? layers)
                0
                (layer-input-size (car layers))))
           ((layer-output-size self)
            (if (null? layers)
                0
                (layer-output-size (last layers))))
           
           ;; Forward pass (chain through all layers)
           ((forward self input)
            (fold (lambda (layer x)
                    (forward layer x))
                  input
                  layers))
           
           ;; Get all parameters from all layers
           ((parameters self)
            (append-map parameters layers))
           
           ;; Zero gradients for all layers
           ((zero-grad-layer! self)
            (for-each zero-grad-layer! layers))
           
           ((layer->serializable self)
            `((type . sequential)
              (name . ,name)
              (layers . ,(map layer->serializable layers))))
           
           ((save-layer self filepath)
            (save-layer-to-file self filepath)))))
       
       (else
        (error 'serializable->layer 
               (format #f "Unknown layer type: ~A" layer-type))))))

  
  ;;; ==================================================================
  ;;; Dense (Fully Connected) Layer
  ;;; ==================================================================

  (define (make-dense-layer input-size output-size 
                           #!key 
                           (activation (make-identity))
                           (dtype 'f32)
                           (name "Dense"))
    (let* ((weight-size (* input-size output-size))
           ;; Xavier/Glorot initialization
           (init-scale (sqrt (/ 2.0 (+ input-size output-size))))
           
           ;; Initialize weights with small random values
           (weight-data (with-dtype dtype
                          (let ((w (vec weight-size 0.0)))
                            (do ((i 0 (+ i 1)))
                                ((= i weight-size) w)
                              (elt-set! w i
                                        (* init-scale
                                           (- (pseudo-random-real) 0.5)))))))
           
           ;; Initialize biases to zero
           (bias-data (with-dtype dtype (vec output-size 0.0)))
           
           ;; Create parameter tensors
           (weights (case dtype
                     ((f32) (make-tensor32 weight-data (list output-size input-size)))
                     ((f64) (make-tensor64 weight-data (list output-size input-size)))))
           
           (biases (case dtype
                    ((f32) (make-tensor32 bias-data (list output-size)))
                    ((f64) (make-tensor64 bias-data (list output-size))))))
      
      (object
       ;; Type predicates
       ((layer? self) #t)
       ((dense-layer? self) #t)
       
       ;; Layer info
       ((layer-name self) name)
       ((layer-input-size self) input-size)
       ((layer-output-size self) output-size)
       ((layer-activation self) activation)
       
       ;; Forward pass
       ((forward self input)
        ;; Check input size
        (let ((input-shape (tensor-shape input)))
          (unless (= (car input-shape) input-size)
            (error 'forward 
                   (format #f "Input size mismatch: expected ~A, got ~A"
                           input-size (car input-shape)))))
        
        ;; Linear transformation: output = W @ input + b
        (let* ((linear-output (matmul-op weights input))
               (output-with-bias (add linear-output biases)))
          ;; Apply activation function
          (activation-forward activation output-with-bias)))
       
       ;; Get all parameters (for optimizer)
       ((parameters self)
        (list weights biases))
       
       ;; Zero gradients
       ((zero-grad-layer! self)
        (zero-grad! weights)
        (zero-grad! biases))
       
       ((set-training-mode! self train?)
        (begin))
       
       ((set-eval-mode! self)
        (begin))

       ((layer->serializable self)
        `((type . dense-layer)
          (name . ,name)
          (input-size . ,input-size)
          (output-size . ,output-size)
          (dtype . ,dtype)
          (weights . ,(tensor->serializable weights))
          (biases . ,(tensor->serializable biases))
          (activation . ,(activation->serializable activation))))
       
       ((save-layer self filepath)
        (save-layer-to-file self filepath))

       ))
    )

  ;;; ==================================================================
  ;;; Sequential Container (chains layers)
  ;;; ==================================================================

  (define (make-sequential layers #!key (name "Sequential"))
    (let ((layer-list layers))
      (object
       ;; Type predicates
       ((layer? self) #t)
       ((sequential? self) #t)
       
       ;; Layer info
       ((layer-name self) name)
       ((layer-input-size self) 
        (if (null? layer-list)
            0
            (layer-input-size (car layer-list))))
       ((layer-output-size self)
        (if (null? layer-list)
            0
            (layer-output-size (last layer-list))))
       
       ;; Forward pass (chain through all layers)
       ((forward self input)
        (fold (lambda (layer x)
                (forward layer x))
              input
              layer-list))
       
       ;; Get all parameters from all layers
       ((parameters self)
        (append-map parameters layer-list))
       
       ;; Zero gradients for all layers
       ((zero-grad-layer! self)
        (for-each zero-grad-layer! layer-list))
       
       ;; Serialize sequential layer with all its sub-layers
       ((layer->serializable self)
        `((type . sequential)
          (name . ,name)
          (layers . ,(map layer->serializable layer-list))))
       
       ((save-layer self filepath)
        (save-layer-to-file self filepath)))))

  ;;; ==================================================================
  ;;; Layer Normalization
  ;;; ==================================================================
  
  (define (layer-norm x gamma beta #!key (epsilon 1e-5))
    ;; Normalize across features, scale by gamma, shift by beta
    (let* ((dtype (tensor-dtype x))
           (data-x (tensor-data x))
           (n (vector-length-for-dtype data-x dtype)))
      
      ;; Compute mean and variance
      (define (compute-stats)
        (let ((sum (with-dtype dtype
                               (let loop ((i 0) (sum 0.0))
                                 (if (= i n) sum
                                     (loop (+ i 1) (+ sum (elt-ref data-x i))))))))
          (let* ((mean (/ sum (exact->inexact n)))
                 (var-sum (with-dtype dtype
                                      (let loop ((i 0) (var-sum 0.0))
                                        (if (= i n) var-sum
                                            (let ((diff (- (elt-ref data-x i) mean)))
                                              (loop (+ i 1) (+ var-sum (* diff diff)))))))))
            (values mean (/ var-sum (exact->inexact n))))))
      
      (let-values (((mean variance) (compute-stats)))
        (let* ((std (sqrt (+ variance epsilon)))
               ;; Normalize, scale, shift
               (normalized
                (let ((norm-data (with-dtype dtype (vec n 0.0))))
                  (with-dtype dtype
                              (do ((i 0 (+ i 1)))
                                  ((= i n))
                                (elt-set! norm-data i
                                          (/ (- (elt-ref data-x i) mean) std))))
                  (make-base-tensor norm-data (tensor-shape x) dtype
                                    (tensor-requires-grad? x)))))
          
          ;; scaled = normalized * gamma
          (define scaled (mul normalized gamma))
          
          ;; output = scaled + beta
          (add scaled beta)))))


  ;;; ==================================================================
  ;;; Convolutional Layer
  ;;; ==================================================================

  (define-predicate conv2d-layer?)
  
  (define (make-conv2d-layer in-channels out-channels kernel-size
                             #!key
                             (stride 1)
                             (padding 0)
                             (activation (make-identity))
                             (dtype 'f32)
                             (name "Conv2D"))
    "Create a 2D convolutional layer"
  
  (let* ((KH kernel-size)
         (KW kernel-size)
         
         ;; He initialization for conv layers
         (fan-in (* in-channels KH KW))
         (init-scale (sqrt (/ 2.0 fan-in)))
         
         ;; Initialize weights: (out_channels, in_channels, KH, KW)
         (weight-size (* out-channels in-channels KH KW))
         (weight-data (with-dtype dtype
                        (let ((w (vec weight-size 0.0)))
                          (do ((i 0 (+ i 1)))
                              ((= i weight-size) w)
                            (elt-set! w i
                                      (* init-scale
                                         (- (* 2.0 (pseudo-random-real)) 1.0)))))))
         
         ;; Initialize biases
         (bias-data (with-dtype dtype (vec out-channels 0.0)))
         
         ;; Create parameter tensors
         (weights (case dtype
                   ((f32) (make-tensor32 weight-data 
                                        (list out-channels in-channels KH KW)))
                   ((f64) (make-tensor64 weight-data 
                                        (list out-channels in-channels KH KW)))))
         
         (biases (case dtype
                  ((f32) (make-tensor32 bias-data (list out-channels)))
                  ((f64) (make-tensor64 bias-data (list out-channels))))))
    
    (object
     ;; Type predicates
     ((layer? self) #t)
     ((conv2d-layer? self) #t)
     
     ;; Layer info
     ((layer-name self) name)
     ((layer-input-size self) in-channels)
     ((layer-output-size self) out-channels)
     ((layer-activation self) activation)
     
     ;; Forward pass
     ((forward self input)
      ;; Input should be (C, H, W)
      (let ((ishape (tensor-shape input)))
        (unless (= (car ishape) in-channels)
          (error 'forward 
                 (format #f "Input channel mismatch: expected ~A, got ~A"
                         in-channels (car ishape)))))
      ;; Apply convolution
      (let* ((conv-output (conv2d input weights biases
                                  stride: stride
                                  padding: padding
                                  )))
        ;; Apply activation
        (activation-forward activation conv-output)))
     
     ;; Get parameters
     ((parameters self)
      (list weights biases))
     
     ;; Zero gradients
     ((zero-grad-layer! self)
      (zero-grad! weights)
      (zero-grad! biases))

     ((set-training-mode! self train?)
      (begin))
     
     ((set-eval-mode! self)
      (begin))

     ((layer->serializable self)
      `((type . conv2d-layer)
        (name . ,name)
        (in-channels . ,in-channels)
        (out-channels . ,out-channels)
        (kernel-size . ,kernel-size)
        (dtype . ,dtype)
        (weights . ,(tensor->serializable weights))
        (biases . ,(tensor->serializable biases))
        (activation . ,(activation->serializable activation)))
      )

     ((save-layer self filepath)
      (save-layer-to-file self filepath))
     
     ))
  )


  
  ;; ==================================================================
  ;; MaxPool2D Layer
  ;; ==================================================================
  
  (define (maxpool2d input kernel-size #!key (stride #f))
    "2D max pooling operation.
   Input shape: (C, H, W)
   Output shape: (C, OH, OW)"
    
    (let* ((dtype (tensor-dtype input))
           (ishape (tensor-shape input))
           (C (car ishape))
           (H (cadr ishape))
           (W (caddr ishape))
           (data (tensor-data input))
           
           (KH kernel-size)
           (KW kernel-size)
           (stride-val (or stride kernel-size))
           
           ;; Output dimensions
           (OH (+ 1 (quotient (- H KH) stride-val)))
           (OW (+ 1 (quotient (- W KW) stride-val)))
           
           (output-data (with-dtype dtype (vec (* C OH OW) 0.0)))
           
           ;; Store indices for backward pass
           (max-indices (make-vector (* C OH OW))))
    
      ;; Forward: find max in each window
      (with-dtype
       dtype
       (do ((c 0 (+ c 1)))
           ((= c C))
         (do ((oh 0 (+ oh 1)))
             ((= oh OH))
           (do ((ow 0 (+ ow 1)))
               ((= ow OW))
             
             (let ((max-val -inf.0)
                   (max-idx 0))
               
               ;; Find max in kernel window
               (do ((kh 0 (+ kh 1)))
                   ((= kh KH))
                 (do ((kw 0 (+ kw 1)))
                     ((= kw KW))
                   
                   (let* ((ih (+ (* oh stride-val) kh))
                          (iw (+ (* ow stride-val) kw))
                          (input-idx (+ (* c H W) (* ih W) iw))
                          (val (elt-ref data input-idx)))
                     
                     (when (> val max-val)
                       (set! max-val val)
                       (set! max-idx input-idx)))))
              
               (let ((output-idx (+ (* c OH OW) (* oh OW) ow)))
                 (elt-set! output-data output-idx max-val)
                 (vector-set! max-indices output-idx max-idx)))))))
    
      (let ((result (make-base-tensor output-data 
                                      (list C OH OW)
                                      dtype
                                      (tensor-requires-grad? input))))
        
        (when (tensor-requires-grad? input)
          (set-backward-fn! result
                            (lambda ()
                              (let ((grad-out (tensor-grad result))
                                    (grad-in (with-dtype dtype (vec (* C H W) 0.0))))
                                
                                ;; Gradient flows only to max positions
                                (with-dtype
                                 dtype
                                 (do ((i 0 (+ i 1)))
                                     ((= i (* C OH OW)))
                                   (let ((max-pos (vector-ref max-indices i))
                                         (grad-val (elt-ref grad-out i)))
                                     (elt-set! grad-in max-pos
                                               (+ (elt-ref grad-in max-pos)
                                                  grad-val))))
                                 )
                                
                                (add-to-grad! input grad-in)))
                            (list input)))
        
        result)))



  ;;; ==================================================================
  ;;; Batch Normalization 2D
  ;;; ==================================================================
  
  (define-predicate batch-norm-2d?)
  
  (define (make-batch-norm-2d num-features 
                             #!key 
                             (epsilon 1e-5)
                             (momentum 0.1)
                             (dtype 'f32)
                             (name "BatchNorm2d"))
    "Batch Normalization for 2D convolutions.
     
     Normalizes activations across batch dimension:
     y = gamma * (x - mu) / sqrt(sigma^2 + epsilon) + beta
     
     Args:
       num-features: Number of channels (C)
       epsilon: Small constant for numerical stability
       momentum: Momentum for running statistics
       dtype: Data type
       
     Input shape: (C, H, W) or (N, C, H, W)
     Output shape: Same as input"
    
    (let* (;; Learnable parameters
           (gamma (case dtype
                   ((f32) (make-tensor32 (make-f32vector num-features 1.0)
                                        (list num-features)))
                   ((f64) (make-tensor64 (make-f64vector num-features 1.0)
                                        (list num-features)))))
           (beta (case dtype
                  ((f32) (make-tensor32 (make-f32vector num-features 0.0)
                                       (list num-features)))
                  ((f64) (make-tensor64 (make-f64vector num-features 0.0)
                                       (list num-features)))))
           
           ;; Running statistics (not trainable)
           (running-mean (with-dtype dtype (vec num-features 0.0)))
           (running-var (with-dtype dtype (vec num-features 1.0)))
           
           ;; Training mode flag
           (training? #t))
      
      (object
       ((layer? self) #t)
       ((batch-norm-2d? self) #t)
       ((layer-name self) name)
       
       ;; Mode control
       ((set-training-mode! self train?)
        (set! training? train?))
       
       ((set-eval-mode! self)
        (set! training? #f))
       
       ((forward self input)
        "Forward pass through batch normalization.
         
         During training: Uses batch statistics
         During eval: Uses running statistics"
        
        (let* ((input-shape (tensor-shape input))
               (C num-features)
               (H (if (= (length input-shape) 3)
                      (cadr input-shape)
                      (caddr input-shape)))
               (W (if (= (length input-shape) 3)
                      (caddr input-shape)
                      (cadddr input-shape)))
               (spatial-size (* H W))
               (input-data (tensor-data input)))
          
          (if training?
              ;; Training mode: compute batch statistics
              (let ((means (with-dtype dtype (vec C 0.0)))
                    (vars (with-dtype dtype (vec C 0.0))))

                ;; Compute mean for each channel
                (with-dtype dtype
                            (do ((c 0 (+ c 1)))
                                ((= c C))
                              (let ((sum (let loop ((sum 0.0) (i 0))
                                           (if (= i spatial-size)
                                               sum
                                               (loop (let ((idx (+ (* c spatial-size) i)))
                                                       (+ sum (elt-ref input-data idx)))
                                                     (+ i 1)))
                                           )))
                                (elt-set! means c (/ sum spatial-size)))
                              ))
                
                ;; Compute variance for each channel
                (with-dtype dtype
                            (do ((c 0 (+ c 1)))
                                ((= c C))
                              (let* ((mean (elt-ref means c))
                                     (sum-sq (let loop ((sum-sq 0.0) (i 0))
                                               (if (= i spatial-size)
                                                   sum-sq
                                                   (loop (let* ((idx (+ (* c spatial-size) i))
                                                                (val (elt-ref input-data idx))
                                                                (diff (- val mean)))
                                                           (+ sum-sq (* diff diff)))
                                                         (+ i 1)))
                                               )))
                                (elt-set! vars c (/ sum-sq spatial-size)))
                              ))
                
                ;; Update running statistics
                (do ((c 0 (+ c 1)))
                    ((= c C))
                  (with-dtype dtype
                    (let ((new-mean (elt-ref means c))
                          (new-var (elt-ref vars c))
                          (old-mean (elt-ref running-mean c))
                          (old-var (elt-ref running-var c)))
                      (elt-set! running-mean c
                               (+ (* (- 1.0 momentum) old-mean)
                                  (* momentum new-mean)))
                      (elt-set! running-var c
                               (+ (* (- 1.0 momentum) old-var)
                                  (* momentum new-var))))))
                
                ;; Normalize using batch statistics
                (let ((normalized-data (with-dtype dtype (vec (* C spatial-size) 0.0))))
                  (do ((c 0 (+ c 1)))
                      ((= c C))
                    (let ((mean (with-dtype dtype (elt-ref means c)))
                          (var (with-dtype dtype (elt-ref vars c)))
                          (gamma-val (with-dtype dtype 
                                       (elt-ref (tensor-data gamma) c)))
                          (beta-val (with-dtype dtype
                                      (elt-ref (tensor-data beta) c))))
                      (let ((std (sqrt (+ var epsilon))))
                        (do ((i 0 (+ i 1)))
                            ((= i spatial-size))
                          (let ((idx (+ (* c spatial-size) i)))
                            (with-dtype dtype
                              (let ((normalized (/ (- (elt-ref input-data idx) mean)
                                                  std)))
                                (elt-set! normalized-data idx
                                         (+ (* gamma-val normalized) beta-val)))))))))
                  
                  (make-base-tensor normalized-data input-shape dtype
                                   (tensor-requires-grad? input))))
              
              ;; Eval mode: use running statistics
              (let ((normalized-data (with-dtype dtype (vec (* C spatial-size) 0.0))))
                (with-dtype dtype
                            (do ((c 0 (+ c 1)))
                                ((= c C))
                              (let ((mean (elt-ref running-mean c))
                                    (var (elt-ref running-var c))
                                    (gamma-val (elt-ref (tensor-data gamma) c))
                                    (beta-val (elt-ref (tensor-data beta) c)))
                                (let ((std (sqrt (+ var epsilon))))
                                  (do ((i 0 (+ i 1)))
                                      ((= i spatial-size))
                                    (let ((idx (+ (* c spatial-size) i)))
                                      (let ((normalized (/ (- (elt-ref input-data idx) mean)
                                                           std)))
                                        (elt-set! normalized-data idx
                                                  (+ (* gamma-val normalized) beta-val))))
                                    ))
                                ))
                            )
                
                (make-base-tensor normalized-data input-shape dtype #f)))))
       
       ((parameters self)
        (list gamma beta))
       
       ((zero-grad-layer! self)
        (zero-grad! gamma)
        (zero-grad! beta)))))

  ;;; ==================================================================
  ;;; Global Average Pooling
  ;;; ==================================================================
  
  (define (global-avg-pool2d input)
    "Global average pooling over spatial dimensions.
     
     Input shape: (C, H, W)
     Output shape: (C,)"
    
    (let* ((dtype (tensor-dtype input))
           (shape (tensor-shape input))
           (C (car shape))
           (H (cadr shape))
           (W (caddr shape))
           (spatial-size (* H W))
           (data (tensor-data input))
           (output-data (with-dtype dtype (vec C 0.0))))
      
      ;; Average over spatial dimensions for each channel
      (with-dtype dtype
                  (do ((c 0 (+ c 1)))
                      ((= c C))
                    (let ((sum
                           (let loop ((sum 0.0) (i 0))
                             (if (< i spatial-size)
                                 (let ((idx (+ (* c spatial-size) i)))
                                   (loop (+ sum (elt-ref data idx))
                                         (+ i 1)))
                                 sum))))
                      (elt-set! output-data c (/ sum spatial-size)))))
      
      (let ((result (make-base-tensor output-data (list C) dtype
                                      (tensor-requires-grad? input))))
        
        ;; Backward pass
        (when (tensor-requires-grad? input)
          (set-backward-fn! result
            (lambda ()
              (let ((grad-out (tensor-grad result))
                    (grad-in (with-dtype dtype (vec (* C H W) 0.0))))
                
                ;; Distribute gradient equally over spatial dimensions
                (with-dtype dtype
                            (do ((c 0 (+ c 1)))
                                ((= c C))
                              (let* ((grad-val (elt-ref grad-out c))
                                     (grad-per-pixel (/ grad-val spatial-size)))
                                (do ((i 0 (+ i 1)))
                                    ((= i spatial-size))
                                  (let ((idx (+ (* c spatial-size) i)))
                                    (elt-set! grad-in idx grad-per-pixel))))))
                
                (add-to-grad! input grad-in)))
            (list input)))
        
        result)))
  
  ;;; ==================================================================
  ;;; Utilities
  ;;; ==================================================================

  ;; Print layer information
  (define (print-layer layer #!optional (indent 0))
    (let ((spaces (make-string indent #\space)))
      (printf "~A~A: " spaces (layer-name layer))
      (cond
       ((dense-layer? layer)
        (printf "Dense(~A → ~A, activation=~A)\n"
                (layer-input-size layer)
                (layer-output-size layer)
                (activation-name (layer-activation layer))))
       ((sequential? layer)
        (printf "Sequential\n")
        (let ((params (parameters layer)))
          (printf "~A  Total parameters: ~A\n" 
                  spaces
                  (fold (lambda (p acc)
                          (let ((data (tensor-data p)))
                            (+ acc (case (tensor-dtype p)
                                    ((f32) (f32vector-length data))
                                    ((f64) (f64vector-length data))))))
                        0
                        params))))
       (else
        (printf "Layer\n")))))

  ;; Print model summary
  (define (summary model)
    (printf "\n=== Model Summary ===\n")
    (printf "Model: ~A\n" (layer-name model))
    (printf "Input size: ~A\n" (layer-input-size model))
    (printf "Output size: ~A\n\n" (layer-output-size model))
    
    (cond
     ((sequential? model)
      (printf "Layers:\n")
      (let ((params (parameters model)))
        (for-each
         (lambda (layer) (print-tensor layer))
         params)
        ))
     
     (else
      (print-layer model)))
    
    (let ((params (parameters model)))
      (printf "\nTotal parameters: ~A\n"
              (fold (lambda (p acc)
                      (let ((data (tensor-data p)))
                        (+ acc (case (tensor-dtype p)
                                ((f32) (f32vector-length data))
                                ((f64) (f64vector-length data))))))
                    0
                    params)))
    (printf "===================\n\n"))

  ;;; ==================================================================
  ;;; File I/O Operations
  ;;; ==================================================================

  (define (save-layer-to-file layer filepath)
    "Save a layer to a file using s11n serialization"
    (let ((serializable (layer->serializable layer)))
      (with-output-to-file filepath
        (lambda ()
          (serialize serializable)))))

  (define (load-layer-from-file filepath)
    "Load a layer from a file using s11n deserialization"
    (let ((serializable (with-input-from-file filepath
                          (lambda ()
                            (deserialize)))))
      (serializable->layer serializable)))

  ;; Public API for layer save/load
  (define (save-layer layer filepath)
    "Public API: Save a layer to file"
    (save-layer-to-file layer filepath))
  
  (define (load-layer filepath)
    "Public API: Load a layer from file"
    (load-layer-from-file filepath))

  ;; Model save/load (alias for sequential models)
  (define (save-model model filepath)
    "Save a model (sequential or single layer) to file"
    (save-layer-to-file model filepath))
  
  (define (load-model filepath)
    "Load a model from file"
    (load-layer-from-file filepath))
  
) ;; end module

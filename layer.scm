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
   layer->serializable
   save-layer load-layer
   
   ;; Activation functions (as objects)
   make-relu make-tanh make-sigmoid make-identity
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

  ;; Replace the make-sigmoid implementation with:
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

  ;; operations for layer serialization
  (define-operation (save-layer layer filepath))
  (define-operation (load-layer filepath))
  (define-operation (layer->serializable layer))

  ;; Helpers for tensor serialization
  (define (tensor->serializable tensor)
    "Convert a tensor to a serializable representation"
    (let ((data (tensor-data tensor))
          (shape (tensor-shape tensor))
          (dtype (tensor-dtype tensor))
          (requires-grad (tensor-requires-grad? tensor)))
      
      `((dtype . ,dtype)
        (shape . ,shape)
        (requires-grad . ,requires-grad)
        (data . ,data))
      ))

  (define (serializable->tensor serializable-tensor)
    "Reconstruct a tensor from serializable representation"
    (let* ((dtype (cdr (assq 'dtype serializable-tensor)))
           (shape (cdr (assq 'shape serializable-tensor)))
           (requires-grad (cdr (assq 'requires-grad serializable-tensor)))
           (data (cdr (assq 'data serializable-tensor))))
      (case dtype
        ((f32) (make-tensor32 data shape requires-grad: requires-grad))
        ((f64) (make-tensor64 data shape requires-grad: requires-grad)))))

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
       (else (error 'serializable->activation 
                    (format #f "Unknown activation function: ~A" name))))))

  
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
           (weight-data (case dtype
                         ((f32)
                          (let ((vec (make-f32vector weight-size 0.0)))
                            (do ((i 0 (+ i 1)))
                                ((= i weight-size) vec)
                              (f32vector-set! vec i
                                             (* init-scale
                                                (- (pseudo-random-real) 0.5))))))
                         ((f64)
                          (let ((vec (make-f64vector weight-size 0.0)))
                            (do ((i 0 (+ i 1)))
                                ((= i weight-size) vec)
                              (f64vector-set! vec i
                                             (* init-scale
                                                (- (pseudo-random-real) 0.5))))))))
           
           ;; Initialize biases to zero
           (bias-data (case dtype
                       ((f32) (make-f32vector output-size 0.0))
                       ((f64) (make-f64vector output-size 0.0))))
           
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
        (for-each zero-grad-layer! layer-list)))))

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
        (let ((sum (case dtype
                     ((f32)
                      (let loop ((i 0) (sum 0.0))
                        (if (= i n) sum
                            (loop (+ i 1) (+ sum (f32vector-ref data-x i))))))
                     ((f64)
                      (let loop ((i 0) (sum 0.0))
                        (if (= i n) sum
                            (loop (+ i 1) (+ sum (f64vector-ref data-x i)))))))))
          (let* ((mean (/ sum (exact->inexact n)))
                 (var-sum (case dtype
                            ((f32)
                             (let loop ((i 0) (var-sum 0.0))
                               (if (= i n) var-sum
                                   (let ((diff (- (f32vector-ref data-x i) mean)))
                                     (loop (+ i 1) (+ var-sum (* diff diff)))))))
                            ((f64)
                             (let loop ((i 0) (var-sum 0.0))
                               (if (= i n) var-sum
                                   (let ((diff (- (f64vector-ref data-x i) mean)))
                                     (loop (+ i 1) (+ var-sum (* diff diff))))))))))
            (values mean (/ var-sum (exact->inexact n))))))
      
      (let-values (((mean variance) (compute-stats)))
        (let ((std (sqrt (+ variance epsilon))))
          ;; Normalize, scale, shift
          (define normalized
            (let ((norm-data (case dtype
                              ((f32) (make-f32vector n 0.0))
                              ((f64) (make-f64vector n 0.0)))))
              (case dtype
                ((f32)
                 (do ((i 0 (+ i 1)))
                     ((= i n))
                   (f32vector-set! norm-data i
                                  (/ (- (f32vector-ref data-x i) mean) std))))
                ((f64)
                 (do ((i 0 (+ i 1)))
                     ((= i n))
                   (f64vector-set! norm-data i
                                  (/ (- (f64vector-ref data-x i) mean) std)))))
              (make-base-tensor norm-data (tensor-shape x) dtype
                              (tensor-requires-grad? x))))
          
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
         (weight-data (case dtype
                       ((f32)
                        (let ((vec (make-f32vector weight-size 0.0)))
                          (do ((i 0 (+ i 1)))
                              ((= i weight-size) vec)
                            (f32vector-set! vec i
                                           (* init-scale
                                              (- (* 2.0 (pseudo-random-real)) 1.0))))))
                       ((f64)
                        (let ((vec (make-f64vector weight-size 0.0)))
                          (do ((i 0 (+ i 1)))
                              ((= i weight-size) vec)
                            (f64vector-set! vec i
                                           (* init-scale
                                              (- (* 2.0 (pseudo-random-real)) 1.0))))))))
         
         ;; Initialize biases
         (bias-data (case dtype
                     ((f32) (make-f32vector out-channels 0.0))
                     ((f64) (make-f64vector out-channels 0.0))))
         
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
                                  debug: (string=? name "Conv2"))))
        ;; Apply activation
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
           
           (output-data (case dtype
                          ((f32) (make-f32vector (* C OH OW) 0.0))
                          ((f64) (make-f64vector (* C OH OW) 0.0))))
           
           ;; Store indices for backward pass
           (max-indices (make-vector (* C OH OW))))
    
      ;; Forward: find max in each window
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
                         (val (case dtype
                                ((f32) (f32vector-ref data input-idx))
                                ((f64) (f64vector-ref data input-idx)))))
                    
                    (when (> val max-val)
                      (set! max-val val)
                      (set! max-idx input-idx)))))
              
              (let ((output-idx (+ (* c OH OW) (* oh OW) ow)))
                (case dtype
                  ((f32) (f32vector-set! output-data output-idx max-val))
                  ((f64) (f64vector-set! output-data output-idx max-val)))
                (vector-set! max-indices output-idx max-idx))))))
    
      (let ((result (make-base-tensor output-data 
                                      (list C OH OW)
                                      dtype
                                      (tensor-requires-grad? input))))
        
        (when (tensor-requires-grad? input)
          (set-backward-fn! result
                            (lambda ()
                              (let ((grad-out (tensor-grad result))
                                    (grad-in (case dtype
                                               ((f32) (make-f32vector (* C H W) 0.0))
                                               ((f64) (make-f64vector (* C H W) 0.0)))))
                                
                                ;; Gradient flows only to max positions
                                (do ((i 0 (+ i 1)))
                                    ((= i (* C OH OW)))
                                  (let ((max-pos (vector-ref max-indices i))
                                        (grad-val (case dtype
                                                    ((f32) (f32vector-ref grad-out i))
                                                    ((f64) (f64vector-ref grad-out i)))))
                                    (case dtype
                                      ((f32) (f32vector-set! grad-in max-pos
                                                             (+ (f32vector-ref grad-in max-pos)
                                                                grad-val)))
                                      ((f64) (f64vector-set! grad-in max-pos
                                                             (+ (f64vector-ref grad-in max-pos)
                                                                grad-val))))))
                                
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

  ;; File I/O Operations

  (define (save-layer-to-file layer filepath)
    "Save a layer to a file using s11n serialization"
    (let ((serializable (layer->serializable layer)))
      (with-output-to-file filepath
        (lambda ()
          (serialize serializable)))))

  #;(define (load-layer-from-file filepath)
    "Load a layer from a file using s11n deserialization"
    (let ((serializable (with-input-from-file filepath
                          (lambda ()
                            (deserialize)))))
      (serializable->layer serializable)))

  
  
) ;; end module

;; YASOS-based layer system for artificial neural networks

(module nanograd-layer
  (
   ;; Layer predicates and operations
   layer? dense-layer? sequential? flatten-layer?
   
   ;; Layer construction
   make-dense-layer make-sequential make-flatten
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
   maxpool2d
   
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
  (define-predicate flatten-layer?)
  
  (define-operation (forward layer . rest))
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

  (define (verify-flatten-transition before-layer flatten-layer after-layer)
    "Verify that conv->flatten->dense transition is valid"
    (let ((conv-output (layer-output-size before-layer))
          (dense-input (layer-input-size after-layer)))
    
      (cond
       ;; If both are known, we can verify the flattening math
       ((and conv-output dense-input)
        ;; For conv layers, output size is num channels
        ;; Dense expects flattened size = channels * height * width
        ;; We need to check if dense-input is a multiple of conv-output
        (unless (and (> dense-input conv-output)
                     (zero? (modulo dense-input conv-output)))
          (printf "Warning: Flatten dimensions may be incompatible: ~A channels -> ~A features\n"
                  conv-output dense-input)))
       
       ;; If either is unknown, we can't verify
       (else
        (void)))))
  
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
               (stride (cdr (assq 'stride serializable-repr)))
               (padding (cdr (assq 'padding serializable-repr)))
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

          (make-conv2d-layer in-channels out-channels kernel-size
                             stride: stride
                             padding: padding
                             activation: activation
                             dtype: dtype
                             name: name
                             weight-values: (tensor-data weights)
                             bias-values: (tensor-data biases))
          ))
       
       ;; Sequential Layer Deserialization
       ((eq? layer-type 'sequential)
        (let* ((name (cdr (assq 'name serializable-repr)))
               (layers-ser (cdr (assq 'layers serializable-repr)))
               
               ;; Recursively deserialize all layers
               (layers (map serializable->layer layers-ser)))

          ;; Verify layer connectivity (allowing for dynamic dimensions)
          (let loop ((remaining-layers layers))
            (when (>= (length remaining-layers) 2)
              (let ((curr-layer (car remaining-layers))
                    (next-layer (cadr remaining-layers)))
                
                (let ((curr-output (layer-output-size curr-layer))
                      (next-input (layer-input-size next-layer)))
                  
                  ;; Only verify when both dimensions are statically known
                  (cond
                   ((and curr-output next-input)
                    (check-dimension-match 
                     curr-output
                     next-input
                     (format #f "Sequential layer connectivity between ~A and ~A"
                             (layer-name curr-layer)
                             (layer-name next-layer))))
                   ;; Flatten in the middle - check if flattening makes sense
                   ((flatten-layer? next-layer)
                    (when (>= (length remaining-layers) 3)
                      (let ((after-flatten (caddr remaining-layers)))
                        (verify-flatten-transition curr-layer next-layer after-flatten))))
                   ))
              
                (loop (cdr remaining-layers)))
              ))
          (make-sequential layers name: name)
          ))

       ((eq? layer-type 'flatten-layer)
        (let ((name (cdr (assq 'name serializable-repr))))
          (make-flatten name: name)))
       
       (else
        (error 'serializable->layer 
               (format #f "Unknown layer type: ~A" layer-type))))))

  
  ;;; ==================================================================
  ;;; Dense (Fully Connected) Layer
  ;;; ==================================================================

  ;; Efficient dense layer that handles both 1D and 2D inputs
  ;; Uses BLAS GEMM for batch operations
  ;; Replace the make-dense-layer in layer.scm with this corrected version
;; The key fix: use the correct GEMM operation to get (batch-size, output-dim) layout

(define (make-dense-layer input-dim output-dim 
                          #!key 
                          (activation (make-identity))
                          (use-bias #t)
                          (dtype 'f32)
                          (name "Dense"))
  "Dense layer supporting both single vectors and batches.
   
   Input shapes:
   1D: (input_dim,) -> output: (output_dim,)
   2D: (batch_size, input_dim) -> output: (batch_size, output_dim)"

  (let* ((weight-data (with-dtype dtype
                                  (let ((w (vec (* output-dim input-dim) 0.0)))
                                    ;; Initialize with small random values
                                    (do ((i 0 (+ i 1)))
                                        ((= i (* output-dim input-dim)) w)
                                      (elt-set! w i 
                                                (* (sqrt (/ 2.0 input-dim))
                                                   (- (pseudo-random-real) 0.5)))))))
         (weight (case dtype
                   ((f32) (make-tensor32 weight-data (list output-dim input-dim)))
                   ((f64) (make-tensor64 weight-data (list output-dim input-dim)))))
       
         (bias (if use-bias
                   (case dtype
                     ((f32) (make-tensor32 (make-f32vector output-dim 0.0)
                                           (list output-dim)))
                     ((f64) (make-tensor64 (make-f64vector output-dim 0.0)
                                           (list output-dim))))
                   #f)))
  
    (object
     
     ((layer? self) #t)
     ((layer-name self) name)
     ((dense-layer? self) #t)

     ((layer-input-size self) input-dim)
     ((layer-output-size self) output-dim)
     ((layer-activation self) activation)
     
     ((forward self input)
      (let* ((input-shape (tensor-shape input))
             (ndim (length input-shape))
             (requires-grad? (or (tensor-requires-grad? input)
                                (tensor-requires-grad? weight)
                                (and bias (tensor-requires-grad? bias)))))
        
        (cond
         ;; Case 1: 1D input (single vector)
         ((= ndim 1)
          (let ((in-dim (car input-shape)))
            (unless (= in-dim input-dim)
              (error 'forward 
                     (format #f "Input size mismatch: expected ~A, got ~A"
                             input-dim in-dim)))
            
            ;; Standard matrix-vector multiplication
            (let ((output (matmul-op weight input)))
              (if bias
                  (activation-forward activation (add output bias))
                  (activation-forward activation output)))))
         
         ;; Case 2: 2D input (batch of vectors)
         ((= ndim 2)
          (let ((batch-size (car input-shape))
                (in-dim (cadr input-shape)))
            (unless (= in-dim input-dim)
              (error 'forward
                     (format #f "Input size mismatch: expected ~A, got ~A"
                             input-dim in-dim)))
            
            ;; Compute output = X @ W^T to get (batch-size, output-dim) directly
            ;; X: (batch-size, input-dim)
            ;; W: (output-dim, input-dim) - will be transposed to (input-dim, output-dim)
            ;; Result: (batch-size, output-dim)
            (let ((output-data (with-dtype dtype 
                                           (vec (* batch-size output-dim) 0.0))))
              
              (with-dtype dtype
                (gemm! RowMajor NoTrans Trans
                       batch-size output-dim input-dim  ; m, n, k
                       1.0 (tensor-data input) (tensor-data weight)
                       0.0 output-data
                       lda: input-dim     ; Leading dimension of X
                       ldb: input-dim     ; Leading dimension of W (before transpose)
                       ldc: output-dim))  ; Leading dimension of output
            
              (let ((output (make-base-tensor output-data 
                                              (list batch-size output-dim)
                                              dtype requires-grad?)))
              
                (when requires-grad?
                  (set-backward-fn!
                   output
                   (lambda ()
                     (let ((grad-out (tensor-grad output))
                           (data-weight (tensor-data weight))
                           (data-input (tensor-data input)))
                       
                       ;; Gradient w.r.t. input: grad_X = grad_out @ W
                       ;; grad_out: (batch-size, output-dim)
                       ;; W: (output-dim, input-dim)
                       ;; Result: (batch-size, input-dim)
                       (when (tensor-requires-grad? input)
                         (let ((grad-input (with-dtype dtype 
                                                       (vec (* batch-size input-dim) 0.0))))
                           (with-dtype dtype
                             (gemm! RowMajor NoTrans NoTrans
                                    batch-size input-dim output-dim  ; m, n, k
                                    1.0 grad-out data-weight
                                    0.0 grad-input
                                    lda: output-dim   ; Leading dim of grad-out
                                    ldb: input-dim    ; Leading dim of W
                                    ldc: input-dim))  ; Leading dim of grad-input
                           (add-to-grad! input grad-input)))
                       
                       ;; Gradient w.r.t. weight: grad_W = grad_out^T @ X
                       ;; grad_out^T: (output-dim, batch-size)
                       ;; X: (batch-size, input-dim)
                       ;; Result: (output-dim, input-dim)
                       (when (tensor-requires-grad? weight)
                         (let ((grad-weight (with-dtype dtype 
                                                        (vec (* output-dim input-dim) 0.0))))
                           (with-dtype dtype
                             (gemm! RowMajor Trans NoTrans
                                    output-dim input-dim batch-size  ; m, n, k
                                    1.0 grad-out data-input
                                    0.0 grad-weight
                                    lda: output-dim   ; Leading dim of grad-out (before transpose)
                                    ldb: input-dim    ; Leading dim of X
                                    ldc: input-dim))  ; Leading dim of grad-weight
                           (add-to-grad! weight grad-weight)))))
                   (list input weight)))
              
                ;; Add bias if present (broadcast across batch)
                (let ((output-with-bias
                       (if bias
                           (let ((bias-data (tensor-data bias))
                                 (biased-data (with-dtype dtype 
                                                          (vec (* batch-size output-dim) 0.0))))
                             ;; Copy output data
                             (with-dtype dtype
                               (copy-to biased-data output-data 
                                        size: (* batch-size output-dim)))
                             
                             ;; Add bias to each row
                             (with-dtype dtype
                               (do ((i 0 (+ i 1)))
                                   ((= i batch-size))
                                 (let ((row-offset (* i output-dim)))
                                   (axpy! output-dim 1.0 bias-data
                                          biased-data offsetY: row-offset))))
                             
                             (let ((biased-output (make-base-tensor biased-data
                                                                    (list batch-size output-dim)
                                                                    dtype requires-grad?)))
                               (when requires-grad?
                                 (set-backward-fn!
                                  biased-output
                                  (lambda ()
                                    (let ((grad-biased (tensor-grad biased-output)))
                                      ;; Gradient flows back to pre-bias output
                                      (when (tensor-requires-grad? output)
                                        (add-to-grad! output grad-biased))
                                      
                                      ;; Gradient w.r.t. bias: sum across batch dimension
                                      (when (and bias (tensor-requires-grad? bias))
                                        (let ((grad-bias (with-dtype dtype (vec output-dim 0.0))))
                                          (with-dtype dtype
                                            (do ((i 0 (+ i 1)))
                                                ((= i batch-size))
                                              (let ((row-offset (* i output-dim)))
                                                (axpy! output-dim 1.0 grad-biased
                                                       grad-bias offsetX: row-offset))))
                                          (add-to-grad! bias grad-bias)))))
                                  (list output bias)))
                               biased-output))
                           output)))
                  
                  ;; Apply activation
                  (activation-forward activation output-with-bias))))))
          
          (else
           (error 'forward 
                  (format #f "Dense layer only supports 1D or 2D inputs, got ~AD"
                          ndim))))))
   
      ((parameters self)
       (if bias
           (list weight bias)
           (list weight)))
      
      ((zero-grad-layer! self)
       (zero-grad! weight)
       (when bias (zero-grad! bias)))

      ((set-training-mode! self train?)
       (begin))
     
      ((set-eval-mode! self)
       (begin))

      ((layer->serializable self)
       `((type . dense-layer)
         (name . ,name)
         (input-size . ,input-dim)
         (output-size . ,output-dim)
         (dtype . ,dtype)
         (weights . ,(tensor->serializable weight))
         (biases . ,(if use-bias (tensor->serializable bias) #f))
         (activation . ,(activation->serializable activation))))
     
      ((save-layer self filepath)
       (save-layer-to-file self filepath))
      )))
  
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


  (define (make-flatten #!key (name "Flatten"))
    "Flatten layer: converts (N, C, H, W) -> (N, C*H*W) or (C, H, W) -> (C*H*W)"
    (object
     ((layer? self) #t)
     ((flatten-layer? self) #t)
     ((layer-name self) name)
     
     ((layer-input-size self) #f)   ; Unknown until runtime
     ((layer-output-size self) #f)  ; Unknown until runtime
     
     ((forward self input)
      (let* ((shape (tensor-shape input))
             (ndim (length shape)))
        (cond
         ;; 4D: (N, C, H, W) -> (N, C*H*W)
         ((= ndim 4)
          (let ((N (car shape))
                (flattened-size (apply * (cdr shape))))
            (reshape input (list N flattened-size))))
         
         ;; 3D: (C, H, W) -> (C*H*W)
         ((= ndim 3)
          (let ((flattened-size (apply * shape)))
            (reshape input (list flattened-size))))
         
         ;; Already flat
         ((= ndim 2) input)
         ((= ndim 1) input)
         
         (else
          (error 'flatten (format #f "Cannot flatten ~AD tensor" ndim))))))
   
     ((parameters self) '())  ; No parameters
     ((zero-grad-layer! self) (void))
     ((set-training-mode! self train?) (void))
     ((set-eval-mode! self) (void))
     
     ((layer->serializable self)
      `((type . flatten-layer)
        (name . ,name)))
     
     ((save-layer self filepath)
      (save-layer-to-file self filepath))))
  
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
                             (weight-values #f)
                             (bias-values #f)
                             (name "Conv2D"))
    "2D convolutional layer with batch support
   
     Input shapes:
       3D: (C, H, W) - single image
       4D: (N, C, H, W) - batched images"

    (let* ((KH kernel-size)
           (KW kernel-size)
           
           ;; He initialization for conv layers
           (fan-in (* in-channels KH KW))
           (init-scale (sqrt (/ 2.0 fan-in)))
           
           ;; Initialize weights: (out_channels, in_channels, KH, KW)
           (weight-size (* out-channels in-channels KH KW))
           (weight-data (or weight-values
                             (with-dtype
                              dtype
                              (let ((w (vec weight-size 0.0)))
                                (do ((i 0 (+ i 1)))
                                    ((= i weight-size) w)
                                  (elt-set! w i
                                            (* init-scale
                                               (- (* 2.0 (pseudo-random-real)) 1.0))))))))
           
           ;; Initialize biases
           (bias-data (or bias-values
                          (with-dtype dtype (vec out-channels 0.0))))
           
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
       
       ;; Forward pass with proper batch support
       ((forward self input)
        (let* ((ishape (tensor-shape input))
               (ndim (length ishape)))
          
          ;; Check input dimensions and validate channel count
          (cond
           ;; 3D input: (C, H, W)
           ((= ndim 3)
            (unless (= (car ishape) in-channels)
              (error 'forward 
                     (format #f "Input channel mismatch: expected ~A, got ~A"
                             in-channels (car ishape)))))
           
           ;; 4D input: (N, C, H, W)
           ((= ndim 4)
            (unless (= (cadr ishape) in-channels)
              (error 'forward 
                     (format #f "Input channel mismatch: expected ~A, got ~A (batch shape: ~A)"
                             in-channels (cadr ishape) ishape))))
           
           (else
            (error 'forward 
                   (format #f "Conv2D expects 3D (C,H,W) or 4D (N,C,H,W) input, got ~AD" 
                           ndim))))
          
          ;; Apply convolution (conv2d handles both 3D and 4D)
          (let ((conv-output (conv2d input weights biases
                                     stride: stride
                                     padding: padding)))
            ;; Apply activation
            (activation-forward activation conv-output))))
       
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
          (stride . ,stride)
          (padding . ,padding)
          (dtype . ,dtype)
          (weights . ,(tensor->serializable weights))
          (biases . ,(tensor->serializable biases))
          (activation . ,(activation->serializable activation))))
       
       ((save-layer self filepath)
        (save-layer-to-file self filepath)))))


  
  ;; ==================================================================
  ;; MaxPool2D Layer
  ;; ==================================================================
  

  (define (maxpool2d input kernel-size #!key (stride #f))
    "2D max pooling operation with batch support.
   
     Input shapes:
       3D: (C, H, W) -> Output: (C, OH, OW)
       4D: (N, C, H, W) -> Output: (N, C, OH, OW)"
  
    (let* ((dtype (tensor-dtype input))
           (ishape (tensor-shape input))
           (ndim (length ishape))
           (KH kernel-size)
           (KW kernel-size)
           (stride-val (or stride kernel-size)))
      
      (cond
       ;; Case 1: 3D input (single image)
       ((= ndim 3)
        (let* ((C (car ishape))
               (H (cadr ishape))
               (W (caddr ishape))
               (data (tensor-data input))
               
               ;; Output dimensions
               (OH (+ 1 (quotient (- H KH) stride-val)))
               (OW (+ 1 (quotient (- W KW) stride-val)))
               
               (output-data (with-dtype dtype (vec (* C OH OW) 0.0)))
               
               ;; Store indices for backward pass
               (max-indices (make-vector (* C OH OW))))
        
          ;; Forward: find max in each window
          (with-dtype dtype
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
                                    (with-dtype dtype
                                                (do ((i 0 (+ i 1)))
                                                    ((= i (* C OH OW)))
                                                  (let ((max-pos (vector-ref max-indices i))
                                                        (grad-val (elt-ref grad-out i)))
                                                    (elt-set! grad-in max-pos
                                                              (+ (elt-ref grad-in max-pos)
                                                                 grad-val)))))
                                    
                                    (add-to-grad! input grad-in)))
                                (list input)))
            
            result)))
       
       ;; Case 2: 4D input (batched images)
       ((= ndim 4)
        (let* ((N (car ishape))
               (C (cadr ishape))
               (H (caddr ishape))
               (W (cadddr ishape))
               (data (tensor-data input))
               
               ;; Output dimensions
               (OH (+ 1 (quotient (- H KH) stride-val)))
               (OW (+ 1 (quotient (- W KW) stride-val)))
               
               (output-data (with-dtype dtype (vec (* N C OH OW) 0.0)))
               
               ;; Store indices for backward pass
               (max-indices (make-vector (* N C OH OW))))
          
          ;; Forward: find max in each window for each batch element
          (with-dtype dtype
                      (do ((n 0 (+ n 1)))
                          ((= n N))
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
                                           (input-idx (+ (* n C H W)
                                                         (* c H W)
                                                         (* ih W)
                                                         iw))
                                           (val (elt-ref data input-idx)))
                                      
                                      (when (> val max-val)
                                        (set! max-val val)
                                        (set! max-idx input-idx)))))
                                
                                (let ((output-idx (+ (* n C OH OW)
                                                     (* c OH OW)
                                                     (* oh OW)
                                                     ow)))
                                  (elt-set! output-data output-idx max-val)
                                  (vector-set! max-indices output-idx max-idx))))))))
          
          (let ((result (make-base-tensor output-data 
                                          (list N C OH OW)
                                          dtype
                                          (tensor-requires-grad? input))))
            
            (when (tensor-requires-grad? input)
              (set-backward-fn! result
                                (lambda ()
                                  (let ((grad-out (tensor-grad result))
                                        (grad-in (with-dtype dtype (vec (* N C H W) 0.0))))
                                    
                                    ;; Gradient flows only to max positions
                                    (with-dtype dtype
                                                (do ((i 0 (+ i 1)))
                                                    ((= i (* N C OH OW)))
                                                  (let ((max-pos (vector-ref max-indices i))
                                                        (grad-val (elt-ref grad-out i)))
                                                    (elt-set! grad-in max-pos
                                                              (+ (elt-ref grad-in max-pos)
                                                                 grad-val)))))
                                    
                                    (add-to-grad! input grad-in)))
                                (list input)))
            
            result)))
       
       (else
        (error 'maxpool2d 
               (format #f "MaxPool2D expects 3D (C,H,W) or 4D (N,C,H,W) input, got ~AD" 
                       ndim))))))


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
    "Batch Normalization for 2D convolutions with proper batch support.
   
     Normalizes activations across batch dimension:
     y = gamma * (x - mu) / sqrt(sigma^2 + epsilon) + beta
   
     Args:
       num-features: Number of channels (C)
       epsilon: Small constant for numerical stability
       momentum: Momentum for running statistics
       dtype: Data type
     
     Input shapes:
       3D: (C, H, W) - treated as batch of 1
       4D: (N, C, H, W) - standard batch"
  
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
         During training: Uses batch statistics and sets up gradients
         During eval: Uses running statistics"
        
        (let* ((input-shape (tensor-shape input))
               (ndim (length input-shape))
               (C num-features)
               (input-data (tensor-data input))
               (requires-grad? (or (tensor-requires-grad? input)
                                   (tensor-requires-grad? gamma)
                                   (tensor-requires-grad? beta))))
          
          ;; Extract dimensions based on input shape
          (let-values (((N H W)
                        (cond
                         ;; 3D input: (C, H, W) - treat as batch of 1
                         ((= ndim 3)
                          (values 1 (cadr input-shape) (caddr input-shape)))
                         
                         ;; 4D input: (N, C, H, W)
                         ((= ndim 4)
                          (values (car input-shape) 
                                  (caddr input-shape) 
                                  (cadddr input-shape)))
                         
                         (else
                          (error 'batch-norm-2d 
                                 (format #f "Expected 3D or 4D input, got ~AD" ndim))))))
            
            (let ((spatial-size (* H W))
                  (batch-spatial (* N H W)))
              
              (if training?
                  ;; Training mode: compute batch statistics and set up gradients
                  (let ((means (with-dtype dtype (vec C 0.0)))
                        (vars (with-dtype dtype (vec C 0.0)))
                        (normalized-data (with-dtype dtype (vec (* N C spatial-size) 0.0)))
                        (output-data (with-dtype dtype (vec (* N C spatial-size) 0.0))))
                    
                    ;; Compute mean for each channel across batch and spatial dims
                    (with-dtype dtype
                                (do ((c 0 (+ c 1)))
                                    ((= c C))
                                  (let ((sum 0.0))
                                    (do ((n 0 (+ n 1)))
                                        ((= n N))
                                      (do ((i 0 (+ i 1)))
                                          ((= i spatial-size))
                                        (let ((idx (+ (* n C spatial-size)
                                                      (* c spatial-size)
                                                      i)))
                                          (set! sum (+ sum (elt-ref input-data idx))))))
                                    (elt-set! means c (/ sum batch-spatial)))))
                    
                    ;; Compute variance for each channel
                    (with-dtype dtype
                                (do ((c 0 (+ c 1)))
                                    ((= c C))
                                  (let ((mean (elt-ref means c))
                                        (sum-sq 0.0))
                                    (do ((n 0 (+ n 1)))
                                        ((= n N))
                                      (do ((i 0 (+ i 1)))
                                          ((= i spatial-size))
                                        (let* ((idx (+ (* n C spatial-size)
                                                       (* c spatial-size)
                                                       i))
                                               (val (elt-ref input-data idx))
                                               (diff (- val mean)))
                                          (set! sum-sq (+ sum-sq (* diff diff))))))
                                    (elt-set! vars c (/ sum-sq batch-spatial)))))
                    
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
                    
                    ;; Normalize and apply gamma/beta
                    (with-dtype dtype
                                (do ((c 0 (+ c 1)))
                                    ((= c C))
                                  (let* ((mean (elt-ref means c))
                                         (var (elt-ref vars c))
                                         (gamma-val (elt-ref (tensor-data gamma) c))
                                         (beta-val (elt-ref (tensor-data beta) c))
                                         (std (sqrt (+ var epsilon))))
                                    
                                    (do ((n 0 (+ n 1)))
                                        ((= n N))
                                      (do ((i 0 (+ i 1)))
                                          ((= i spatial-size))
                                        (let ((idx (+ (* n C spatial-size)
                                                      (* c spatial-size)
                                                      i)))
                                          (let* ((x-val (elt-ref input-data idx))
                                                 (normalized (/ (- x-val mean) std)))
                                            ;; Store normalized value for backward
                                            (elt-set! normalized-data idx normalized)
                                            ;; Apply scale and shift
                                            (elt-set! output-data idx
                                                      (+ (* gamma-val normalized) 
                                                         beta-val)))))))))
                    
                    (let ((result (make-base-tensor output-data input-shape dtype
                                                    requires-grad?)))
                      
                      ;; Set up backward function
                      (when requires-grad?
                        (set-backward-fn!
                         result
                         (lambda ()

                           ;(printf "BatchNorm2D backward: vars exists? ~A\n" (if vars #t #f))
                           ;(printf "BatchNorm2D backward: normalized-data exists? ~A\n" (if normalized-data #t #f))
                           ;(printf "BatchNorm2D backward: C = ~A, N = ~A\n" C N)
                           ;(when vars
                           ;  (printf "BatchNorm2D backward: mean[0] = ~A\n" (f32vector-ref means 0)))
                           ;(when vars
                           ;  (printf "BatchNorm2D backward: var[0] = ~A\n" (f32vector-ref vars 0)))
                           ;(when normalized-data
                           ;  (printf "BatchNorm2D backward: norm[0] = ~A\n" (f32vector-ref normalized-data 0)))
                           
                           (let ((grad-out (tensor-grad result)))
                             
                             ;; Gradient w.r.t. gamma: sum of grad_out * normalized
                             (when (tensor-requires-grad? gamma)
                               (let ((grad-gamma (with-dtype dtype (vec C 0.0))))
                                 (with-dtype dtype
                                             (do ((c 0 (+ c 1)))
                                                 ((= c C))
                                               (let ((sum 0.0))
                                                 (do ((n 0 (+ n 1)))
                                                     ((= n N))
                                                   (do ((i 0 (+ i 1)))
                                                       ((= i spatial-size))
                                                     (let ((idx (+ (* n C spatial-size)
                                                                   (* c spatial-size)
                                                                   i)))
                                                       (set! sum (+ sum (* (elt-ref grad-out idx)
                                                                           (elt-ref normalized-data idx)))))))
                                                 (elt-set! grad-gamma c sum))))
                                 (add-to-grad! gamma grad-gamma)))
                             
                             ;; Gradient w.r.t. beta: sum of grad_out
                             (when (tensor-requires-grad? beta)
                               (let ((grad-beta (with-dtype dtype (vec C 0.0))))
                                 (with-dtype dtype
                                             (do ((c 0 (+ c 1)))
                                                 ((= c C))
                                               (let ((sum 0.0))
                                                 (do ((n 0 (+ n 1)))
                                                     ((= n N))
                                                   (do ((i 0 (+ i 1)))
                                                       ((= i spatial-size))
                                                     (let ((idx (+ (* n C spatial-size)
                                                                   (* c spatial-size)
                                                                   i)))
                                                       (set! sum (+ sum (elt-ref grad-out idx))))))
                                                 (elt-set! grad-beta c sum))))
                                 (add-to-grad! beta grad-beta)))
                             
                             ;; Gradient w.r.t. input
                             (when (tensor-requires-grad? input)
                               (let ((grad-input (with-dtype dtype (vec (* N C spatial-size) 0.0))))
                                 
                                 (with-dtype dtype
                                             (do ((c 0 (+ c 1)))
                                                 ((= c C))
                                               (let* ((var (elt-ref vars c))
                                                      (std (sqrt (+ var epsilon)))
                                                      (gamma-val (elt-ref (tensor-data gamma) c)))
                                                 
                                                 ;; Compute mean of grad_out for this channel
                                                 (let ((grad-out-sum 0.0))
                                                   (do ((n 0 (+ n 1)))
                                                       ((= n N))
                                                     (do ((i 0 (+ i 1)))
                                                         ((= i spatial-size))
                                                       (let ((idx (+ (* n C spatial-size)
                                                                     (* c spatial-size)
                                                                     i)))
                                                         (set! grad-out-sum (+ grad-out-sum (elt-ref grad-out idx))))))
                                                   
                                                   ;; Compute mean of grad_out * normalized
                                                   (let ((grad-norm-sum 0.0))
                                                     (do ((n 0 (+ n 1)))
                                                         ((= n N))
                                                       (do ((i 0 (+ i 1)))
                                                           ((= i spatial-size))
                                                         (let ((idx (+ (* n C spatial-size)
                                                                       (* c spatial-size)
                                                                       i)))
                                                           (set! grad-norm-sum (+ grad-norm-sum 
                                                                                  (* (elt-ref grad-out idx)
                                                                                     (elt-ref normalized-data idx)))))))
                                                     
                                                     (let ((grad-out-mean (/ grad-out-sum batch-spatial))
                                                           (grad-norm-mean (/ grad-norm-sum batch-spatial)))
                                                       
                                                       ;; Compute gradient for each element
                                                       (do ((n 0 (+ n 1)))
                                                           ((= n N))
                                                         (do ((i 0 (+ i 1)))
                                                             ((= i spatial-size))
                                                           (let ((idx (+ (* n C spatial-size)
                                                                         (* c spatial-size)
                                                                         i)))
                                                             (let ((grad-val (elt-ref grad-out idx))
                                                                   (norm-val (elt-ref normalized-data idx)))
                                                               ;; dL/dx = (gamma/std) * (dL/dy - mean(dL/dy) - norm * mean(dL/dy * norm))
                                                               (elt-set! grad-input idx
                                                                         (* (/ gamma-val std)
                                                                            (- grad-val
                                                                               grad-out-mean
                                                                               (* norm-val grad-norm-mean))))))))))))))
                                 
                                 (add-to-grad! input grad-input)))))
                         (list input gamma beta)))
                      
                      result))
                  
                  ;; Eval mode: use running statistics (no gradients)
                  (let ((output-data (with-dtype dtype (vec (* N C spatial-size) 0.0))))
                    
                    (with-dtype dtype
                                (do ((c 0 (+ c 1)))
                                    ((= c C))
                                  (let* ((mean (elt-ref running-mean c))
                                         (var (elt-ref running-var c))
                                         (gamma-val (elt-ref (tensor-data gamma) c))
                                         (beta-val (elt-ref (tensor-data beta) c))
                                         (std (sqrt (+ var epsilon))))
                                    
                                    (do ((n 0 (+ n 1)))
                                        ((= n N))
                                      (do ((i 0 (+ i 1)))
                                          ((= i spatial-size))
                                        (let ((idx (+ (* n C spatial-size)
                                                      (* c spatial-size)
                                                      i)))
                                          (let ((normalized 
                                                 (/ (- (elt-ref input-data idx) mean) std)))
                                            (elt-set! output-data idx
                                                      (+ (* gamma-val normalized) 
                                                         beta-val)))))))))
                    
                    (make-base-tensor output-data input-shape dtype #f)))))))
       
       ((parameters self)
        (list gamma beta))
     
       ((zero-grad-layer! self)
        (zero-grad! gamma)
        (zero-grad! beta)))))

  ;;; ==================================================================
  ;;; Global Average Pooling
  ;;; ==================================================================

  (define (global-avg-pool2d input)
    "Global average pooling over spatial dimensions with batch support.
   
     Input shapes:
       3D: (C, H, W) -> Output: (C,)
       4D: (N, C, H, W) -> Output: (N, C)"
  
    (let* ((dtype (tensor-dtype input))
           (shape (tensor-shape input))
           (ndim (length shape)))
      
      (cond
       ;; 3D input: (C, H, W)
       ((= ndim 3)
        (let* ((C (car shape))
               (H (cadr shape))
               (W (caddr shape))
               (spatial-size (* (cadr shape) (caddr shape)))
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
              (set-backward-fn!
               result
               (lambda ()
                 (let ((grad-out (tensor-grad result))
                       (grad-in (with-dtype dtype (vec (* C H W) 0.0))))
                   
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
       
       ;; 4D input: (N, C, H, W)
       ((= ndim 4)
        (let* ((N (car shape))
               (C (cadr shape))
               (H (caddr shape))
               (W (cadddr shape))
               (spatial-size (* (caddr shape) (cadddr shape)))
               (data (tensor-data input))
               (output-data (with-dtype dtype (vec (* N C) 0.0))))
          
          ;; Average over spatial dimensions for each (batch, channel) pair
          (with-dtype dtype
                      (do ((n 0 (+ n 1)))
                          ((= n N))
                        (do ((c 0 (+ c 1)))
                            ((= c C))
                          (let ((sum
                                 (let loop ((sum 0.0) (i 0))
                                   (if (< i spatial-size)
                                       (let ((idx (+ (* n C spatial-size)
                                                     (* c spatial-size)
                                                     i)))
                                         (loop (+ sum (elt-ref data idx))
                                               (+ i 1)))
                                       sum))))
                            (elt-set! output-data (+ (* n C) c) 
                                      (/ sum spatial-size))))))
          
          (let ((result (make-base-tensor output-data (list N C) dtype
                                          (tensor-requires-grad? input))))
            
            ;; Backward pass
            (when (tensor-requires-grad? input)
              (set-backward-fn!
               result
               (lambda ()
                 (let ((grad-out (tensor-grad result))
                       (grad-in (with-dtype dtype (vec (* N C H W) 0.0))))
                   
                   (with-dtype dtype
                               (do ((n 0 (+ n 1)))
                                   ((= n N))
                                 (do ((c 0 (+ c 1)))
                                     ((= c C))
                                   (let* ((grad-val (elt-ref grad-out (+ (* n C) c)))
                                          (grad-per-pixel (/ grad-val spatial-size)))
                                     (do ((i 0 (+ i 1)))
                                         ((= i spatial-size))
                                       (let ((idx (+ (* n C spatial-size)
                                                     (* c spatial-size)
                                                     i)))
                                         (elt-set! grad-in idx grad-per-pixel)))))))
                   
                   (add-to-grad! input grad-in)))
               (list input)))
            
            result)))
       
       (else
        (error 'global-avg-pool2d 
               (format #f "Expected 3D or 4D input, got ~AD" ndim))))))
  
  ;;; ==================================================================
  ;;; Utilities
  ;;; ==================================================================

  ;; Print layer information
  (define (print-layer layer #!optional (indent 0))
    (let ((spaces (make-string indent #\space)))
      (printf "~A~A: " spaces (layer-name layer))
      (cond
       ((dense-layer? layer)
        (printf "Dense(~A -> ~A, activation=~A)\n"
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

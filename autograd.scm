;; YASOS-based automatic differentiation framework

(module nanograd-autograd
  (
   ;; Tensor constructors
   make-tensor32 make-tensor64
   tensor? tensor32? tensor64?
   
   ;; Tensor accessors
   tensor-data tensor-grad tensor-shape
   tensor-requires-grad? tensor-dtype
   
   ;; Gradient operations
   zero-grad! backward!
   set-backward-fn! add-to-grad!
   
   ;; Arithmetic operations
   add sub mul div safe-div
   
   ;; BLAS-based operations
   matmul-op dot-op scale-op
   
   ;; Activation functions
   relu tanh-op sigmoid
   sigmoid-stable
   softplus leaky-relu 
   softmax log-softmax
   
   ;; Loss functions
   mse-loss cross-entropy-loss
   
   ;; Utilities
   reshape
   flatten-tensor
   rmsnorm conv2d
   l2-normalize cosine-similarity
   make-base-tensor
   tensor->list print-tensor
   vector-length-for-dtype
   )
  
  (import
   scheme
   (chicken base)
   (chicken format)
   (chicken flonum)
   (srfi 1)
   (srfi 4)
   (srfi 42)  ; Comprehensions
   (srfi 69)
   mathh
   yasos
   blas
   )

  ;;; ==================================================================
  ;;; Tensor interface using YASOS
  ;;; ==================================================================

  ;; Define tensor predicate and operations
  (define-predicate tensor?)
  (define-predicate tensor32?)
  (define-predicate tensor64?)
  
  (define-operation (tensor-data obj))
  (define-operation (tensor-grad obj))
  (define-operation (tensor-shape obj))
  (define-operation (tensor-requires-grad? obj))
  (define-operation (tensor-dtype obj))
  (define-operation (tensor-backward-fn obj))
  (define-operation (tensor-children obj))

  (define-operation (reshape tensor new-shape))
  (define-operation (transpose-tensor tensor axes))
  (define-operation (get-strides tensor))

  
  (define-operation (set-grad! obj grad))
  (define-operation (set-backward-fn! obj fn input-tensors))
  (define-operation (add-to-grad! obj delta))
  
  (define-operation (zero-grad! obj))
  (define-operation (backward! obj))
  
  ;;; ==================================================================
  ;;; Base tensor implementation
  ;;; ==================================================================

  (define (make-base-tensor data shape dtype requires-grad?)
    (let ((grad (if requires-grad?
                    (case dtype
                      ((f32) (make-f32vector (apply * shape) 0.0))
                      ((f64) (make-f64vector (apply * shape) 0.0)))
                    #f))
          (backward-fn #f)
          (children '()))  ; store tensor dependencies
    
    (object
     ;; Type predicates
     ((tensor? self) #t)
     ((tensor32? self) (eq? dtype 'f32))
     ((tensor64? self) (eq? dtype 'f64))
     
     ;; Accessors
     ((tensor-data self) data)
     ((tensor-grad self) grad)
     ((tensor-shape self) shape)
     ((tensor-requires-grad? self) requires-grad?)
     ((tensor-dtype self) dtype)
     ((tensor-backward-fn self) backward-fn)
     ((tensor-children self) children)

     ;; Shape manipulation
     ((reshape self new-shape)
      ;; Reshape tensor (must preserve total number of elements)
      (let ((old-size (apply * shape))
            (new-size (apply * new-shape)))
        (unless (= old-size new-size)
          (error 'reshape 
                 (format #f "Cannot reshape ~A to ~A: size mismatch" 
                         shape new-shape)))
        
        ;; Create new tensor with separate gradient buffer
        (let ((reshaped (make-base-tensor data new-shape dtype requires-grad?)))
     
          (when requires-grad?
            ;; Set up backward function to pass gradients back
            ;; Add self to children so topological sort works!
            (set-backward-fn!
             reshaped
             (lambda ()
               (let ((reshaped-grad (tensor-grad reshaped)))
                 (when reshaped-grad
                   ;; Add reshaped gradient to original gradient
                   ;; They share the same underlying data vector
                   (add-to-grad! self reshaped-grad))))
             (list self)))  ; Register self as dependency
          
     reshaped)))
     
     ((transpose-tensor self axes)
      ;; Transpose tensor dimensions according to axes permutation
      ;; e.g., (0 1 2 3) -> (0 2 3 1) for NCHW -> NHWC
      (let* ((rank (length shape))
             (new-shape (map (lambda (axis) (list-ref shape axis)) axes))
             (size (apply * shape))
             (new-data (case dtype
                         ((f32) (make-f32vector size 0.0))
                         ((f64) (make-f64vector size 0.0)))))
   
        ;; Compute strides for old and new layouts
        (define (compute-strides shape)
          (reverse
           (cdr (reverse
                 (fold (lambda (dim acc)
                         (cons (* dim (car acc)) acc))
                       '(1)
                       (reverse shape))))))
        
        (let ((old-strides (compute-strides shape))
              (new-strides (compute-strides new-shape)))
          
          ;; Copy data with transposed indices
          (let loop ((i 0))
            (when (< i size)
              ;; Convert flat index to multi-dimensional index
              (let* ((indices (let idx-loop ((n i) (strides old-strides) (result '()))
                                (if (null? strides)
                                    (reverse result)
                                    (let ((idx (quotient n (car strides))))
                                      (idx-loop (remainder n (car strides))
                                                (cdr strides)
                                                (cons idx result))))))
                     ;; Permute indices according to axes
                     (new-indices (map (lambda (axis) (list-ref indices axis)) axes))
                     ;; Convert back to flat index in new layout
                     (new-idx (fold (lambda (idx stride acc)
                                      (+ acc (* idx stride)))
                                    0 new-indices new-strides)))
                
                (case dtype
                  ((f32) (f32vector-set! new-data new-idx (f32vector-ref data i)))
                  ((f64) (f64vector-set! new-data new-idx (f64vector-ref data i))))
                
                (loop (+ i 1)))))
          
          (make-base-tensor new-data new-shape dtype requires-grad?))))

     ((get-strides self)
      ;; Calculate strides for the current shape
      (reverse
       (cdr (reverse
             (fold (lambda (dim acc)
                     (cons (* dim (car acc)) acc))
                   '(1)
                   (reverse shape))))))
     
     ;; Mutators
     ((set-grad! self new-grad)
      (set! grad new-grad))
     
     ((set-backward-fn! self fn input-tensors)
      ;; input-tensors is a list of tensors this operation depends on
      (set! backward-fn fn)
      (set! children (filter tensor-requires-grad? input-tensors)))
     
     ((add-to-grad! self delta)
      (when grad
        (let ((n (vector-length-for-dtype grad dtype)))
          (case dtype
            ((f32) (saxpy! n 1.0 delta grad))
            ((f64) (daxpy! n 1.0 delta grad))))))
     
     ;; Gradient operations
     ((zero-grad! self)
      (when grad
        (let ((n (vector-length-for-dtype grad dtype)))
          (case dtype
            ((f32) (do ((i 0 (+ i 1)))
                       ((= i n))
                     (f32vector-set! grad i 0.0)))
            ((f64) (do ((i 0 (+ i 1)))
                       ((= i n))
                     (f64vector-set! grad i 0.0)))))))

     ;; Topological sort backward (correct for DAGs with shared nodes)
     ((backward! self)
      (when requires-grad?

        (when (detect-cycles self)
          (error 'backward! 
                 "Computation graph contains cycles - cannot compute gradients"))

        
        ;; Initialize gradient
        (unless (gradient-initialized? grad dtype)
          (fill-ones! grad dtype))
        
        ;; Build topological ordering using depth-first search
        (let ((visited (make-hash-table eq?))
              (topo-order '()))
          
          ;; DFS to build reverse topological order
          (define (visit node)
            (unless (hash-table-ref/default visited node #f)
              (hash-table-set! visited node #t)
              ;; Visit children first (post-order traversal)
              (for-each visit (tensor-children node))
              ;; Add to order after visiting children
              (set! topo-order (cons node topo-order))))
          
          ;; Start DFS from this (output) tensor
          (visit self)
          
          ;; Execute backward functions in topological order
          ;; This ensures each backward function is called exactly once
          ;; and gradients are accumulated in the correct order

          (for-each
           (lambda (node)
             (let ((bwd-fn (tensor-backward-fn node)))
               (when bwd-fn
                 (bwd-fn))))
           topo-order))))
     
     ))
    )

  (define (detect-cycles tensor)
    "Returns #t if the computation graph contains cycles, #f otherwise"
    (let ((visiting (make-hash-table eq?))  ; Currently in DFS stack
          (visited (make-hash-table eq?)))   ; Completely processed
      
      (define (has-cycle? node)
        (cond
         ;; Already completely visited - no cycle from this node
         ((hash-table-ref/default visited node #f) #f)
         ;; Currently visiting - found a back edge (cycle!)
         ((hash-table-ref/default visiting node #f) #t)
         (else
          ;; Mark as visiting
          (hash-table-set! visiting node #t)
          ;; Check children
          (let ((cycle-found? (any has-cycle? (tensor-children node))))
            ;; Mark as visited and remove from visiting
            (hash-table-set! visiting node #f)
            (hash-table-set! visited node #t)
            cycle-found?))))
      
      (has-cycle? tensor)))
     

  (define (gradient-initialized? grad dtype)
    (and grad
         (let ((n (vector-length-for-dtype grad dtype)))
           (and (> n 0)
                (case dtype
                  ((f32) (not (= (f32vector-ref grad 0) 0.0)))
                  ((f64) (not (= (f64vector-ref grad 0) 0.0))))))))

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
  ;;; Tensor constructors
  ;;; ==================================================================

  (define (make-tensor32 data shape #!key (requires-grad? #t))
    (make-base-tensor data shape 'f32 requires-grad?))

  (define (make-tensor64 data shape #!key (requires-grad? #t))
    (make-base-tensor data shape 'f64 requires-grad?))

  ;;; ==================================================================
  ;;; Arithmetic operations
  ;;; ==================================================================

  ;; Element-wise addition
  (define (add a b)
    (assert (eq? (tensor-dtype a) (tensor-dtype b)))
    (let* ((dtype (tensor-dtype a))
           (data-a (tensor-data a))
           (data-b (tensor-data b))
           (n (vector-length-for-dtype data-a dtype))
           (result-data (case dtype
                          ((f32) (make-f32vector n 0.0))
                          ((f64) (make-f64vector n 0.0))))
           (requires-grad? (or (tensor-requires-grad? a)
                              (tensor-requires-grad? b))))
      
      ;; Forward: result = a + b
      (case dtype
        ((f32)
         (do ((i 0 (+ i 1)))
             ((= i n))
           (f32vector-set! result-data i
                          (+ (f32vector-ref data-a i)
                             (f32vector-ref data-b i)))))
        ((f64)
         (do ((i 0 (+ i 1)))
             ((= i n))
           (f64vector-set! result-data i
                          (+ (f64vector-ref data-a i)
                             (f64vector-ref data-b i))))))
      
      (let ((result (make-base-tensor result-data (tensor-shape a) dtype requires-grad?)))
        (when requires-grad?
          ;; Pass list of input tensors explicitly
          (set-backward-fn! result
                            (lambda ()
                              (let ((grad-out (tensor-grad result)))
                                (when (tensor-requires-grad? a)
                                  (add-to-grad! a grad-out))
                                (when (tensor-requires-grad? b)
                                  (add-to-grad! b grad-out))))
                            (list a b)))  ; <-- Explicitly pass children
        result)))
  
  ;; Element-wise subtraction
  (define (sub a b)
    (assert (eq? (tensor-dtype a) (tensor-dtype b)))
    (let* ((dtype (tensor-dtype a))
           (data-a (tensor-data a))
           (data-b (tensor-data b))
           (n (vector-length-for-dtype data-a dtype))
           (result-data (case dtype
                          ((f32) (make-f32vector n 0.0))
                          ((f64) (make-f64vector n 0.0))))
           (requires-grad? (or (tensor-requires-grad? a)
                              (tensor-requires-grad? b))))
      
      ;; Forward: result = a - b
      (case dtype
        ((f32)
         (do ((i 0 (+ i 1)))
             ((= i n))
           (f32vector-set! result-data i
                          (- (f32vector-ref data-a i)
                             (f32vector-ref data-b i)))))
        ((f64)
         (do ((i 0 (+ i 1)))
             ((= i n))
           (f64vector-set! result-data i
                          (- (f64vector-ref data-a i)
                             (f64vector-ref data-b i))))))
      
      (let ((result (make-base-tensor result-data (tensor-shape a) dtype requires-grad?)))
        (when requires-grad?
          (set-backward-fn! result
            (lambda ()
              (let ((grad-out (tensor-grad result)))
                (when (tensor-requires-grad? a)
                  (add-to-grad! a grad-out))
                (when (tensor-requires-grad? b)
                  ;; Negate gradient for subtraction
                  (let ((neg-grad (case dtype
                                   ((f32) (make-f32vector n 0.0))
                                   ((f64) (make-f64vector n 0.0)))))
                    (case dtype
                      ((f32) (saxpy! n -1.0 grad-out neg-grad))
                      ((f64) (daxpy! n -1.0 grad-out neg-grad)))
                    (add-to-grad! b neg-grad)))))
            (list a b))
          )
        result)))

  ;; Element-wise multiplication
  (define (mul a b)
    (assert (eq? (tensor-dtype a) (tensor-dtype b)))
    (let* ((dtype (tensor-dtype a))
           (data-a (tensor-data a))
           (data-b (tensor-data b))
           (n (vector-length-for-dtype data-a dtype))
           (result-data (case dtype
                          ((f32) (make-f32vector n 0.0))
                          ((f64) (make-f64vector n 0.0))))
           (requires-grad? (or (tensor-requires-grad? a)
                              (tensor-requires-grad? b))))
      
      ;; Forward: result = a * b (element-wise)
      (case dtype
        ((f32)
         (do ((i 0 (+ i 1)))
             ((= i n))
           (f32vector-set! result-data i
                          (* (f32vector-ref data-a i)
                             (f32vector-ref data-b i)))))
        ((f64)
         (do ((i 0 (+ i 1)))
             ((= i n))
           (f64vector-set! result-data i
                          (* (f64vector-ref data-a i)
                             (f64vector-ref data-b i))))))
      
      (let ((result (make-base-tensor result-data (tensor-shape a) dtype requires-grad?)))
        (when requires-grad?
          (set-backward-fn! result
            (lambda ()
              (let ((grad-out (tensor-grad result)))
                ;; d(a*b)/da = b
                (when (tensor-requires-grad? a)
                  (let ((grad-a (case dtype
                                 ((f32) (make-f32vector n 0.0))
                                 ((f64) (make-f64vector n 0.0)))))
                    (case dtype
                      ((f32)
                       (do ((i 0 (+ i 1)))
                           ((= i n))
                         (f32vector-set! grad-a i
                                        (* (f32vector-ref grad-out i)
                                           (f32vector-ref data-b i)))))
                      ((f64)
                       (do ((i 0 (+ i 1)))
                           ((= i n))
                         (f64vector-set! grad-a i
                                        (* (f64vector-ref grad-out i)
                                           (f64vector-ref data-b i))))))
                    (add-to-grad! a grad-a)))
                ;; d(a*b)/db = a
                (when (tensor-requires-grad? b)
                  (let ((grad-b (case dtype
                                 ((f32) (make-f32vector n 0.0))
                                 ((f64) (make-f64vector n 0.0)))))
                    (case dtype
                      ((f32)
                       (do ((i 0 (+ i 1)))
                           ((= i n))
                         (f32vector-set! grad-b i
                                        (* (f32vector-ref grad-out i)
                                           (f32vector-ref data-a i)))))
                      ((f64)
                       (do ((i 0 (+ i 1)))
                           ((= i n))
                         (f64vector-set! grad-b i
                                        (* (f64vector-ref grad-out i)
                                           (f64vector-ref data-a i))))))
                    (add-to-grad! b grad-b)))))
          (list a b))
        result))))

  ;; Element-wise division
  (define (div a b)
  "Element-wise division: z = a / b
   Gradients:
     dL/da = dL/dz * (1/b)
     dL/db = dL/dz * (-a/b^2)"
  
  (assert (eq? (tensor-dtype a) (tensor-dtype b)))
  
  (let* ((dtype (tensor-dtype a))
         (data-a (tensor-data a))
         (data-b (tensor-data b))
         (n (vector-length-for-dtype data-a dtype))
         (result-data (case dtype
                        ((f32) (make-f32vector n 0.0))
                        ((f64) (make-f64vector n 0.0))))
         (requires-grad? (or (tensor-requires-grad? a)
                            (tensor-requires-grad? b))))
    
    ;; Check for shapes match
    (unless (equal? (tensor-shape a) (tensor-shape b))
      (error 'div "Shape mismatch: cannot divide tensors with different shapes"))
    
    ;; Forward: result = a / b (element-wise)
    (case dtype
      ((f32)
       (do ((i 0 (+ i 1)))
           ((= i n))
         (let ((b-val (f32vector-ref data-b i)))
           (when (= b-val 0.0)
             (error 'div "Division by zero"))
           (f32vector-set! result-data i
                          (/ (f32vector-ref data-a i) b-val)))))
      ((f64)
       (do ((i 0 (+ i 1)))
           ((= i n))
         (let ((b-val (f64vector-ref data-b i)))
           (when (= b-val 0.0)
             (error 'div "Division by zero"))
           (f64vector-set! result-data i
                          (/ (f64vector-ref data-a i) b-val))))))
    
    (let ((result (make-base-tensor result-data (tensor-shape a) dtype requires-grad?)))
      (when requires-grad?
        (set-backward-fn! result
          (lambda ()
            (let ((grad-out (tensor-grad result)))
              
              ;; Gradient w.r.t. a: dL/da = dL/dz * (1/b)
              (when (tensor-requires-grad? a)
                (let ((grad-a (case dtype
                               ((f32) (make-f32vector n 0.0))
                               ((f64) (make-f64vector n 0.0)))))
                  (case dtype
                    ((f32)
                     (do ((i 0 (+ i 1)))
                         ((= i n))
                       (f32vector-set! grad-a i
                                      (/ (f32vector-ref grad-out i)
                                         (f32vector-ref data-b i)))))
                    ((f64)
                     (do ((i 0 (+ i 1)))
                         ((= i n))
                       (f64vector-set! grad-a i
                                      (/ (f64vector-ref grad-out i)
                                         (f64vector-ref data-b i))))))
                  (add-to-grad! a grad-a)))
              
              ;; Gradient w.r.t. b: dL/db = dL/dz * (-a/b²)
              (when (tensor-requires-grad? b)
                (let ((grad-b (case dtype
                               ((f32) (make-f32vector n 0.0))
                               ((f64) (make-f64vector n 0.0)))))
                  (case dtype
                    ((f32)
                     (do ((i 0 (+ i 1)))
                         ((= i n))
                       (let ((b-val (f32vector-ref data-b i)))
                         (f32vector-set! grad-b i
                                        (* (f32vector-ref grad-out i)
                                           (- (/ (f32vector-ref data-a i)
                                                (* b-val b-val))))))))
                    ((f64)
                     (do ((i 0 (+ i 1)))
                         ((= i n))
                       (let ((b-val (f64vector-ref data-b i)))
                         (f64vector-set! grad-b i
                                        (* (f64vector-ref grad-out i)
                                           (- (/ (f64vector-ref data-a i)
                                                (* b-val b-val)))))))))
                  (add-to-grad! b grad-b)))))
          (list a b)))
      result)))


  (define (safe-div a b #!key (epsilon 1e-8))
    "Safe division: a / (b + epsilon) to avoid division by zero"
  
    (let* ((dtype (tensor-dtype b))
           (data-b (tensor-data b))
           (n (vector-length-for-dtype data-b dtype))
           (b-safe-data (case dtype
                          ((f32) (make-f32vector n 0.0))
                          ((f64) (make-f64vector n 0.0)))))
      
      ;; Add epsilon to b
      (case dtype
        ((f32)
         (do ((i 0 (+ i 1)))
             ((= i n))
           (f32vector-set! b-safe-data i
                           (+ (f32vector-ref data-b i) epsilon))))
        ((f64)
         (do ((i 0 (+ i 1)))
             ((= i n))
           (f64vector-set! b-safe-data i
                           (+ (f64vector-ref data-b i) epsilon)))))
      
      ;; Create safe denominator tensor
      (let ((b-safe (make-base-tensor b-safe-data 
                                      (tensor-shape b)
                                      dtype
                                      (tensor-requires-grad? b))))
        ;; Perform safe division
        (div a b-safe))))
  
  
  ;;; ==================================================================
  ;;; BLAS Operations
  ;;; ==================================================================

  ;; Matrix multiplication: C = A @ B
  ;; A: (m, k), B: (k, n), C: (m, n)
  (define (matmul-op a b)
    (assert (eq? (tensor-dtype a) (tensor-dtype b)))
  
    (let* ((dtype (tensor-dtype a))
           (shape-a (tensor-shape a))
           (shape-b (tensor-shape b))
           (data-a (tensor-data a))
           (data-b (tensor-data b))
           (requires-grad? (or (tensor-requires-grad? a)
                               (tensor-requires-grad? b))))
      
      ;; Determine dimensions based on shapes
      (let* ((is-a-matrix? (= (length shape-a) 2))
             (is-b-matrix? (= (length shape-b) 2))
             (is-b-vector? (= (length shape-b) 1))
             
             ;; Extract dimensions
             (m (if is-a-matrix? (car shape-a) 1))
             (k (if is-a-matrix? (cadr shape-a) (car shape-a)))
             (n (if is-b-vector? 1 (if is-b-matrix? (cadr shape-b) 1)))
             
             ;; Check compatibility
             (k-b (if is-b-vector? (car shape-b) (car shape-b))))
        
        (unless (= k k-b)
        (error 'matmul-op 
               (format #f "Incompatible dimensions: (~A,~A) @ (~A,~A)" 
                       m k k-b n)))
      
      ;; Create result
      (let* ((result-shape (if is-b-vector? (list m) (list m n)))
             (result-size (* m n))
             (result-data (case dtype
                           ((f32) (make-f32vector result-size 0.0))
                           ((f64) (make-f64vector result-size 0.0)))))
        
        ;; Forward pass
        (cond
         ;; Case 1: Matrix @ Matrix using GEMM
         ((and is-a-matrix? is-b-matrix?)
          (case dtype
            ((f32) (sgemm! RowMajor NoTrans NoTrans 
                          m n k 
                          1.0 data-a      ; A: m*k, lda=k
                          data-b          ; B: k*n, ldb=n
                          0.0 result-data
                          lda: k
                          ldb: n
                          ldc: n)) ; C: m*n, ldc=n
            ((f64) (dgemm! RowMajor NoTrans NoTrans 
                          m n k 
                          1.0 data-a
                          data-b
                          0.0 result-data
                          lda: k
                          ldb: n
                          ldc: n))))
         
         ;; Case 2: Matrix @ Vector using GEMV
         ((and is-a-matrix? is-b-vector?)
          (case dtype
            ((f32) (sgemv! RowMajor NoTrans m k 
                          1.0 data-a data-b 
                          0.0 result-data))
            ((f64) (dgemv! RowMajor NoTrans m k 
                          1.0 data-a data-b 
                          0.0 result-data))))
         
         ;; Case 3: Vector @ Matrix (treat as 1×k @ k×n)
         ((and (not is-a-matrix?) is-b-matrix?)
          (case dtype
            ((f32) (sgemv! RowMajor Trans n k 
                          1.0 data-b data-a 
                          0.0 result-data))
            ((f64) (dgemv! RowMajor Trans n k 
                          1.0 data-b data-a 
                          0.0 result-data))))
         
         ;; Case 4: Vector @ Vector (dot product)
         (else
          (let ((dot-result (case dtype
                             ((f32) (sdot k data-a data-b))
                             ((f64) (ddot k data-a data-b)))))
            (case dtype
              ((f32) (f32vector-set! result-data 0 dot-result))
              ((f64) (f64vector-set! result-data 0 dot-result))))))
        
        ;; Create result tensor
        (let ((result (make-base-tensor result-data result-shape dtype requires-grad?)))
          (when requires-grad?
            (set-backward-fn! result
              (lambda ()
                (let ((grad-out (tensor-grad result)))
                  
                  ;; Gradient w.r.t. A: dL/dA = dL/dC @ B^T
                  (when (tensor-requires-grad? a)
                    (let ((grad-a (case dtype
                                   ((f32) (make-f32vector (* m k) 0.0))
                                   ((f64) (make-f64vector (* m k) 0.0)))))
                      
                      (cond
                       ;; Matrix @ Matrix case
                       ((and is-a-matrix? is-b-matrix?)
                        ;; dL/dA = dL/dC @ B^T
                        ;; (m×n) @ (n×k) = (m×k)
                        (case dtype
                          ((f32) (sgemm! RowMajor NoTrans Trans
                                        m k n
                                        1.0 grad-out      ; dL/dC: m*n
                                        data-b            ; B: k*n (transposed)
                                        0.0 grad-a
                                        lda: n
                                        ldb: n
                                        ldc: k))      ; dL/dA: m*k
                          ((f64) (dgemm! RowMajor NoTrans Trans
                                        m k n
                                        1.0 grad-out
                                        data-b
                                        0.0 grad-a
                                        lda: n
                                        ldb: n
                                        ldc: k))))
                       
                       ;; Matrix @ Vector case
                       ((and is-a-matrix? is-b-vector?)
                        ;; dL/dA = dL/dC ⊗ B (outer product)
                        ;; Each row i of dL/dA = (dL/dC)[i] * B
                        (case dtype
                          ((f32)
                           (do ((i 0 (+ i 1)))
                               ((= i m))
                             (let ((scale (f32vector-ref grad-out i)))
                               (do ((j 0 (+ j 1)))
                                   ((= j k))
                                 (f32vector-set! grad-a (+ (* i k) j)
                                                (* scale (f32vector-ref data-b j)))))))
                          ((f64)
                           (do ((i 0 (+ i 1)))
                               ((= i m))
                             (let ((scale (f64vector-ref grad-out i)))
                               (do ((j 0 (+ j 1)))
                                   ((= j k))
                                 (f64vector-set! grad-a (+ (* i k) j)
                                                (* scale (f64vector-ref data-b j)))))))))
                       
                       ;; Vector @ Matrix case
                       ((and (not is-a-matrix?) is-b-matrix?)
                        ;; dL/dA = B @ dL/dC (vector result)
                        (case dtype
                          ((f32) (sgemv! RowMajor NoTrans k n
                                        1.0 data-b grad-out
                                        0.0 grad-a))
                          ((f64) (dgemv! RowMajor NoTrans k n
                                        1.0 data-b grad-out
                                        0.0 grad-a))))
                       
                       ;; Vector @ Vector (dot product)
                       (else
                        ;; dL/dA = dL/dC * B (element-wise)
                        (let ((scale (case dtype
                                      ((f32) (f32vector-ref grad-out 0))
                                      ((f64) (f64vector-ref grad-out 0)))))
                          (case dtype
                            ((f32) (do ((i 0 (+ i 1)))
                                       ((= i k))
                                     (f32vector-set! grad-a i
                                                    (* scale (f32vector-ref data-b i)))))
                            ((f64) (do ((i 0 (+ i 1)))
                                       ((= i k))
                                     (f64vector-set! grad-a i
                                                    (* scale (f64vector-ref data-b i)))))))))
                      
                      (add-to-grad! a grad-a)))
                  
                  ;; Gradient w.r.t. B: dL/dB = A^T @ dL/dC
                  (when (tensor-requires-grad? b)
                    (let ((grad-b (case dtype
                                   ((f32) (make-f32vector (* k n) 0.0))
                                   ((f64) (make-f64vector (* k n) 0.0)))))
                      
                      (cond
                       ;; Matrix @ Matrix case
                       ((and is-a-matrix? is-b-matrix?)
                        ;; dL/dB = A^T @ dL/dC
                        ;; (k×m) @ (m×n) = (k×n)
                        (case dtype
                          ((f32) (sgemm! RowMajor Trans NoTrans
                                        k n m
                                        1.0 data-a        ; A: m*k (transposed)
                                        grad-out          ; dL/dC: m*n
                                        0.0 grad-b
                                        lda: k
                                        ldb: n
                                        ldc: n))      ; dL/dB: k×n
                          ((f64) (dgemm! RowMajor Trans NoTrans
                                        k n m
                                        1.0 data-a
                                        grad-out
                                        0.0 grad-b
                                        lda: k
                                        ldb: n
                                        ldc: n))))
                       
                       ;; Matrix @ Vector case
                       ((and is-a-matrix? is-b-vector?)
                        ;; dL/dB = A^T @ dL/dC
                        ;; (k×m) @ (m×1) = (k×1)
                        (case dtype
                          ((f32) (sgemv! RowMajor Trans m k
                                        1.0 data-a grad-out
                                        0.0 grad-b))
                          ((f64) (dgemv! RowMajor Trans m k
                                        1.0 data-a grad-out
                                        0.0 grad-b))))
                       
                       ;; Vector @ Matrix case
                       ((and (not is-a-matrix?) is-b-matrix?)
                        ;; dL/dB = A ⊗ dL/dC (outer product)
                        ;; Result is k×n matrix
                        (case dtype
                          ((f32)
                           (do ((i 0 (+ i 1)))
                               ((= i k))
                             (let ((a-val (f32vector-ref data-a i)))
                               (do ((j 0 (+ j 1)))
                                   ((= j n))
                                 (f32vector-set! grad-b (+ (* i n) j)
                                                (* a-val (f32vector-ref grad-out j)))))))
                          ((f64)
                           (do ((i 0 (+ i 1)))
                               ((= i k))
                             (let ((a-val (f64vector-ref data-a i)))
                               (do ((j 0 (+ j 1)))
                                   ((= j n))
                                 (f64vector-set! grad-b (+ (* i n) j)
                                                (* a-val (f64vector-ref grad-out j)))))))))
                       
                       ;; Vector @ Vector (dot product)
                       (else
                        ;; dL/dB = dL/dC * A
                        (let ((scale (case dtype
                                      ((f32) (f32vector-ref grad-out 0))
                                      ((f64) (f64vector-ref grad-out 0)))))
                          (case dtype
                            ((f32) (do ((i 0 (+ i 1)))
                                       ((= i k))
                                     (f32vector-set! grad-b i
                                                    (* scale (f32vector-ref data-a i)))))
                            ((f64) (do ((i 0 (+ i 1)))
                                       ((= i k))
                                     (f64vector-set! grad-b i
                                                    (* scale (f64vector-ref data-a i)))))))))
                      
                      (add-to-grad! b grad-b)))))
              (list a b)))
          result)))))


  ;; Scalar multiplication
  (define (scale-op tensor scalar)
    (let* ((dtype (tensor-dtype tensor))
           (data (tensor-data tensor))
           (n (vector-length-for-dtype data dtype))
           (result-data (case dtype
                          ((f32) (make-f32vector n 0.0))
                          ((f64) (make-f64vector n 0.0))))
           (requires-grad? (tensor-requires-grad? tensor)))
      
      ;; Forward: copy and scale
      (case dtype
        ((f32)
         (do ((i 0 (+ i 1)))
             ((= i n))
           (f32vector-set! result-data i (f32vector-ref data i)))
         (sscal! n scalar result-data))
        ((f64)
         (do ((i 0 (+ i 1)))
             ((= i n))
           (f64vector-set! result-data i (f64vector-ref data i)))
         (dscal! n scalar result-data)))
      
      (let ((result (make-base-tensor result-data (tensor-shape tensor) dtype requires-grad?)))
        (when requires-grad?
          (set-backward-fn! result
            (lambda ()
              (let ((grad-out (tensor-grad result))
                    (grad-scaled (case dtype
                                  ((f32) (make-f32vector n 0.0))
                                  ((f64) (make-f64vector n 0.0)))))
                (case dtype
                  ((f32)
                   (saxpy! n scalar grad-out grad-scaled))
                  ((f64)
                   (daxpy! n scalar grad-out grad-scaled)))
                (add-to-grad! tensor grad-scaled)))
            (list tensor)))
        result)))

  (define (dot-op a b)
    "Dot product (inner product): scalar = a · b
   Uses BLAS sdot/ddot for efficient computation.
   Gradients:
     dL/da = dL/dscalar * b
     dL/db = dL/dscalar * a"
   
  
    (assert (eq? (tensor-dtype a) (tensor-dtype b)))
  
  (let* ((dtype (tensor-dtype a))
         (shape-a (tensor-shape a))
         (shape-b (tensor-shape b))
         (data-a (tensor-data a))
         (data-b (tensor-data b))
         (requires-grad? (or (tensor-requires-grad? a)
                            (tensor-requires-grad? b))))
    
    (unless (and (= (length shape-a) 1)
                (= (length shape-b) 1))
      (error 'dot-op "Both tensors must be 1D vectors"))
    
    (let ((n-a (car shape-a))
          (n-b (car shape-b)))
      (unless (= n-a n-b)
        (error 'dot-op 
               (format #f "Vector length mismatch: ~A vs ~A" n-a n-b)))
      
      (let* ((n n-a)
             (dot-result (case dtype
                          ((f32) (sdot n data-a data-b))
                          ((f64) (ddot n data-a data-b))))
             (result-data (case dtype
                           ((f32) (f32vector dot-result))
                           ((f64) (f64vector dot-result)))))
        
        (let ((result (make-base-tensor result-data '(1) dtype requires-grad?)))
          (when requires-grad?
            (set-backward-fn! result
              (lambda ()
                (let* ((grad-out (tensor-grad result))
                       (grad-scalar (case dtype
                                      ((f32) (f32vector-ref grad-out 0))
                                      ((f64) (f64vector-ref grad-out 0)))))
                  
                  ;; Gradient w.r.t. a: use BLAS for efficiency
                  (when (tensor-requires-grad? a)
                    (let ((grad-a (case dtype
                                   ((f32) (make-f32vector n 0.0))
                                   ((f64) (make-f64vector n 0.0)))))
                      ;; Copy b to grad-a, then scale by grad-scalar
                      (case dtype
                        ((f32)
                         (sblit grad-a data-b)
                         (sscal! n grad-scalar grad-a))
                        ((f64)
                         (dblit grad-a data-b)
                         (dscal! n grad-scalar grad-a)))
                      (add-to-grad! a grad-a)))
                  
                  ;; Gradient w.r.t. b: use BLAS for efficiency
                  (when (tensor-requires-grad? b)
                    (let ((grad-b (case dtype
                                   ((f32) (make-f32vector n))
                                   ((f64) (make-f64vector n)))))
                      ;; Copy a to grad-b, then scale by grad-scalar
                      (case dtype
                        ((f32)
                         (sblit grad-b data-a)
                         (sscal! n grad-scalar grad-b))
                        ((f64)
                         (dblit grad-b data-a)
                         (dscal! n grad-scalar grad-b)))
                      (add-to-grad! b grad-b)))))
              (list a b)))
          result)))))
  
  ;;; ==================================================================
  ;;; Convolution Operations
  ;;; ==================================================================
  
  ;; im2col: Convert image to column matrix for convolution
  (define (im2col input kernel-h kernel-w stride-h stride-w pad-h pad-w)
    "Convert image tensor to column matrix.
   Input shape: (C, H, W)
   Output shape: (C*KH*KW, OH*OW) where OH, OW are output dims"
  
    (let* ((dtype (tensor-dtype input))
           (ishape (tensor-shape input))
           (C (car ishape))
           (H (cadr ishape))
           (W (caddr ishape))
           (data (tensor-data input))
         
           ;; Output dimensions
           (OH (+ 1 (quotient (+ H (* 2 pad-h) (- kernel-h)) stride-h)))
           (OW (+ 1 (quotient (+ W (* 2 pad-w) (- kernel-w)) stride-w)))

           ;; Column matrix dimensions
           (col-height (* C kernel-h kernel-w))
           (col-width (* OH OW))
           
           (col-data (case dtype
                       ((f32) (make-f32vector (* col-height col-width) 0.0))
                       ((f64) (make-f64vector (* col-height col-width) 0.0))))
           )
    
      ;; Fill column matrix
      (do ((c 0 (+ c 1)))
          ((= c C))
        (do ((kh 0 (+ kh 1)))
            ((= kh kernel-h))
          (do ((kw 0 (+ kw 1)))
              ((= kw kernel-w))
            
            ;; Current position in output column
            (let ((col-row (+ (* c kernel-h kernel-w)
                              (* kh kernel-w)
                              kw)))
            
              (do ((oh 0 (+ oh 1)))
                  ((= oh OH))
                (do ((ow 0 (+ ow 1)))
                    ((= ow OW))
                
                  ;; Corresponding position in input
                  (let* ((ih (+ (* oh stride-h) kh (- pad-h)))
                         (iw (+ (* ow stride-w) kw (- pad-w)))
                         (col-col (+ (* oh OW) ow))
                         (col-idx (+ (* col-row col-width) col-col)))
                    
                    ;; Copy value (with padding check)
                    (if (and (>= ih 0) (< ih H)
                             (>= iw 0) (< iw W))
                        (let ((input-idx (+ (* c H W)
                                            (* ih W)
                                            iw)))
                          (case dtype
                            ((f32) (f32vector-set! col-data col-idx
                                                   (f32vector-ref data input-idx)))
                            ((f64) (f64vector-set! col-data col-idx
                                                   (f64vector-ref data input-idx)))))
                        ;; Padding: value is already 0
                        #f))))))))
      
      (make-base-tensor col-data 
                        (list col-height col-width)
                        dtype
                        (tensor-requires-grad? input))
      ))

  ;; col2im: Reverse of im2col (for backward pass)
  (define (col2im col C H W kernel-h kernel-w stride-h stride-w pad-h pad-w)
    "Convert column matrix back to image tensor.
   Input shape: (C*KH*KW, OH*OW)
   Output shape: (C, H, W)"
  
    (let* ((dtype (tensor-dtype col))
           (col-shape (tensor-shape col))
           (col-data (tensor-data col))
           
           ;; Output dimensions
           (OH (+ 1 (quotient (+ H (* 2 pad-h) (- kernel-h)) stride-h)))
           (OW (+ 1 (quotient (+ W (* 2 pad-w) (- kernel-w)) stride-w)))
           (col-width (* OH OW))
           
           (img-data (case dtype
                       ((f32) (make-f32vector (* C H W) 0.0))
                       ((f64) (make-f64vector (* C H W) 0.0)))))
      
      ;; Accumulate values from column matrix
      (do ((c 0 (+ c 1)))
          ((= c C))
        (do ((kh 0 (+ kh 1)))
            ((= kh kernel-h))
          (do ((kw 0 (+ kw 1)))
              ((= kw kernel-w))
            
            (let ((col-row (+ (* c kernel-h kernel-w)
                              (* kh kernel-w)
                              kw)))
              
              (do ((oh 0 (+ oh 1)))
                  ((= oh OH))
                (do ((ow 0 (+ ow 1)))
                    ((= ow OW))
                  
                  (let* ((ih (+ (* oh stride-h) kh (- pad-h)))
                         (iw (+ (* ow stride-w) kw (- pad-w)))
                         (col-col (+ (* oh OW) ow))
                         (col-idx (+ (* col-row col-width) col-col)))
                    
                    ;; Accumulate value (with bounds check)
                    (when (and (>= ih 0) (< ih H)
                               (>= iw 0) (< iw W))
                      (let ((img-idx (+ (* c H W)
                                        (* ih W)
                                        iw)))
                        (case dtype
                          ((f32) (f32vector-set! img-data img-idx
                                                 (+ (f32vector-ref img-data img-idx)
                                                    (f32vector-ref col-data col-idx))))
                          ((f64) (f64vector-set! img-data img-idx
                                                 (+ (f64vector-ref img-data img-idx)
                                                    (f64vector-ref col-data col-idx))))))))))))))
    
      (make-base-tensor img-data 
                        (list C H W)
                        dtype
                        (tensor-requires-grad? col))
      )
    )


  (define (compensated-sum-f32vector vec start end)
    "Compute sum of f32vector elements from start to end using compensated summation."
    (let loop ((i start) (sum 0.0) (c 0.0))
      (if (= i end)
          sum
          (let* ((y (- (f32vector-ref vec i) c))   ; Subtract compensation
                 (t (+ sum y))                      ; New sum
                 (new-c (- (- t sum) y)))           ; Update compensation
            (loop (+ i 1) t new-c)))))

  (define (compensated-sum-f64vector vec start end)
    "Compute sum of f64vector elements from start to end using compensated summation."
    (let loop ((i start) (sum 0.0) (c 0.0))
      (if (= i end)
          sum
          (let* ((y (- (f64vector-ref vec i) c))
                 (t (+ sum y))
                 (new-c (- (- t sum) y)))
            (loop (+ i 1) t new-c)))))
  
  (define (conv2d input weight bias 
                  #!key 
                  (stride 1)
                  (padding 0)
                  (debug #f))
    "2D Convolution operation.
   Input shape: (Cin, H, W)
   Weight shape: (Cout, Cin, KH, KW)
   Bias shape: (Cout,)
   Output shape: (Cout, OH, OW)"
    
    (assert (eq? (tensor-dtype input) (tensor-dtype weight)))
  
  (let* ((dtype (tensor-dtype input))
         (ishape (tensor-shape input))
         (wshape (tensor-shape weight))
         
         (Cin (car ishape))
         (H (cadr ishape))
         (W (caddr ishape))
         
         (Cout (car wshape))
         (Cin-w (cadr wshape))
         (KH (caddr wshape))
         (KW (cadddr wshape))
         
         (stride-h stride)
         (stride-w stride)
         (pad-h padding)
         (pad-w padding)
         
         ;; Output dimensions
         (OH (+ 1 (quotient (+ H (* 2 pad-h) (- KH)) stride-h)))
         (OW (+ 1 (quotient (+ W (* 2 pad-w) (- KW)) stride-w)))

         (requires-grad? (or (tensor-requires-grad? input)
                            (tensor-requires-grad? weight)
                            (and bias (tensor-requires-grad? bias)))))
    
    (unless (= Cin Cin-w)
      (error 'conv2d "Input channels mismatch"))
    
    ;; Forward pass using im2col + matrix multiplication
    ;; 1. Convert input to column matrix
    (let* ((col (im2col input KH KW stride-h stride-w pad-h pad-w))
           (col-data (tensor-data col))
           
           ;; 2. Reshape weight to (Cout, Cin*KH*KW)
           (weight-data (tensor-data weight))
           (weight-rows Cout)
           (weight-cols (* Cin KH KW))
           
           ;; 3. Matrix multiply: output = weight @ col
           (output-data (case dtype
                         ((f32) (make-f32vector (* Cout OH OW) 0.0))
                         ((f64) (make-f64vector (* Cout OH OW) 0.0)))))

      (case dtype
        ((f32) (sgemm! RowMajor NoTrans NoTrans
                      weight-rows (* OH OW) weight-cols
                      1.0 weight-data col-data 
                      0.0 output-data lda: weight-cols
                      ldb: (* OH OW)
                      ldc: (* OH OW)))
        ((f64) (dgemm! RowMajor NoTrans NoTrans
                      weight-rows (* OH OW) weight-cols
                      1.0 weight-data
                      col-data 
                      0.0 output-data
                      lda: weight-cols
                      ldb: (* OH OW)
                      ldc: (* OH OW))))

      
      ;; 4. Add bias if provided
      (when bias
        (let ((bias-data (tensor-data bias)))
          (do ((cout 0 (+ cout 1)))
              ((= cout Cout))
            (let ((b (case dtype
                      ((f32) (f32vector-ref bias-data cout))
                      ((f64) (f64vector-ref bias-data cout)))))
              (do ((i 0 (+ i 1)))
                  ((= i (* OH OW)))
                (let ((idx (+ (* cout OH OW) i)))
                  (case dtype
                    ((f32) (f32vector-set! output-data idx
                                          (+ (f32vector-ref output-data idx) b)))
                    ((f64) (f64vector-set! output-data idx
                                          (+ (f64vector-ref output-data idx) b))))))))))
      
      ;; Create result tensor
      (let ((result (make-base-tensor output-data 
                                     (list Cout OH OW)
                                     dtype
                                     requires-grad?)))

        (when requires-grad?
          (set-backward-fn! result
            (lambda ()
              (let ((grad-out (tensor-grad result)))
                
                ;; Gradient w.r.t. input
                (when (tensor-requires-grad? input)
                  ;; dL/dInput = weight^T @ dL/dOutput (in column form)
                  ;; Then apply col2im
                  (let ((grad-col-data (case dtype
                                        ((f32) (make-f32vector (* weight-cols OH OW) 0.0))
                                        ((f64) (make-f64vector (* weight-cols OH OW) 0.0)))))
                    
                    (case dtype
                      ((f32) (sgemm! RowMajor Trans NoTrans
                                    weight-cols (* OH OW) weight-rows
                                    1.0 weight-data grad-out 
                                    0.0 grad-col-data
                                    lda: weight-cols
                                    ldb: (* OH OW)
                                    ldc: (* OH OW)))
                      ((f64) (dgemm! RowMajor Trans NoTrans
                                    weight-cols (* OH OW) weight-rows
                                    1.0 weight-data grad-out 
                                    0.0 grad-col-data
                                    lda: weight-cols
                                    ldb: (* OH OW)
                                    ldc: (* OH OW))))
                    
                    (let* ((grad-col (make-base-tensor grad-col-data
                                                      (list weight-cols (* OH OW))
                                                      dtype #f))
                           (grad-input (col2im grad-col Cin H W 
                                               KH KW stride-h stride-w pad-h pad-w)))
                      (add-to-grad! input (tensor-data grad-input)))))
                
                ;; Gradient w.r.t. weight
                (when (tensor-requires-grad? weight)
                  ;; dL/dWeight = dL/dOutput @ col^T
                  ;; Reshape to (Cout, Cin*KH*KW)
                  (let ((grad-weight-data (case dtype
                                           ((f32) (make-f32vector (* Cout weight-cols) 0.0))
                                           ((f64) (make-f64vector (* Cout weight-cols) 0.0)))))
                    
                    (case dtype
                      ((f32) (sgemm! RowMajor NoTrans Trans
                                    weight-rows weight-cols (* OH OW)
                                    1.0 grad-out 
                                    col-data 
                                    0.0 grad-weight-data
                                    lda: (* OH OW)
                                    ldb: (* OH OW)
                                    ldc: weight-cols))
                      ((f64) (dgemm! RowMajor NoTrans Trans
                                    weight-rows weight-cols (* OH OW)
                                    1.0 grad-out 
                                    col-data 
                                    0.0 grad-weight-data
                                    lda: (* OH OW)
                                    ldb: (* OH OW)
                                    ldc: weight-cols)))
                    
                    (add-to-grad! weight grad-weight-data)))
                
                ;; Gradient w.r.t. bias
                (when (and bias (tensor-requires-grad? bias))
                  ;; dL/dBias = sum of dL/dOutput over spatial dimensions
                  (let ((grad-bias-data (case dtype
                                         ((f32) (make-f32vector Cout 0.0))
                                         ((f64) (make-f64vector Cout 0.0))))
                        (vref (case dtype
                                ((f32) f32vector-ref)
                                ((f64) f64vector-ref))))
                    
                    (do ((cout 0 (+ cout 1)))
                        ((= cout Cout))
                      ;; Use compensated summation for better numerical accuracy
                      (let ((start-idx (* cout OH OW))
                            (end-idx (* (+ cout 1) OH OW)))
        
                        (case dtype
                          ((f32)
                           (let ((sum (compensated-sum-f32vector grad-out start-idx end-idx)))
                             (f32vector-set! grad-bias-data cout sum)))
                          ((f64)
                           (let ((sum (compensated-sum-f64vector grad-out start-idx end-idx)))
                             (f64vector-set! grad-bias-data cout sum))))))
                    
                    (add-to-grad! bias grad-bias-data)))))
            
            (if bias
                (list input weight bias)
                (list input weight))))
        
        result))))

  ;;; ==================================================================
  ;;; Activation Functions
  ;;; ==================================================================

  ;; ReLU activation
  (define (relu tensor)
    (let* ((dtype (tensor-dtype tensor))
           (data (tensor-data tensor))
           (n (vector-length-for-dtype data dtype))
           (result-data (case dtype
                          ((f32) (make-f32vector n 0.0))
                          ((f64) (make-f64vector n 0.0))))
           (requires-grad? (tensor-requires-grad? tensor)))
      
      ;; Forward: max(0, x)
      (case dtype
        ((f32)
         (do ((i 0 (+ i 1)))
             ((= i n))
           (f32vector-set! result-data i
                          (max 0.0 (f32vector-ref data i)))))
        ((f64)
         (do ((i 0 (+ i 1)))
             ((= i n))
           (f64vector-set! result-data i
                          (max 0.0 (f64vector-ref data i))))))
      
      (let ((result (make-base-tensor result-data (tensor-shape tensor) dtype requires-grad?)))
        (when requires-grad?
          (set-backward-fn! result
            (lambda ()
              (let ((grad-out (tensor-grad result))
                    (grad-in (case dtype
                              ((f32) (make-f32vector n 0.0))
                              ((f64) (make-f64vector n 0.0)))))
                ;; Gradient is 1 where x > 0, else 0
                (case dtype
                  ((f32)
                   (do ((i 0 (+ i 1)))
                       ((= i n))
                     (f32vector-set! grad-in i
                                    (if (> (f32vector-ref data i) 0.0)
                                        (f32vector-ref grad-out i)
                                        0.0))))
                  ((f64)
                   (do ((i 0 (+ i 1)))
                       ((= i n))
                     (f64vector-set! grad-in i
                                    (if (> (f64vector-ref data i) 0.0)
                                        (f64vector-ref grad-out i)
                                        0.0)))))
                (add-to-grad! tensor grad-in)))
             (list tensor)))
        result)))

  (define (tanh-op tensor)
  "Element-wise hyperbolic tangent: tanh(x) = (e^x - e^-x)/(e^x + e^-x)
   Gradient: d/dx tanh(x) = 1 - tanh^2(x) = sech^2(x)"
  
  (let* ((dtype (tensor-dtype tensor))
         (data (tensor-data tensor))
         (n (vector-length-for-dtype data dtype))
         (result-data (case dtype
                        ((f32) (make-f32vector n 0.0))
                        ((f64) (make-f64vector n 0.0))))
         (requires-grad? (tensor-requires-grad? tensor)))
    
    ;; Forward: apply tanh element-wise
    (case dtype
      ((f32)
       (do ((i 0 (+ i 1)))
           ((= i n))
         (f32vector-set! result-data i
                        (fptanh (f32vector-ref data i)))))
      ((f64)
       (do ((i 0 (+ i 1)))
           ((= i n))
         (f64vector-set! result-data i
                        (fptanh (f64vector-ref data i))))))
    
    (let ((result (make-base-tensor result-data (tensor-shape tensor) dtype requires-grad?)))
      (when requires-grad?
        (set-backward-fn! result
          (lambda ()
            ;; Gradient: (1 - tanh²(x)) * grad_out
            (let ((grad-out (tensor-grad result))
                  (grad-in (case dtype
                            ((f32) (make-f32vector n 0.0))
                            ((f64) (make-f64vector n 0.0)))))
              
              (case dtype
                ((f32)
                 (do ((i 0 (+ i 1)))
                     ((= i n))
                   (let ((tanh-val (f32vector-ref result-data i)))
                     (f32vector-set! grad-in i
                                    (* (f32vector-ref grad-out i)
                                       (- 1.0 (* tanh-val tanh-val)))))))
                ((f64)
                 (do ((i 0 (+ i 1)))
                     ((= i n))
                   (let ((tanh-val (f64vector-ref result-data i)))
                     (f64vector-set! grad-in i
                                    (* (f64vector-ref grad-out i)
                                       (- 1.0 (* tanh-val tanh-val))))))))
              
              (add-to-grad! tensor grad-in)))
          (list tensor)))
      result)))

  ;; ==================================================================
  ;; Sigmoid (Logistic) Operation
  ;; ==================================================================
  
  (define (sigmoid tensor)
    "Element-wise sigmoid: sigm(x) = 1 / (1 + e^-x)
     Gradient: d/dx sigm(x) = sigm(x) * (1 - sigm(x))"
    
  (let* ((dtype (tensor-dtype tensor))
         (data (tensor-data tensor))
         (n (vector-length-for-dtype data dtype))
         (result-data (case dtype
                        ((f32) (make-f32vector n 0.0))
                        ((f64) (make-f64vector n 0.0))))
         (requires-grad? (tensor-requires-grad? tensor)))
    
    ;; Forward: apply sigmoid element-wise
    (case dtype
      ((f32)
       (do ((i 0 (+ i 1)))
           ((= i n))
         (let ((x (f32vector-ref data i)))
           (f32vector-set! result-data i
                          (/ 1.0 (+ 1.0 (exp (- x))))))))
      ((f64)
       (do ((i 0 (+ i 1)))
           ((= i n))
         (let ((x (f64vector-ref data i)))
           (f64vector-set! result-data i
                          (/ 1.0 (+ 1.0 (exp (- x)))))))))
    
    (let ((result (make-base-tensor result-data (tensor-shape tensor) dtype requires-grad?)))
      (when requires-grad?
        (set-backward-fn! result
          (lambda ()
            ;; Gradient: σ(x) * (1 - σ(x)) * grad_out
            (let ((grad-out (tensor-grad result))
                  (grad-in (case dtype
                            ((f32) (make-f32vector n 0.0))
                            ((f64) (make-f64vector n 0.0)))))
              
              (case dtype
                ((f32)
                 (do ((i 0 (+ i 1)))
                     ((= i n))
                   (let ((sig-val (f32vector-ref result-data i)))
                     (f32vector-set! grad-in i
                                    (* (f32vector-ref grad-out i)
                                       sig-val
                                       (- 1.0 sig-val))))))
                ((f64)
                 (do ((i 0 (+ i 1)))
                     ((= i n))
                   (let ((sig-val (f64vector-ref result-data i)))
                     (f64vector-set! grad-in i
                                    (* (f64vector-ref grad-out i)
                                       sig-val
                                       (- 1.0 sig-val)))))))
              
              (add-to-grad! tensor grad-in)))
          (list tensor)))
      result)))


  ;; Sigmoid implementation for better numerical stability, especially
  ;; with large negative values:

  (define (sigmoid-stable tensor)
    "Numerically stable sigmoid implementation.
     Uses different formulas for positive and negative x to avoid overflow."
  
  (let* ((dtype (tensor-dtype tensor))
         (data (tensor-data tensor))
         (n (vector-length-for-dtype data dtype))
         (result-data (case dtype
                        ((f32) (make-f32vector n 0.0))
                        ((f64) (make-f64vector n 0.0))))
         (requires-grad? (tensor-requires-grad? tensor)))
    
    ;; Forward: numerically stable sigmoid
    (case dtype
      ((f32)
       (do ((i 0 (+ i 1)))
           ((= i n))
         (let ((x (f32vector-ref data i)))
           (f32vector-set! result-data i
                          (if (>= x 0.0)
                              ;; For x >= 0: sigm(x) = 1 / (1 + e^-x)
                              (/ 1.0 (+ 1.0 (exp (- x))))
                              ;; For x < 0: sigm(x) = e^x / (1 + e^x)
                              (let ((exp-x (exp x)))
                                (/ exp-x (+ 1.0 exp-x))))))))
      ((f64)
       (do ((i 0 (+ i 1)))
           ((= i n))
         (let ((x (f64vector-ref data i)))
           (f64vector-set! result-data i
                          (if (>= x 0.0)
                              (/ 1.0 (+ 1.0 (exp (- x))))
                              (let ((exp-x (exp x)))
                                (/ exp-x (+ 1.0 exp-x)))))))))
    
    (let ((result (make-base-tensor result-data (tensor-shape tensor) dtype requires-grad?)))
      (when requires-grad?
        (set-backward-fn! result
          (lambda ()
            (let ((grad-out (tensor-grad result))
                  (grad-in (case dtype
                            ((f32) (make-f32vector n 0.0))
                            ((f64) (make-f64vector n 0.0)))))
              
              (case dtype
                ((f32)
                 (do ((i 0 (+ i 1)))
                     ((= i n))
                   (let ((sig-val (f32vector-ref result-data i)))
                     (f32vector-set! grad-in i
                                    (* (f32vector-ref grad-out i)
                                       sig-val
                                       (- 1.0 sig-val))))))
                ((f64)
                 (do ((i 0 (+ i 1)))
                     ((= i n))
                   (let ((sig-val (f64vector-ref result-data i)))
                     (f64vector-set! grad-in i
                                    (* (f64vector-ref grad-out i)
                                       sig-val
                                       (- 1.0 sig-val)))))))
              
              (add-to-grad! tensor grad-in)))
          (list tensor)))
      result)))

  
  ;; Softplus: log(1 + e^x) - smooth approximation of ReLU
  (define (softplus tensor #!key (beta 1.0))
    "Softplus activation: log(1 + e^(beta*x)) / beta
   Gradient: sigmoid(beta*x)"
  
  (let* ((dtype (tensor-dtype tensor))
         (data (tensor-data tensor))
         (n (vector-length-for-dtype data dtype))
         (result-data (case dtype
                        ((f32) (make-f32vector n 0.0))
                        ((f64) (make-f64vector n 0.0))))
         (requires-grad? (tensor-requires-grad? tensor)))
    
    (case dtype
      ((f32)
       (do ((i 0 (+ i 1)))
           ((= i n))
         (let ((x (* beta (f32vector-ref data i))))
           (f32vector-set! result-data i
                          (/ (log (+ 1.0 (exp x))) beta)))))
      ((f64)
       (do ((i 0 (+ i 1)))
           ((= i n))
         (let ((x (* beta (f64vector-ref data i))))
           (f64vector-set! result-data i
                          (/ (log (+ 1.0 (exp x))) beta))))))
    
    (let ((result (make-base-tensor result-data (tensor-shape tensor) dtype requires-grad?)))
      (when requires-grad?
        (set-backward-fn! result
          (lambda ()
            (let ((grad-out (tensor-grad result))
                  (grad-in (case dtype
                            ((f32) (make-f32vector n 0.0))
                            ((f64) (make-f64vector n 0.0)))))
              
              ;; Gradient is sigmoid(beta*x)
              (case dtype
                ((f32)
                 (do ((i 0 (+ i 1)))
                     ((= i n))
                   (let* ((x (* beta (f32vector-ref data i)))
                          (sig (/ 1.0 (+ 1.0 (exp (- x))))))
                     (f32vector-set! grad-in i
                                    (* (f32vector-ref grad-out i) sig)))))
                ((f64)
                 (do ((i 0 (+ i 1)))
                     ((= i n))
                   (let* ((x (* beta (f64vector-ref data i)))
                          (sig (/ 1.0 (+ 1.0 (exp (- x))))))
                     (f64vector-set! grad-in i
                                    (* (f64vector-ref grad-out i) sig))))))
              
              (add-to-grad! tensor grad-in)))
          (list tensor)))
      result)))

  ;; Leaky ReLU: max(alpha*x, x)
  (define (leaky-relu tensor #!key (alpha 0.01))
  "Leaky ReLU activation: max(alpha*x, x)
   Gradient: 1 if x > 0, else alpha"
  
  (let* ((dtype (tensor-dtype tensor))
         (data (tensor-data tensor))
         (n (vector-length-for-dtype data dtype))
         (result-data (case dtype
                        ((f32) (make-f32vector n 0.0))
                        ((f64) (make-f64vector n 0.0))))
         (requires-grad? (tensor-requires-grad? tensor)))
    
    (case dtype
      ((f32)
       (do ((i 0 (+ i 1)))
           ((= i n))
         (let ((x (f32vector-ref data i)))
           (f32vector-set! result-data i
                          (if (> x 0.0) x (* alpha x))))))
      ((f64)
       (do ((i 0 (+ i 1)))
           ((= i n))
         (let ((x (f64vector-ref data i)))
           (f64vector-set! result-data i
                          (if (> x 0.0) x (* alpha x)))))))
    
    (let ((result (make-base-tensor result-data (tensor-shape tensor) dtype requires-grad?)))
      (when requires-grad?
        (set-backward-fn! result
          (lambda ()
            (let ((grad-out (tensor-grad result))
                  (grad-in (case dtype
                            ((f32) (make-f32vector n 0.0))
                            ((f64) (make-f64vector n 0.0)))))
              
              (case dtype
                ((f32)
                 (do ((i 0 (+ i 1)))
                     ((= i n))
                   (f32vector-set! grad-in i
                                  (if (> (f32vector-ref data i) 0.0)
                                      (f32vector-ref grad-out i)
                                      (* alpha (f32vector-ref grad-out i))))))
                ((f64)
                 (do ((i 0 (+ i 1)))
                     ((= i n))
                   (f64vector-set! grad-in i
                                  (if (> (f64vector-ref data i) 0.0)
                                      (f64vector-ref grad-out i)
                                      (* alpha (f64vector-ref grad-out i)))))))
              
              (add-to-grad! tensor grad-in)))
          (list tensor)))
      result)))


  (define (softmax x #!key (dim #f))
    "Softmax activation"
  
    (let* ((dtype (tensor-dtype x))
           (shape-x (tensor-shape x))
           (data-x (tensor-data x))
           (requires-grad? (tensor-requires-grad? x)))
      
      (unless (= (length shape-x) 1)
        (error 'softmax "Currently only supports 1D tensors"))
    
      (let* ((n (car shape-x))
             (result-data (case dtype
                            ((f32) (make-f32vector n 0.0))
                            ((f64) (make-f64vector n 0.0)))))
      
        ;; Find max
        (let ((max-val (case dtype
                         ((f32) (f32vector-fold max -inf.0 data-x))
                         ((f64) (f64vector-fold max -inf.0 data-x)))))
          
          ;; Compute exp(x - max) and sum with compensated summation
          (let ((exp-sum (case dtype
                           ((f32)
                            (let loop ((i 0) (sum 0.0) (c 0.0))
                              (if (= i n)
                                  sum
                                  (let* ((exp-val (exp (- (f32vector-ref data-x i) max-val)))
                                         ;; Store exp value for later
                                         (_ (f32vector-set! result-data i exp-val))
                                         ;; compensated summation
                                         (y (- exp-val c))
                                         (t (+ sum y))
                                         (new-c (- (- t sum) y)))
                                    (loop (+ i 1) t new-c)))))
                           ((f64)
                            (let loop ((i 0) (sum 0.0) (c 0.0))
                              (if (= i n)
                                  sum
                                  (let* ((exp-val (exp (- (f64vector-ref data-x i) max-val)))
                                         (_ (f64vector-set! result-data i exp-val))
                                         (y (- exp-val c))
                                         (t (+ sum y))
                                         (new-c (- (- t sum) y)))
                                    (loop (+ i 1) t new-c))))))))
            
            ;; Normalize using BLAS scal
            (case dtype
              ((f32) (sscal! n (/ 1.0 exp-sum) result-data))
              ((f64) (dscal! n (/ 1.0 exp-sum) result-data)))
            
            (let ((result (make-base-tensor result-data shape-x dtype requires-grad?)))
              
              (when requires-grad?
                (set-backward-fn! result
                                  (lambda ()
                                    (let ((grad-out (tensor-grad result)))
                                      
                                      ;; Use BLAS dot for efficiency
                                      (let* ((dot-prod (case dtype
                                                         ((f32) (sdot n grad-out result-data))
                                                         ((f64) (ddot n grad-out result-data))))
                                             (grad-x (case dtype
                                                       ((f32) (make-f32vector n 0.0))
                                                       ((f64) (make-f64vector n 0.0)))))
                                        
                                        ;; Method using BLAS operations:
                                        ;; 1. grad_x = grad_out (copy)
                                        ;; 2. grad_x -= dot_prod (subtract scalar from all elements)
                                        ;; 3. grad_x *= softmax (element-wise multiply)
                      
                                        (case dtype
                                          ((f32)
                                           ;; Copy grad_out to grad_x
                                           (sblit grad-x grad-out)
                                           ;; Subtract dot_prod from each element and multiply by softmax
                                           (do ((i 0 (+ i 1)))
                                               ((= i n))
                                             (f32vector-set! grad-x i
                                                             (* (f32vector-ref result-data i)
                                                                (- (f32vector-ref grad-x i)
                                                                   dot-prod)))))
                                          ((f64)
                                           (dblit grad-x grad-out)
                                           (do ((i 0 (+ i 1)))
                                               ((= i n))
                                             (f64vector-set! grad-x i
                                                             (* (f64vector-ref result-data i)
                                                                (- (f64vector-ref grad-x i)
                                                                   dot-prod))))))
                                        
                                        (add-to-grad! x grad-x))))
                                  
                                  (list x)))
              
              result))))))

  (define (log-softmax x #!key (dim #f))
    "Log-softmax: log(softmax(x))
     More numerically stable than computing log(softmax(x)) separately.
   
   Forward: log_softmax(x)[i] = x[i] - max(x) - log(sum(exp(x[j] - max(x))))
   
   Gradient: dL/dx = dL/dy - exp(log_softmax) * sum(dL/dy)"
  
    (let* ((dtype (tensor-dtype x))
           (shape-x (tensor-shape x))
           (data-x (tensor-data x))
           (requires-grad? (tensor-requires-grad? x)))
      
      (unless (= (length shape-x) 1)
        (error 'log-softmax "Currently only supports 1D tensors"))
      
      (let* ((n (car shape-x))
             (result-data (case dtype
                            ((f32) (make-f32vector n 0.0))
                            ((f64) (make-f64vector n 0.0)))))
      
        ;; Find max
        (let ((max-val (case dtype
                         ((f32) (f32vector-fold max -inf.0 data-x))
                         ((f64) (f64vector-fold max -inf.0 data-x)))))
          
          ;; Compute sum of exp(x - max)
          (let ((exp-sum (case dtype
                           ((f32)
                            (let loop ((i 0) (sum 0.0) (c 0.0))
                              (if (= i n)
                                  sum
                                  (let* ((exp-val (exp (- (f32vector-ref data-x i) max-val)))
                                         ;; Store exp value for later
                                         (_ (f32vector-set! result-data i exp-val))
                                         ;; compensated summation
                                         (y (- exp-val c))
                                         (t (+ sum y))
                                         (new-c (- (- t sum) y)))
                                    (loop (+ i 1) t new-c)))))
                           ((f64)
                            (let loop ((i 0) (sum 0.0) (c 0.0))
                              (if (= i n)
                                  sum
                                  (let* ((exp-val (exp (- (f64vector-ref data-x i) max-val)))
                                         (_ (f64vector-set! result-data i exp-val))
                                         (y (- exp-val c))
                                         (t (+ sum y))
                                         (new-c (- (- t sum) y)))
                                    (loop (+ i 1) t new-c))))))))
            
            (let ((log-sum-exp (log exp-sum)))
              
              ;; Compute: log_softmax[i] = x[i] - max - log_sum_exp
              (case dtype
                ((f32)
                 (do ((i 0 (+ i 1)))
                     ((= i n))
                   (f32vector-set! result-data i
                                   (- (f32vector-ref data-x i) max-val log-sum-exp))))
                ((f64)
                 (do ((i 0 (+ i 1)))
                     ((= i n))
                   (f64vector-set! result-data i
                                   (- (f64vector-ref data-x i) max-val log-sum-exp)))))
              
              (let ((result (make-base-tensor result-data shape-x dtype requires-grad?)))
                
                (when requires-grad?
                  (set-backward-fn! result
                                    (lambda ()
                                      (let ((grad-out (tensor-grad result)))
                                        
                                        ;; Compute sum of grad_out
                                        (let ((grad-sum (case dtype
                                                          ((f32)
                                                           (let loop ((i 0) (sum 0.0))
                                                             (if (= i n)
                                                                 sum
                                                                 (loop (+ i 1) 
                                                                       (+ sum (f32vector-ref grad-out i))))))
                                                          ((f64)
                                                           (let loop ((i 0) (sum 0.0))
                                                             (if (= i n)
                                                                 sum
                                                                 (loop (+ i 1) 
                                                                       (+ sum (f64vector-ref grad-out i)))))))))
                                          
                                          (let ((grad-x (case dtype
                                                          ((f32) (make-f32vector n 0.0))
                                                          ((f64) (make-f64vector n 0.0)))))
                                            
                                            ;; grad_x[i] = grad_out[i] - exp(log_softmax[i]) * grad_sum
                                            (case dtype
                                              ((f32)
                                               (do ((i 0 (+ i 1)))
                                                   ((= i n))
                                                 (f32vector-set! grad-x i
                                                                 (- (f32vector-ref grad-out i)
                                                                    (* (exp (f32vector-ref result-data i))
                                                                       grad-sum)))))
                                              ((f64)
                                               (do ((i 0 (+ i 1)))
                                                   ((= i n))
                                                 (f64vector-set! grad-x i
                                                                 (- (f64vector-ref grad-out i)
                                                                    (* (exp (f64vector-ref result-data i))
                                                                       grad-sum))))))
                                            
                                            (add-to-grad! x grad-x)))))
                                    
                                    (list x)))
                
                result)))))))

  ;;; ==================================================================
  ;;; Loss Functions
  ;;; ==================================================================

  ;; Mean Squared Error Loss
  (define (mse-loss pred target)
    (assert (eq? (tensor-dtype pred) (tensor-dtype target)))
    (let* ((dtype (tensor-dtype pred))
           (data-pred (tensor-data pred))
           (data-target (tensor-data target))
           (n (vector-length-for-dtype data-pred dtype))
           (requires-grad? (tensor-requires-grad? pred)))
      
      ;; Forward: L = (1/n) * \sum(pred - target)^2
      (define loss-value
        (case dtype
          ((f32)
           (let loop ((i 0) (sum 0.0))
             (if (= i n) (/ sum (exact->inexact n))
               (let ((diff (- (f32vector-ref data-pred i)
                              (f32vector-ref data-target i))))
                 (loop (+ i 1)
                       (+ sum (* diff diff)))))))
          ((f64)
           (let loop ((i 0) (sum 0.0))
             (if (= i n) (/ sum (exact->inexact n))
                 (let ((diff (- (f64vector-ref data-pred i)
                                (f64vector-ref data-target i))))
                   (loop (+ i 1)
                         (+ sum (* diff diff)))))))))
      
      ;; Create scalar tensor for loss
      (let ((loss-tensor (make-base-tensor
                         (case dtype
                           ((f32) (f32vector loss-value))
                           ((f64) (f64vector loss-value)))
                         '(1) dtype requires-grad?)))
        (when requires-grad?
          (set-backward-fn! loss-tensor
            (lambda ()
              ;; Gradient: dL/d pred_i = (2/n) * (pred_i - target_i)
              (let ((grad-pred (case dtype
                                ((f32) (make-f32vector n 0.0))
                                ((f64) (make-f64vector n 0.0))))
                    (scale-factor (/ 2.0 (exact->inexact n))))
                (case dtype
                  ((f32)
                   (do ((i 0 (+ i 1)))
                       ((= i n))
                     (f32vector-set! grad-pred i
                                    (* scale-factor
                                       (- (f32vector-ref data-pred i)
                                          (f32vector-ref data-target i))))))
                  ((f64)
                   (do ((i 0 (+ i 1)))
                       ((= i n))
                     (f64vector-set! grad-pred i
                                    (* scale-factor
                                       (- (f64vector-ref data-pred i)
                                          (f64vector-ref data-target i)))))))
                (add-to-grad! pred grad-pred)))
            (list pred)
            ))
        loss-tensor)))

  ;; Cross Entropy Loss (simplified, assumes softmax already applied)
  (define (cross-entropy-loss pred target)
    (assert (eq? (tensor-dtype pred) (tensor-dtype target)))
    (let* ((dtype (tensor-dtype pred))
           (data-pred (tensor-data pred))
           (data-target (tensor-data target))
           (n (vector-length-for-dtype data-pred dtype))
           (requires-grad? (tensor-requires-grad? pred)))

      
      ;; Forward: L = -\sum(target * log(pred))
      (define loss-value
        (case dtype
          ((f32)
           (let loop ((i 0) (sum 0.0))
             (if (= i n) (- sum)
                 (loop (+ i 1)
                       (+ sum (* (f32vector-ref data-target i)
                                 (log (max 1e-10 (f32vector-ref data-pred i)))))))))
          ((f64)
           (let loop ((i 0) (sum 0.0))
             (if (= i n) (- sum)
                 (loop (+ i 1)
                       (+ sum (* (f64vector-ref data-target i)
                                 (log (max 1e-10 (f64vector-ref data-pred i)))))))))))
      
      (let ((loss-tensor (make-base-tensor
                         (case dtype
                           ((f32) (f32vector loss-value))
                           ((f64) (f64vector loss-value)))
                         '(1) dtype requires-grad?)))
        (when requires-grad?
          (set-backward-fn!
           loss-tensor
           (lambda ()
                              
              ;; Gradient: dL/dpred_i = -target_i / pred_i
              (let ((grad-pred (case dtype
                                ((f32) (make-f32vector n 0.0))
                                ((f64) (make-f64vector n 0.0)))))
                (case dtype
                  ((f32)
                   (do ((i 0 (+ i 1)))
                       ((= i n))
                     (f32vector-set! grad-pred i
                                    (- (/ (f32vector-ref data-target i)
                                          (max 1e-10 (f32vector-ref data-pred i)))))))
                  ((f64)
                   (do ((i 0 (+ i 1)))
                       ((= i n))
                     (f64vector-set! grad-pred i
                                    (- (/ (f64vector-ref data-target i)
                                          (max 1e-10 (f64vector-ref data-pred i))))))))
                (add-to-grad! pred grad-pred)
                ))
            (list pred)))
        loss-tensor)))

  
  ;;; ==================================================================
  ;;; Utilities
  ;;; ==================================================================

  
  (define (rmsnorm x weight #!key (epsilon 1e-5))
    "Root Mean Square Normalization."
  
    (assert (eq? (tensor-dtype x) (tensor-dtype weight)))
    
    (let* ((dtype (tensor-dtype x))
           (data-x (tensor-data x))
           (data-w (tensor-data weight))
           (requires-grad? (or (tensor-requires-grad? x)
                               (tensor-requires-grad? weight))))
      
      (let* ((n (case dtype
                  ((f32) (f32vector-length data-x))
                  ((f64) (f64vector-length data-x))))
             
             ;; Compute RMS using BLAS dot
             (x-dot-x (case dtype
                        ((f32) (sdot n data-x data-x))
                        ((f64) (ddot n data-x data-x))))
             (rms (sqrt (+ epsilon (/ x-dot-x (exact->inexact n)))))
             (inv-rms (/ 1.0 rms))
             
             (result-data (case dtype
                            ((f32) (make-f32vector n 0.0))
                            ((f64) (make-f64vector n 0.0))))
             (normalized-data (case dtype
                                ((f32) (make-f32vector n 0.0))
                                ((f64) (make-f64vector n 0.0)))))
        
        ;; Copy and scale: normalized = x / rms (using BLAS)
        (case dtype
          ((f32)
           (sblit normalized-data data-x)
           (sscal! n inv-rms normalized-data))
          ((f64)
           (dblit normalized-data data-x)
           (dscal! n inv-rms normalized-data)))
        
        ;; Element-wise multiply with weight
        (case dtype
          ((f32)
           (do ((i 0 (+ i 1)))
               ((= i n))
             (f32vector-set! result-data i
                             (* (f32vector-ref normalized-data i)
                                (f32vector-ref data-w i)))))
          ((f64)
           (do ((i 0 (+ i 1)))
               ((= i n))
             (f64vector-set! result-data i
                             (* (f64vector-ref normalized-data i)
                                (f64vector-ref data-w i))))))
        
        (let ((result (make-base-tensor result-data (tensor-shape x) dtype requires-grad?)))
          
          (when requires-grad?
            (set-backward-fn! result
                              (lambda ()
                                (let ((grad-out (tensor-grad result)))
                                  
                                  ;; Gradient for weight
                                  (when (tensor-requires-grad? weight)
                                    (let ((grad-weight (case dtype
                                                         ((f32) (make-f32vector n 0.0))
                                                         ((f64) (make-f64vector n 0.0)))))
                                      (case dtype
                                        ((f32)
                                         (do ((i 0 (+ i 1)))
                                             ((= i n))
                                           (f32vector-set! grad-weight i
                                                           (* (f32vector-ref grad-out i)
                                                              (f32vector-ref normalized-data i)))))
                                        ((f64)
                                         (do ((i 0 (+ i 1)))
                                             ((= i n))
                                           (f64vector-set! grad-weight i
                                                           (* (f64vector-ref grad-out i)
                                                              (f64vector-ref normalized-data i))))))
                                      (add-to-grad! weight grad-weight)))
                                  
                                  ;; Gradient for x
                                  (when (tensor-requires-grad? x)
                                    (let ((grad-normalized (case dtype
                                                             ((f32) (make-f32vector n 0.0))
                                                             ((f64) (make-f64vector n 0.0)))))
                                      
                                      ;; grad_normalized = grad_out * weight
                                      (case dtype
                                        ((f32)
                                         (do ((i 0 (+ i 1)))
                                             ((= i n))
                                           (f32vector-set! grad-normalized i
                                                           (* (f32vector-ref grad-out i)
                                                              (f32vector-ref data-w i)))))
                                        ((f64)
                                         (do ((i 0 (+ i 1)))
                                             ((= i n))
                                           (f64vector-set! grad-normalized i
                                                           (* (f64vector-ref grad-out i)
                                                              (f64vector-ref data-w i))))))
                                      
                                      ;; Use BLAS for dot product
                                      (let* ((dot-prod (case dtype
                                                         ((f32) (sdot n grad-normalized data-x))
                                                         ((f64) (ddot n grad-normalized data-x))))
                                             (mean-term (/ dot-prod (exact->inexact n)))
                                             (grad-x (case dtype
                                                       ((f32) (make-f32vector n 0.0))
                                                       ((f64) (make-f64vector n 0.0)))))
                                        
                                        ;; Compute: grad_x = (grad_normalized - normalized * mean_term) / rms
                                        ;; Use BLAS for subtraction: grad_x = grad_normalized
                                        (case dtype
                                          ((f32)
                                           (sblit grad-x grad-normalized)
                                           ;; Subtract: grad_x -= normalized * mean_term
                                           (saxpy! n (- mean-term) normalized-data grad-x)
                                           ;; Scale: grad_x /= rms
                                           (sscal! n inv-rms grad-x))
                                          ((f64)
                                           (dblit grad-x grad-normalized)
                                           (daxpy! n (- mean-term) normalized-data grad-x)
                                           (dscal! n inv-rms grad-x)))
                                        
                                        (add-to-grad! x grad-x)))))
                                )
                                
                              (list x weight)))
            
          result)))
    )
    

  
  ; Normalized dot product (cosine similarity)
  (define (cosine-similarity a b)
    "Cosine similarity: (a \\dot b) / (||a|| * ||b||)"
    (let* ((dot-ab (dot-op a b))
           (dot-aa (dot-op a a))
           (dot-bb (dot-op b b))
           
           ;; Extract scalar values
           (dtype (tensor-dtype a))
           (ab (case dtype
                 ((f32) (f32vector-ref (tensor-data dot-ab) 0))
                 ((f64) (f64vector-ref (tensor-data dot-ab) 0))))
           (aa (case dtype
                 ((f32) (f32vector-ref (tensor-data dot-aa) 0))
                 ((f64) (f64vector-ref (tensor-data dot-aa) 0))))
           (bb (case dtype
                 ((f32) (f32vector-ref (tensor-data dot-bb) 0))
                 ((f64) (f64vector-ref (tensor-data dot-bb) 0))))
           
           ;; Compute cosine similarity
           (norm-a (sqrt aa))
           (norm-b (sqrt bb))
           (cos-sim (/ ab (* norm-a norm-b)))
           
           (result-data (case dtype
                          ((f32) (f32vector cos-sim))
                          ((f64) (f64vector cos-sim)))))
      
      (make-base-tensor result-data '(1) dtype #f)))

  ;; L2 normalization
  (define (l2-normalize tensor #!key (epsilon 1e-8))
    "L2 normalization: x / ||x||"
    (let* ((dot-self (dot-op tensor tensor))
           (dtype (tensor-dtype tensor))
           (norm-squared (case dtype
                           ((f32) (f32vector-ref (tensor-data dot-self) 0))
                           ((f64) (f64vector-ref (tensor-data dot-self) 0))))
           (norm (sqrt (+ norm-squared epsilon)))
           (n (vector-length-for-dtype (tensor-data tensor) dtype))
           (norm-tensor-data (case dtype
                               ((f32) (make-f32vector n norm))
                               ((f64) (make-f64vector n norm))))
           (norm-tensor (make-base-tensor norm-tensor-data
                                          (tensor-shape tensor)
                                          dtype
                                          #f)))
      (div tensor norm-tensor)))
  
  
  ;; Helper functions for tensor creation
  (define (vector-length-for-dtype vec dtype)
    (case dtype
      ((f32) (f32vector-length vec))
      ((f64) (f64vector-length vec))))
  
  (define (tensor->list tensor)
    (let* ((data (tensor-data tensor))
           (dtype (tensor-dtype tensor))
           (n (vector-length-for-dtype data dtype)))
      (case dtype
        ((f32) (f32vector->list data))
        ((f64) (f64vector->list data)))
      ))

  (define (flatten-tensor tensor)
    "Flatten a multi-dimensional tensor to 1D"
    (let* ((shape (tensor-shape tensor))
           (total-size (apply * shape)))
      (reshape tensor (list total-size))))

  (define (print-tensor tensor)
    (printf "Tensor(dtype=~a, shape=~a, requires_grad=~a)\n"
            (tensor-dtype tensor)
            (tensor-shape tensor)
            (tensor-requires-grad? tensor))
    (printf "  data: ~a\n" (tensor->list tensor))
    (when (tensor-grad tensor)
      (let* ((grad (tensor-grad tensor))
             (dtype (tensor-dtype tensor))
             (n (vector-length-for-dtype grad dtype)))
        (printf "  grad: ~a\n"
                (case dtype
                  ((f32) (f32vector->list grad))
                  ((f64) (f64vector->list grad))))
        ))
    )
  
  (define (f64vector-fold f x0 v . rest)
    (let ((n   (f64vector-length v))
	  (vs  (cons v rest)))
      (fold-ec x0 (:range i 0 n)
	       (map (lambda (v) (f64vector-ref v i)) vs)
	       (lambda (x ax) (apply f (append x (list ax)))))))

  (define (f32vector-fold f x0 v . rest)
    (let ((n   (f32vector-length v))
	  (vs  (cons v rest)))
      (fold-ec x0 (:range i 0 n)
	       (map (lambda (v) (f32vector-ref v i)) vs)
	       (lambda (x ax) (apply f (append x (list ax)))))))

  ;; Vector subset
  (define (dsub x start len)
    (subf64vector x start (+ start len)))
  (define (ssub x start len)
    (subf32vector x start (+ start len)))

  ;; Vector copy with offset
  (define (dblit out x #!key (Xoffset 0) (Yoffset 0) (size #f))
    (let ((size1 (or size (- (f64vector-length x) Xoffset))))
      (dicopy size1 x y: out offsetX: Xoffset offsetY: Yoffset)))
  
  ;; Vector copy with offset
  (define (sblit out x #!key (Xoffset 0) (Yoffset 0) (size #f))
    (let ((size1 (or size (- (f32vector-length x) Xoffset))))
      (sicopy size1 x y: out offsetX: Xoffset offsetY: Yoffset)))

) ;; end module

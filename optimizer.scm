;; YASOS-based optimizers for artificial neural networks

(module nanograd-optimizer
  (
   ;; Optimizer predicates
   optimizer? sgd? adam? adamw? rmsprop?

   ;; Optimizer constructors
   make-sgd make-adam make-adamw make-rmsprop
   
   ;; Optimizer operations
   step! get-learning-rate set-learning-rate!
   optimizer-state

   ;; Whole-vector update kernels
   register-optimizer-kernel! optimizer-kernel
   )
  
  (import
   scheme
   (chicken base)
   (chicken format)
   (srfi 1)
   (srfi 4)
   (srfi 69)  ; Hash tables
   yasos
   blas
   nanograd-autograd
   )

  ;; Hygienic macro for dtype-based operation dispatch
  (include "with-dtype.scm")

  ;;; ==================================================================
  ;;; Optimizer kernels
  ;;; ==================================================================

  ;; Kernels that perform an optimizer's whole update of one parameter
  ;; vector in a single call, keyed by (name . dtype).  An adam kernel is
  ;; a procedure (n hyper param grad m v) -> void updating the first n
  ;; elements of the SRFI-4 vectors param, m and v (and grad, which
  ;; receives the weight-decay term) in place.  hyper is an f64vector
  ;; holding the learning rate, beta1, beta2, epsilon, the two bias
  ;; corrections 1 - beta1^t and 1 - beta2^t, and the weight decay.
  ;; An adamw kernel has the same arguments, but applies the weight decay
  ;; by scaling param by 1 - lr * wd before the Adam update, and leaves
  ;; grad unchanged.  Adam with decoupled weight decay uses the adamw
  ;; kernel when one is registered; otherwise it scales param itself and
  ;; calls the adam kernel with a weight decay of 0.
  (define *optimizer-kernels* '())

  (define (register-optimizer-kernel! name dtype kernel)
    "Use kernel for the updates of optimizer name (a symbol, such as adam)
    on dtype ('f32 or 'f64) parameters.  A kernel of #f removes the
    registration, so that the optimizer's Scheme code is used again."
    (unless (memq dtype '(f32 f64))
      (error 'register-optimizer-kernel! "unsupported dtype" dtype))
    (unless (or (not kernel) (procedure? kernel))
      (error 'register-optimizer-kernel! "kernel must be a procedure or #f" kernel))
    (set! *optimizer-kernels*
          (let ((rest (remove (lambda (e) (equal? (car e) (cons name dtype)))
                              *optimizer-kernels*)))
            (if kernel (cons (cons (cons name dtype) kernel) rest) rest))))

  (define (optimizer-kernel name dtype)
    "The kernel registered for optimizer name on dtype parameters, or #f."
    (let ((e (assoc (cons name dtype) *optimizer-kernels*)))
      (and e (cdr e))))

  
  ;;; ==================================================================
  ;;; Optimizer Base Operations
  ;;; ==================================================================

  (define-predicate optimizer?)
  (define-predicate sgd?)
  (define-predicate adam?)
  (define-predicate adamw?)
  (define-predicate rmsprop?)
  
  (define-operation (step! opt))
  (define-operation (get-learning-rate opt))
  (define-operation (set-learning-rate! opt lr))
  (define-operation (optimizer-state opt))

  ;;; ==================================================================
  ;;; SGD (Stochastic Gradient Descent) Optimizer
  ;;; ==================================================================

  (define (make-sgd parameters 
                    #!key 
                    (learning-rate 0.01)
                    (momentum 0.0)
                    (weight-decay 0.0)
                    (nesterov #f))
    (let ((lr learning-rate)
          (mom momentum)
          (wd weight-decay)
          (use-nesterov nesterov)
          ;; Velocity buffers for momentum (hash table: tensor -> velocity)
          (velocity-buffers (make-hash-table eq?)))
      
      ;; Initialize velocity buffers if using momentum
      (when (> momentum 0.0)
        (for-each
         (lambda (param)
           (let* ((dtype (tensor-dtype param))
                  (data (tensor-data param))
                  (n (vector-length-for-dtype data dtype))
                  (v (with-dtype dtype (vec n 0.0))))

             (hash-table-set! velocity-buffers param v)))
         parameters))
      
      (object
       ;; Type predicates
       ((optimizer? self) #t)
       ((sgd? self) #t)
       
       ;; Learning rate accessors
       ((get-learning-rate self) lr)
       ((set-learning-rate! self new-lr) (set! lr new-lr))
       
       ;; Optimizer state
       ((optimizer-state self)
        `((learning-rate . ,lr)
          (momentum . ,mom)
          (weight-decay . ,wd)
          (nesterov . ,use-nesterov)))
       
       ;; Perform optimization step
       ((step! self)
        (for-each
         (lambda (param)
           (let* ((dtype (tensor-dtype param))
                  (data (tensor-data param))
                  (grad (tensor-grad param))
                  (n (vector-length-for-dtype data dtype)))
             
             (when grad
               ;; Apply weight decay if specified
               (when (> wd 0.0)
                 (with-dtype dtype
                   (axpy! n wd data grad)))
               
               (if (> mom 0.0)
                   ;; SGD with momentum
                   (let ((v (hash-table-ref velocity-buffers param)))
                     (with-dtype dtype
                        ;; v = momentum * v + grad
                        (scal! n mom v)
                        (axpy! n 1.0 grad v)
                        
                        (if use-nesterov
                            ;; Nesterov: param = param - lr * (momentum * v + grad)
                            (let ((update (vec n 0.0)))
                              (axpy! n mom v update)
                              (axpy! n 1.0 grad update)
                              (axpy! n (- lr) update data))
                            ;; Standard: param = param - lr * v
                            (axpy! n (- lr) v data))))
                   
                   ;; Standard SGD: param = param - lr * grad
                   (with-dtype dtype (axpy! n (- lr) grad data)))
               )))
         parameters)))))

  ;;; ==================================================================
  ;;; Adam Optimizer
  ;;; ==================================================================

  (define (make-adam parameters
                     #!key
                     (learning-rate 0.001)
                     (beta1 0.9)
                     (beta2 0.999)
                     (epsilon 1e-8)
                     (weight-decay 0.0)
                     (decoupled-weight-decay #f))
    "Adam optimizer over the list of parameter tensors PARAMETERS.

     weight-decay is applied in one of two ways:
       decoupled-weight-decay #f (default): L2 regularization; wd * p is
         added to the gradient before the moment updates.
       decoupled-weight-decay #t (AdamW, Loshchilov and Hutter): each
         parameter is first scaled by (1 - lr * wd), and the Adam update
         then uses the plain gradient."

    (let ((lr learning-rate)
          (b1 beta1)
          (b2 beta2)
          (eps epsilon)
          (wd weight-decay)
          (t 0)  ; Time step
          ;; First moment (mean) buffers
          (m-buffers (make-hash-table eq?))
          ;; Second moment (variance) buffers
          (v-buffers (make-hash-table eq?)))
      
      ;; Initialize moment buffers
      (for-each
       (lambda (param)
         (let* ((dtype (tensor-dtype param))
                (data (tensor-data param))
                (n (vector-length-for-dtype data dtype))
                (m (with-dtype dtype (vec n 0.0)))
                (v (with-dtype dtype (vec n 0.0))))
           (hash-table-set! m-buffers param m)
           (hash-table-set! v-buffers param v)))
       parameters)
      
      (object
       ;; Type predicates
       ((optimizer? self) #t)
       ((adam? self) #t)
       ((adamw? self) decoupled-weight-decay)

       ;; Learning rate accessors
       ((get-learning-rate self) lr)
       ((set-learning-rate! self new-lr) (set! lr new-lr))
       
       ;; Optimizer state
       ((optimizer-state self)
        `((learning-rate . ,lr)
          (beta1 . ,b1)
          (beta2 . ,b2)
          (epsilon . ,eps)
          (weight-decay . ,wd)
          (decoupled-weight-decay . ,decoupled-weight-decay)
          (step . ,t)))
       
       ;; Perform optimization step
       ((step! self)
        (set! t (+ t 1))
        
        ;; Bias correction terms
        (let ((bias-correction1 (- 1.0 (expt b1 t)))
              (bias-correction2 (- 1.0 (expt b2 t))))
          
          (for-each
           (lambda (param)
             (let* ((dtype (tensor-dtype param))
                    (data (tensor-data param))
                    (grad (tensor-grad param))
                    (n (vector-length-for-dtype data dtype)))

               (when grad
                 (let* ((m (hash-table-ref m-buffers param))
                        (v (hash-table-ref v-buffers param))
                        (adamw-kernel (and decoupled-weight-decay
                                           (optimizer-kernel 'adamw dtype)))
                        (kernel (or adamw-kernel (optimizer-kernel 'adam dtype)))
                        ;; Decoupled decay not handled by an adamw kernel
                        (scale-here? (and decoupled-weight-decay (not adamw-kernel)))
                        ;; Weight decay passed to the kernel
                        (kernel-wd (if scale-here? 0.0 wd))
                        ;; Weight decay that enters through the gradient
                        (coupled-wd (if decoupled-weight-decay 0.0 wd)))
                  ;; Decoupled weight decay: p <- (1 - lr * wd) p
                  (when (and scale-here? (> wd 0.0))
                    (with-dtype dtype (scal! n (- 1.0 (* lr wd)) data)))
                  (if kernel
                   (kernel n (f64vector lr b1 b2 eps bias-correction1 bias-correction2 kernel-wd)
                           data grad m v)
                   (begin
                   ;; Apply weight decay
                   (when (> coupled-wd 0.0)
                     (with-dtype dtype (axpy! n coupled-wd data grad)))

                   (with-dtype dtype
                      ;; Update biased first moment: m = beta1 * m + (1-beta1) * grad
                      (scal! n b1 m)
                      (axpy! n (- 1.0 b1) grad m)
                      
                      ;; Update biased second moment: v = beta2 * v + (1-beta2) * grad^2
                      (scal! n b2 v)
                      (do ((i 0 (+ i 1)))
                          ((= i n))
                        (let ((g (elt-ref grad i)))
                          (elt-set! v i
                                    (+ (elt-ref v i)
                                       (* (- 1.0 b2) g g)))))
                      
                      ;; Compute bias-corrected step
                      (do ((i 0 (+ i 1)))
                          ((= i n))
                        (let ((m-hat (/ (elt-ref m i) bias-correction1))
                              (v-hat (/ (elt-ref v i) bias-correction2)))
                          (elt-set! data i
                                          (- (elt-ref data i)
                                             (/ (* lr m-hat)
                                                (+ (sqrt v-hat) eps))))))
                      ))))
                   ))
                 )
           parameters))))))

  ;;; ==================================================================
  ;;; AdamW Optimizer
  ;;; ==================================================================

  (define (make-adamw parameters
                      #!key
                      (learning-rate 0.001)
                      (beta1 0.9)
                      (beta2 0.999)
                      (epsilon 1e-8)
                      (weight-decay 0.01))
    "AdamW: Adam with decoupled weight decay.  Each step first scales
     every parameter by (1 - learning-rate * weight-decay) and then
     applies the Adam update computed from the plain gradient.  The
     result answers #t to both adam? and adamw?."
    (make-adam parameters
               learning-rate: learning-rate
               beta1: beta1
               beta2: beta2
               epsilon: epsilon
               weight-decay: weight-decay
               decoupled-weight-decay: #t))

  ;;; ==================================================================
  ;;; RMSprop Optimizer
  ;;; ==================================================================

  (define (make-rmsprop parameters
                        #!key
                        (learning-rate 0.01)
                        (alpha 0.99)
                        (epsilon 1e-8)
                        (weight-decay 0.0)
                        (momentum 0.0))
    
    (let ((lr learning-rate)
          (a alpha)
          (eps epsilon)
          (wd weight-decay)
          (mom momentum)
          ;; Square average buffers
          (v-buffers (make-hash-table eq?))
          ;; Momentum buffers
          (m-buffers (make-hash-table eq?)))
      
      ;; Initialize buffers
      (for-each
       (lambda (param)
         (let* ((dtype (tensor-dtype param))
                (data (tensor-data param))
                (n (vector-length-for-dtype data dtype))
                (v (with-dtype dtype (vec n 0.0))))
           (hash-table-set! v-buffers param v)
           (when (> momentum 0.0)
             (let ((m (with-dtype dtype (vec n 0.0))))
               (hash-table-set! m-buffers param m)))))
       parameters)
      
      (object
       ;; Type predicates
       ((optimizer? self) #t)
       ((rmsprop? self) #t)
       
       ;; Learning rate accessors
       ((get-learning-rate self) lr)
       ((set-learning-rate! self new-lr) (set! lr new-lr))
       
       ;; Optimizer state
       ((optimizer-state self)
        `((learning-rate . ,lr)
          (alpha . ,a)
          (epsilon . ,eps)
          (weight-decay . ,wd)
          (momentum . ,mom)))
       
       ;; Perform optimization step
       ((step! self)
        (for-each
         (lambda (param)
           (let* ((dtype (tensor-dtype param))
                  (data (tensor-data param))
                  (grad (tensor-grad param))
                  (n (vector-length-for-dtype data dtype)))
             
             (when grad
               ;; Apply weight decay
               (when (> wd 0.0)
                 (with-dtype dtype (axpy! n wd data grad)))
               
               (let ((v (hash-table-ref v-buffers param)))
                 (with-dtype dtype
                    ;; Update running average: v = alpha * v + (1-alpha) * grad^2
                    (scal! n a v)
                    (do ((i 0 (+ i 1)))
                        ((= i n))
                      (let ((g (elt-ref grad i)))
                        (elt-set! v i
                                  (+ (elt-ref v i)
                                     (* (- 1.0 a) g g)))))
                    
                    (if (> mom 0.0)
                        ;; With momentum
                        (let ((m (hash-table-ref m-buffers param)))
                          (do ((i 0 (+ i 1)))
                              ((= i n))
                            (let* ((g (elt-ref grad i))
                                   (avg (elt-ref v i))
                                   (buf (elt-ref m i))
                                   (new-buf (+ (* mom buf)
                                              (/ (* lr g)
                                                 (+ (sqrt avg) eps)))))
                              (elt-set! m i new-buf)
                              (elt-set! data i
                                             (- (elt-ref data i) new-buf)))))
                        ;; Without momentum
                        (do ((i 0 (+ i 1)))
                            ((= i n))
                          (let ((g (elt-ref grad i))
                                (avg (elt-ref v i)))
                            (elt-set! data i
                                           (- (elt-ref data i)
                                              (/ (* lr g)
                                                 (+ (sqrt avg) eps))))))))
                   
                    ))))
         parameters)))))

) ;; end module

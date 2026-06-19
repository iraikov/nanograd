;;; nanograd/array-morphisms-nanograd.scm
;;;
;;; Integration: lazy (array-morphisms-backed) tensors for nanograd.
;;;
;;; Extends the YASOS tensor protocol with non-strict tensors that wrap
;;; morph-variable nodes from array-morphisms-grad.  The existing nanograd
;;; backward! topo-sort and all optimizers work unchanged because:
;;;
;;;   1. make-lazy-tensor implements the full YASOS tensor protocol.
;;;   2. tensor-data and tensor-grad return SRFI-4 vectors (realized
;;;      on demand), satisfying the optimizer's expectations.
;;;   3. tensor-children and tensor-backward-fn route through the AM
;;;      gradient graph, so backward! on a lazy tensor delegates to
;;;      array-morphisms-grad:backward! which handles morphism gradients.
;;;
;;; Key optimizer enhancement: am-realize-grads/ctx realizes ALL parameter
;;; gradient morphisms through a shared morphism-context, enabling the
;;; context's graph-coloring buffer allocator to find reuse opportunities
;;; ACROSS the entire backward graph.  After the first training step the
;;; context is finalized and subsequent steps replay into the buffer pool.

(module nanograd-array-morphisms

  (;; Core lazy tensor
   make-lazy-tensor
   lazy-tensor-morph-variable
   get-or-make-lazy

   ;; Activation bridge
   activation-forward-am

   ;; Loss functions
   am-mse-loss
   am-cross-entropy-loss

   ;; AM-backed layers (satisfy nanograd layer? protocol)
   make-am-dense-layer
   make-am-conv2d-layer
   make-am-batch-norm
   make-am-sequential

   ;; Parameter / gradient helpers
   am-parameters
   am-zero-grad!

   ;; Memory reuse integration
   make-am-training-context
   am-forward/ctx
   am-backward!/ctx
   am-realize-grads/ctx
   am-training-step
   am-training-step/ssa
   am-finalize-context!
   am-reset-context!

   ;; Buffer utilities
   fresh-copy-morphism)

  (import scheme (chicken base) (chicken random))
  (import (only srfi-1 map filter fold for-each any every append-map filter-map iota))
  (import srfi-4)
  (import (only srfi-69 make-hash-table hash-table-ref/default hash-table-set!))
  (import yasos)
  (import datatype matchable)
  (import array-morphisms-core)
  (import array-morphisms-basic-ops)
  (import array-morphisms-structural-ops)
  (import array-morphisms-realization)
  (import array-morphisms-context)
  ;; Prefix all array-morphisms-grad exports with "am:" to avoid name
  ;; collisions with the YASOS operations of the same name from nanograd-autograd.
  (import (prefix array-morphisms-grad am:))
  (import array-morphisms-ssa)
  (import nanograd-autograd)
  (import nanograd-layer)
  (import (only nanograd-optimizer step!))

  ;; YASOS operation to retrieve sub-layers from a sequential model.
  ;; Defined here (not in nanograd-layer) to keep changes local.
  (define-operation (inner-layers layer))


  ;;;; ================================================================
  ;;;; morph-variable <-> lazy-tensor identity caches
  ;;;;
  ;;;; Each morph-variable maps to at most one lazy-tensor and vice
  ;;;; versa.  get-or-make-lazy is the canonical constructor that
  ;;;; maintains this 1-to-1 correspondence.
  ;;;; ================================================================

  ;; Keyed by (am:var-name mv) — a gensym symbol — rather than the mv
  ;; record itself.  Symbol hashing in CHICKEN is content-based (the name
  ;; string), so the key is stable across minor GC moves.  Keying by the
  ;; record address would break: a minor GC that moves the record changes
  ;; its address and therefore its hash bucket, causing lookups to miss.
  (define *mv->lazy* (make-hash-table))

  (define (lazy-tensor-morph-variable lt)
    "Return the morph-variable underlying lazy tensor lt, or #f.
     Dispatches the lazy-tensor-mv method directly on the YASOS object
     rather than using an address-keyed hash table (which breaks after GC)."
    (and (lazy-tensor? lt) (lazy-tensor-mv lt)))

  (define (get-or-make-lazy mv)
    "Return the canonical lazy-tensor for morph-variable mv,
     creating it if necessary."
    (or (hash-table-ref/default *mv->lazy* (am:var-name mv) #f)
        (make-lazy-tensor mv)))


  ;;;; ================================================================
  ;;;; Type conversion helpers
  ;;;; ================================================================

  (define (morphism->srfi4 m)
    "Realize morphism m and extract its SRFI-4 data vector."
    (cases array-morphism (realize m)
      (concrete-array (data shape strides offset dtype alloc-id batch-axis)
        data)
      (else (error "morphism->srfi4: realization produced non-concrete result"))))

  (define (srfi4->morphism vec shape dtype)
    "Wrap a SRFI-4 vector as a zero-copy concrete-array morphism."
    (make-morphism vec (vector->list shape) dtype))

  (define (delta->morphism delta shape dtype)
    "Convert any gradient delta to a morphism.
     Accepts: array-morphism, lazy-tensor, strict tensor, or raw SRFI-4 vector."
    (cond
      ((array-morphism? delta) delta)
      ((lazy-tensor? delta)
       (am:var-value (lazy-tensor-morph-variable delta)))
      ((tensor? delta)
       (srfi4->morphism (tensor-data delta)
                        (list->vector (tensor-shape delta))
                        (tensor-dtype delta)))
      (else
       ;; raw SRFI-4 vector from existing nanograd backward-fns
       (srfi4->morphism delta shape dtype))))


  ;;;; ================================================================
  ;;;; make-lazy-tensor
  ;;;;
  ;;;; Creates a YASOS object implementing the full nanograd tensor
  ;;;; protocol.  The object wraps a morph-variable mv.
  ;;;;
  ;;;; Shape / dtype / requires-grad? are read from the morphism
  ;;;; without triggering realization.
  ;;;;
  ;;;; tensor-data: zero-copy for concrete-array parameters; triggers
  ;;;;   realize for abstract morphism-expr values.
  ;;;; tensor-grad: returns a freshly realized SRFI-4 vector each call
  ;;;;   (or #f if no gradient has been accumulated).
  ;;;;   Note: after am-realize-grads/ctx the gradient morphism is a
  ;;;;   concrete-array, so realize is instantaneous (data extraction).
  ;;;;
  ;;;; backward!: delegates to am:backward! mv (array-morphisms-grad)
  ;;;;   which knows how to handle lazy morphism gradients.  Does NOT
  ;;;;   use backward-impl! from nanograd-autograd (which expects SRFI-4
  ;;;;   gradient buffers for seeding).
  ;;;; ================================================================

  (define (make-lazy-tensor mv)
    "Create a YASOS lazy tensor wrapping morph-variable mv."
    (let ((lt
      (object
       ;; ---- Type predicates ----
       ((tensor?        self) #t)
       ((tensor32?      self) (eq? (morph-dtype (am:var-value mv)) 'f32))
       ((tensor64?      self) (eq? (morph-dtype (am:var-value mv)) 'f64))
       ((strict-tensor? self) #f)
       ((lazy-tensor?   self) #t)

       ;; ---- Stable morph-variable accessor ----
       ;; Returns mv directly from the closure — no hash table needed.
       ;; This is GC-safe because mv is a live closure variable, not a
       ;; hash table key that could become stale after a minor GC moves mv.
       ((lazy-tensor-mv self) mv)

       ;; ---- Shape / dtype (no realization) ----
       ((tensor-shape self)
        (vector->list (morph-shape (am:var-value mv))))
       ((tensor-dtype self)
        (morph-dtype (am:var-value mv)))
       ((tensor-requires-grad? self)
        (am:var-requires-grad? mv))

       ;; ---- Data access ----
       ;; concrete-array: extract SRFI-4 vector zero-copy.
       ;; morphism-expr: realize on demand (may allocate).
       ((tensor-data self)
        (cases array-morphism (am:var-value mv)
          (concrete-array (data shape strides offset dtype alloc-id batch-axis)
            data)
          (else (morphism->srfi4 (am:var-value mv)))))

       ;; ---- Force to strict tensor ----
       ((tensor-force self)
        (make-base-tensor (tensor-data self)
                          (tensor-shape self)
                          (tensor-dtype self)
                          #f))

       ;; ---- Morphism access ----
       ((tensor-morphism-value self) (am:var-value mv))

       ;; ---- Gradient (realized on demand) ----
       ((tensor-grad self)
        (let ((mg (am:var-grad mv)))
          (if mg (morphism->srfi4 mg) #f)))

       ;; ---- Gradient accumulation ----
       ;; Accepts SRFI-4 vector (from strict backward-fns), strict tensor,
       ;; lazy tensor, or morphism.
       ((add-to-grad! self delta)
        (let ((g-morph (delta->morphism delta
                                        (morph-shape (am:var-value mv))
                                        (morph-dtype (am:var-value mv)))))
          (am:accumulate-grad! mv g-morph)))

       ;; ---- Gradient zeroing ----
       ((zero-grad! self) (am:zero-grad! mv))

       ;; ---- set-grad! ----
       ((set-grad! self g)
        (am:zero-grad! mv)
        (am:accumulate-grad! mv
          (delta->morphism g
                           (morph-shape (am:var-value mv))
                           (morph-dtype (am:var-value mv)))))

       ;; ---- Children for backward topo-sort ----
       ;; Maps AM parents to lazy tensors, then filters by requires-grad?.
       ((tensor-children self)
        (filter tensor-requires-grad?
                (map get-or-make-lazy (am:var-parents mv))))

       ;; ---- Backward function ----
       ;; Called by backward-impl! if used in a mixed strict+lazy graph.
       ;; For pure lazy graphs, backward! on a lazy tensor calls am:backward!
       ;; directly and this method is not invoked.
       ((tensor-backward-fn self)
        (let ((gfn (am:var-grad-fn mv)))
          (if gfn
              (lambda ()
                (let ((g (am:var-grad mv)))
                  (when g (gfn g))))
              #f)))

       ;; ---- set-backward-fn! ----
       ;; No-op: the AM gradient graph owns the backward connections.
       ((set-backward-fn! self fn inputs) (if #f #f))

       ;; ---- backward! ----
       ;; Delegates to array-morphisms-grad:backward! which handles
       ;; morphism gradients (morph-ones-like seed, lazy accumulation).
       ((backward! self)
        (when (am:var-requires-grad? mv)
          (am:backward! mv)))

       ;; ---- Structural operations (zero-copy) ----
       ((reshape self new-shape)
        (get-or-make-lazy (am:var-reshape mv new-shape)))
       ((transpose-tensor self axes)
        (get-or-make-lazy (am:var-transpose mv axes)))

       )))
      ;; Cache by stable gensym symbol so get-or-make-lazy can deduplicate.
      (hash-table-set! *mv->lazy* (am:var-name mv) lt)
      lt))


  ;;;; ================================================================
  ;;;; Activation bridge
  ;;;;
  ;;;; Applies a nanograd activation object to a lazy tensor using
  ;;;; array-morphisms-grad var-* operations (lazy, no realization).
  ;;;; ================================================================

  (define (activation-forward-am act lt)
    "Apply nanograd activation act to lazy tensor lt.
     Returns a lazy tensor built from AM var-* operations."
    (let* ((mv (lazy-tensor-morph-variable lt))
           (dt (morph-dtype (am:var-value mv)))
           (name (activation-name act)))
      (cond
        ((string=? name "Identity") lt)

        ((string=? name "ReLU")
         ;; relu(x) = 0.5 * (x + |x|)
         (let* ((abs-x  (am:var-abs mv))
                (sum    (am:var+ mv abs-x))
                (half   (am:make-var (morph-from-list '(0.5) '(1) dt) #f)))
           (get-or-make-lazy (am:var* sum half))))

        ((string=? name "Sigmoid")
         ;; sigmoid(x) = 1 / (1 + exp(-x))
         (let* ((neg-x  (am:var-negate mv))
                (ex     (am:var-exp neg-x))
                (one    (am:make-var (morph-from-list '(1.0) '(1) dt) #f))
                (denom  (am:var+ one ex)))
           (get-or-make-lazy (am:var/ one denom))))

        ((string=? name "Tanh")
         ;; tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
         (let* ((two    (am:make-var (morph-from-list '(2.0) '(1) dt) #f))
                (two-x  (am:var* mv two))
                (e2x    (am:var-exp two-x))
                (one    (am:make-var (morph-from-list '(1.0) '(1) dt) #f)))
           (get-or-make-lazy
            (am:var/ (am:var- e2x one)
                     (am:var+ e2x one)))))

        ((string=? name "GeLU")
         ;; gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
         (let* ((pi-factor 0.7978845608028654) ; sqrt(2/pi)
                (pf    (am:make-var (morph-from-list (list pi-factor) '(1) dt) #f))
                (c     (am:make-var (morph-from-list '(0.044715) '(1) dt) #f))
                (three (am:make-var (morph-from-list '(3.0) '(1) dt) #f))
                (two   (am:make-var (morph-from-list '(2.0) '(1) dt) #f))
                (half  (am:make-var (morph-from-list '(0.5) '(1) dt) #f))
                (one   (am:make-var (morph-from-list '(1.0) '(1) dt) #f))
                (x3    (am:var-pow mv three))
                (inner (am:var+ mv (am:var* c x3)))
                (arg   (am:var* pf inner))
                (t     (let* ((e2  (am:var* two arg))
                              (e2x (am:var-exp e2))
                              (em1 (am:var- e2x one))
                              (ep1 (am:var+ e2x one)))
                         (am:var/ em1 ep1)))
                (cdf   (am:var* half (am:var+ one t))))
           (get-or-make-lazy (am:var* mv (am:var* half cdf)))))

        ((string=? name "SiLU")
         ;; silu(x) = x * sigmoid(x)
         (let* ((sig-lt (activation-forward-am (make-sigmoid) lt))
                (sig-mv (lazy-tensor-morph-variable sig-lt)))
           (get-or-make-lazy (am:var* mv sig-mv))))

        (else
         (error "activation-forward-am: unknown activation" name)))))


  ;;;; ================================================================
  ;;;; Loss functions
  ;;;; ================================================================

  (define (am-mse-loss pred-lt target-lt)
    "Mean squared error loss.  Both arguments are lazy tensors.
     Returns a scalar lazy tensor."
    (let* ((p   (lazy-tensor-morph-variable pred-lt))
           (t   (lazy-tensor-morph-variable target-lt))
           (dt  (morph-dtype (am:var-value p)))
           (two (am:make-var (morph-from-list '(2.0) '(1) dt) #f)))
      (get-or-make-lazy
       (am:var-mean (am:var-pow (am:var- p t) two)))))

  (define (am-cross-entropy-loss logits-lt target-lt)
    "Numerically stable softmax cross-entropy.
     logits: [batch, C] lazy tensor of unnormalized scores.
     target: [batch, C] lazy tensor of one-hot labels.
     Returns a scalar lazy tensor."
    (let* ((p     (lazy-tensor-morph-variable logits-lt))
           (t     (lazy-tensor-morph-variable target-lt))
           (pv    (am:var-value p))
           (dt    (morph-dtype pv))
           ;; shift by max for numerical stability
           (max-v (am:make-var (morph-reduce 'max pv '(1) #t) #f))
           (shift (am:var- p max-v))
           (exp-s (am:var-exp shift))
           ;; log-sum-exp
           (sum-e (am:var-sum exp-s '(1) #t))
           (log-s (am:var-log sum-e))
           ;; log-softmax
           (lp    (am:var- shift log-s))
           ;; per-sample cross-entropy
           (per-s (am:var-sum (am:var* t lp) '(1))))
      (get-or-make-lazy
       (am:var-negate (am:var-mean per-s)))))


  ;;;; ================================================================
  ;;;; He initialization helper
  ;;;; ================================================================

  (define (random-normal)
    "Box-Muller standard normal sample."
    (let* ((u1 (pseudo-random-real))
           (u2 (pseudo-random-real))
           (u1 (max u1 1e-10)))
      (* (sqrt (* -2.0 (log u1)))
         (cos (* 2.0 3.141592653589793 u2)))))

  (define (he-init-vector n fan-in dtype)
    "Allocate and He-initialize a typed vector of length n.
     Values are ~ N(0, sqrt(2/fan_in))."
    (let* ((std (sqrt (/ 2.0 (exact->inexact fan-in))))
           (v   (allocate-typed-vector dtype n)))
      (let loop ((i 0))
        (when (< i n)
          (typed-vector-set! v dtype i (* std (random-normal)))
          (loop (+ i 1))))
      v))

  (define (zero-vector n dtype)
    "Allocate a zero-filled typed vector of length n."
    (let ((v (allocate-typed-vector dtype n)))
      (let loop ((i 0))
        (when (< i n)
          (typed-vector-set! v dtype i 0.0)
          (loop (+ i 1))))
      v))

  (define (ones-vector n dtype)
    "Allocate a ones-filled typed vector of length n."
    (let ((v (allocate-typed-vector dtype n)))
      (let loop ((i 0))
        (when (< i n)
          (typed-vector-set! v dtype i 1.0)
          (loop (+ i 1))))
      v))


  ;;;; ================================================================
  ;;;; make-am-dense-layer
  ;;;;
  ;;;; Dense (fully connected) layer with AM-backed parameters.
  ;;;;   W: [out-size, in-size]  He-initialized
  ;;;;   b: [out-size]           zero-initialized
  ;;;;
  ;;;; Forward: activation(x @ W^T + b)
  ;;;;   x shape: [batch, in-size]
  ;;;;   result:  [batch, out-size]
  ;;;; ================================================================

  (define (make-am-dense-layer in-size out-size
                               #!key
                               (activation (make-identity))
                               (dtype 'f64))
    (let* ((W-data (he-init-vector (* out-size in-size) in-size dtype))
           (b-data (zero-vector out-size dtype))
           (W-mv   (am:make-var
                    (make-morphism W-data (list out-size in-size) dtype)
                    #t))
           (b-mv   (am:make-var
                    (make-morphism b-data (list out-size) dtype)
                    #t))
           (W-lt   (get-or-make-lazy W-mv))
           (b-lt   (get-or-make-lazy b-mv)))
      (object
       ((layer? self) #t)
       ((dense-layer? self) #t)
       ((layer-name self) "am-dense")
       ((layer-input-size self) in-size)
       ((layer-output-size self) out-size)
       ((layer-activation self) activation)
       ((layer-norm self) #f)

       ((parameters self) (list W-lt b-lt))

       ((zero-grad-layer! self)
        (am:zero-grad! W-mv)
        (am:zero-grad! b-mv))

       ((set-training-mode! self training?) (if #f #f))
       ((set-eval-mode! self) (if #f #f))

       ;; forward: x-lt has shape [batch, in-size]
       ;; W has shape [out-size, in-size], so x @ W^T = [batch, in-size] @ [in-size, out-size]
       ;; We use var-matmul(x, W^T) = [batch, out-size]
       ((forward self x-lt)
        (let* ((x-mv  (lazy-tensor-morph-variable x-lt))
               (WT-mv (am:var-transpose W-mv '(1 0)))
               (xW    (am:var-matmul x-mv WT-mv))
               ;; broadcast bias: b has shape [out-size], xW has [batch, out-size]
               (pre   (am:var+ xW b-mv)))
          (activation-forward-am activation (get-or-make-lazy pre))))

       ((layer->serializable self) #f)
       ((save-layer self filepath) (if #f #f))
       )))


  ;;;; ================================================================
  ;;;; var-im2col helper
  ;;;;
  ;;;; Wraps im2col-morph as a proper morph-variable node with a
  ;;;; grad-fn that calls col2im-morph for the backward pass.
  ;;;; ================================================================

  (define (var-im2col x-mv kernel-size stride padding)
    "Differentiable im2col: wraps structural im2col-morph with col2im backward."
    (let* ((x-morph  (am:var-value x-mv))
           (col      (im2col-morph x-morph kernel-size stride padding))
           (col-mv   (am:make-var col #f)))
      (when (am:var-requires-grad? x-mv)
        ;; grad-fn: g_col -> g_x via col2im
        (am:morph-variable-grad-fn-set! col-mv
          (lambda (g-col)
            (let* ((x-shape (morph-shape x-morph))
                   (g-col-c (realize g-col))
                   (g-x     (col2im-morph g-col-c
                                          (vector->list x-shape)
                                          kernel-size
                                          stride
                                          padding)))
              (am:accumulate-grad! x-mv g-x))))
        (am:morph-variable-parents-set! col-mv (list x-mv)))
      col-mv))


  ;;;; ================================================================
  ;;;; make-am-conv2d-layer
  ;;;;
  ;;;; 2-D convolution layer using im2col + matmul + col2im.
  ;;;;   W: [out-ch, in-ch*KH*KW]  He-initialized
  ;;;;   b: [out-ch]               zero-initialized
  ;;;;
  ;;;; Forward:
  ;;;;   col = im2col(x)          ; [N, in_ch*KH*KW, OH*OW]
  ;;;;   y   = W @ col + b        ; [N, out_ch, OH*OW]
  ;;;; ================================================================

  (define (make-am-conv2d-layer in-channels out-channels kernel-size
                                #!key
                                (stride 1)
                                (padding 0)
                                (activation (make-identity))
                                (dtype 'f64))
    (let* ((KH (if (pair? kernel-size) (car kernel-size) kernel-size))
           (KW (if (pair? kernel-size) (cdr kernel-size) kernel-size))
           (fan-in  (* in-channels KH KW))
           (W-data  (he-init-vector (* out-channels fan-in) fan-in dtype))
           (b-data  (zero-vector out-channels dtype))
           (W-mv    (am:make-var
                     (make-morphism W-data (list out-channels fan-in) dtype)
                     #t))
           (b-mv    (am:make-var
                     (make-morphism b-data (list out-channels) dtype)
                     #t))
           (W-lt    (get-or-make-lazy W-mv))
           (b-lt    (get-or-make-lazy b-mv)))
      (object
       ((layer? self) #t)
       ((conv2d-layer? self) #t)
       ((layer-name self) "am-conv2d")
       ((layer-input-size self) in-channels)
       ((layer-output-size self) out-channels)
       ((layer-activation self) activation)
       ((layer-norm self) #f)

       ((parameters self) (list W-lt b-lt))

       ((zero-grad-layer! self)
        (am:zero-grad! W-mv)
        (am:zero-grad! b-mv))

       ((set-training-mode! self training?) (if #f #f))
       ((set-eval-mode! self) (if #f #f))

       ((forward self x-lt)
        (let* ((x-mv   (lazy-tensor-morph-variable x-lt))
               (x-morph (am:var-value x-mv))
               (x-shape (morph-shape x-morph))
               (N      (if (= (vector-length x-shape) 4)
                           (vector-ref x-shape 0) 1))
               ;; col shape: [N, in_ch*KH*KW, OH*OW]
               (col-mv  (var-im2col x-mv
                                    (list KH KW)
                                    stride
                                    padding))
               ;; W @ col: [out_ch, in_ch*KH*KW] @ [N, in_ch*KH*KW, OH*OW]
               ;; We need batched matmul. For each batch element:
               ;;   W @ col[n] = [out_ch, OH*OW]
               ;; Simplify: reshape col to [N*OH*OW, in_ch*KH*KW], transpose W,
               ;; multiply, reshape back.
               (col-v   (am:var-value col-mv))
               (col-sh  (morph-shape col-v))
               (OH_OW   (vector-ref col-sh (- (vector-length col-sh) 1)))
               ;; Reshape col to [N*OH_OW, fan_in]
               (col-r   (am:var-reshape col-mv (list (* N OH_OW) fan-in)))
               ;; W^T: [fan_in, out_channels]
               (WT-mv   (am:var-transpose W-mv '(1 0)))
               ;; result: [N*OH_OW, out_channels]
               (res-mv  (am:var-matmul col-r WT-mv))
               ;; Reshape to [N, OH_OW, out_channels] then transpose to [N, out_ch, OH_OW]
               (res-r   (am:var-reshape res-mv (list N OH_OW out-channels)))
               (res-t   (am:var-transpose res-r '(0 2 1)))
               ;; Add bias: b shape [out_ch], broadcast over N and OH_OW
               (pre     (am:var+ res-t b-mv)))
          (activation-forward-am activation (get-or-make-lazy pre))))

       ((layer->serializable self) #f)
       ((save-layer self filepath) (if #f #f))
       )))


  ;;;; ================================================================
  ;;;; make-am-batch-norm
  ;;;;
  ;;;; Batch normalization with learnable gamma and beta.
  ;;;;   gamma: [C]  ones-initialized
  ;;;;   beta:  [C]  zero-initialized
  ;;;;   running_mean, running_var: concrete-arrays (no gradient)
  ;;;; ================================================================

  (define (make-am-batch-norm num-features
                              #!key
                              (epsilon 1e-5)
                              (momentum 0.1)
                              (dtype 'f64))
    (let* ((gamma-data (ones-vector num-features dtype))
           (beta-data  (zero-vector num-features dtype))
           (rm-data    (zero-vector num-features dtype))
           (rv-data    (ones-vector num-features dtype))
           (gamma-mv   (am:make-var
                        (make-morphism gamma-data (list num-features) dtype)
                        #t))
           (beta-mv    (am:make-var
                        (make-morphism beta-data (list num-features) dtype)
                        #t))
           ;; Running stats: non-grad morph-variables (concrete-array wrappers)
           (rm-mv      (am:make-var
                        (make-morphism rm-data (list num-features) dtype)
                        #f))
           (rv-mv      (am:make-var
                        (make-morphism rv-data (list num-features) dtype)
                        #f))
           (gamma-lt   (get-or-make-lazy gamma-mv))
           (beta-lt    (get-or-make-lazy beta-mv))
           (training?  #t))
      (object
       ((layer? self) #t)
       ((batch-norm-2d? self) #t)
       ((layer-name self) "am-batch-norm")
       ((layer-input-size self) num-features)
       ((layer-output-size self) num-features)
       ((layer-activation self) (make-identity))
       ((layer-norm self) #f)

       ((parameters self) (list gamma-lt beta-lt))

       ((zero-grad-layer! self)
        (am:zero-grad! gamma-mv)
        (am:zero-grad! beta-mv))

       ((set-training-mode! self mode) (set! training? mode))
       ((set-eval-mode! self) (set! training? #f))

       ((forward self x-lt)
        (let* ((x-mv  (lazy-tensor-morph-variable x-lt))
               (x-v   (am:var-value x-mv))
               (dt    (morph-dtype x-v))
               (eps-v (am:make-var (morph-from-list (list epsilon) '(1) dt) #f)))
          (if training?
              ;; Training: normalize over batch dimension
              (let* ((mu    (am:var-mean x-mv '(0) #t))
                     (diff  (am:var- x-mv mu))
                     (var   (am:var-mean (am:var-pow diff
                                                     (am:make-var (morph-from-list '(2.0) '(1) dt) #f))
                                         '(0) #t))
                     (std   (am:var-sqrt (am:var+ var eps-v)))
                     (xhat  (am:var/ diff std))
                     (out   (am:var+ (am:var* gamma-mv xhat) beta-mv)))
                ;; Update running stats (in-place mutation of the concrete vector)
                (let* ((mu-c    (realize (am:var-value mu)))
                       (var-c   (realize (am:var-value var)))
                       (mu-data  (cases array-morphism mu-c
                                   (concrete-array (data shape strides offset dtype2 alloc-id batch-axis) data)
                                   (else (morphism->srfi4 mu-c))))
                       (var-data (cases array-morphism var-c
                                   (concrete-array (data shape strides offset dtype2 alloc-id batch-axis) data)
                                   (else (morphism->srfi4 var-c))))
                       (n       num-features))
                  (let loop ((i 0))
                    (when (< i n)
                      (typed-vector-set! rm-data dtype i
                        (+ (* (- 1.0 momentum) (typed-vector-ref rm-data dtype i))
                           (* momentum (typed-vector-ref mu-data dtype i))))
                      (typed-vector-set! rv-data dtype i
                        (+ (* (- 1.0 momentum) (typed-vector-ref rv-data dtype i))
                           (* momentum (typed-vector-ref var-data dtype i))))
                      (loop (+ i 1)))))
                (get-or-make-lazy out))
              ;; Eval: use running statistics
              (let* ((rm-lt  (get-or-make-lazy rm-mv))
                     (rv-lt  (get-or-make-lazy rv-mv))
                     (std    (am:var-sqrt (am:var+ rv-mv eps-v)))
                     (xhat   (am:var/ (am:var- x-mv rm-mv) std))
                     (out    (am:var+ (am:var* gamma-mv xhat) beta-mv)))
                (get-or-make-lazy out)))))

       ((layer->serializable self) #f)
       ((save-layer self filepath) (if #f #f))
       )))


  ;;;; ================================================================
  ;;;; make-am-sequential
  ;;;; ================================================================

  (define (make-am-sequential layers)
    "Chain a list of AM-backed layers.  forward threads input through each."
    (object
     ((layer? self) #t)
     ((sequential? self) #t)
     ((inner-layers self) layers)
     ((layer-name self) "am-sequential")
     ((layer-input-size self)
      (layer-input-size (car layers)))
     ((layer-output-size self)
      (layer-output-size (car (reverse layers))))
     ((layer-activation self) (make-identity))
     ((layer-norm self) #f)

     ((parameters self)
      (append-map parameters layers))

     ((zero-grad-layer! self)
      (for-each zero-grad-layer! layers))

     ((set-training-mode! self mode)
      (for-each (lambda (l) (set-training-mode! l mode)) layers))
     ((set-eval-mode! self)
      (for-each set-eval-mode! layers))

     ((forward self x-lt)
      (fold (lambda (layer acc) (forward layer acc))
            x-lt
            layers))

     ((layer->serializable self) #f)
     ((save-layer self filepath) (if #f #f))
     ))


  ;;;; ================================================================
  ;;;; Parameter / gradient helpers
  ;;;; ================================================================

  (define (am-parameters layer)
    "Return the list of lazy tensor parameters for layer."
    (parameters layer))

  (define (am-zero-grad! layer)
    "Zero all parameter gradients for layer."
    (zero-grad-layer! layer))


  ;;;; ================================================================
  ;;;; Memory reuse integration
  ;;;;
  ;;;; Two contexts are used per training session:
  ;;;;   ctx-fwd  traces forward-pass realizations
  ;;;;   ctx-bwd  traces backward-pass gradient realizations
  ;;;;
  ;;;; KEY ENHANCEMENT: am-realize-grads/ctx realizes all leaf parameter
  ;;;; gradients through a SHARED context, so the context's graph-coloring
  ;;;; buffer allocator can find reuse opportunities across the entire
  ;;;; backward graph.  After the first training step the context is
  ;;;; finalized (buffer pool created) and subsequent steps replay into
  ;;;; pre-allocated buffers (~20% faster, ~70-80% fewer allocations).
  ;;;;
  ;;;; After am-realize-grads/ctx, each parameter's morph-variable-grad
  ;;;; is a concrete-array.  When the optimizer calls (tensor-grad param),
  ;;;; it calls morphism->srfi4 on the concrete-array which just extracts
  ;;;; the data pointer -- no re-computation.
  ;;;; ================================================================

  (define (make-am-training-context)
    "Return two fresh trace-mode contexts: forward and backward."
    (values (make-morphism-context) (make-morphism-context)))

  (define (am-forward/ctx ctx layer x-lt)
    "Run forward pass through ctx, returning a lazy tensor with concrete value.

     For sequential models, each sub-layer is processed individually so that
     each layer receives a concrete-valued input.  This keeps the lazy
     morphism-expr trees shallow (bounded by activation complexity, not by
     the number of layers), eliminating the O(N^2) lazy-subgraph embedding
     that occurs when the full sequential model is realized in one shot.

     For a single layer, the output is realized through ctx and packaged as
     a new morph-variable with concrete value, preserving the backward graph
     (parents + grad-fn) from the original forward computation."
    (if (sequential? layer)
        ;; Sequential: recurse per sub-layer so each gets a concrete input.
        (fold (lambda (sub-layer acc-lt)
                (am-forward/ctx ctx sub-layer acc-lt))
              x-lt
              (inner-layers layer))
        ;; Single layer: run forward, realize output, return concrete lazy tensor.
        (let* ((out-lt   (forward layer x-lt))
               (out-mv   (lazy-tensor-morph-variable out-lt))
               (val      (am:var-value out-mv))
               (realized (realize/ctx ctx val))
               (new-mv   (am:make-var realized (am:var-requires-grad? out-mv))))
          (am:morph-variable-grad-fn-set! new-mv (am:var-grad-fn out-mv))
          (am:morph-variable-parents-set! new-mv (am:var-parents out-mv))
          (get-or-make-lazy new-mv))))

  (define (am-backward!/ctx ctx loss-lt)
    "Run backward pass with eager intermediate gradient materialization.

     Unlike plain backward! (which builds the full lazy gradient graph and
     leaves realization to the caller), this function materializes each
     node's accumulated gradient through ctx BEFORE calling its grad-fn.
     This ensures every grad-fn receives a concrete upstream gradient, so
     the deltas it computes (and stores in parent grad fields) are shallow
     one-level morphisms rather than deep lazy chains.

     Combined with the layer-by-layer am-forward/ctx (which ensures forward
     activations are concrete), this gives O(N) backward instead of O(N^2):
       - each grad-fn receives a concrete g
       - each grad-fn reads concrete (var-value v) activations
       - each resulting delta is a 1-level morphism -> O(1) to realize

     For leaf parameters (no parents), the gradient is copied to a fresh
     non-pooled buffer so the optimizer can safely read it after the context
     pool is reused in the next training step."
    (let* ((root-mv (lazy-tensor-morph-variable loss-lt))
           (sorted  (am:topo-sort root-mv)))
      ;; Seed the root gradient (loss = 1.0)
      (am:accumulate-grad! root-mv
        (am:morph-ones-like (am:var-value root-mv)))
      ;; Process each node: materialize its accumulated grad, then propagate
      (for-each
       (lambda (mv)
         (let ((g   (am:var-grad mv))
               (gfn (am:var-grad-fn mv)))
           (when g
             (let* ((leaf?      (null? (am:var-parents mv)))
                    ;; Leaf parameters need a fresh copy: the pool buffer
                    ;; could be reused for intermediates in the next step.
                    (concrete-g (if leaf?
                                    (fresh-copy-morphism (realize/ctx ctx g))
                                    (realize/ctx ctx g))))
               (am:zero-grad! mv)
               (am:accumulate-grad! mv concrete-g)
               ;; Propagate concrete gradient to parents
               (when gfn (gfn concrete-g))))))
       sorted)))

  (define (fresh-copy-morphism m)
    "Copy a concrete-array to a fresh non-pooled buffer (alloc-id = -1).
     Used to protect gradient outputs from being overwritten by the
     buffer pool's greedy reuse of short-lived slots.

     Background: when realize/ctx returns a concrete-array, its buffer
     lives in the pool.  The greedy interval scheduler assigns lifetime
     [N, N] to an output that nothing in the context reads again.  On
     the next realize/ctx call in the same loop, the scheduler may
     hand out the SAME physical buffer for an intermediate result,
     overwriting the gradient before the optimizer sees it.  Copying
     to a fresh alloc-id=-1 buffer breaks this aliasing."
    (cases array-morphism m
      (concrete-array (data shape strides offset dtype alloc-id batch-axis)
        (let* ((n     (shape-size shape))
               (fresh (allocate-typed-vector dtype n)))
          (let loop ((i 0))
            (when (< i n)
              (typed-vector-set! fresh dtype i (typed-vector-ref data dtype i))
              (loop (+ i 1))))
          (concrete-array fresh shape (compute-strides shape) 0 dtype -1 batch-axis)))
      (else (error "fresh-copy-morphism: expected concrete-array" m))))

  (define (am-realize-grads/ctx ctx parameters)
    "Realize all parameter gradient morphisms through a shared context.

     This is the KEY optimizer enhancement.  After backward! all leaf
     parameter gradients are lazy morphism-expr trees.  Realizing them
     one by one (as the optimizer does via tensor-grad) misses buffer-
     reuse opportunities across the backward graph.

     By routing all gradient realizations through one shared context:
       - The context traces each allocation and its dependencies.
       - The graph-coloring allocator finds buffers that can be reused
         across different gradient computations (non-overlapping lifetimes).
       - After finalize-context! the buffer pool is created.
       - Subsequent training steps replay ALL gradient computations into
         the pool, achieving 70-80% allocation reduction.

     After this call, each parameter's morph-variable-grad is a
     concrete-array in a fresh (non-pooled) buffer.  The optimizer's
     (tensor-grad param) call then extracts the SRFI-4 data vector
     without re-computation, and without risk of aliasing."
    (for-each
     (lambda (param)
       (let* ((mv (lazy-tensor-morph-variable param))
              (g  (am:var-grad mv)))
         (when g
           ;; realize/ctx uses the pool for intermediate computations.
           ;; fresh-copy-morphism copies the FINAL result to a non-pooled
           ;; buffer so the pool can safely reuse the gradient slot for
           ;; the next parameter's intermediate computations.
           (let ((concrete-g (fresh-copy-morphism (realize/ctx ctx g))))
             (am:zero-grad! mv)
             (am:accumulate-grad! mv concrete-g)))))
     parameters))

  (define (am-finalize-context! ctx)
    "Finalize ctx: run lifetime analysis and create buffer pool."
    (finalize-context! ctx))

  (define (am-reset-context! ctx)
    "Reset ctx to replay mode for the next iteration."
    (reset-context! ctx))

  (define (am-training-step ctx-fwd ctx-bwd optimizer forward-fn loss-fn x-lt)
    "Integrated training step with two-context trace/replay.

     First call: both contexts trace buffer allocations.  After forward
     and backward realization, each context is finalized (buffer pool
     created from lifetime analysis).

     Subsequent calls: both contexts replay, reusing pre-allocated buffers.
     This yields ~20% faster wall time and ~70-80% fewer allocations per step.

     Workflow:
       1. am-forward/ctx ctx-fwd  -- forward, output realized via fwd pool
       2. loss-fn out-lt          -- loss (lazy morphism)
       3. backward! loss-lt       -- builds lazy gradient graph
       4. am-realize-grads/ctx    -- realizes ALL gradients via bwd pool
       5. [finalize if first step]
       6. step! optimizer         -- optimizer sees concrete SRFI-4 grads
       7. am-zero-grad! + reset   -- prepare for next step"
    (let* ((params   (am-parameters forward-fn))
           ;; Step 1: layer-by-layer forward pass (concrete intermediate outputs)
           (out-lt   (am-forward/ctx ctx-fwd forward-fn x-lt))
           ;; Step 2: loss (lazy morphism, no allocation)
           (loss-lt  (loss-fn out-lt)))
      ;; Step 3: backward with eager gradient materialization through ctx-bwd.
      ;; am-backward!/ctx processes nodes in topological order, realizing each
      ;; node's gradient through ctx-bwd before calling its grad-fn.  This
      ;; ensures each grad-fn receives a concrete upstream gradient and produces
      ;; shallow (1-level) deltas, giving O(N) total backward work.
      (am-backward!/ctx ctx-bwd loss-lt)
      ;; Step 4: after am-backward!/ctx, each parameter's grad is already a
      ;; concrete-array (fresh-copied from pool).  am-realize-grads/ctx is a
      ;; no-op for concrete grads but still needed for optimizer compatibility.
      (am-realize-grads/ctx ctx-bwd params)
      ;; Step 5: finalize contexts after the first step (check per-context mode
      ;; so that multiple context pairs can be used independently across calls).
      (unless (eq? (context-mode ctx-fwd) 'replay)
        (am-finalize-context! ctx-fwd))
      (unless (eq? (context-mode ctx-bwd) 'replay)
        (am-finalize-context! ctx-bwd))
      ;; Step 6: optimizer step (tensor-grad now returns concrete SRFI-4 vectors)
      (step! optimizer)
      ;; Step 7: zero gradients and reset contexts for next step
      (am-zero-grad! forward-fn)
      (am-reset-context! ctx-fwd)
      (am-reset-context! ctx-bwd)
      loss-lt))


  ;; ============================================================
  ;; SSA-based training step (Approach B)
  ;;
  ;; First call: builds the morph-variable graph via (forward model x-lt),
  ;; compiles it to a joint forward+backward SSA program, and caches the result.
  ;; Every call: updates the input constant for the current batch, realizes the
  ;; joint program through ctx, applies optimizer, and returns the loss tensor.
  ;;
  ;; Uses a single context (no ctx-bwd), eliminating the eager/lazy boundary
  ;; and fresh-copy overhead present in am-training-step (Approach A).
  ;; ============================================================

  (define *ssa-state* #f)
  ; State: (list joint-prog param-const-vals input-const-val)
  ;   joint-prog       -- ssa-program with fwd+bwd bindings
  ;   param-const-vals -- list of (const-ref cid) for trainable parameters
  ;   input-const-val  -- (const-ref cid) for the input batch, or #f

  (define (am-training-step/ssa ctx optimizer model loss-fn x-lt)
    "Training step using pre-compiled SSA forward+backward program.

     First call compiles the joint program; subsequent calls replay it.
     Parameters updated in-place by the optimizer are automatically
     reflected in SSA constants across steps without re-registration."
    (let ((params (am-parameters model)))
      (unless *ssa-state*
        (let* ((out-lt       (forward model x-lt))
               (loss-lt      (loss-fn out-lt))
               (loss-mv      (lazy-tensor-morph-variable loss-lt))
               (fwd-prog     (morphism-to-ssa loss-mv))
               (x-mv         (lazy-tensor-morph-variable x-lt))
               (x-const-val  (ssa-constant-id fwd-prog (am:var-value x-mv)))
               (p-vals       (filter-map
                               (lambda (p)
                                 (ssa-constant-id fwd-prog
                                   (am:var-value (lazy-tensor-morph-variable p))))
                               params))
               (loss-val     (ssa-loss-binding-val fwd-prog))
               (joint        (ssa-vjp fwd-prog p-vals loss-val)))
          (set! *ssa-state* (list joint p-vals x-const-val))))

      (let* ((joint         (car  *ssa-state*))
             (p-ids         (cadr *ssa-state*))
             (x-const-val   (caddr *ssa-state*))
             (x-mv          (lazy-tensor-morph-variable x-lt))
             (x-concrete    (am:var-value x-mv)))
        ;; Replace input constant with this step's batch
        (when x-const-val
          (hash-table-set! (ssa-program-constants joint)
                           (ssa-value-id x-const-val)
                           x-concrete))
        ;; Execute joint forward+backward program
        (let* ((results   (ssa-realize/ctx ctx joint))
               (loss-arr  (car results))
               (grads     (cdr results)))
          ;; Install gradients into morph-variables so optimizer can read them
          (for-each (lambda (p g)
                      (let ((mv (lazy-tensor-morph-variable p)))
                        (am:zero-grad! mv)
                        (am:accumulate-grad! mv g)))
                    params grads)
          ;; Finalize context after trace run, then reset for next step
          (unless (eq? (context-mode ctx) 'replay)
            (am-finalize-context! ctx))
          (step! optimizer)
          (am-zero-grad! model)
          (am-reset-context! ctx)
          ;; Return loss as a lazy tensor
          (get-or-make-lazy (am:make-var loss-arr #f))))))

) ; end module array-morphisms-nanograd

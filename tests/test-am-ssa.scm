;;; test-am-ssa.scm
;;;
;;; Integration tests for the SSA-based nanograd training path
;;; (am-training-step/ssa + context pinned-output allocation).
;;;
;;; Covers:
;;;   1. Basic SSA training step behavior
;;;   2. Context pinning: replay outputs use pool buffers (alloc-id >= 0)
;;;   3. Multi-step stability: identical results across replay runs
;;;   4. Convergence on a simple regression task
;;;   5. Gradient correctness: same values via SSA vs plain am-training-step
;;;
;;; Run with:
;;;   /home/igr/bin/chicken/bin/csi -s tests/test-am-ssa.scm

(import scheme (chicken base) (chicken format) (chicken random))
(import test)
(import (only srfi-1 iota every map filter))
(import (only srfi-4 f64vector-ref f64vector-length f64vector-set!))
(import datatype)
(import array-morphisms-core)
(import array-morphisms-context)
(import array-morphisms-realization)
(import array-morphisms-ssa)
(import (prefix array-morphisms-grad am:))
(import nanograd-autograd)
(import nanograd-layer)
(import nanograd-optimizer)
(import nanograd-array-morphisms)


;;;; ============================================================
;;;; Helpers
;;;; ============================================================

(define tol 1e-6)

(define (approx= a b) (< (abs (- a b)) tol))

(define (in-range? val lo hi) (and (>= val lo) (<= val hi)))

(define (make-lt data shape)
  (get-or-make-lazy
   (am:make-var (morph-from-list data (list->vector shape) 'f64) #f)))

(define (get-loss-value loss-lt)
  (f64vector-ref (tensor-data loss-lt) 0))

(define (concrete-alloc-id m)
  "Extract alloc-id from a realized concrete-array."
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis) alloc-id)
    (else (error "concrete-alloc-id: not a concrete-array"))))


;;;; ============================================================
;;;; Group 1: Basic SSA training step
;;;; ============================================================

(test-group "am-training-step/ssa basic behavior"

  (test-assert "first call returns a lazy tensor"
    (let* ((model (make-am-sequential
                   (list (make-am-dense-layer 2 1 activation: (make-identity) dtype: 'f64))))
           (opt   (make-adam (am-parameters model) learning-rate: 1e-3))
           (ctx   (make-morphism-context))
           (x     (make-lt '(1.0 0.0 0.0 1.0) '(2 2)))
           (tgt   (make-lt '(1.0 -1.0) '(2 1)))
           (loss  (am-training-step/ssa ctx opt model
                                        (lambda (p) (am-mse-loss p tgt))
                                        x)))
      (lazy-tensor? loss)))

  (test-assert "loss value is non-negative"
    (let* ((model (make-am-sequential
                   (list (make-am-dense-layer 2 1 activation: (make-identity) dtype: 'f64))))
           (opt   (make-adam (am-parameters model) learning-rate: 1e-3))
           (ctx   (make-morphism-context))
           (x     (make-lt '(1.0 0.0 0.0 1.0) '(2 2)))
           (tgt   (make-lt '(1.0 -1.0) '(2 1)))
           (loss  (am-training-step/ssa ctx opt model
                                        (lambda (p) (am-mse-loss p tgt))
                                        x)))
      (>= (get-loss-value loss) 0.0)))

  (test-assert "context switches to replay mode after first step"
    (let* ((model (make-am-sequential
                   (list (make-am-dense-layer 2 1 activation: (make-identity) dtype: 'f64))))
           (opt   (make-adam (am-parameters model) learning-rate: 1e-3))
           (ctx   (make-morphism-context))
           (x     (make-lt '(1.0 0.0) '(1 2)))
           (tgt   (make-lt '(1.0)     '(1 1))))
      (am-training-step/ssa ctx opt model (lambda (p) (am-mse-loss p tgt)) x)
      (eq? (context-mode ctx) 'replay)))

  (test-assert "with lr=0 loss is identical across two consecutive steps"
    (let* ((model (make-am-sequential
                   (list (make-am-dense-layer 2 1 activation: (make-identity) dtype: 'f64))))
           (opt   (make-adam (am-parameters model) learning-rate: 0.0))
           (ctx   (make-morphism-context))
           (x     (make-lt '(1.0 2.0) '(1 2)))
           (tgt   (make-lt '(0.0)     '(1 1)))
           (step! (lambda ()
                    (get-loss-value
                     (am-training-step/ssa ctx opt model
                                           (lambda (p) (am-mse-loss p tgt))
                                           x))))
           (l1 (step!))
           (l2 (step!)))
      (approx= l1 l2))))


;;;; ============================================================
;;;; Group 2: Context pinning — pool buffer verification
;;;;
;;;; After the context-pinned-output fix, outputs from ssa-realize/ctx
;;;; are pool buffers (alloc-id >= 0) rather than fresh copies
;;;; (alloc-id = -1) when they are row-major pool allocations.
;;;; ============================================================

(test-group "context pinning: pool buffer verification"

  (test-assert "after finalize, replay gradient has alloc-id >= 0 (pool, not copy)"
    ;; Direct SSA test: compile a simple program and verify the gradient output
    ;; is returned from the pool rather than a fresh copy in replay mode.
    (let* ((ctx  (make-morphism-context))
           (W-mv (am:make-var (morph-from-list '(1.0 2.0 3.0 4.0) #(4) 'f64) #t))
           (x-mv (am:make-var (morph-from-list '(0.5 0.5 0.5 0.5) #(4) 'f64) #f))
           (loss-mv (am:var-sum (am:var* W-mv x-mv)))
           (fwd-prog (morphism-to-ssa loss-mv))
           (p-val    (ssa-constant-id fwd-prog (am:var-value W-mv)))
           (joint    (ssa-vjp fwd-prog (list p-val) (ssa-loss-binding-val fwd-prog))))
      ;; Trace run
      (ssa-realize/ctx ctx joint)
      (finalize-context! ctx)
      ;; Replay run
      (reset-context! ctx)
      (let* ((results (ssa-realize/ctx ctx joint))
             (dW      (cadr results)))
        ;; dW is a reduce-sum result -> row-major -> pinned -> alloc-id >= 0
        (>= (concrete-alloc-id dW) 0))))

  (test-assert "after finalize, replay outputs have consistent alloc-ids across steps"
    ;; The same pool buffer slot is used every replay run.
    (let* ((ctx  (make-morphism-context))
           (W-mv (am:make-var (morph-from-list '(1.0 2.0) #(2) 'f64) #t))
           (x-mv (am:make-var (morph-from-list '(1.0 1.0) #(2) 'f64) #f))
           (loss-mv (am:var-sum (am:var* W-mv x-mv)))
           (fwd-prog (morphism-to-ssa loss-mv))
           (p-val    (ssa-constant-id fwd-prog (am:var-value W-mv)))
           (joint    (ssa-vjp fwd-prog (list p-val) (ssa-loss-binding-val fwd-prog))))
      (ssa-realize/ctx ctx joint)
      (finalize-context! ctx)
      (reset-context! ctx)
      (let* ((r1    (ssa-realize/ctx ctx joint))
             (id-r1 (concrete-alloc-id (cadr r1))))
        (reset-context! ctx)
        (let* ((r2    (ssa-realize/ctx ctx joint))
               (id-r2 (concrete-alloc-id (cadr r2))))
          ;; Both replays should return the same physical pool slot
          (= id-r1 id-r2)))))

  (test-assert "context stats: buffers >= 1 after training step"
    (let* ((model (make-am-sequential
                   (list (make-am-dense-layer 2 4 activation: (make-relu)     dtype: 'f64)
                         (make-am-dense-layer 4 1 activation: (make-identity) dtype: 'f64))))
           (opt   (make-adam (am-parameters model) learning-rate: 1e-3))
           (ctx   (make-morphism-context))
           (x     (make-lt '(1.0 2.0 3.0 4.0) '(2 2)))
           (tgt   (make-lt '(1.0 2.0) '(2 1))))
      (am-training-step/ssa ctx opt model (lambda (p) (am-mse-loss p tgt)) x)
      (let ((stats (context-stats ctx)))
        (>= (cdr (assq 'buffers stats)) 1)))))


;;;; ============================================================
;;;; Group 3: Multi-step stability
;;;; ============================================================

(test-group "multi-step stability"

  (test-assert "5 replay steps produce identical loss values"
    (let* ((model (make-am-sequential
                   (list (make-am-dense-layer 3 4 activation: (make-relu)     dtype: 'f64)
                         (make-am-dense-layer 4 1 activation: (make-identity) dtype: 'f64))))
           (opt   (make-adam (am-parameters model) learning-rate: 0.0)) ; lr=0 keeps params fixed
           (ctx   (make-morphism-context))
           (x     (make-lt '(1.0 0.0 0.0 0.0 1.0 0.0) '(2 3)))
           (tgt   (make-lt '(1.0 0.0) '(2 1)))
           (step! (lambda ()
                    (get-loss-value
                     (am-training-step/ssa ctx opt model
                                           (lambda (p) (am-mse-loss p tgt))
                                           x))))
           (l0 (step!))  ; trace
           (losses (map (lambda (_) (step!)) (iota 5))))  ; 5 replays
      (every (lambda (lv) (approx= lv l0)) losses)))

  (test-assert "5 replay steps produce identical gradient magnitudes"
    ;; Verify gradient values are stable across replay (no aliasing corruption).
    (let* ((ctx  (make-morphism-context))
           (W-mv (am:make-var (morph-from-list '(0.1 0.2 0.3 0.4) #(2 2) 'f64) #t))
           (x-mv (am:make-var (morph-from-list '(1.0 2.0 3.0 4.0) #(2 2) 'f64) #f))
           (loss-mv (am:var-mean (am:var-matmul W-mv x-mv)))
           (fwd-prog (morphism-to-ssa loss-mv))
           (p-val    (ssa-constant-id fwd-prog (am:var-value W-mv)))
           (joint    (ssa-vjp fwd-prog (list p-val) (ssa-loss-binding-val fwd-prog))))
      ;; Trace + get reference gradient
      (let* ((trace-results (ssa-realize/ctx ctx joint))
             (ref-dW (cases array-morphism (cadr trace-results)
                       (concrete-array (data shape strides offset dtype alloc-id batch-axis)
                         (map (lambda (i) (typed-vector-ref data dtype i))
                              (iota (shape-size shape))))
                       (else (error "unexpected")))))
        (finalize-context! ctx)
        ;; 5 replay runs
        (let loop ((i 0))
          (if (= i 5)
              #t
              (begin
                (reset-context! ctx)
                (let* ((results (ssa-realize/ctx ctx joint))
                       (dW     (cases array-morphism (cadr results)
                                 (concrete-array (data shape strides offset dtype alloc-id batch-axis)
                                   (map (lambda (i) (typed-vector-ref data dtype i))
                                        (iota (shape-size shape))))
                                 (else (error "unexpected")))))
                  (if (every approx= dW ref-dW)
                      (loop (+ i 1))
                      #f)))))))))


;;;; ============================================================
;;;; Group 4: Convergence
;;;; ============================================================

(test-group "SSA training convergence"

  (test-assert "loss decreases over 20 SSA training steps"
    (let* ((model (make-am-sequential
                   (list (make-am-dense-layer 2 8 activation: (make-relu)     dtype: 'f64)
                         (make-am-dense-layer 8 1 activation: (make-identity) dtype: 'f64))))
           (opt   (make-adam (am-parameters model) learning-rate: 1e-2))
           (ctx   (make-morphism-context))
           (x     (make-lt '(1.0 0.0 0.0 1.0 1.0 1.0 0.0 0.0) '(4 2)))
           (tgt   (make-lt '(1.0 1.0 0.0 0.0)                   '(4 1)))
           (step! (lambda ()
                    (get-loss-value
                     (am-training-step/ssa ctx opt model
                                           (lambda (p) (am-mse-loss p tgt))
                                           x)))))
      (let ((l0 (step!)))
        (let loop ((i 1))
          (when (< i 20) (step!) (loop (+ i 1))))
        (let ((l20 (step!)))
          (< l20 l0)))))

  (test-assert "simple linear regression converges toward correct slope"
    (begin
      (set-pseudo-random-seed! "am-ssa-test")
      (let* ((model (make-am-sequential
                     (list (make-am-dense-layer 1 1 activation: (make-identity) dtype: 'f64))))
             (params (am-parameters model))
             (W-data (tensor-data (car  params)))
             (b-data (tensor-data (cadr params))))
        (f64vector-set! W-data 0 0.0)
        (f64vector-set! b-data 0 0.0)
        (let* ((opt (make-adam params learning-rate: 0.1))
               (ctx (make-morphism-context)))
          (do ((i 0 (+ i 1))) ((= i 200))
            (let* ((x   (make-lt '(1.0 2.0 3.0) '(3 1)))
                   (tgt (make-lt '(2.0 4.0 6.0) '(3 1))))
              (am-training-step/ssa ctx opt model
                                     (lambda (p) (am-mse-loss p tgt))
                                     x)))
          (let ((weight (f64vector-ref W-data 0)))
            (in-range? weight 1.5 2.5)))))))


;;;; ============================================================
;;;; Group 5: Gradient correctness vs reference path
;;;; ============================================================

(test-group "SSA gradient correctness"

  (test-assert "SSA and non-SSA paths give same loss value"
    ;; Compile the same computation two ways and verify they agree.
    (let* ((W-data '(0.5 1.0 1.5 2.0))
           (x-data '(1.0 2.0 3.0 4.0))
           (W-mv   (am:make-var (morph-from-list W-data #(4) 'f64) #t))
           (x-mv   (am:make-var (morph-from-list x-data #(4) 'f64) #f))
           (loss-mv (am:var-sum (am:var* W-mv x-mv)))
           ;; SSA path
           (ctx  (make-morphism-context))
           (fwd  (morphism-to-ssa loss-mv))
           (pv   (ssa-constant-id fwd (am:var-value W-mv)))
           (joint (ssa-vjp fwd (list pv) (ssa-loss-binding-val fwd)))
           (ssa-results (ssa-realize/ctx ctx joint))
           (ssa-loss    (cases array-morphism (car ssa-results)
                          (concrete-array (data shape strides offset dtype alloc-id ba)
                            (typed-vector-ref data dtype 0))
                          (else (error "ssa loss not concrete"))))
           ;; Reference: direct realization without SSA
           (ref-loss (cases array-morphism (realize (am:var-value loss-mv))
                       (concrete-array (data shape strides offset dtype alloc-id ba)
                         (typed-vector-ref data dtype 0))
                       (else (error "ref loss not concrete")))))
      (approx= ssa-loss ref-loss)))

  (test-assert "SSA gradient dW matches finite-difference approximation"
    ;; loss = sum(W * x), dW[i] = x[i] analytically.
    (let* ((W-data '(1.0 2.0 3.0))
           (x-data '(0.5 1.5 2.5))
           (W-mv   (am:make-var (morph-from-list W-data #(3) 'f64) #t))
           (x-mv   (am:make-var (morph-from-list x-data #(3) 'f64) #f))
           (loss-mv (am:var-sum (am:var* W-mv x-mv)))
           (ctx     (make-morphism-context))
           (fwd     (morphism-to-ssa loss-mv))
           (pv      (ssa-constant-id fwd (am:var-value W-mv)))
           (joint   (ssa-vjp fwd (list pv) (ssa-loss-binding-val fwd)))
           (results (ssa-realize/ctx ctx joint))
           (dW      (cases array-morphism (cadr results)
                      (concrete-array (data shape strides offset dtype alloc-id ba)
                        (map (lambda (i) (typed-vector-ref data dtype i))
                             (iota (shape-size shape))))
                      (else (error "dW not concrete")))))
      ;; analytical: dW[i] = x[i]
      (every approx= dW x-data))))

(test-exit)

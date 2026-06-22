;;; nanograd/examples/am-regression-perf.scm
;;;
;;; Performance analysis for the array-morphisms regression backend.
;;; Uses the same architecture as regression.scm (4 -> 64 -> 32 -> 16 -> 1)
;;; and instruments every phase of the training step to identify bottlenecks.
;;;
;;; Measurements reported per step:
;;;   forward/ctx   -- am-forward/ctx: build + realize forward morphism graph
;;;   loss fn       -- construct lazy MSE morphisms (no realization)
;;;   am-backward!/ctx -- eager gradient materialization (Approach A)
;;;   grad sizes    -- per-parameter gradient element count (already concrete)
;;;   optimizer     -- step! (Adam parameter update)
;;;   zero+reset    -- zero gradients + reset context counters
;;;
;;; The first step runs in trace mode (fresh allocations).
;;; The second step runs in replay mode (buffer pool reuse).
;;; Comparing them isolates allocation cost from computation cost.
;;;
;;; Run with:
;;;   csi -q am-regression-perf.scm

(import scheme (chicken base) (chicken format) (chicken random) (chicken time))
(import (only srfi-1 map fold filter take drop last iota append-map))
(import srfi-4)
(import array-morphisms-core array-morphisms-realization array-morphisms-context)
(import array-morphisms-blas-exec)
(import array-morphisms-blas-egg-backend)
(import (prefix array-morphisms-grad am:))
(import nanograd-autograd nanograd-layer nanograd-optimizer)
(import nanograd-array-morphisms)

(register-blas-backend! (make-blas-egg-backend))


;;; ============================================================
;;; Timing helper
;;; ============================================================

(define (call/timed thunk)
  "Call thunk and return (values result elapsed-ms)."
  (let* ((t0  (cpu-time))
         (res (thunk))
         (dt  (- (cpu-time) t0)))
    (values res dt)))

(define (fmt-ms ms) (string-append (number->string ms) "ms"))


;;; ============================================================
;;; Synthetic dataset (same function as regression.scm)
;;; y = sin(x1) + cos(x2) + x3^2 - 0.5*x4
;;; ============================================================

(define num-features 4)

(define (make-random-batch n)
  "Return (x-lt . y-lt) with shapes [n,4] and [n,1], unnormalized."
  (let* ((xs (map (lambda (_)
                    (list (- (* 6.0 (pseudo-random-real)) 3.0)
                          (- (* 6.0 (pseudo-random-real)) 3.0)
                          (- (* 4.0 (pseudo-random-real)) 2.0)
                          (- (* 2.0 (pseudo-random-real)) 1.0)))
                  (iota n)))
         (ys (map (lambda (x)
                    (+ (sin (list-ref x 0))
                       (cos (list-ref x 1))
                       (* (list-ref x 2) (list-ref x 2))
                       (* -0.5 (list-ref x 3))))
                  xs))
         (x-flat (apply append xs))
         (x-lt   (get-or-make-lazy
                  (am:make-var (morph-from-list x-flat (list n num-features) 'f64) #f)))
         (y-lt   (get-or-make-lazy
                  (am:make-var (morph-from-list ys (list n 1) 'f64) #f))))
    (cons x-lt y-lt)))


;;; ============================================================
;;; Model -- matches regression.scm architecture exactly
;;; Input(4) -> Dense(64,relu) -> Dense(32,relu) -> Dense(16,relu) -> Output(1)
;;; ============================================================

(define batch-n 32)

(define model
  (make-am-sequential
   (list (make-am-dense-layer  4 64 activation: (make-relu)     dtype: 'f64)
         (make-am-dense-layer 64 32 activation: (make-relu)     dtype: 'f64)
         (make-am-dense-layer 32 16 activation: (make-relu)     dtype: 'f64)
         (make-am-dense-layer 16  1 activation: (make-identity) dtype: 'f64))))

(define params (am-parameters model))

(define (param-label p)
  "Human-readable name for a lazy tensor parameter."
  (let* ((mv (lazy-tensor-morph-variable p))
         (sh (vector->list (morph-shape (am:var-value mv)))))
    (string-append "["
                   (fold (lambda (d acc)
                           (if (string=? acc "")
                               (number->string d)
                               (string-append acc "x" (number->string d))))
                         ""
                         sh)
                   "]")))

(let ((n-params (fold (lambda (p acc)
                        (+ acc (f64vector-length (tensor-data p))))
                      0 params)))
  (printf "\nModel: ~A -> 64 -> 32 -> 16 -> 1  (~A parameters, f64)\n"
          num-features n-params)
  (printf "Batch size: ~A\n\n" batch-n))


;;; ============================================================
;;; Instrumented training step
;;;
;;; Replaces am-training-step with a manual loop that times each
;;; phase and reports per-parameter gradient realization costs.
;;; ============================================================

(define (instrumented-step ctx-fwd ctx-bwd opt x-lt y-lt step-label)
  "Run one training step and print a timing breakdown."

  (printf "~A\n" step-label)

  ;; Phase 1: forward pass through the context
  (define-values (out-lt t-fwd)
    (call/timed (lambda () (am-forward/ctx ctx-fwd model x-lt))))
  (printf "  forward/ctx:    ~A\n" (fmt-ms t-fwd))

  ;; Phase 2: loss construction (no realization -- builds lazy morphism tree)
  (define-values (loss-lt t-loss)
    (call/timed (lambda () (am-mse-loss out-lt y-lt))))
  (printf "  loss fn:        ~A  (lazy morphism, no alloc)\n" (fmt-ms t-loss))

  ;; Phase 3: am-backward!/ctx -- topo-sort, realize each node's gradient
  ;; eagerly through ctx-bwd, propagate concrete g to parents via grad-fn.
  ;; Leaf parameter grads are fresh-copied out of the pool.
  ;; After this call every parameter's morph-variable-grad is a concrete-array.
  (define-values (_ t-bwd)
    (call/timed (lambda () (am-backward!/ctx ctx-bwd loss-lt))))
  (printf "  am-backward!/ctx: ~A\n" (fmt-ms t-bwd))

  ;; Phase 4: per-parameter gradient sizes (grads are already concrete -- no realization).
  ;; We just report which parameters received gradients and their shapes.
  (printf "  grad sizes:\n")
  (define per-param-concrete
    (map (lambda (p)
           (let* ((mv (lazy-tensor-morph-variable p))
                  (g  (am:var-grad mv)))
             (and g (concrete-array? g) (shape-size (morph-shape g)))))
         params))

  (for-each (lambda (p sz)
              (if sz
                  (printf "    ~A  ~A elements (concrete)\n" (param-label p) sz)
                  (printf "    ~A  no grad or not concrete\n" (param-label p))))
            params per-param-concrete)

  ;; Context stats after gradient realization
  (let* ((fs (context-stats ctx-fwd))
         (bs (context-stats ctx-bwd)))
    (printf "  ctx-fwd:  ~A allocs, ~A buffers (~A)\n"
            (cdr (assq 'allocations fs))
            (or (and (assq 'buffers fs) (cdr (assq 'buffers fs))) "n/a (trace)")
            (cdr (assq 'mode fs)))
    (printf "  ctx-bwd:  ~A allocs, ~A buffers (~A)\n"
            (cdr (assq 'allocations bs))
            (or (and (assq 'buffers bs) (cdr (assq 'buffers bs))) "n/a (trace)")
            (cdr (assq 'mode bs))))

  ;; Finalize contexts after first step so second step replays
  (when (eq? (context-mode ctx-fwd) 'trace) (am-finalize-context! ctx-fwd))
  (when (eq? (context-mode ctx-bwd) 'trace) (am-finalize-context! ctx-bwd))

  ;; Phase 5: optimizer (Adam update -- operates on concrete SRFI-4 vectors)
  (define-values (_ t-opt)
    (call/timed (lambda () (step! opt))))
  (printf "  optimizer:      ~A\n" (fmt-ms t-opt))

  ;; Phase 6: zero gradients and reset context counters for next step
  (define-values (_ t-zero)
    (call/timed (lambda ()
                  (am-zero-grad! model)
                  (am-reset-context! ctx-fwd)
                  (am-reset-context! ctx-bwd))))
  (printf "  zero+reset:     ~A\n\n" (fmt-ms t-zero)))


;;; ============================================================
;;; Main: run three instrumented steps
;;; Step 1 -- trace mode (both contexts allocate fresh buffers)
;;; Step 2 -- replay mode (both contexts reuse buffer pool)
;;; Step 3 -- replay mode again (confirm steady state)
;;; ============================================================

(set-pseudo-random-seed! "42")

(define opt (make-adam params learning-rate: 1e-3))
(define-values (ctx-fwd ctx-bwd) (make-am-training-context))

(let* ((bp1 (make-random-batch batch-n))
       (bp2 (make-random-batch batch-n))
       (bp3 (make-random-batch batch-n)))

  (instrumented-step ctx-fwd ctx-bwd opt
                     (car bp1) (cdr bp1)
                     "Step 1 (trace mode -- first call allocates and records):")

  (instrumented-step ctx-fwd ctx-bwd opt
                     (car bp2) (cdr bp2)
                     "Step 2 (replay mode -- reuses buffer pool):")

  (instrumented-step ctx-fwd ctx-bwd opt
                     (car bp3) (cdr bp3)
                     "Step 3 (replay mode -- steady state):"))

;;; ============================================================
;;; SSA (Approach B) -- single context, fused forward+backward
;;;
;;; Step 1: SSA compilation + first trace realization
;;; Step 2: replay with updated input
;;; Step 3: replay (steady state)
;;; ============================================================

(printf "~A\n" (make-string 60 #\-))
(printf "\n--- SSA Approach B (single context, fused fwd+bwd) ---\n\n")

(define model-ssa
  (make-am-sequential
   (list (make-am-dense-layer  4 64 activation: (make-relu)     dtype: 'f64)
         (make-am-dense-layer 64 32 activation: (make-relu)     dtype: 'f64)
         (make-am-dense-layer 32 16 activation: (make-relu)     dtype: 'f64)
         (make-am-dense-layer 16  1 activation: (make-identity) dtype: 'f64))))

(define params-ssa (am-parameters model-ssa))
(define opt-ssa (make-adam params-ssa learning-rate: 1e-3))
(define ctx-ssa (make-morphism-context))

(define (instrumented-step/ssa ctx opt x-lt y-lt step-label)
  "Run one SSA training step and print a timing breakdown."
  (printf "~A\n" step-label)

  ;; Full step (compile on first call, replay on subsequent).
  ;; y-lt is passed as an extra-input so its SSA constant is updated each step.
  (define-values (loss-lt t-total)
    (call/timed (lambda ()
                  (am-training-step/ssa ctx opt model-ssa
                                        (lambda (out-lt) (am-mse-loss out-lt y-lt))
                                        x-lt y-lt))))
  (printf "  total step:     ~A\n" (fmt-ms t-total))

  ;; Context stats
  (let ((s (context-stats ctx)))
    (printf "  ctx:  ~A allocs, ~A buffers (~A)\n"
            (cdr (assq 'allocations s))
            (or (and (assq 'buffers s) (cdr (assq 'buffers s))) "n/a (trace)")
            (cdr (assq 'mode s))))
  (printf "\n"))

(let* ((bp4 (make-random-batch batch-n))
       (bp5 (make-random-batch batch-n))
       (bp6 (make-random-batch batch-n)))

  (instrumented-step/ssa ctx-ssa opt-ssa
                         (car bp4) (cdr bp4)
                         "SSA Step 1 (compile + trace):")

  (instrumented-step/ssa ctx-ssa opt-ssa
                         (car bp5) (cdr bp5)
                         "SSA Step 2 (replay):")

  (instrumented-step/ssa ctx-ssa opt-ssa
                         (car bp6) (cdr bp6)
                         "SSA Step 3 (replay, steady state):"))


;;; ============================================================
;;; Summary of findings
;;; ============================================================

(printf "~A\n" (make-string 60 #\-))
(printf "Performance notes:\n")
(printf "\n")
(printf "Approach A (am-backward!/ctx): layer-by-layer forward realization\n")
(printf "  + eager gradient materialization through two contexts.\n")
(printf "  O(N_nodes) total backward work.  Two buffer pools.\n")
(printf "\n")
(printf "Approach B (am-training-step/ssa): one-time SSA compilation of the\n")
(printf "  joint fwd+bwd computation.  Single context, no fresh-copy overhead.\n")
(printf "  First step includes compilation; subsequent steps are pure replay.\n")
(printf "~A\n" (make-string 60 #\-))

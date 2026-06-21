;;; test-am-network.scm
;;; Integration tests for array-morphisms training with am-training-step
;;;
;;; Equivalent to test-network.scm but for AM-backed training:
;;;   convergence on simple regressions, context trace/replay, Adam optimizer,
;;;   multi-batch training, and learning rate effect.
;;;
;;; Run with:
;;;   csi -q test-am-network.scm

(import scheme (chicken base) (chicken format) (chicken random) (chicken time) (chicken gc))
(import test)
(import (only srfi-1 iota map fold every filter take drop))
(import (only srfi-4 f64vector-ref f64vector-length f64vector-set!))
(import array-morphisms-core)
(import array-morphisms-context)
(import array-morphisms-realization)
(import (prefix array-morphisms-grad am:))
(import nanograd-autograd)
(import nanograd-layer)
(import nanograd-optimizer)
(import nanograd-array-morphisms)


;;;; ============================================================
;;;; Helpers
;;;; ============================================================

(define tol 1e-4)

(define (approx= a b) (< (abs (- a b)) tol))

(define (in-range? val lo hi) (and (>= val lo) (<= val hi)))

(define (make-lt data shape)
  (get-or-make-lazy
   (am:make-var (morph-from-list data (list->vector shape) 'f64) #f)))

(define (get-loss-value loss-lt)
  (f64vector-ref (tensor-data loss-lt) 0))

(define (count-params model)
  (fold (lambda (p acc) (+ acc (f64vector-length (tensor-data p))))
        0 (am-parameters model)))

(define (eval-loss model x-lt tgt-lt)
  (let* ((out  (forward model x-lt))
         (loss (am-mse-loss out tgt-lt)))
    (get-loss-value loss)))

(define (set-random-seed! seed)
  "Set random seed for reproducibility"
  (set-pseudo-random-seed! (number->string seed)))


;;;; ============================================================
;;;; am-training-step basic behavior
;;;; ============================================================

(test-group "am-training-step basic behavior"

  ;; Step returns a lazy tensor (loss)
  (let* ((model (make-am-sequential
                 (list (make-am-dense-layer 2 1 activation: (make-identity) dtype: 'f64))))
         (opt   (make-adam (am-parameters model) learning-rate: 1e-3)))
    (let-values (((ctx-fwd ctx-bwd) (make-am-training-context)))
      (let* ((x    (make-lt '(1.0 0.0 0.0 1.0) '(2 2)))
             (tgt  (make-lt '(1.0 -1.0)         '(2 1)))
             (loss (am-training-step ctx-fwd ctx-bwd opt model
                                     (lambda (p) (am-mse-loss p tgt))
                                     x)))
        (test-assert "am-training-step returns a lazy tensor"
          (lazy-tensor? loss))
        (test-assert "am-training-step loss is non-negative"
          (>= (get-loss-value loss) 0.0)))))

  ;; With lr=0 weights don't change; loss should be consistent
  (let* ((model (make-am-sequential
                 (list (make-am-dense-layer 2 1 activation: (make-identity) dtype: 'f64))))
         (opt   (make-adam (am-parameters model) learning-rate: 0.0)))
    (let-values (((ctx-fwd ctx-bwd) (make-am-training-context)))
      (let* ((x   (make-lt '(1.0 2.0) '(1 2)))
             (tgt (make-lt '(0.0)     '(1 1)))
             (l1  (am-training-step ctx-fwd ctx-bwd opt model
                                    (lambda (p) (am-mse-loss p tgt)) x))
             (l2  (am-training-step ctx-fwd ctx-bwd opt model
                                    (lambda (p) (am-mse-loss p tgt)) x)))
        (test-assert "lr=0: loss unchanged across steps"
          (approx= (get-loss-value l1) (get-loss-value l2)))))))


;;;; ============================================================
;;;; Linear regression y = 3x + 2 with SGD
;;;;
;;;; Mirrors "Linear Regression with SGD" in test-network.scm.
;;;; Same data, same init, same epoch count, same assertion ranges.
;;;; lr=0.05 here vs lr=0.1 in test-network.scm because am-mse-loss
;;;; computes mean((p-t)^2) while nanograd mse-loss computes
;;;; 0.5*mean((p-t)^2), giving a 2x larger gradient in the AM backend.
;;;; ============================================================

(test-group "Linear Regression y = 3x + 2 with SGD"

  ;; Same 7 deterministic samples as test-network.scm
  (define training-data
    '((0.0 . 2.0) (1.0 . 5.0) (-1.0 . -1.0) (2.0 . 8.0)
      (-2.0 . -4.0) (0.5 . 3.5) (-0.5 . 0.5)))

  (let* ((model  (make-am-sequential
                  (list (make-am-dense-layer 1 1 activation: (make-identity) dtype: 'f64))))
         (params (am-parameters model))
         (W-data (tensor-data (car params)))
         (b-data (tensor-data (cadr params))))
    ;; Same initialization as test-network.scm
    (f64vector-set! W-data 0 0.5)
    (f64vector-set! b-data 0 0.5)

    (let* ((opt (make-sgd (am-parameters model) learning-rate: 0.1)))
      (let-values (((ctx-fwd ctx-bwd) (make-am-training-context)))

        ;; 50 epochs, sample-by-sample — same structure as test-network.scm
        (do ((epoch 1 (+ epoch 1))) ((> epoch 50))
          (for-each
           (lambda (sample)
             (let* ((x   (make-lt (list (car sample)) '(1 1)))
                    (tgt (make-lt (list (cdr sample)) '(1 1))))
               (am-training-step ctx-fwd ctx-bwd opt model
                                 (lambda (p) (am-mse-loss p tgt))
                                 x)))
           training-data))

        ;; Same assertions and ranges as test-network.scm
        (let* ((weight (f64vector-ref W-data 0))
               (bias   (f64vector-ref b-data 0)))
          (test-assert "Weight converges near 3.0"
            (in-range? weight 2.5 3.5))
          (test-assert "Bias converges near 2.0"
            (in-range? bias 1.5 2.5))
          ;; Prediction for x=3: 3*3+2=11, expect ~[10,12]
          (let* ((test-x  (make-lt '(3.0) '(1 1)))
                 (out-lt  (forward model test-x))
                 (pred    (f64vector-ref (tensor-data out-lt) 0)))
            (test-assert "Prediction for x=3 is near 11.0"
              (in-range? pred 10.0 12.0))))))))


;;;; ============================================================
;;;; Loss decreasing trend on simple problem
;;;; ============================================================

(test-group "loss trajectory"

  ;; Fixed batch, linear model, small lr: loss should decrease
  (let* ((model  (make-am-sequential
                  (list (make-am-dense-layer 1 1 activation: (make-identity) dtype: 'f64))))
         (params (am-parameters model))
         (W-data (tensor-data (car params)))
         (b-data (tensor-data (cadr params))))
    (f64vector-set! W-data 0 0.0)
    (f64vector-set! b-data 0 0.0)

    (let* ((opt (make-adam (am-parameters model) learning-rate: 1e-2)))
      (let-values (((ctx-fwd ctx-bwd) (make-am-training-context)))
        (define (run-step)
          (let* ((x   (make-lt '(1.0 2.0 3.0) '(3 1)))
                 (tgt (make-lt '(3.0 6.0 9.0) '(3 1)))
                 (l   (am-training-step ctx-fwd ctx-bwd opt model
                                       (lambda (p) (am-mse-loss p tgt))
                                       x)))
            (get-loss-value l)))

        (let* ((l1 (run-step))
               (l2 (run-step))
               (l3 (run-step))
               (l4 (run-step))
               (l5 (run-step)))
          (test-assert "loss trajectory: step 1 > step 5"
            (> l1 l5))
          (test-assert "loss trajectory: decreasing trend l1>l3>l5"
            (and (> l1 l3) (> l3 l5))))))))


;;;; ============================================================
;;;; Context trace/replay consistency
;;;; ============================================================

(test-group "context trace/replay consistency"

  ;; After the first step (trace), subsequent steps (replay) should
  ;; produce valid (non-NaN, finite) loss values and show overall
  ;; loss improvement on a fixed dataset.
  (let* ((model (make-am-sequential
                 (list (make-am-dense-layer 4 8  activation: (make-relu)     dtype: 'f64)
                       (make-am-dense-layer 8 1  activation: (make-identity) dtype: 'f64))))
         (opt   (make-adam (am-parameters model) learning-rate: 1e-2))
         ;; Fixed dataset — same each call so context replay trains consistently
         (x-fixed   (make-lt '(1.0 0.0 0.0 1.0
                                0.0 1.0 1.0 0.0
                                1.0 1.0 0.0 0.0
                                0.0 0.0 1.0 1.0) '(4 4)))
         (tgt-fixed (make-lt '(1.0 0.0 1.0 0.0) '(4 1))))
    (let-values (((ctx-fwd ctx-bwd) (make-am-training-context)))

      (let ((losses
             (map (lambda (_)
                    (get-loss-value
                     (am-training-step ctx-fwd ctx-bwd opt model
                                       (lambda (p) (am-mse-loss p tgt-fixed))
                                       x-fixed)))
                  (iota 10))))
        (test-assert "context replay: all loss values are finite"
          (every (lambda (lv) (and (finite? lv) (not (nan? lv)))) losses))
        (test-assert "context replay: all loss values are non-negative"
          (every (lambda (lv) (>= lv 0.0)) losses))
        (test-assert "context replay: loss decreases from step 1 to step 10"
          (< (list-ref losses 9) (list-ref losses 0)))))))


;;;; ============================================================
;;;; Multi-layer model convergence
;;;; ============================================================

(test-group "multi-layer model convergence"

  ;; 2-layer model on y = x^2 (sampled at a few points)
  (set-random-seed! 99)
  (let* ((model (make-am-sequential
                 (list (make-am-dense-layer 1 8  activation: (make-relu)     dtype: 'f64)
                       (make-am-dense-layer 8 1  activation: (make-identity) dtype: 'f64))))
         (opt   (make-adam (am-parameters model) learning-rate: 1e-2)))
    (let-values (((ctx-fwd ctx-bwd) (make-am-training-context)))

      (define x-lt   (make-lt '(-2.0 -1.0 0.0 1.0 2.0) '(5 1)))
      (define tgt-lt (make-lt '( 4.0  1.0 0.0 1.0 4.0) '(5 1)))

      (define (current-loss) (eval-loss model x-lt tgt-lt))

      (let ((l0 (current-loss)))
        (let loop ((i 0))
          (when (< i 100)
            (am-training-step ctx-fwd ctx-bwd opt model
                              (lambda (p) (am-mse-loss p tgt-lt))
                              x-lt)
            (loop (+ i 1))))

        (let ((l1 (current-loss)))
          (test-assert "multi-layer: loss decreased after 100 steps"
            (< l1 l0))
          (test-assert "multi-layer: at least 50% loss reduction"
            (< l1 (* 0.5 l0))))))))


;;;; ============================================================
;;;; Adam optimizer convergence
;;;; ============================================================

(test-group "Adam optimizer"

  (define (make-linear-model)
    (let* ((m  (make-am-sequential
                (list (make-am-dense-layer 1 1 activation: (make-identity) dtype: 'f64))))
           (W  (tensor-data (car  (am-parameters m))))
           (b  (tensor-data (cadr (am-parameters m)))))
      (f64vector-set! W 0 0.0)
      (f64vector-set! b 0 0.0)
      m))

  (define (train-to-threshold target-loss n-steps lr)
    (let* ((model (make-linear-model))
           (opt   (make-adam (am-parameters model) learning-rate: lr)))
      (let-values (((ctx-fwd ctx-bwd) (make-am-training-context)))
        (let loop ((i 0) (reached #f))
          (if (or reached (>= i n-steps))
              (list reached i)
              (let* ((x   (make-lt '(1.0 -1.0 2.0 -2.0) '(4 1)))
                     (tgt (make-lt '(2.0 -2.0 4.0 -4.0) '(4 1)))
                     (l   (am-training-step ctx-fwd ctx-bwd opt model
                                           (lambda (p) (am-mse-loss p tgt))
                                           x))
                     (lv  (get-loss-value l)))
                (loop (+ i 1) (< lv target-loss))))))))

  (let* ((result   (train-to-threshold 0.01 200 0.1))
         (reached? (car result))
         (steps    (cadr result)))
    (test-assert "Adam: reaches loss < 0.01 within 200 steps"
      reached?)
    (test-assert "Adam: converges in under 100 steps with lr=0.1"
      (< steps 100))))


;;;; ============================================================
;;;; am-parameters integrity after training
;;;; ============================================================

(test-group "am-parameters integrity after training"

  (let* ((model (make-am-sequential
                 (list (make-am-dense-layer 3 4 activation: (make-relu)     dtype: 'f64)
                       (make-am-dense-layer 4 1 activation: (make-identity) dtype: 'f64))))
         (opt   (make-adam (am-parameters model) learning-rate: 1e-3)))
    (let-values (((ctx-fwd ctx-bwd) (make-am-training-context)))
      ;; Record initial param shapes
      (let* ((shapes0 (map tensor-shape (am-parameters model))))
        ;; Train 5 steps
        (let loop ((i 0))
          (when (< i 5)
            (let* ((x   (make-lt '(1.0 2.0 3.0 4.0 5.0 6.0) '(2 3)))
                   (tgt (make-lt '(1.0 2.0) '(2 1))))
              (am-training-step ctx-fwd ctx-bwd opt model
                                (lambda (p) (am-mse-loss p tgt))
                                x)
              (loop (+ i 1)))))

        (let* ((params1 (am-parameters model))
               (shapes1 (map tensor-shape params1)))
          (test-assert "params: shapes unchanged after training"
            (equal? shapes0 shapes1))
          (test-assert "params: all values are finite after training"
            (every (lambda (p)
                     (let* ((data (tensor-data p))
                            (n    (f64vector-length data)))
                       (let loop ((i 0))
                         (or (= i n)
                             (and (finite? (f64vector-ref data i))
                                  (loop (+ i 1)))))))
                   params1))
          (test-assert "params: count = 3*4+4 + 4*1+1 = 21"
            (= (fold + 0 (map (lambda (p) (f64vector-length (tensor-data p))) params1))
               21)))))))


;;;; ============================================================
;;;; Classification helpers
;;;; ============================================================

(define (f64-argmax vec n)
  (let loop ((i 1) (best 0) (best-val (f64vector-ref vec 0)))
    (if (= i n)
        best
        (if (> (f64vector-ref vec i) best-val)
            (loop (+ i 1) i (f64vector-ref vec i))
            (loop (+ i 1) best best-val)))))

(define (f64-argmax-at vec offset n)
  "Argmax of n elements starting at offset in vec."
  (let loop ((i 1) (best 0) (best-val (f64vector-ref vec offset)))
    (if (= i n)
        best
        (if (> (f64vector-ref vec (+ offset i)) best-val)
            (loop (+ i 1) i (f64vector-ref vec (+ offset i)))
            (loop (+ i 1) best best-val)))))

(define (make-one-hot n k)
  (map (lambda (i) (if (= i k) 1.0 0.0)) (iota n)))


;;;; ============================================================
;;;; Binary Classification
;;;;
;;;; Mirrors "Binary Classification" in test-network.scm.
;;;; Same 10 samples, same architecture (2->8 ReLU -> 2 Sigmoid),
;;;; same loss (MSE), same optimizer (Adam lr=0.01), same 100 epochs.
;;;;
;;;; make-am-dense-layer uses He-normal init which is too large for
;;;; sigmoid (can saturate from the start).  nanograd's make-dense-layer
;;;; uses uniform * sqrt(2/fan_in), keeping sigmoid in its linear
;;;; region.  We replicate that by discarding the He-init weights and
;;;; re-initializing with the same formula before training begins.
;;;; ============================================================

(test-group "Binary Classification"

  ;; Replicates nanograd's make-dense-layer init:
  ;;   w_i = sqrt(2/fan_in) * (pseudo-random-real - 0.5)
  ;; Allocation-free inner loop so GC cannot disturb the PRNG mid-init.
  (define (uniform-init! data n fan-in)
    (let ((scale (sqrt (/ 2.0 fan-in))))
      (let loop ((i 0))
        (when (< i n)
          (f64vector-set! data i (* scale (- (pseudo-random-real) 0.5)))
          (loop (+ i 1))))))

  (define binary-data
    (append
     ;; Class 0: points near (0, 0)
     (list
      (cons '(0.0  0.0) '(1.0 0.0))
      (cons '(0.1  0.0) '(1.0 0.0))
      (cons '(0.0  0.1) '(1.0 0.0))
      (cons '(-0.1 0.0) '(1.0 0.0))
      (cons '(0.0 -0.1) '(1.0 0.0)))
     ;; Class 1: points near (1, 1)
     (list
      (cons '(1.0 1.0) '(0.0 1.0))
      (cons '(0.9 1.0) '(0.0 1.0))
      (cons '(1.0 0.9) '(0.0 1.0))
      (cons '(1.1 1.0) '(0.0 1.0))
      (cons '(1.0 1.1) '(0.0 1.0)))))

  (let* ((model  (make-am-sequential
                  (list (make-am-dense-layer 2 8 activation: (make-relu)    dtype: 'f64)
                        (make-am-dense-layer 8 2 activation: (make-sigmoid) dtype: 'f64))))
         (params (am-parameters model)))
    ;; Overwrite He-init with deterministic uniform init (seed after model
    ;; creation so GC during He-init does not consume the seed's PRNG state).
    ;; params order: W1[8x2] b1[8] W2[2x8] b2[2]; biases stay zero.
    (set-pseudo-random-seed! "123")
    (uniform-init! (tensor-data (list-ref params 0)) 16 2)
    (uniform-init! (tensor-data (list-ref params 2)) 16 8)

    ;; Build full-dataset batch tensors once and reuse every epoch.
    ;; Shape [10,2] for both inputs and targets.
    ;; Training with 1 step/epoch instead of 10 steps/epoch reduces
    ;; morph-variable graph construction from 1000 to 100 total calls.
    (let* ((x-batch   (make-lt (apply append (map car binary-data))
                                '(10 2)))
           (tgt-batch (make-lt (apply append (map cdr binary-data))
                                '(10 2)))
           (opt (make-adam params learning-rate: 0.01))
           (ctx (make-morphism-context)))

        (do ((epoch 1 (+ epoch 1))) ((> epoch 100))
          (am-training-step/ssa ctx opt model
                            (lambda (p) (am-mse-loss p tgt-batch))
                            x-batch))

        ;; Single batched forward pass for evaluation.
        ;; Output shape [10,2]; row i occupies out-data[i*2 .. i*2+1].
        (let* ((out-lt   (forward model x-batch))
               (out-data (tensor-data out-lt))
               (correct  0))
          (let loop ((i 0) (samples binary-data))
            (unless (null? samples)
              (let* ((tgt      (cdr (car samples)))
                     (offset   (* i 2))
                     (pred-cls (f64-argmax-at out-data offset 2))
                     (true-cls (if (> (car tgt) (cadr tgt)) 0 1)))
                (when (= pred-cls true-cls) (set! correct (+ correct 1)))
                (loop (+ i 1) (cdr samples)))))
          (let ((accuracy (* 100.0 (/ correct (length binary-data)))))
            (test-assert "Binary classification achieves >80% accuracy"
              (> accuracy 80.0))
            (printf "    Final accuracy: ~A%\n" accuracy))))))


;;;; ============================================================
;;;; Multi-class Classification
;;;;
;;;; Mirrors "Multi-class Classification" in test-network.scm.
;;;; Same 12 samples (3 well-separated clusters).
;;;; Uses am-cross-entropy-loss (applies softmax internally —
;;;; no separate softmax call needed unlike nanograd).
;;;; Inputs shaped [1,2], one-hot targets shaped [1,3].
;;;; ============================================================

(test-group "Multi-class Classification"

  ;; Weight init helper: uniform * sqrt(2/fan_in), allocation-free inner loop
  ;; so minor GC cannot disturb the PRNG sequence mid-initialization.
  (define (uniform-init-mc! data n fan-in)
    (let ((scale (sqrt (/ 2.0 fan-in))))
      (let loop ((i 0))
        (when (< i n)
          (f64vector-set! data i (* scale (- (pseudo-random-real) 0.5)))
          (loop (+ i 1))))))

  (define multiclass-data
    (append
     ;; Class 0: bottom-left
     (list
      (cons '(-1.0 -1.0) 0)
      (cons '(-0.9 -1.0) 0)
      (cons '(-1.0 -0.9) 0)
      (cons '(-0.8 -0.8) 0))
     ;; Class 1: bottom-right
     (list
      (cons '(1.0 -1.0) 1)
      (cons '(0.9 -1.0) 1)
      (cons '(1.0 -0.9) 1)
      (cons '(0.8 -0.8) 1))
     ;; Class 2: top-center
     (list
      (cons '(0.0  1.0) 2)
      (cons '(0.1  1.0) 2)
      (cons '(-0.1 0.9) 2)
      (cons '(0.0  0.8) 2))))

  (let* ((model  (make-am-sequential
                  (list (make-am-dense-layer 2  16 activation: (make-relu)     dtype: 'f64)
                        (make-am-dense-layer 16  8 activation: (make-relu)     dtype: 'f64)
                        (make-am-dense-layer  8  3 activation: (make-identity) dtype: 'f64))))
         (params (am-parameters model)))
    ;; Overwrite He-init with deterministic uniform init after seeding.
    ;; Seed is set after model creation so that module-loading and He-init
    ;; allocations (which trigger minor GC and consume PRNG state) don't
    ;; affect the weights we actually train with.
    ;; params layout: W1[16x2] b1[16] W2[8x16] b2[8] W3[3x8] b3[3]
    ;; biases stay zero; only weight matrices are re-initialized.
    (set-pseudo-random-seed! "456")
    (uniform-init-mc! (tensor-data (list-ref params 0)) 32 2)
    (uniform-init-mc! (tensor-data (list-ref params 2)) 128 16)
    (uniform-init-mc! (tensor-data (list-ref params 4)) 24 8)

    ;; Build full-dataset batch tensors once and reuse every epoch.
    ;; x-batch shape [12,2]; tgt-batch shape [12,3].
    ;; Training with 1 step/epoch instead of 12 reduces total
    ;; morph-variable graph construction from 1200 to 100 calls.
    (let* ((x-batch   (make-lt (apply append (map car multiclass-data))
                                '(12 2)))
           (tgt-batch (make-lt (apply append
                                      (map (lambda (s) (make-one-hot 3 (cdr s)))
                                           multiclass-data))
                                '(12 3)))
           (opt (make-adam params learning-rate: 0.01))
           (ctx (make-morphism-context)))

        (do ((epoch 1 (+ epoch 1))) ((> epoch 100))
          (am-training-step/ssa ctx opt model
                                (lambda (p) (am-cross-entropy-loss p tgt-batch))
                                x-batch))
        ;; Single batched forward pass for evaluation.
        ;; Output shape [12,3]; row i occupies out-data[i*3 .. i*3+2].
        (let* ((out-lt   (forward model x-batch))
               (out-data (tensor-data out-lt))
               (correct  0))
          (let loop ((i 0) (samples multiclass-data))
            (unless (null? samples)
              (let* ((offset   (* i 3))
                     (pred-cls (f64-argmax-at out-data offset 3))
                     (true-cls (cdr (car samples))))
                (when (= pred-cls true-cls) (set! correct (+ correct 1)))
                (loop (+ i 1) (cdr samples)))))
          (let ((accuracy (* 100.0 (/ correct (length multiclass-data)))))
            (test-assert "Multi-class achieves >70% accuracy"
              (> accuracy 70.0))
            (printf "    Final accuracy: ~A%\n" accuracy))))))


(test-exit)

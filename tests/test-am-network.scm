;;; test-am-network.scm
;;; Integration tests for array-morphisms training with am-training-step
;;;
;;; Equivalent to test-network.scm but for AM-backed training:
;;;   convergence on simple regressions, context trace/replay, Adam optimizer,
;;;   multi-batch training, and learning rate effect.
;;;
;;; Run with:
;;;   csi -q test-am-network.scm

(import scheme (chicken base) (chicken format) (chicken random) (chicken time))
(import test)
(import (only srfi-1 iota map fold every filter take drop))
(import (only srfi-4 f64vector-ref f64vector-length f64vector-set!))
(import array-morphisms-core)
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


;;;; ============================================================
;;;; Group 1: am-training-step basic behavior
;;;; ============================================================

(test-group "am-training-step basic behavior"

  ;; Step returns a lazy tensor (loss)
  (let* ((model   (make-am-sequential
                   (list (make-am-dense-layer 2 1 activation: (make-identity) dtype: 'f64))))
         (opt     (make-adam (am-parameters model) learning-rate: 1e-3))
         (ctx     (make-am-training-context)))
    (let-values (((ctx-fwd ctx-bwd) ctx))
      (let* ((x    (make-lt '(1.0 0.0 0.0 1.0) '(2 2)))
             (tgt  (make-lt '(1.0 -1.0)         '(2 1)))
             (loss (am-training-step ctx-fwd ctx-bwd opt model
                                     (lambda (p) (am-mse-loss p tgt))
                                     x)))
        (test-assert "am-training-step returns a lazy tensor"
          (lazy-tensor? loss))
        (test-assert "am-training-step loss is a positive scalar"
          (>= (get-loss-value loss) 0.0)))))

  ;; Multiple steps on identical batches: loss is consistent
  (let* ((model   (make-am-sequential
                   (list (make-am-dense-layer 2 1 activation: (make-identity) dtype: 'f64))))
         (opt     (make-adam (am-parameters model) learning-rate: 0.0))   ; lr=0 => no update
         (ctx     (make-am-training-context)))
    (let-values (((ctx-fwd ctx-bwd) ctx))
      (let* ((x    (make-lt '(1.0 2.0) '(1 2)))
             (tgt  (make-lt '(0.0)     '(1 1)))
             (l1   (am-training-step ctx-fwd ctx-bwd opt model
                                     (lambda (p) (am-mse-loss p tgt)) x))
             (l2   (am-training-step ctx-fwd ctx-bwd opt model
                                     (lambda (p) (am-mse-loss p tgt)) x))
             (lv1  (get-loss-value l1))
             (lv2  (get-loss-value l2)))
        ;; With lr=0 weights don't change, so loss should be identical
        (test-assert "lr=0: loss unchanged across steps"
          (approx= lv1 lv2))))))


;;;; ============================================================
;;;; Group 2: Linear regression y = 2x convergence
;;;; ============================================================

(test-group "linear regression y = 2x"

  ;; Model: 1 -> 1 linear, learn W~2, b~0
  ;; Train for 50 steps with Adam lr=0.1, batch=4
  (set-pseudo-random-seed! "42")
  (let* ((model  (make-am-sequential
                  (list (make-am-dense-layer 1 1 activation: (make-identity) dtype: 'f64))))
         (params (am-parameters model))
         (W-data (tensor-data (car params)))
         (b-data (tensor-data (cadr params))))
    ;; Initialize W=0, b=0
    (f64vector-set! W-data 0 0.0)
    (f64vector-set! b-data 0 0.0)

    (let* ((opt (make-adam (am-parameters model) learning-rate: 0.1))
           (ctx (make-am-training-context)))
      (let-values (((ctx-fwd ctx-bwd) ctx))

        ;; Evaluate initial loss
        (let* ((x0   (make-lt '(1.0 2.0 3.0 4.0) '(4 1)))
               (tgt0 (make-lt '(2.0 4.0 6.0 8.0) '(4 1)))
               (l0   (eval-loss model x0 tgt0)))

          ;; Train 50 steps
          (let loop ((i 0))
            (when (< i 50)
              (let* ((xs  (map (lambda (v) (exact->inexact (+ v 1))) (iota 4)))
                     (ys  (map (lambda (v) (* 2.0 v)) xs))
                     (x   (make-lt xs  '(4 1)))
                     (tgt (make-lt ys  '(4 1))))
                (am-training-step ctx-fwd ctx-bwd opt model
                                  (lambda (p) (am-mse-loss p tgt))
                                  x)
                (loop (+ i 1)))))

          ;; Evaluate final loss
          (let* ((x1   (make-lt '(1.0 2.0 3.0 4.0) '(4 1)))
                 (tgt1 (make-lt '(2.0 4.0 6.0 8.0) '(4 1)))
                 (l1   (eval-loss model x1 tgt1)))
            (test-assert "y=2x regression: loss decreased after 50 steps"
              (< l1 l0))
            (test-assert "y=2x regression: final loss < 0.01"
              (< l1 0.01))
            ;; W should be close to 2
            (test-assert "y=2x regression: W close to 2.0"
              (approx= (f64vector-ref W-data 0) 2.0))))))))


;;;; ============================================================
;;;; Group 3: Constant prediction y = c convergence
;;;; ============================================================

(test-group "constant regression y = 5"

  ;; Model: 1 -> 1 linear with bias; targets = constant 5
  ;; After enough steps, b should converge toward 5, W toward 0
  (let* ((model  (make-am-sequential
                  (list (make-am-dense-layer 1 1 activation: (make-identity) dtype: 'f64))))
         (params (am-parameters model))
         (W-data (tensor-data (car params)))
         (b-data (tensor-data (cadr params))))
    (f64vector-set! W-data 0 0.1)
    (f64vector-set! b-data 0 0.0)

    (let* ((opt (make-adam (am-parameters model) learning-rate: 0.3))
           (ctx (make-am-training-context)))
      (let-values (((ctx-fwd ctx-bwd) ctx))
        ;; inputs are all 0 so W contribution is 0; only bias matters
        (let loop ((i 0) (last-loss 1e10))
          (when (< i 200)
            (let* ((x   (make-lt '(0.0 0.0 0.0 0.0) '(4 1)))
                   (tgt (make-lt '(5.0 5.0 5.0 5.0) '(4 1)))
                   (l   (am-training-step ctx-fwd ctx-bwd opt model
                                         (lambda (p) (am-mse-loss p tgt))
                                         x)))
              (loop (+ i 1) (get-loss-value l)))))

        (let* ((x2   (make-lt '(0.0 0.0 0.0 0.0) '(4 1)))
               (tgt2 (make-lt '(5.0 5.0 5.0 5.0) '(4 1)))
               (l2   (eval-loss model x2 tgt2)))
          (test-assert "constant regression: loss < 0.01 after 200 steps"
            (< l2 0.01))
          (test-assert "constant regression: bias close to 5"
            (approx= (f64vector-ref b-data 0) 5.0)))))))


;;;; ============================================================
;;;; Group 4: Loss monotonically decreases on simple problem
;;;; ============================================================

(test-group "loss trajectory"

  ;; For a small enough LR on a convex problem (linear regression),
  ;; loss should not increase after each step
  (let* ((model  (make-am-sequential
                  (list (make-am-dense-layer 1 1 activation: (make-identity) dtype: 'f64))))
         (params (am-parameters model))
         (W-data (tensor-data (car params)))
         (b-data (tensor-data (cadr params))))
    (f64vector-set! W-data 0 0.0)
    (f64vector-set! b-data 0 0.0)

    (let* ((opt (make-adam (am-parameters model) learning-rate: 1e-2))
           (ctx (make-am-training-context)))
      (let-values (((ctx-fwd ctx-bwd) ctx))
        ;; Fixed batch (no noise)
        (define (run-step)
          (let* ((x   (make-lt '(1.0 2.0 3.0) '(3 1)))
                 (tgt (make-lt '(3.0 6.0 9.0) '(3 1)))  ; y = 3x
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
          (test-assert "loss trajectory: decreasing trend (l1>l3>l5)"
            (and (> l1 l3) (> l3 l5))))))))


;;;; ============================================================
;;;; Group 5: Context trace/replay consistency
;;;; ============================================================

(test-group "context trace/replay consistency"

  ;; After the first step (trace), subsequent steps (replay) should
  ;; produce valid (non-NaN, non-zero) loss values
  (let* ((model  (make-am-sequential
                  (list (make-am-dense-layer 4 8  activation: (make-relu)     dtype: 'f64)
                        (make-am-dense-layer 8 1  activation: (make-identity) dtype: 'f64))))
         (opt    (make-adam (am-parameters model) learning-rate: 1e-3))
         (ctx    (make-am-training-context)))
    (let-values (((ctx-fwd ctx-bwd) ctx))

      ;; Make several batches; test that loss values are all finite and positive
      (define (make-batch)
        (let* ((x-data (map (lambda (_) (pseudo-random-real)) (iota 16)))
               (y-data (map (lambda (_) (pseudo-random-real)) (iota  4))))
          (cons (make-lt x-data '(4 4))
                (make-lt y-data '(4 1)))))

      (set-pseudo-random-seed! "123")
      (let ((losses
             (map (lambda (_)
                    (let* ((bp  (make-batch))
                           (x   (car bp))
                           (tgt (cdr bp))
                           (l   (am-training-step ctx-fwd ctx-bwd opt model
                                                  (lambda (p) (am-mse-loss p tgt))
                                                  x)))
                      (get-loss-value l)))
                  (iota 6))))
        (test-assert "context replay: all loss values are finite"
          (every (lambda (lv) (and (finite? lv) (not (nan? lv)))) losses))
        (test-assert "context replay: all loss values are non-negative"
          (every (lambda (lv) (>= lv 0.0)) losses))
        (test-assert "context replay: loss decreases over 6 steps"
          (< (list-ref losses 5) (list-ref losses 0)))))))


;;;; ============================================================
;;;; Group 6: Multi-layer model convergence
;;;; ============================================================

(test-group "multi-layer model convergence"

  ;; 2-layer model on a nonlinear function: y = x^2 (sampled at few points)
  ;; After enough steps, loss should be significantly lower than initial
  (set-pseudo-random-seed! "99")
  (let* ((model  (make-am-sequential
                  (list (make-am-dense-layer 1 8  activation: (make-relu)     dtype: 'f64)
                        (make-am-dense-layer 8 1  activation: (make-identity) dtype: 'f64))))
         (opt    (make-adam (am-parameters model) learning-rate: 1e-2))
         (ctx    (make-am-training-context)))
    (let-values (((ctx-fwd ctx-bwd) ctx))

      ;; Fixed dataset: x in {-2,-1,0,1,2}, y = x^2
      (define xs   '(-2.0 -1.0 0.0 1.0 2.0))
      (define ys   '( 4.0  1.0 0.0 1.0 4.0))
      (define x-lt  (make-lt xs '(5 1)))
      (define tgt-lt (make-lt ys '(5 1)))

      (define (current-loss)
        (eval-loss model x-lt tgt-lt))

      (let ((l0 (current-loss)))
        ;; Train 100 steps
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
;;;; Group 7: Adam optimizer convergence vs SGD
;;;; ============================================================

(test-group "Adam optimizer"

  ;; Compare number of steps to reach low loss: Adam should be more efficient
  ;; than a high-LR plain gradient step on this simple problem
  (define (train-to-threshold model-fn target-loss n-steps lr)
    (let* ((model (model-fn))
           (opt   (make-adam (am-parameters model) learning-rate: lr))
           (ctx   (make-am-training-context)))
      (let-values (((ctx-fwd ctx-bwd) ctx))
        (let loop ((i 0) (reached #f))
          (if (or reached (>= i n-steps))
              (list reached i)
              (let* ((x   (make-lt '(1.0 -1.0 2.0 -2.0) '(4 1)))
                     (tgt (make-lt '(2.0 -2.0 4.0 -4.0) '(4 1)))  ; y = 2x
                     (l   (am-training-step ctx-fwd ctx-bwd opt model
                                           (lambda (p) (am-mse-loss p tgt))
                                           x))
                     (lv  (get-loss-value l)))
                (loop (+ i 1) (< lv target-loss))))))))

  (define (make-linear-model)
    (let* ((m  (make-am-sequential
                (list (make-am-dense-layer 1 1 activation: (make-identity) dtype: 'f64))))
           (W  (tensor-data (car  (am-parameters m))))
           (b  (tensor-data (cadr (am-parameters m)))))
      (f64vector-set! W 0 0.0)
      (f64vector-set! b 0 0.0)
      m))

  (let* ((result (train-to-threshold make-linear-model 0.01 200 0.1))
         (reached? (car result))
         (steps    (cadr result)))
    (test-assert "Adam: reaches loss < 0.01 within 200 steps"
      reached?)
    (test-assert "Adam: converges in under 100 steps with lr=0.1"
      (< steps 100))))


;;;; ============================================================
;;;; Group 8: am-parameters integrity after training
;;;; ============================================================

(test-group "am-parameters integrity after training"

  ;; Parameters should be readable and their shapes unchanged after training
  (let* ((model  (make-am-sequential
                  (list (make-am-dense-layer 3 4 activation: (make-relu)     dtype: 'f64)
                        (make-am-dense-layer 4 1 activation: (make-identity) dtype: 'f64))))
         (opt    (make-adam (am-parameters model) learning-rate: 1e-3))
         (ctx    (make-am-training-context)))
    (let-values (((ctx-fwd ctx-bwd) ctx))
      ;; Record initial param shapes
      (let* ((params0 (am-parameters model))
             (shapes0 (map tensor-shape params0)))
        ;; Train 5 steps
        (let loop ((i 0))
          (when (< i 5)
            (let* ((x   (make-lt '(1.0 2.0 3.0 4.0 5.0 6.0) '(2 3)))
                   (tgt (make-lt '(1.0 2.0) '(2 1))))
              (am-training-step ctx-fwd ctx-bwd opt model
                                (lambda (p) (am-mse-loss p tgt))
                                x)
              (loop (+ i 1)))))

        ;; Check parameters still accessible with same shapes
        (let* ((params1 (am-parameters model))
               (shapes1 (map tensor-shape params1)))
          (test-assert "params: shapes unchanged after training"
            (equal? shapes0 shapes1))
          (test-assert "params: all parameter values are finite"
            (every (lambda (p)
                     (let* ((data (tensor-data p))
                            (n    (f64vector-length data)))
                       (let loop ((i 0))
                         (or (= i n)
                             (and (finite? (f64vector-ref data i))
                                  (loop (+ i 1)))))))
                   params1)))
          (test-assert "params: count matches model architecture (3*4+4 + 4*1+1 = 21)"
            (= (fold + 0 (map (lambda (p) (f64vector-length (tensor-data p))) params1))
               21)))))))


(test-exit)

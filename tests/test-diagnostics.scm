;; Tests for nanograd-diagnostics: gradient statistics against direct
;; computation, detection of exploding, vanishing, non-finite and
;; unbalanced gradients, clipping, and the gradient monitor.

(import scheme
        (chicken base)
        (srfi 1)
        (srfi 4)
        test
        nanograd-autograd
        nanograd-diagnostics)

(define tol 1e-12)

;; A parameter tensor whose gradient is GRAD (a list of numbers)
(define (param64 grad)
  (let ((t (make-tensor64 (make-f64vector (length grad) 0.0) (list (length grad)))))
    (set-grad! t (list->f64vector (map exact->inexact grad)))
    t))

(define (param32 grad)
  (let ((t (make-tensor32 (make-f32vector (length grad) 0.0) (list (length grad)))))
    (set-grad! t (list->f32vector (map exact->inexact grad)))
    t))

(define (stat key alist) (cdr (assq key alist)))
(define (issues ps . keys)
  (stat 'issues (apply check-gradient-health ps warn?: #f keys)))

(test-group "gradient statistics"
  (let* ((ps (list (param64 '(3 -4)) (param32 '(1 -2 2)) (param64 '(0.5))))
         (stats (gradient-stats ps))
         (layers (stat 'layer-stats stats)))
    (test "global L2 norm" (sqrt (+ 25 9 0.25)) (stat 'global-l2-norm stats))
    (test "global infinity norm" 4.0 (stat 'global-inf-norm stats))
    (test "compute-gradient-norm l2"
          (sqrt (+ 25 9 0.25)) (compute-gradient-norm ps))
    (test "compute-gradient-norm inf" 4.0 (compute-gradient-norm ps norm-type: 'inf))
    (test "per-parameter L2" '(5.0 3.0 0.5) (map (lambda (l) (stat 'l2-norm l)) layers))
    (test "per-parameter infinity norm" '(4.0 2.0 0.5)
          (map (lambda (l) (stat 'inf-norm l)) layers))
    (test "mean absolute value" '(3.5 1.6666666666666667 0.5)
          (map (lambda (l) (stat 'mean-abs l)) layers))
    (test "sizes" '(2 3 1) (map (lambda (l) (stat 'size l)) layers))
    (test "gradient-l2-norm" 5.0 (gradient-l2-norm (f64vector 3 -4) 'f64))
    (test "gradient-inf-norm" 4.0 (gradient-inf-norm (f32vector 3 -4) 'f32)))
  (let ((ps (list (param64 '(1 2)) (make-tensor64 (f64vector 0 0) '(2) requires-grad?: #f))))
    (test "a parameter without gradient is skipped" '(0)
          (map (lambda (l) (stat 'layer l)) (stat 'layer-stats (gradient-stats ps))))))

(test-group "health checks"
  (test "healthy" '() (issues (list (param64 '(0.3 0.4)))))
  (test-assert "exploding"
    (memq 'exploding-gradients (issues (list (param64 '(30 40))))))
  (test-assert "has-exploding-gradients?"
    (has-exploding-gradients? (list (param64 '(30 40)))))
  (test-assert "vanishing"
    (memq 'vanishing-gradients (issues (list (param64 '(1e-9 0))))))
  (test-assert "has-vanishing-gradients?"
    (has-vanishing-gradients? (list (param64 '(1e-9 0)))))
  (test-assert "NaN gradients are reported"
    (memq 'nan-gradients (issues (list (param64 (list 0.1 +nan.0))))))
  (test-assert "the infinity norm of a NaN gradient is NaN"
    (nan? (gradient-inf-norm (f64vector 1.0 +nan.0 2.0) 'f64)))
  (test-assert "infinite gradients are reported"
    (memq 'nan-gradients (issues (list (param32 (list 0.1 +inf.0)))))))

(test-group "imbalance threshold"
  (let ((ps (list (param64 '(1.0)) (param64 '(1e-4)))))
    (test-assert "flagged at the default threshold of 1000"
      (memq 'gradient-imbalance (issues ps)))
    (test "not flagged with a threshold of 1e5" '()
          (issues ps imbalance-threshold: 1e5))))

(test-group "clipping"
  (let ((ps (list (param64 '(3 0)) (param64 '(0 4)))))
    (test "returns the norm before clipping" 5.0
          (clip-gradients! ps max-norm: 1.0 verbose?: #f))
    (test "clipped norm" 1.0 (compute-gradient-norm ps))
    (test-assert "directions are kept"
      (every (lambda (x y) (< (abs (- x y)) tol))
             '(0.6 0.0 0.0 0.8)
             (append-map (lambda (p) (f64vector->list (tensor-grad p))) ps)))
    (test "returns #f when the norm is within the limit" #f
          (clip-gradients! ps max-norm: 2.0 verbose?: #f))))

(test-group "gradient monitor"
  (let ((m (make-gradient-monitor exploding-threshold: 10.0 history-size: 3)))
    (test-assert "monitor predicate" (gradient-monitor? m))
    (test "first step is healthy" #t
          (stat 'healthy? (record-step! m 1 (list (param64 '(0.3 0.4))))))
    (test "an exploding step is flagged" '(exploding-gradients)
          (stat 'issues (record-step! m 2 (list (param64 '(30 40))))))
    (record-step! m 3 (list (param64 '(0.6 0.8))))
    (record-step! m 4 (list (param64 '(1.2 1.6))))
    (let ((h (get-gradient-history m))
          (d (diagnose-training m)))
      (test "history keeps the last steps, oldest first" '(2 3 4)
            (map (lambda (e) (stat 'step e)) h))
      (test "recorded norms" '(50.0 1.0 2.0)
            (map (lambda (e) (stat 'global-l2-norm e)) h))
      (test "total steps" 4 (stat 'total-steps d))
      (test "unhealthy steps among the recent ones" 1 (stat 'unhealthy-steps d))
      (test "warning count" 1 (stat 'warning-count d))
      (test "max gradient norm" 50.0 (stat 'max-gradient-norm d)))))

(test-exit)

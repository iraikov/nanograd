;;; nanograd/examples/array-morphisms-integration.scm
;;;
;;; Demonstrates the unified strict/non-strict tensor architecture:
;;;   - AM-backed dense layers with lazy evaluation
;;;   - am-training-step with two-context trace/replay optimization
;;;   - Buffer reuse stats showing backward context efficiency
;;;
;;; Run with:
;;;   csi -q array-morphisms-integration.scm

(import srfi-4)
(import array-morphisms-core)
(import array-morphisms-basic-ops)
(import array-morphisms-realization)
(import array-morphisms-context)
(import (prefix array-morphisms-grad am:))
(import nanograd-autograd)
(import nanograd-layer)
(import nanograd-array-morphisms)
(import nanograd-optimizer)


;;; ============================================================
;;; XOR dataset
;;; Input:  [[0,0],[0,1],[1,0],[1,1]]
;;; Target: [[0], [1], [1], [0]]
;;; ============================================================

(define X-data '(0.0 0.0  0.0 1.0  1.0 0.0  1.0 1.0))
(define Y-data '(0.0  1.0  1.0  0.0))

(define X-mv (am:make-var (morph-from-list X-data '(4 2) 'f64) #f))
(define Y-mv (am:make-var (morph-from-list Y-data '(4 1) 'f64) #f))
(define X-lt (get-or-make-lazy X-mv))
(define Y-lt (get-or-make-lazy Y-mv))


;;; ============================================================
;;; Model: 2 -> 8 -> 8 -> 1
;;; ============================================================

(define model
  (make-am-sequential
   (list (make-am-dense-layer 2  8 activation: (make-relu)     dtype: 'f64)
         (make-am-dense-layer 8  8 activation: (make-relu)     dtype: 'f64)
         (make-am-dense-layer 8  1 activation: (make-identity) dtype: 'f64))))

(let ((params (am-parameters model)))
  (display "Model parameters: ")
  (display (length params))
  (display " tensors\n"))


;;; ============================================================
;;; Training with am-training-step
;;; ============================================================

(define params (am-parameters model))
(define opt    (make-adam params learning-rate: 5e-2))
(define-values (ctx-fwd ctx-bwd) (make-am-training-context))

(define (mse-loss-fn pred-lt) (am-mse-loss pred-lt Y-lt))

(display "Training (200 steps):\n")

(let loop ((step 0) (prev-loss +inf.0))
  (when (< step 200)
    (let* ((loss-lt  (am-training-step ctx-fwd ctx-bwd opt model mse-loss-fn X-lt))
           (loss-val (f64vector-ref (tensor-data loss-lt) 0)))
      (when (zero? (remainder step 40))
        (display "  step=")
        (display step)
        (display " loss=")
        (display (/ (round (* loss-val 1e6)) 1e6))
        (newline))
      (loop (+ step 1) loss-val))))

;; ============================================================
;; Finalize contexts and report buffer stats
;; ============================================================

(display "\nBuffer reuse (after finalization):\n")
(let ((fs (context-stats ctx-fwd)) (bs (context-stats ctx-bwd)))
  (display "  Forward: ") (display (assq 'allocations fs)) (display " -> ")
  (display (assq 'buffers fs)) (newline)
  (display "  Backward: ") (display (assq 'allocations bs)) (display " -> ")
  (display (assq 'buffers bs)) (newline))

(display "\nFinal predictions:\n")
(let* ((out-lt  (forward model X-lt))
       (out-vec (tensor-data out-lt)))
  (let loop ((i 0))
    (when (< i 4)
      (let* ((x1 (list-ref X-data (* i 2)))
             (x2 (list-ref X-data (+ (* i 2) 1)))
             (y  (list-ref Y-data i))
             (p  (f64vector-ref out-vec i)))
        (display "  [")
        (display (inexact->exact (round x1)))
        (display ",")
        (display (inexact->exact (round x2)))
        (display "] -> pred=")
        (display (/ (round (* p 100)) 100))
        (display "  target=")
        (display y)
        (newline))
      (loop (+ i 1)))))


(display "\nDone.\n")

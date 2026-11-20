;; Unit tests for binary-cross-entropy-with-logits and for the per-row
;; mask of cross-entropy-loss.  Finite-difference gradient checks of
;; both losses are in test-gradcheck.scm.

(import scheme
        (chicken base)
        (chicken condition)
        (chicken format)
        (srfi 1)
        (srfi 4)
        test
        nanograd-autograd)

;;; ==================================================================
;;; Helper Functions
;;; ==================================================================

(define (approx-equal? actual expected tolerance)
  "Check if two numbers are approximately equal within tolerance"
  (<= (abs (- actual expected)) tolerance))

(define-syntax test-approximate
  (syntax-rules ()
    ((test-approximate name expected actual tolerance)
     (test-assert name (approx-equal? actual expected tolerance)))))

(define tol 1e-12)

(define (t64 values shape #!key (requires-grad? #t))
  (make-tensor64 (list->f64vector values) shape requires-grad?: requires-grad?))

(define (const64 values shape)
  (t64 values shape requires-grad?: #f))

(define (value tensor)
  (f64vector-ref (tensor-data tensor) 0))

(define (grad-list tensor)
  (f64vector->list (tensor-grad tensor)))

(define (lists-approx-equal? actual expected tolerance)
  (and (= (length actual) (length expected))
       (every (lambda (a e) (approx-equal? a e tolerance)) actual expected)))

(define (sigmoid-value q) (/ 1.0 (+ 1.0 (exp (- q)))))

(define (naive-bce q t)
  "-[t log s + (1 - t) log(1 - s)] with s = sigmoid(q)."
  (let ((s (sigmoid-value q)))
    (- (+ (* t (log s)) (* (- 1.0 t) (log (- 1.0 s)))))))

(define (raises-error? thunk)
  (condition-case (begin (thunk) #f)
    ((exn) #t)))

;;; ==================================================================
;;; Unit Tests: binary-cross-entropy-with-logits
;;; ==================================================================

(test-group "BCE with Logits - Values"
  (let* ((qs '(-2.0 -0.5 0.0 0.7 3.0))
         (ts '(0.0 1.0 0.5 1.0 0.0))
         (expected (map naive-bce qs ts))
         (q (t64 qs '(5)))
         (t (const64 ts '(5))))
    (test-approximate "mean equals the textbook formula"
      (/ (fold + 0.0 expected) 5.0)
      (value (binary-cross-entropy-with-logits q t)) tol)
    (test-approximate "sum equals the textbook formula"
      (fold + 0.0 expected)
      (value (binary-cross-entropy-with-logits q t reduction: 'sum)) tol)
    (let ((per-element (binary-cross-entropy-with-logits q t reduction: 'none)))
      (test "none keeps the input shape" '(5) (tensor-shape per-element))
      (test-assert "none gives per-element losses"
        (lists-approx-equal? (f64vector->list (tensor-data per-element)) expected tol))))
  (test-approximate "logit 0 with target 0 gives log 2" (log 2.0)
    (value (binary-cross-entropy-with-logits (t64 '(0.0) '(1)) (const64 '(0.0) '(1))))
    tol))

(test-group "BCE with Logits - Gradient is sigmoid(q) - t"
  (let* ((qs '(-1.0 0.0 2.0))
         (ts '(1.0 0.0 0.25))
         (q (t64 qs '(3))))
    (backward! (binary-cross-entropy-with-logits q (const64 ts '(3)) reduction: 'sum))
    (test-assert "sum reduction"
      (lists-approx-equal? (grad-list q)
                           (map (lambda (qi ti) (- (sigmoid-value qi) ti)) qs ts)
                           tol)))
  (let* ((qs '(-1.0 0.0 2.0 0.5))
         (ts '(1.0 0.0 0.25 1.0))
         (q (t64 qs '(2 2))))
    (backward! (binary-cross-entropy-with-logits q (const64 ts '(2 2))))
    (test-assert "mean reduction divides by the element count"
      (lists-approx-equal? (grad-list q)
                           (map (lambda (qi ti) (/ (- (sigmoid-value qi) ti) 4.0)) qs ts)
                           tol)))
  (let ((q (t64 '(1.5) '(1)))
        (target (t64 '(1.0) '(1))))
    (backward! (binary-cross-entropy-with-logits q target))
    (test-assert "target receives no gradient"
      (every zero? (grad-list target)))))

(test-group "BCE with Logits - Extreme Logits"
  (let* ((q (t64 '(1000.0 -1000.0 1000.0 -1000.0) '(4)))
         (t (const64 '(0.0 1.0 1.0 0.0) '(4)))
         (losses (binary-cross-entropy-with-logits q t reduction: 'none)))
    (test-assert "wrong-sign logits cost |q|"
      (lists-approx-equal? (take (f64vector->list (tensor-data losses)) 2)
                           '(1000.0 1000.0) 1e-9))
    (test-assert "right-sign logits cost nothing"
      (lists-approx-equal? (drop (f64vector->list (tensor-data losses)) 2)
                           '(0.0 0.0) 1e-9))
    (backward! (sum-tensor losses))
    (test-assert "gradients are finite and saturate at +/-1 or 0"
      (lists-approx-equal? (grad-list q) '(1.0 -1.0 0.0 0.0) 1e-9))))

(test-group "BCE with Logits - Mask"
  (let* ((qs '(0.3 -1.2 2.0 0.0))
         (ts '(1.0 0.0 0.0 1.0))
         (mask (f64vector 1.0 0.0 1.0 0.0))
         (q (t64 qs '(4)))
         (loss (binary-cross-entropy-with-logits q (const64 ts '(4)) mask: mask)))
    (test-approximate "mean over the unmasked elements"
      (/ (+ (naive-bce 0.3 1.0) (naive-bce 2.0 0.0)) 2.0)
      (value loss) tol)
    (backward! loss)
    (test-assert "masked elements get no gradient"
      (and (zero? (second (grad-list q))) (zero? (fourth (grad-list q))))))
  (let* ((q (t64 '(0.3 -1.2) '(2)))
         (loss (binary-cross-entropy-with-logits
                q (const64 '(1.0 0.0) '(2))
                mask: (make-tensor32 (f32vector 0.0 0.0) '(2) requires-grad?: #f))))
    (test "fully masked loss is 0" 0.0 (value loss))
    (backward! loss)
    (test-assert "fully masked gradient is 0" (every zero? (grad-list q))))
  (test-assert "mask of the wrong length is an error"
    (raises-error? (lambda ()
                     (binary-cross-entropy-with-logits
                      (t64 '(0.0 1.0) '(2)) (const64 '(0.0 1.0) '(2))
                      mask: (f64vector 1.0))))))

(test-group "BCE with Logits - Argument Checks"
  (test-assert "target of a different shape is an error"
    (raises-error? (lambda ()
                     (binary-cross-entropy-with-logits
                      (t64 '(0.0 1.0) '(2)) (const64 '(0.0 1.0) '(1 2))))))
  (test-assert "unknown reduction is an error"
    (raises-error? (lambda ()
                     (binary-cross-entropy-with-logits
                      (t64 '(0.0) '(1)) (const64 '(0.0) '(1)) reduction: 'max)))))

;;; ==================================================================
;;; Unit Tests: cross-entropy-loss with a row mask
;;; ==================================================================

;; Logits for 4 rows of 3 classes, flattened row-major.
(define ce-logits '(1.0 2.0 0.5   0.0 0.0 3.0   -1.0 0.5 0.2   2.0 -2.0 0.0))

(define (row-ce row target-class)
  "-log softmax(logits of ROW)[TARGET-CLASS]."
  (let* ((zs (take (drop ce-logits (* 3 row)) 3))
         (lse (log (fold + 0.0 (map exp zs)))))
    (- lse (list-ref zs target-class))))

(test-group "Cross-Entropy Mask - Class Indices"
  (let* ((pred (t64 ce-logits '(4 3)))
         (target (const64 '(1.0 2.0 0.0 0.0) '(4)))
         (mask (f64vector 1.0 0.0 1.0 1.0))
         (loss (cross-entropy-loss pred target from-logits: #t mask: mask)))
    (test-approximate "mean over unmasked rows"
      (/ (+ (row-ce 0 1) (row-ce 2 0) (row-ce 3 0)) 3.0)
      (value loss) tol)
    (backward! loss)
    (test-assert "masked row gets no gradient"
      (every zero? (take (drop (grad-list pred) 3) 3)))
    (test-assert "unmasked row gradient is (softmax - onehot) / 3"
      (let* ((zs (take ce-logits 3))
             (z (fold + 0.0 (map exp zs)))
             (expected (map (lambda (zi k) (/ (- (/ (exp zi) z) (if (= k 1) 1.0 0.0)) 3.0))
                            zs '(0 1 2))))
        (lists-approx-equal? (take (grad-list pred) 3) expected tol))))
  (let* ((pred (t64 ce-logits '(4 3)))
         (loss (cross-entropy-loss pred (const64 '(1.0 2.0 0.0 0.0) '(4))
                                   from-logits: #t reduction: 'sum
                                   mask: (f64vector 1.0 0.0 1.0 1.0))))
    (test-approximate "sum over unmasked rows"
      (+ (row-ce 0 1) (row-ce 2 0) (row-ce 3 0))
      (value loss) tol))
  (let ((loss (cross-entropy-loss (t64 ce-logits '(4 3))
                                  (const64 '(1.0 -1.0 0.0 7.0) '(4))
                                  from-logits: #t
                                  mask: (f64vector 1.0 0.0 1.0 0.0))))
    (test-approximate "masked rows may hold invalid targets"
      (/ (+ (row-ce 0 1) (row-ce 2 0)) 2.0)
      (value loss) tol))
  (let ((loss (cross-entropy-loss (t64 ce-logits '(4 3))
                                  (const64 '(1.0 2.0 0.0 0.0) '(4))
                                  from-logits: #t
                                  mask: (f64vector 2.0 0.0 1.0 1.0))))
    (test-approximate "weights other than 0 and 1 give a weighted mean"
      (/ (+ (* 2.0 (row-ce 0 1)) (row-ce 2 0) (row-ce 3 0)) 4.0)
      (value loss) tol)))

(test-group "Cross-Entropy Mask - One-Hot Targets"
  (let* ((pred (t64 ce-logits '(4 3)))
         (onehot (const64 '(0 1 0  0 0 1  1 0 0  1 0 0) '(4 3)))
         (masked (cross-entropy-loss pred onehot from-logits: #t
                                     mask: (f64vector 0.0 1.0 1.0 0.0))))
    (test-approximate "mean over unmasked rows"
      (/ (+ (row-ce 1 2) (row-ce 2 0)) 2.0)
      (value masked) tol)
    (backward! masked)
    (test-assert "masked rows get no gradient"
      (and (every zero? (take (grad-list pred) 3))
           (every zero? (drop (grad-list pred) 9))))))

(test-group "Cross-Entropy Mask - Probabilities"
  ;; With from-logits #f the predictions are probabilities.
  (let* ((probs '(0.2 0.5 0.3  0.1 0.1 0.8  0.6 0.3 0.1))
         (pred (t64 probs '(3 3)))
         (loss (cross-entropy-loss pred (const64 '(1.0 2.0 0.0) '(3))
                                   mask: (f64vector 1.0 0.0 1.0))))
    (test-approximate "mean of -log p over unmasked rows"
      (/ (+ (- (log 0.5)) (- (log 0.6))) 2.0)
      (value loss) tol)
    (backward! loss)
    (test-assert "gradient is -1 / (2 p) at the targets of unmasked rows"
      (lists-approx-equal? (grad-list pred)
                           (list 0.0 (/ -1.0 (* 2.0 0.5)) 0.0
                                 0.0 0.0 0.0
                                 (/ -1.0 (* 2.0 0.6)) 0.0 0.0)
                           tol))))

(test-group "Cross-Entropy Mask - Agrees With Unmasked Loss"
  (let* ((all-ones (cross-entropy-loss (t64 ce-logits '(4 3))
                                       (const64 '(1.0 2.0 0.0 0.0) '(4))
                                       from-logits: #t
                                       mask: (f32vector 1.0 1.0 1.0 1.0)))
         (unmasked (cross-entropy-loss (t64 ce-logits '(4 3))
                                       (const64 '(1.0 2.0 0.0 0.0) '(4))
                                       from-logits: #t)))
    (test-approximate "mask of ones changes nothing" (value unmasked) (value all-ones) tol))
  (let ((loss (cross-entropy-loss (t64 ce-logits '(4 3))
                                  (const64 '(1.0 2.0 0.0 0.0) '(4))
                                  from-logits: #t
                                  mask: (f64vector 0.0 0.0 0.0 0.0))))
    (test "fully masked loss is 0" 0.0 (value loss))))

(test-group "Cross-Entropy - Flattened Sequence Logits"
  ;; (B, L, V) = (2, 3, 4) flattened to (B*L, V) = (6, 4), with the
  ;; given cells (mask 0) excluded, as in a puzzle where only blank
  ;; cells are predicted.
  (let* ((logits (map (lambda (i) (* 0.1 (- (modulo (* 7 i) 11) 5))) (iota 24)))
         (targets '(0.0 3.0 1.0 2.0 2.0 0.0))
         (mask (f64vector 1.0 0.0 1.0 1.0 0.0 1.0))
         (pred (t64 logits '(6 4)))
         (loss (cross-entropy-loss pred (const64 targets '(6)) from-logits: #t mask: mask))
         (row-loss (lambda (r)
                     (let* ((zs (take (drop logits (* 4 r)) 4))
                            (k (inexact->exact (list-ref targets r))))
                       (- (log (fold + 0.0 (map exp zs))) (list-ref zs k))))))
    (test-approximate "mean over the four unmasked cells"
      (/ (+ (row-loss 0) (row-loss 2) (row-loss 3) (row-loss 5)) 4.0)
      (value loss) tol)
    (backward! loss)
    (test-assert "every unmasked row's gradient sums to 0"
      (every (lambda (r)
               (approx-equal? (fold + 0.0 (take (drop (grad-list pred) (* 4 r)) 4)) 0.0 1e-12))
             '(0 2 3 5)))))

(test-group "Cross-Entropy Mask - Argument Checks"
  (test-assert "mask with 1-D predictions is an error"
    (raises-error? (lambda ()
                     (cross-entropy-loss (t64 '(1.0 2.0 3.0) '(3)) (const64 '(1.0) '(1))
                                         from-logits: #t mask: (f64vector 1.0)))))
  (test-assert "mask of the wrong length is an error"
    (raises-error? (lambda ()
                     (cross-entropy-loss (t64 ce-logits '(4 3))
                                         (const64 '(1.0 2.0 0.0 0.0) '(4))
                                         from-logits: #t mask: (f64vector 1.0 1.0))))))

(test-exit)

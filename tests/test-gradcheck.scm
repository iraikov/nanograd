;; Finite-difference gradient checks for transpose-tensor, layer-norm,
;; rmsnorm and the losses.
;;
;; Each check builds the scalar loss L = sum(w * f(x)) for a fixed
;; random weight tensor w, runs backward!, and compares every analytic
;; partial derivative with a central difference computed in f64.

(import scheme
        (chicken base)
        (chicken format)
        (chicken random)
        (chicken blob)
        (srfi 1)
        (srfi 4)
        test
        nanograd-autograd
        nanograd-layer)

(set-pseudo-random-seed! (string->blob "gradcheck"))

(define (random-f64vector n)
  (let ((v (make-f64vector n 0.0)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (f64vector-set! v i (- (* 2.0 (pseudo-random-real)) 1.0)))))

(define (weighted-sum y w)
  (sum-tensor (mul y w)))

;; F maps a list of input tensors to an output tensor.  Returns the
;; largest absolute difference between the analytic and numeric
;; gradients over every element of every input.
(define (max-gradient-error f shapes #!key (h 1e-6))
  (let* ((datas (map (lambda (s) (random-f64vector (apply * s))) shapes))
         (make-inputs (lambda ()
                        (map (lambda (d s) (make-tensor64 (f64vector-copy d) s))
                             datas shapes)))
         (probe (f (make-inputs)))
         (w-data (random-f64vector (apply * (tensor-shape probe))))
         (w-shape (tensor-shape probe))
         (loss-of (lambda (inputs)
                    (f64vector-ref
                     (tensor-data
                      (weighted-sum (f inputs)
                                    (make-tensor64 w-data w-shape
                                                   requires-grad?: #f)))
                     0)))
         (inputs (make-inputs))
         (loss (weighted-sum (f inputs)
                             (make-tensor64 w-data w-shape requires-grad?: #f))))
    (backward! loss)
    (fold
     (lambda (k inp err)
       (let ((n (apply * (list-ref shapes k)))
             (g (tensor-grad inp)))
         (let loop ((i 0) (err err))
           (if (= i n)
               err
               (let* ((bump (lambda (delta)
                              (let ((ins (make-inputs)))
                                (let ((d (tensor-data (list-ref ins k))))
                                  (f64vector-set! d i (+ (f64vector-ref d i) delta)))
                                (loss-of ins))))
                      (numeric (/ (- (bump h) (bump (- h))) (* 2.0 h))))
                 (loop (+ i 1)
                       (max err (abs (- numeric (f64vector-ref g i))))))))))
     0.0
     (iota (length shapes))
     inputs)))

(define (f64vector-copy v)
  (let* ((n (f64vector-length v)) (c (make-f64vector n)))
    (do ((i 0 (+ i 1))) ((= i n) c)
      (f64vector-set! c i (f64vector-ref v i)))))

(test-group "transpose-tensor"
  (test "2-D transpose data"
        '(1.0 4.0 2.0 5.0 3.0 6.0)
        (f64vector->list
         (tensor-data (transpose-tensor
                       (make-tensor64 (f64vector 1 2 3 4 5 6) '(2 3))
                       '(1 0)))))
  (test "3-D transpose shape" '(4 2 3)
        (tensor-shape (transpose-tensor
                       (make-tensor64 (make-f64vector 24 0.0) '(2 3 4))
                       '(2 0 1))))
  (test "3-D transpose element (1,2,3) -> (3,1,2)"
        (+ (* 1 12) (* 2 4) 3)
        (let* ((d (make-f64vector 24 0.0)))
          (do ((i 0 (+ i 1))) ((= i 24)) (f64vector-set! d i (exact->inexact i)))
          (inexact->exact
           (f64vector-ref
            (tensor-data (transpose-tensor (make-tensor64 d '(2 3 4)) '(2 0 1)))
            (+ (* 3 6) (* 1 3) 2)))))
  (test-assert "2-D transpose gradient"
    (< (max-gradient-error
        (lambda (xs) (transpose-tensor (car xs) '(1 0)))
        '((3 5)))
       1e-6))
  (test-assert "4-D transpose gradient"
    (< (max-gradient-error
        (lambda (xs) (transpose-tensor (car xs) '(0 2 3 1)))
        '((2 3 2 2)))
       1e-6)))

(test-group "layer-norm"
  (test-assert "rows have zero mean"
    (let* ((x (make-tensor64 (random-f64vector 12) '(3 4)))
           (y (layer-norm x
                          (make-tensor64 (make-f64vector 4 1.0) '(4))
                          (make-tensor64 (make-f64vector 4 0.0) '(4))))
           (d (tensor-data y)))
      (every (lambda (r)
               (< (abs (+ (f64vector-ref d (* r 4)) (f64vector-ref d (+ 1 (* r 4)))
                          (f64vector-ref d (+ 2 (* r 4))) (f64vector-ref d (+ 3 (* r 4)))))
                  1e-9))
             '(0 1 2))))
  (test-assert "gradient w.r.t. x, gamma, beta"
    (< (max-gradient-error
        (lambda (xs) (layer-norm (car xs) (cadr xs) (caddr xs)))
        '((3 5) (5) (5)))
       1e-6)))

(test-group "rmsnorm"
  (test-assert "1-D gradient w.r.t. x and weight"
    (< (max-gradient-error
        (lambda (xs) (rmsnorm (car xs) (cadr xs)))
        '((6) (6)))
       1e-6))
  (test-assert "2-D gradient w.r.t. x and weight"
    (< (max-gradient-error
        (lambda (xs) (rmsnorm (car xs) (cadr xs)))
        '((3 5) (5)))
       1e-6)))

;; The loss is multiplied by the random weight of max-gradient-error, so
;; these checks also cover the scaling by the upstream gradient.
(define (constant-tensor data shape)
  (make-tensor64 data shape requires-grad?: #f))

(test-group "losses"
  (test-assert "mse-loss, mean"
    (< (max-gradient-error
        (lambda (xs) (mse-loss (car xs) (constant-tensor (f64vector 0.1 -0.2 0.3 0.0 0.5 0.4) '(2 3))))
        '((2 3)))
       1e-6))
  (test-assert "mse-loss, sum"
    (< (max-gradient-error
        (lambda (xs) (mse-loss (car xs) (constant-tensor (f64vector 0.1 -0.2 0.3) '(3))
                               reduction: 'sum))
        '((3)))
       1e-6))
  (test-assert "cross-entropy from logits, class indices"
    (< (max-gradient-error
        (lambda (xs) (cross-entropy-loss (car xs) (constant-tensor (f64vector 2.0 1.0) '(2))
                                         from-logits: #t))
        '((2 4)))
       1e-6))
  (test-assert "cross-entropy from logits, one-hot"
    (< (max-gradient-error
        (lambda (xs) (cross-entropy-loss (car xs)
                                         (constant-tensor (f64vector 0 0 1 0 1 0 0 0) '(2 4))
                                         from-logits: #t))
        '((2 4)))
       1e-6))
  (test-assert "cross-entropy from logits, single sample"
    (< (max-gradient-error
        (lambda (xs) (cross-entropy-loss (car xs) (constant-tensor (f64vector 3.0) '(1))
                                         from-logits: #t))
        '((5)))
       1e-6)))

(test-group "dense layer on a batch"
  (let* ((layer (make-dense-layer 3 2 dtype: 'f64))
         (w (tensor-data (car (parameters layer))))
         (b (tensor-data (cadr (parameters layer))))
         (_ (begin (f64vector-set! b 0 0.5) (f64vector-set! b 1 -1.5)))
         (xd (f64vector 1 2 3 4 5 6 7 8 9 10 11 12))
         (x (make-tensor64 xd '(4 3) requires-grad?: #f))
         (y (forward layer x))
         (g (f64vector 1 -2 3 0.5 -1 4 2 2)))
    (test-assert "every row gets the bias"
      (every (lambda (r)
               (every (lambda (j)
                        (let ((expected
                               (+ (f64vector-ref b j)
                                  (fold + 0.0
                                        (map (lambda (k)
                                               (* (f64vector-ref w (+ (* j 3) k))
                                                  (f64vector-ref xd (+ (* r 3) k))))
                                             '(0 1 2))))))
                          (< (abs (- expected (f64vector-ref (tensor-data y) (+ (* r 2) j))))
                             1e-12)))
                      '(0 1)))
             '(0 1 2 3)))
    ;; L = sum(G * y), so dL/db_j is the sum of column j of G.
    (backward! (sum-tensor (mul y (make-tensor64 g '(4 2) requires-grad?: #f))))
    (test "bias gradient is the column sum of the upstream gradient"
          '(5.0 4.5)
          (f64vector->list (tensor-grad (cadr (parameters layer)))))))

(test-exit)

;;; test-am-autograd.scm
;;; Unit tests for array-morphisms automatic differentiation
;;;
;;; Equivalent to test-autograd.scm but for the AM grad backend.
;;; Tests morph-variable forward values and backward gradients for all
;;; gradient-tracked operations exported by array-morphisms-grad.
;;;
;;; Run with:
;;;   csi -q test-am-autograd.scm

(import scheme (chicken base) (chicken format))
(import test)
(import (only srfi-1 iota map every))
(import (only srfi-4 f64vector-ref))
(import datatype matchable)
(import array-morphisms-core)
(import array-morphisms-realization)
(import array-morphisms-grad)


;;;; ============================================================
;;;; Helpers
;;;; ============================================================

(define tol 1e-6)

(define (approx= a b) (< (abs (- a b)) tol))

(define (lists-approx= l1 l2)
  (and (= (length l1) (length l2))
       (every approx= l1 l2)))

(define (concrete->list c)
  "Read a concrete-array into a row-major list, respecting strides and offset."
  (cases array-morphism c
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
      (let ((rank (vector-length shape)))
        (define (traverse dim base)
          (if (= dim rank)
              (list (f64vector-ref data base))
              (apply append
                     (map (lambda (i)
                            (traverse (+ dim 1) (+ base (* i (vector-ref strides dim)))))
                          (iota (vector-ref shape dim))))))
        (traverse 0 offset)))
    (else (error "concrete->list: not a concrete-array"))))

(define (value-list mv)
  (concrete->list (realize (var-value mv))))

(define (grad-list mv)
  (let ((g (var-grad mv)))
    (unless g (error "grad-list: no gradient on" mv))
    (concrete->list (realize g))))

(define (make-fv lst shape)
  "Make a requires-grad=#t morph-variable from a flat f64 list."
  (make-var (morph-from-list lst (list->vector shape) 'f64) #t))

(define (make-const lst shape)
  "Make a requires-grad=#f morph-variable (constant)."
  (make-var (morph-from-list lst (list->vector shape) 'f64) #f))


;;;; ============================================================
;;;; Group 1: make-var and accessors
;;;; ============================================================

(test-group "make-var and accessors"

  (let* ((m (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
         (v (make-var m #t)))
    (test-assert "morph-variable? predicate is true"
      (morph-variable? v))
    (test-assert "var-value returns the morphism"
      (eq? m (var-value v)))
    (test-assert "var-grad is #f initially"
      (not (var-grad v)))
    (test-assert "var-requires-grad? is #t"
      (var-requires-grad? v)))

  (let* ((v (make-var (morph-from-list '(1.0) #(1) 'f64))))
    (test-assert "var-requires-grad? defaults to #f"
      (not (var-requires-grad? v))))

  (let* ((v (make-fv '(1.0 2.0) '(2)))
         (_ (zero-grad! v)))
    (test-assert "zero-grad! on fresh var leaves grad #f"
      (not (var-grad v))))

  (let* ((v (make-fv '(1.0) '(1)))
         (g (morph-from-list '(3.0) #(1) 'f64))
         (_ (accumulate-grad! v g))
         (_ (zero-grad! v)))
    (test-assert "zero-grad! clears previously set grad"
      (not (var-grad v)))))


;;;; ============================================================
;;;; Group 2: var+ forward and backward
;;;; ============================================================

(test-group "var+ forward and backward"

  (let* ((v1  (make-fv '(1.0 2.0 3.0) '(3)))
         (v2  (make-fv '(4.0 5.0 6.0) '(3)))
         (out (var+ v1 v2))
         (_   (backward! out)))
    (test-assert "var+ forward: [1,2,3]+[4,5,6]=[5,7,9]"
      (lists-approx= (value-list out) '(5.0 7.0 9.0)))
    (test-assert "var+ grad v1 = ones"
      (lists-approx= (grad-list v1) '(1.0 1.0 1.0)))
    (test-assert "var+ grad v2 = ones"
      (lists-approx= (grad-list v2) '(1.0 1.0 1.0))))

  ;; Broadcast: scalar + vector, grad of scalar sums upstream
  (let* ((v1  (make-fv '(2.0) '(1)))
         (v2  (make-fv '(1.0 2.0 3.0) '(3)))
         (out (var+ v1 v2))
         (_   (backward! out)))
    (test-assert "var+ broadcast: grad of scalar = sum(ones) = 3"
      (lists-approx= (grad-list v1) '(3.0)))))


;;;; ============================================================
;;;; Group 3: var- forward and backward
;;;; ============================================================

(test-group "var- forward and backward"

  (let* ((v1  (make-fv '(5.0 7.0) '(2)))
         (v2  (make-fv '(1.0 2.0) '(2)))
         (out (var- v1 v2))
         (_   (backward! out)))
    (test-assert "var- forward: [5,7]-[1,2]=[4,5]"
      (lists-approx= (value-list out) '(4.0 5.0)))
    (test-assert "var- grad v1 = ones"
      (lists-approx= (grad-list v1) '(1.0 1.0)))
    (test-assert "var- grad v2 = -ones"
      (lists-approx= (grad-list v2) '(-1.0 -1.0)))))


;;;; ============================================================
;;;; Group 4: var* forward and backward (product rule)
;;;; ============================================================

(test-group "var* forward and backward"

  (let* ((v1  (make-fv '(2.0 3.0) '(2)))
         (v2  (make-fv '(4.0 5.0) '(2)))
         (out (var* v1 v2))
         (_   (backward! out)))
    (test-assert "var* forward: [2,3]*[4,5]=[8,15]"
      (lists-approx= (value-list out) '(8.0 15.0)))
    (test-assert "var* grad v1 = v2 = [4,5]"
      (lists-approx= (grad-list v1) '(4.0 5.0)))
    (test-assert "var* grad v2 = v1 = [2,3]"
      (lists-approx= (grad-list v2) '(2.0 3.0)))))


;;;; ============================================================
;;;; Group 5: var/ forward and backward (quotient rule)
;;;; ============================================================

(test-group "var/ forward and backward"

  ;; f = x/y,  df/dx = 1/y,  df/dy = -x/y^2
  (let* ((v1  (make-fv '(6.0 8.0) '(2)))
         (v2  (make-fv '(2.0 4.0) '(2)))
         (out (var/ v1 v2))
         (_   (backward! out)))
    (test-assert "var/ forward: [6,8]/[2,4]=[3,2]"
      (lists-approx= (value-list out) '(3.0 2.0)))
    (test-assert "var/ grad v1 = 1/v2 = [0.5, 0.25]"
      (lists-approx= (grad-list v1) '(0.5 0.25)))
    (test-assert "var/ grad v2 = -v1/v2^2 = [-1.5, -0.5]"
      (lists-approx= (grad-list v2) '(-1.5 -0.5)))))


;;;; ============================================================
;;;; Group 6: var-negate forward and backward
;;;; ============================================================

(test-group "var-negate forward and backward"

  (let* ((v   (make-fv '(1.0 -2.0 3.0) '(3)))
         (out (var-negate v))
         (_   (backward! out)))
    (test-assert "var-negate forward: [1,-2,3]->[-1,2,-3]"
      (lists-approx= (value-list out) '(-1.0 2.0 -3.0)))
    (test-assert "var-negate grad = -ones"
      (lists-approx= (grad-list v) '(-1.0 -1.0 -1.0)))))


;;;; ============================================================
;;;; Group 7: var-pow forward and backward
;;;; ============================================================

(test-group "var-pow forward and backward"

  ;; f = x^p,  df/dx = p * x^(p-1),  df/dp = x^p * ln(x)
  (let* ((x  (make-fv '(2.0) '(1)))
         (p  (make-fv '(3.0) '(1)))
         (out (var-pow x p))
         (_   (backward! out)))
    (test-assert "var-pow forward: 2^3 = 8"
      (lists-approx= (value-list out) '(8.0)))
    ;; df/dx = 3 * 2^2 = 12
    (test-assert "var-pow grad x: p * x^(p-1) = 12"
      (lists-approx= (grad-list x) '(12.0)))
    ;; df/dp = 8 * ln(2)
    (test-assert "var-pow grad p: x^p * ln(x)"
      (lists-approx= (grad-list p) (list (* 8.0 (log 2.0)))))))


;;;; ============================================================
;;;; Group 8: var-sqrt forward and backward
;;;; ============================================================

(test-group "var-sqrt forward and backward"

  ;; f = sqrt(x),  df/dx = 1/(2*sqrt(x))
  (let* ((x-vals '(1.0 4.0 9.0))
         (v    (make-fv x-vals '(3)))
         (out  (var-sqrt v))
         (_    (backward! out))
         (expected (map (lambda (x) (/ 1.0 (* 2.0 (sqrt x)))) x-vals)))
    (test-assert "var-sqrt forward: [1,4,9]->[1,2,3]"
      (lists-approx= (value-list out) '(1.0 2.0 3.0)))
    (test-assert "var-sqrt grad = 1/(2*sqrt(x))"
      (lists-approx= (grad-list v) expected))))


;;;; ============================================================
;;;; Group 9: var-exp forward and backward
;;;; ============================================================

(test-group "var-exp forward and backward"

  ;; f = exp(x),  df/dx = exp(x)
  (let* ((x-vals '(0.0 1.0 2.0))
         (v    (make-fv x-vals '(3)))
         (out  (var-exp v))
         (_    (backward! out))
         (expected (map exp x-vals)))
    (test-assert "var-exp forward: [e^0, e^1, e^2]"
      (lists-approx= (value-list out) expected))
    (test-assert "var-exp grad = exp(x)"
      (lists-approx= (grad-list v) expected))))


;;;; ============================================================
;;;; Group 10: var-log forward and backward
;;;; ============================================================

(test-group "var-log forward and backward"

  ;; f = ln(x),  df/dx = 1/x
  (let* ((x-vals '(1.0 2.0 4.0))
         (v    (make-fv x-vals '(3)))
         (out  (var-log v))
         (_    (backward! out))
         (expected (map (lambda (x) (/ 1.0 x)) x-vals)))
    (test-assert "var-log forward: [0, ln2, ln4]"
      (lists-approx= (value-list out) (map log x-vals)))
    (test-assert "var-log grad = 1/x"
      (lists-approx= (grad-list v) expected))))


;;;; ============================================================
;;;; Group 11: var-sin and var-cos
;;;; ============================================================

(test-group "var-sin forward and backward"

  (let* ((x-vals '(0.0 1.0 2.0))
         (v    (make-fv x-vals '(3)))
         (out  (var-sin v))
         (_    (backward! out)))
    (test-assert "var-sin forward: [sin0, sin1, sin2]"
      (lists-approx= (value-list out) (map sin x-vals)))
    (test-assert "var-sin grad = cos(x)"
      (lists-approx= (grad-list v) (map cos x-vals)))))

(test-group "var-cos forward and backward"

  (let* ((x-vals '(0.0 1.0 2.0))
         (v    (make-fv x-vals '(3)))
         (out  (var-cos v))
         (_    (backward! out)))
    (test-assert "var-cos forward: [cos0, cos1, cos2]"
      (lists-approx= (value-list out) (map cos x-vals)))
    (test-assert "var-cos grad = -sin(x)"
      (lists-approx= (grad-list v) (map (lambda (x) (- (sin x))) x-vals)))))


;;;; ============================================================
;;;; Group 12: var-abs forward and backward
;;;; ============================================================

(test-group "var-abs forward and backward"

  ;; f = |x|,  df/dx = sign(x)
  (let* ((v   (make-fv '(-2.0 3.0) '(2)))
         (out (var-abs v))
         (_   (backward! out)))
    (test-assert "var-abs forward: [|-2|, |3|] = [2, 3]"
      (lists-approx= (value-list out) '(2.0 3.0)))
    (test-assert "var-abs grad: sign([-2,3]) = [-1,1]"
      (lists-approx= (grad-list v) '(-1.0 1.0)))))


;;;; ============================================================
;;;; Group 13: var-sum forward and backward
;;;; ============================================================

(test-group "var-sum forward and backward"

  (let* ((v   (make-fv '(1.0 2.0 3.0 4.0) '(4)))
         (out (var-sum v))
         (_   (backward! out)))
    (test-assert "var-sum forward: sum([1,2,3,4]) = 10"
      (lists-approx= (value-list out) '(10.0)))
    (test-assert "var-sum grad = ones"
      (lists-approx= (grad-list v) '(1.0 1.0 1.0 1.0))))

  ;; 2D: sum of 2x3 matrix
  (let* ((v   (make-fv '(1.0 2.0 3.0 4.0 5.0 6.0) '(2 3)))
         (out (var-sum v))
         (_   (backward! out)))
    (test-assert "var-sum 2D: sum of 2x3 = 21"
      (lists-approx= (value-list out) '(21.0)))
    (test-assert "var-sum 2D grad = ones (shape [2,3])"
      (lists-approx= (grad-list v) '(1.0 1.0 1.0 1.0 1.0 1.0)))))


;;;; ============================================================
;;;; Group 14: var-mean forward and backward
;;;; ============================================================

(test-group "var-mean forward and backward"

  (let* ((v   (make-fv '(1.0 2.0 3.0 4.0) '(4)))
         (out (var-mean v))
         (_   (backward! out))
         (n   4.0))
    (test-assert "var-mean forward: mean([1,2,3,4]) = 2.5"
      (lists-approx= (value-list out) '(2.5)))
    (test-assert "var-mean grad = 1/n each"
      (lists-approx= (grad-list v) (map (lambda (_) (/ 1.0 n)) '(1 2 3 4))))))


;;;; ============================================================
;;;; Group 15: var-reshape forward and backward
;;;; ============================================================

(test-group "var-reshape forward and backward"

  ;; Reshape [4] -> [2,2], backward should restore [4]
  (let* ((v   (make-fv '(1.0 2.0 3.0 4.0) '(4)))
         (r   (var-reshape v '(2 2)))
         (out (var-sum r))
         (_   (backward! out)))
    (test-assert "var-reshape forward: values preserved"
      (lists-approx= (value-list r) '(1.0 2.0 3.0 4.0)))
    (test-assert "var-reshape backward: grad flows back to original shape"
      (lists-approx= (grad-list v) '(1.0 1.0 1.0 1.0)))))


;;;; ============================================================
;;;; Group 16: var-transpose forward and backward
;;;; ============================================================

(test-group "var-transpose forward and backward"

  ;; A = [[1,2],[3,4]] -> A^T = [[1,3],[2,4]]
  ;; perm #(1 0) swaps the two axes
  (let* ((v   (make-fv '(1.0 2.0 3.0 4.0) '(2 2)))
         (t   (var-transpose v '(1 0)))
         (out (var-sum t))
         (_   (backward! out)))
    (test-assert "var-transpose forward: [[1,2],[3,4]]^T = [[1,3],[2,4]]"
      (lists-approx= (value-list t) '(1.0 3.0 2.0 4.0)))
    (test-assert "var-transpose backward: grad flows back as ones"
      (lists-approx= (grad-list v) '(1.0 1.0 1.0 1.0)))))


;;;; ============================================================
;;;; Group 17: var-matmul forward and backward
;;;; ============================================================

(test-group "var-matmul forward and backward"

  ;; A [2x2], B [2x2]: C = A @ B
  ;; A = [[1,2],[3,4]], B = [[5,6],[7,8]]
  ;; C = [[19,22],[43,50]]
  ;; backward with seed=ones [2x2]:
  ;;   dA = ones @ B^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
  ;;   dB = A^T @ ones = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
  (let* ((A   (make-fv '(1.0 2.0 3.0 4.0) '(2 2)))
         (B   (make-fv '(5.0 6.0 7.0 8.0) '(2 2)))
         (C   (var-matmul A B))
         (_   (backward! C)))
    (test-assert "var-matmul forward: [[1,2],[3,4]]@[[5,6],[7,8]]"
      (lists-approx= (value-list C) '(19.0 22.0 43.0 50.0)))
    (test-assert "var-matmul grad A = ones @ B^T = [[11,15],[11,15]]"
      (lists-approx= (grad-list A) '(11.0 15.0 11.0 15.0)))
    (test-assert "var-matmul grad B = A^T @ ones = [[4,4],[6,6]]"
      (lists-approx= (grad-list B) '(4.0 4.0 6.0 6.0))))

  ;; Non-square: A [2x3], B [3x2], C [2x2]
  ;; A = [[1,0,1],[0,1,1]], B = [[1,0],[0,1],[1,1]]
  ;; C = [[2,1],[1,2]]
  (let* ((A   (make-fv '(1.0 0.0 1.0 0.0 1.0 1.0) '(2 3)))
         (B   (make-fv '(1.0 0.0 0.0 1.0 1.0 1.0) '(3 2)))
         (C   (var-matmul A B))
         (_   (backward! C)))
    (test-assert "var-matmul non-square forward"
      (lists-approx= (value-list C) '(2.0 1.0 1.0 2.0)))))


;;;; ============================================================
;;;; Group 18: ReLU (via var-abs)
;;;; ============================================================

(test-group "ReLU (0.5*(x+|x|)) forward and backward"

  ;; ReLU(x) = 0.5 * (x + |x|)
  ;; gradient at x>0: 1; at x<0: 0; at x=0: 0.5 (sign(0)=0, so 0.5*(1+0))
  (let* ((x     (make-fv '(-2.0 0.0 3.0) '(3)))
         (half  (make-const '(0.5) '(1)))
         (relu  (var* half (var+ x (var-abs x))))
         (loss  (var-sum relu))
         (_     (backward! loss)))
    (test-assert "ReLU forward: max(0,x)"
      (lists-approx= (value-list relu) '(0.0 0.0 3.0)))
    (test-assert "ReLU backward: 0 for x<0, 0.5 at x=0, 1 for x>0"
      (lists-approx= (grad-list x) '(0.0 0.5 1.0)))))


;;;; ============================================================
;;;; Group 19: MSE loss (var-mean of squared diff)
;;;; ============================================================

(test-group "MSE loss forward and backward"

  ;; L = mean((pred - target)^2)
  ;; pred = [1,2,3], target = [0,0,0]
  ;; diff = [1,2,3], diff^2 = [1,4,9], mean = 14/3
  ;; dL/dpred = 2*(pred-target)/N = [2/3, 4/3, 6/3]
  (let* ((pred   (make-fv '(1.0 2.0 3.0) '(3)))
         (target (make-const '(0.0 0.0 0.0) '(3)))
         (diff   (var- pred target))
         (sq     (var* diff diff))
         (loss   (var-mean sq))
         (_      (backward! loss))
         (n      3.0)
         (expected-grad (map (lambda (p) (/ (* 2.0 p) n)) '(1.0 2.0 3.0))))
    (test-assert "MSE loss forward: mean([1,4,9]) = 14/3"
      (lists-approx= (value-list loss) (list (/ 14.0 3.0))))
    (test-assert "MSE loss backward: 2*(pred-target)/N"
      (lists-approx= (grad-list pred) expected-grad))))


;;;; ============================================================
;;;; Group 20: Chain rule
;;;; ============================================================

(test-group "chain rule"

  ;; f(x) = exp(x^2), df/dx = 2x * exp(x^2)
  ;; at x = 0.5: df/dx = 1.0 * exp(0.25) = e^0.25
  (let* ((x   (make-fv '(0.5) '(1)))
         (x2  (var* x x))
         (out (var-exp x2))
         (_   (backward! out)))
    (test-assert "chain rule: d/dx[exp(x^2)] at x=0.5"
      (lists-approx= (grad-list x) (list (* 1.0 (exp 0.25))))))

  ;; f(x) = sin(cos(x)), df/dx = -sin(x)*cos(cos(x))
  ;; at x = 0: df/dx = 0 * cos(1) = 0
  (let* ((x   (make-fv '(0.0) '(1)))
         (cx  (var-cos x))
         (out (var-sin cx))
         (_   (backward! out)))
    (test-assert "chain rule: d/dx[sin(cos(0))] = 0"
      (lists-approx= (grad-list x) '(0.0)))))


;;;; ============================================================
;;;; Group 21: Multi-use gradient accumulation
;;;; ============================================================

(test-group "multi-use gradient accumulation"

  ;; z = x^2 + x, at x=2: dz/dx = 2x+1 = 5
  (let* ((x   (make-fv '(2.0) '(1)))
         (x2  (var* x x))
         (z   (var+ x2 x))
         (out (var-sum z))
         (_   (backward! out)))
    (test-assert "multi-use: d/dx[x^2+x] at x=2 = 5"
      (lists-approx= (grad-list x) '(5.0))))

  ;; w = a * b + a, at a=3, b=2: dw/da = b+1 = 3, dw/db = a = 3
  (let* ((a   (make-fv '(3.0) '(1)))
         (b   (make-fv '(2.0) '(1)))
         (ab  (var* a b))
         (w   (var+ ab a))
         (out (var-sum w))
         (_   (backward! out)))
    (test-assert "multi-use: d/da[a*b+a] at a=3,b=2 = 3"
      (lists-approx= (grad-list a) '(3.0)))
    (test-assert "multi-use: d/db[a*b+a] at a=3,b=2 = 3"
      (lists-approx= (grad-list b) '(3.0)))))


(test-exit)

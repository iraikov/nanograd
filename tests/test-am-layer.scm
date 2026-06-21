;;; test-am-layer.scm
;;; Unit tests for array-morphisms layer operations
;;;
;;; Equivalent to test-layer.scm but for AM-backed layers:
;;;   make-am-dense-layer, make-am-sequential, activations,
;;;   am-parameters, am-zero-grad!, and a single training step.
;;;
;;; Run with:
;;;   csi -q test-am-layer.scm

(import scheme (chicken base) (chicken format) (chicken random))
(import test)
(import (only srfi-1 iota map fold every filter))
(import (only srfi-4 f64vector-ref f64vector-length f64vector-set!))
(import datatype matchable)
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

(define tol 1e-5)

(define (approx= a b) (< (abs (- a b)) tol))

(define (lists-approx= l1 l2)
  (and (= (length l1) (length l2))
       (every approx= l1 l2)))

(define (make-lt data shape)
  "Build a lazy tensor from flat f64 list and shape (list of ints)."
  (get-or-make-lazy
   (am:make-var (morph-from-list data (list->vector shape) 'f64) #f)))

(define (make-lt-grad data shape)
  "Build a requires-grad=#t lazy tensor."
  (get-or-make-lazy
   (am:make-var (morph-from-list data (list->vector shape) 'f64) #t)))

(define (lt-data lt)
  "Read lazy tensor values as list of f64."
  (let* ((v (tensor-data lt))
         (n (f64vector-length v)))
    (map (lambda (i) (f64vector-ref v i)) (iota n))))

(define (lt-grad lt)
  "Read lazy tensor gradient as list of f64, or #f if none."
  (let ((g (tensor-grad lt)))
    (if g
        (let ((n (f64vector-length g)))
          (map (lambda (i) (f64vector-ref g i)) (iota n)))
        #f)))

(define (count-params model)
  (fold (lambda (p acc) (+ acc (f64vector-length (tensor-data p))))
        0 (am-parameters model)))


;;;; ============================================================
;;;; Group 1: AM activations via activation-forward-am
;;;; ============================================================

(test-group "AM activation functions"

  (test-group "ReLU via activation-forward-am"
    (let* ((relu (make-relu))
           (x    (make-lt '(-2.0 0.0 1.0 3.0) '(4)))
           (y    (activation-forward-am relu x)))
      (test-assert "ReLU: output non-negative"
        (every (lambda (v) (>= v 0.0)) (lt-data y)))
      (test-assert "ReLU: relu(-2)=0, relu(0)=0, relu(1)=1, relu(3)=3"
        (lists-approx= (lt-data y) '(0.0 0.0 1.0 3.0)))))

  (test-group "Identity via activation-forward-am"
    (let* ((act (make-identity))
           (x   (make-lt '(1.0 2.0 -3.0) '(3)))
           (y   (activation-forward-am act x)))
      (test-assert "Identity: output = input"
        (lists-approx= (lt-data y) '(1.0 2.0 -3.0)))))

  (test-group "Sigmoid via activation-forward-am"
    (let* ((act (make-sigmoid))
           (x   (make-lt '(0.0) '(1)))
           (y   (activation-forward-am act x)))
      (test-assert "Sigmoid(0) = 0.5"
        (approx= (car (lt-data y)) 0.5)))))


;;;; ============================================================
;;;; Group 2: make-am-dense-layer construction
;;;; ============================================================

(test-group "make-am-dense-layer construction"

  (let* ((layer (make-am-dense-layer 4 8 activation: (make-identity) dtype: 'f64))
         (params (am-parameters layer)))

    (test-assert "dense 4->8: am-parameters returns 2 tensors"
      (= (length params) 2))

    (let* ((W (car params))
           (b (cadr params)))
      (test-assert "dense 4->8: W shape is [8,4]"
        (equal? (tensor-shape W) '(8 4)))
      (test-assert "dense 4->8: b shape is [8]"
        (equal? (tensor-shape b) '(8)))
      (test-assert "dense 4->8: W has 32 elements"
        (= (f64vector-length (tensor-data W)) 32))
      (test-assert "dense 4->8: b has 8 elements"
        (= (f64vector-length (tensor-data b)) 8))))

  ;; Different dimensions
  (let* ((layer (make-am-dense-layer 16 32 activation: (make-relu) dtype: 'f64))
         (params (am-parameters layer)))
    (test-assert "dense 16->32: W shape is [32,16]"
      (equal? (tensor-shape (car params)) '(32 16)))
    (test-assert "dense 16->32: 512+32 = 544 params"
      (= (count-params layer) 544))))


;;;; ============================================================
;;;; Group 3: make-am-dense-layer forward pass
;;;; ============================================================

(test-group "make-am-dense-layer forward"

  ;; Single-layer, known weights: W=[1,0;0,1] (identity), b=[0,0]
  ;; Input [2,2] (batch=2): each row is [1,1] -> output = [1,1] per row
  (let* ((layer (make-am-dense-layer 2 2 activation: (make-identity) dtype: 'f64))
         (params (am-parameters layer))
         (W-data (tensor-data (car params)))
         (b-data (tensor-data (cadr params))))
    ;; Set W = identity, b = 0
    (f64vector-set! W-data 0 1.0) (f64vector-set! W-data 1 0.0)
    (f64vector-set! W-data 2 0.0) (f64vector-set! W-data 3 1.0)
    (f64vector-set! b-data 0 0.0) (f64vector-set! b-data 1 0.0)

    (let* ((x   (make-lt '(1.0 0.0 0.0 1.0) '(2 2)))  ; batch=2, features=2
           (out (forward layer x)))
      (test-assert "identity layer: output shape [2,2]"
        (equal? (tensor-shape out) '(2 2)))
      (test-assert "identity layer: [1,0] -> [1,0], [0,1] -> [0,1]"
        (lists-approx= (lt-data out) '(1.0 0.0 0.0 1.0)))))

  ;; Output dimensions: batch=4, in=3, out=5 -> [4,5]
  (let* ((layer (make-am-dense-layer 3 5 activation: (make-relu) dtype: 'f64))
         (x    (make-lt (map exact->inexact (iota 12)) '(4 3)))
         (out  (forward layer x)))
    (test-assert "dense 3->5 output shape: [4,5]"
      (equal? (tensor-shape out) '(4 5)))
    (test-assert "dense 3->5 with relu: output non-negative"
      (every (lambda (v) (>= v 0.0)) (lt-data out)))))


;;;; ============================================================
;;;; Group 4: dense layer gradient flow
;;;; ============================================================

(test-group "make-am-dense-layer gradient flow"

  ;; 1-layer 2->1 with known weights
  ;; W = [1, -1], b = 0, input = [[1, 1]] (batch=1)
  ;; pred = 1*1 + (-1)*1 + 0 = 0
  ;; target = [0], loss = mean((0-0)^2) = 0 (trivially)
  ;; Use non-zero target to get non-trivial gradients
  ;; target = [1]: loss = mean((0-1)^2) = 1
  ;; dL/dpred = 2*(pred-target)/N = -2
  ;; dW = dL/dpred * x = [-2, -2], db = -2
  (let* ((layer  (make-am-dense-layer 2 1 activation: (make-identity) dtype: 'f64))
         (params (am-parameters layer))
         (W-data (tensor-data (car params)))
         (b-data (tensor-data (cadr params))))
    (f64vector-set! W-data 0  1.0)
    (f64vector-set! W-data 1 -1.0)
    (f64vector-set! b-data 0  0.0)

    (let* ((x      (make-lt '(1.0 1.0) '(1 2)))
           (tgt    (make-lt '(1.0)     '(1 1)))
           (out-mv (lazy-tensor-morph-variable (forward layer x)))
           (tgt-mv (lazy-tensor-morph-variable tgt))
           (loss   (am:var- out-mv tgt-mv))
           (sq     (am:var* loss loss))
           (l      (am:var-mean sq))
           (_      (am:backward! l)))
      (test-assert "dense grad: W parameter has gradient"
        (not (not (lt-grad (car params)))))
      (test-assert "dense grad: b parameter has gradient"
        (not (not (lt-grad (cadr params)))))))

  ;; Gradient requires-grad check: only layer params get grads, not input
  (let* ((layer (make-am-dense-layer 4 2 activation: (make-relu) dtype: 'f64))
         (x     (make-lt (map exact->inexact (iota 8)) '(2 4)))
         (out   (forward layer x))
         (loss  (am:var-mean (lazy-tensor-morph-variable out)))
         (_     (am:backward! loss)))
    (test-assert "dense relu: W gets gradient"
      (not (not (lt-grad (car (am-parameters layer))))))
    (test-assert "dense relu: b gets gradient"
      (not (not (lt-grad (cadr (am-parameters layer))))))))


;;;; ============================================================
;;;; Group 5: make-am-sequential construction
;;;; ============================================================

(test-group "make-am-sequential"

  ;; 2-layer model: 4 -> 8 -> 2
  (let* ((model (make-am-sequential
                 (list (make-am-dense-layer 4 8 activation: (make-relu) dtype: 'f64)
                       (make-am-dense-layer 8 2 activation: (make-identity) dtype: 'f64))))
         (x   (make-lt (map exact->inexact (iota 8)) '(2 4)))
         (out (forward model x)))
    (test-assert "sequential 4->8->2: output shape [2,2]"
      (equal? (tensor-shape out) '(2 2)))
    (test-assert "sequential 4->8->2: parameter count = (4*8+8)+(8*2+2) = 58"
      (= (count-params model) 58)))

  ;; 3-layer model: 3 -> 5 -> 4 -> 1
  (let* ((model (make-am-sequential
                 (list (make-am-dense-layer 3 5 activation: (make-relu)     dtype: 'f64)
                       (make-am-dense-layer 5 4 activation: (make-relu)     dtype: 'f64)
                       (make-am-dense-layer 4 1 activation: (make-identity) dtype: 'f64))))
         (x   (make-lt '(1.0 2.0 3.0 4.0 5.0 6.0) '(2 3)))
         (out (forward model x)))
    (test-assert "sequential 3->5->4->1: output shape [2,1]"
      (equal? (tensor-shape out) '(2 1)))
    (test-assert "sequential 3->5->4->1: param count = (3*5+5)+(5*4+4)+(4*1+1) = 49"
      (= (count-params model) 49))))


;;;; ============================================================
;;;; Group 6: am-parameters and am-zero-grad!
;;;; ============================================================

(test-group "am-parameters"

  (let* ((model (make-am-sequential
                 (list (make-am-dense-layer 2 4 activation: (make-relu) dtype: 'f64)
                       (make-am-dense-layer 4 1 activation: (make-identity) dtype: 'f64))))
         (params (am-parameters model)))
    (test-assert "am-parameters returns list of tensors"
      (and (list? params) (every tensor? params)))
    (test-assert "2-layer model: 4 parameter tensors (W1,b1,W2,b2)"
      (= (length params) 4))
    (test-assert "param 0: W1 shape [4,2] (8 elts)"
      (= (f64vector-length (tensor-data (list-ref params 0))) 8))
    (test-assert "param 1: b1 shape [4] (4 elts)"
      (= (f64vector-length (tensor-data (list-ref params 1))) 4))
    (test-assert "param 2: W2 shape [1,4] (4 elts)"
      (= (f64vector-length (tensor-data (list-ref params 2))) 4))
    (test-assert "param 3: b2 shape [1] (1 elt)"
      (= (f64vector-length (tensor-data (list-ref params 3))) 1))))

(test-group "am-zero-grad!"

  ;; Run a forward + backward, then zero-grad, verify grads cleared
  (let* ((model (make-am-sequential
                 (list (make-am-dense-layer 2 1 activation: (make-identity) dtype: 'f64))))
         (params (am-parameters model))
         (x      (make-lt '(1.0 2.0) '(1 2)))
         (out-mv (lazy-tensor-morph-variable (forward model x)))
         (loss   (am:var-mean out-mv))
         (_      (am:backward! loss)))
    ;; Gradients should be set after backward
    (test-assert "after backward: W has gradient"
      (not (not (lt-grad (car params)))))
    ;; Zero them
    (am-zero-grad! model)
    (test-assert "after zero-grad: W gradient cleared"
      (not (lt-grad (car params))))
    (test-assert "after zero-grad: b gradient cleared"
      (not (lt-grad (cadr params))))))


;;;; ============================================================
;;;; Group 7: am-mse-loss
;;;; ============================================================

(test-group "am-mse-loss"

  ;; pred = [[2,3]], target = [[0,0]], MSE = mean([4,9]) = 6.5
  (let* ((pred   (make-lt '(2.0 3.0) '(1 2)))
         (target (make-lt '(0.0 0.0) '(1 2)))
         (loss   (am-mse-loss pred target))
         (lv     (f64vector-ref (tensor-data loss) 0)))
    (test-assert "am-mse-loss: mean((pred-target)^2)"
      (approx= lv 6.5)))

  ;; pred = target => loss = 0
  (let* ((pred   (make-lt '(1.0 2.0 3.0) '(1 3)))
         (target (make-lt '(1.0 2.0 3.0) '(1 3)))
         (loss   (am-mse-loss pred target))
         (lv     (f64vector-ref (tensor-data loss) 0)))
    (test-assert "am-mse-loss: perfect prediction => 0"
      (approx= lv 0.0))))


;;;; ============================================================
;;;; Group 8: single training step decreases loss
;;;; ============================================================

(test-group "single am-training-step decreases loss"

  ;; Model: 1->1 identity, learn y=0 from x=1 batch
  ;; Loss should decrease after one Adam step
  (let* ((model (make-am-sequential
                 (list (make-am-dense-layer 1 1 activation: (make-identity) dtype: 'f64))))
         (opt   (make-adam (am-parameters model) learning-rate: 0.1)))
    (let-values (((ctx-fwd ctx-bwd) (make-am-training-context)))
      (define (make-input) (make-lt '(1.0) '(1 1)))
      (define (make-target) (make-lt '(0.0) '(1 1)))

      ;; Evaluate initial loss without training
      (let* ((x0   (make-input))
             (tgt0 (make-target))
             (out0 (forward model x0))
             (l0   (am-mse-loss out0 tgt0))
             (lv0  (f64vector-ref (tensor-data l0) 0)))

        ;; Training step
        (let* ((x1   (make-input))
               (tgt1 (make-target))
               (loss (am-training-step ctx-fwd ctx-bwd opt model
                                       (lambda (p) (am-mse-loss p tgt1))
                                       x1)))

          ;; Evaluate new loss
          (let* ((x2   (make-input))
                 (tgt2 (make-target))
                 (out2 (forward model x2))
                 (l2   (am-mse-loss out2 tgt2))
                 (lv2  (f64vector-ref (tensor-data l2) 0)))

            (test-assert "single step: returned loss is non-negative"
              (>= (f64vector-ref (tensor-data loss) 0) 0.0))
            (test-assert "single step: loss decreased"
              (< lv2 lv0))))))))


(test-exit)

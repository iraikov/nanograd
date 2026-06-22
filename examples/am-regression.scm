;;; nanograd/examples/am-regression.scm
;;;
;;; Array-morphisms regression example.
;;; Same task as regression.scm: learn y = sin(x1) + cos(x2) + x3^2 - 0.5*x4
;;; using AM-backed lazy tensors and the two-context trace/replay optimizer.
;;;
;;; Differences from regression.scm:
;;;   - make-am-dense-layer / make-am-sequential instead of make-dense-layer
;;;   - am-training-step with two contexts instead of manual forward/backward/step
;;;   - f64 throughout (AM backend is f64-native)
;;;   - Context stats at the end showing buffer reuse

(import scheme (chicken base) (chicken format) (chicken random) (chicken time))
(import (only srfi-1 map fold filter take drop last iota))
(import srfi-4)
(import array-morphisms-core array-morphisms-context)
(import array-morphisms-blas-exec)
(import array-morphisms-blas-egg-backend)
(import (prefix array-morphisms-grad am:))
(import nanograd-autograd nanograd-layer nanograd-optimizer)
(import nanograd-array-morphisms)

(register-blas-backend! (make-blas-egg-backend))


;;; ============================================================
;;; Data generation
;;; ============================================================

(define num-features 4)

(define feature-ranges
  '((-3.0 3.0)   ; x1: argument to sin
    (-3.0 3.0)   ; x2: argument to cos
    (-2.0 2.0)   ; x3: squared term
    (-1.0 1.0))) ; x4: linear term

(define (target-fn x1 x2 x3 x4)
  (+ (sin x1) (cos x2) (* x3 x3) (* -0.5 x4)))

(define (generate-sample)
  (let ((feats (map (lambda (r)
                      (+ (car r)
                         (* (pseudo-random-real) (- (cadr r) (car r)))))
                    feature-ranges)))
    (cons feats (apply target-fn feats))))

(define (generate-dataset n)
  (let loop ((i 0) (acc '()))
    (if (= i n) acc (loop (+ i 1) (cons (generate-sample) acc)))))

(define (shuffle lst)
  (let* ((v (list->vector lst)) (n (vector-length v)))
    (do ((i (- n 1) (- i 1)))
        ((< i 1) (vector->list v))
      (let* ((j (pseudo-random-integer (+ i 1))) (tmp (vector-ref v i)))
        (vector-set! v i (vector-ref v j))
        (vector-set! v j tmp)))))


;;; ============================================================
;;; Normalization (feature z-score + target z-score)
;;; ============================================================

(define-record-type norm-stats
  (make-norm-stats feat-means feat-stds tgt-mean tgt-std)
  norm-stats?
  (feat-means norm-feat-means)
  (feat-stds  norm-feat-stds)
  (tgt-mean   norm-tgt-mean)
  (tgt-std    norm-tgt-std))

(define (compute-norm-stats dataset)
  (let* ((n         (length dataset))
         (feat-sums (make-vector num-features 0.0))
         (feat-sq   (make-vector num-features 0.0))
         (tgt-sum   0.0)
         (tgt-sq    0.0))
    (for-each
     (lambda (s)
       (let ((feats (car s)) (tgt (cdr s)))
         (do ((i 0 (+ i 1))) ((= i num-features))
           (let ((v (list-ref feats i)))
             (vector-set! feat-sums i (+ (vector-ref feat-sums i) v))
             (vector-set! feat-sq   i (+ (vector-ref feat-sq   i) (* v v)))))
         (set! tgt-sum (+ tgt-sum tgt))
         (set! tgt-sq  (+ tgt-sq  (* tgt tgt)))))
     dataset)
    (let* ((fmeans (list->vector
                    (map (lambda (i) (/ (vector-ref feat-sums i) n))
                         (iota num-features))))
           (fstds  (list->vector
                    (map (lambda (i)
                           (let* ((m (vector-ref fmeans i))
                                  (v (- (/ (vector-ref feat-sq i) n) (* m m))))
                             (max (sqrt (max 0.0 v)) 1e-8)))
                         (iota num-features))))
           (tmean  (/ tgt-sum n))
           (tstd   (max (sqrt (max 0.0 (- (/ tgt-sq n) (* tmean tmean)))) 1e-8)))
      (make-norm-stats fmeans fstds tmean tstd))))

(define (normalize-sample s stats)
  (let ((feats (car s)) (tgt (cdr s)))
    (cons (map (lambda (i)
                 (/ (- (list-ref feats i) (vector-ref (norm-feat-means stats) i))
                    (vector-ref (norm-feat-stds stats) i)))
               (iota num-features))
          (/ (- tgt (norm-tgt-mean stats)) (norm-tgt-std stats)))))

(define (denormalize pred stats)
  (+ (* pred (norm-tgt-std stats)) (norm-tgt-mean stats)))


;;; ============================================================
;;; Batch construction: normalized samples -> lazy tensors
;;;
;;; x-lt:  [mb-size, num-features]   (input)
;;; tgt-lt: [mb-size, 1]             (targets, column vector for mse-loss)
;;; ============================================================

(define (build-batch normalized-samples mb-size)
  (let loop ((i 0) (xs '()) (ys '()) (rem normalized-samples))
    (if (or (= i mb-size) (null? rem))
        (let* ((x-flat  (apply append (reverse xs)))
               (y-flat  (reverse ys))
               (x-lt    (get-or-make-lazy
                         (am:make-var (morph-from-list x-flat
                                                       (list mb-size num-features)
                                                       'f64)
                                      #f)))
               (tgt-lt  (get-or-make-lazy
                         (am:make-var (morph-from-list y-flat
                                                       (list mb-size 1)
                                                       'f64)
                                      #f))))
          (cons x-lt tgt-lt))
        (loop (+ i 1)
              (cons (car (car rem)) xs)
              (cons (cdr (car rem)) ys)
              (cdr rem)))))


;;; ============================================================
;;; Evaluation metrics
;;; ============================================================

(define (mse preds actuals)
  (/ (fold + 0.0 (map (lambda (p a) (* (- p a) (- p a))) preds actuals))
     (length preds)))

(define (mae preds actuals)
  (/ (fold + 0.0 (map (lambda (p a) (abs (- p a))) preds actuals))
     (length preds)))

(define (r-squared preds actuals)
  (let* ((n      (length actuals))
         (mu     (/ (fold + 0.0 actuals) n))
         (ss-tot (fold + 0.0 (map (lambda (a) (* (- a mu) (- a mu))) actuals)))
         (ss-res (fold + 0.0 (map (lambda (p a) (* (- a p) (- a p)))
                                  preds actuals))))
    (if (< ss-tot 1e-10) 1.0 (- 1.0 (/ ss-res ss-tot)))))


;;; ============================================================
;;; Evaluation pass (forward only, no context)
;;; ============================================================

(define (evaluate model test-data stats mb-size)
  (let loop ((rem test-data) (preds '()) (actuals '()))
    (if (< (length rem) mb-size)
        (values (mse (map (lambda (p) (denormalize p stats)) preds)
                     (map (lambda (a) (denormalize a stats)) actuals))
                (mae (map (lambda (p) (denormalize p stats)) preds)
                     (map (lambda (a) (denormalize a stats)) actuals))
                (r-squared preds actuals))
        (let* ((batch   (take rem mb-size))
               (normed  (map (lambda (s) (normalize-sample s stats)) batch))
               (bp      (build-batch normed mb-size))
               (x-lt    (car bp))
               (out-lt  (forward model x-lt))
               (out-vec (tensor-data out-lt)))
          (loop (drop rem mb-size)
                (append preds  (map (lambda (i) (f64vector-ref out-vec i))
                                    (iota mb-size)))
                (append actuals (map (lambda (s) (cdr (normalize-sample s stats)))
                                     batch)))))))


;;; ============================================================
;;; Main
;;; ============================================================

(printf "\n")
(printf "  NanoGrad AM Regression Example\n")
(printf "  y = sin(x1) + cos(x2) + x3^2 - 0.5*x4\n")
(printf "  (array-morphisms lazy-tensor backend)\n")
(printf "\n")

(set-pseudo-random-seed! "42")

;; Dataset
(define train-data (shuffle (generate-dataset 500)))
(define test-data  (shuffle (generate-dataset 100)))
(define stats (compute-norm-stats train-data))
(printf "Dataset: ~A train, ~A test\n" (length train-data) (length test-data))
(printf "Target: mean=~A std=~A\n\n"
        (/ (round (* (norm-tgt-mean stats) 1e3)) 1e3)
        (/ (round (* (norm-tgt-std  stats) 1e3)) 1e3))

;; Model: 4 -> 64 -> 32 -> 16 -> 1 with ReLU hidden layer
(define mb-size 32)

#;(define model
  (make-am-sequential
   (list (make-am-dense-layer num-features 16
                              activation: (make-relu)
                              dtype: 'f64)
         (make-am-dense-layer 16 1
                              activation: (make-identity)
                              dtype: 'f64))))

(define model
  (make-am-sequential
   (list (make-am-dense-layer  4 64 activation: (make-relu)     dtype: 'f64)
         (make-am-dense-layer 64 32 activation: (make-relu)     dtype: 'f64)
         (make-am-dense-layer 32 16 activation: (make-relu)     dtype: 'f64)
         (make-am-dense-layer 16  1 activation: (make-identity) dtype: 'f64))))

(let ((n-params (fold (lambda (p acc)
                        (+ acc (f64vector-length (tensor-data p))))
                      0 (am-parameters model))))
  (printf "Model: ~A -> 64 -> 32 -> 16 -> 1  (~A parameters, f64)\n\n" num-features n-params))

;; Optimizer and AM training contexts
(define opt (make-adam (am-parameters model) learning-rate: 1e-3))
(define-values (ctx-fwd ctx-bwd) (make-am-training-context))

;; Training
(define num-epochs 100)
(define best-r2 -inf.0)

(printf "Training: ~A epochs, mb-size ~A, lr=~A\n"
        num-epochs mb-size 1e-3)
(printf "~A\n" (make-string 50 #\-))

(define t0 (cpu-time))

(let epoch-loop ((epoch 1))
  (when (<= epoch num-epochs)
    (let* ((shuffled    (shuffle train-data))
           (total-loss  0.0)
           (n-batches   0))

      ;; Mini-batch loop; drop the last incomplete batch so every
      ;; batch has the same shape (required for context replay).
      (let batch-loop ((rem shuffled))
        (when (>= (length rem) mb-size)
          (let* ((batch   (take rem mb-size))
                 (normed  (map (lambda (s) (normalize-sample s stats)) batch))
                 (bp      (build-batch normed mb-size))
                 (x-lt    (car bp))
                 (tgt-lt  (cdr bp))
                 ;; Loss closure captures the current batch's target.
                 ;; am-training-step traces on the first call, then
                 ;; replays into the buffer pool on every subsequent call.
                 (loss-fn (lambda (pred-lt) (am-mse-loss pred-lt tgt-lt)))
                 (loss-lt (am-training-step ctx-fwd ctx-bwd opt
                                            model loss-fn x-lt))
                 (lv      (f64vector-ref (tensor-data loss-lt) 0)))
            (set! total-loss (+ total-loss lv))
            (set! n-batches  (+ n-batches  1))
            (batch-loop (drop rem mb-size)))))

      (let ((avg-loss (/ total-loss (max n-batches 1))))
        (let-values (((e-mse e-mae e-r2)
                      (evaluate model test-data stats mb-size)))
          (printf "epoch ~A/~A  train-loss=~A  test-MSE=~A  MAE=~A  R2=~A\n"
                  epoch num-epochs
                  (/ (round (* avg-loss 1e4)) 1e4)
                  (/ (round (* e-mse    1e4)) 1e4)
                  (/ (round (* e-mae    1e4)) 1e4)
                  (/ (round (* e-r2     1e4)) 1e4))
          (when (> e-r2 best-r2) (set! best-r2 e-r2))))

      ;; LR decay
      (when (member epoch '(50 75))
        (let ((new-lr (* (get-learning-rate opt) 0.5)))
          (set-learning-rate! opt new-lr)
          (printf "  lr -> ~A\n" new-lr))))

    (epoch-loop (+ epoch 1))))

(define elapsed (- (cpu-time) t0))
(printf "~A\n" (make-string 50 #\-))
(printf "Training time: ~Ams\n\n" elapsed)


;;; ============================================================
;;; Final evaluation
;;; ============================================================

(let-values (((test-mse test-mae test-r2)
              (evaluate model test-data stats mb-size)))
  (printf "Final test metrics:\n")
  (printf "  MSE: ~A\n" test-mse)
  (printf "  MAE: ~A\n" test-mae)
  (printf "  R^2: ~A\n\n" test-r2))


;;; ============================================================
;;; Context stats: buffer reuse summary
;;; ============================================================

(let* ((total-steps (* num-epochs (quotient (length train-data) mb-size)))
       (fs (context-stats ctx-fwd))
       (bs (context-stats ctx-bwd))
       (fa (cdr (assq 'allocations fs)))
       (fb (cdr (assq 'buffers     fs)))
       (ba (cdr (assq 'allocations bs)))
       (bb (cdr (assq 'buffers     bs))))
  (printf "Context buffer reuse (~A total steps):\n" total-steps)
  (printf "  Forward:  ~A allocs -> ~A buffers (~A% reduction)\n"
          fa fb (round (* 100 (- 1 (/ fb fa)))))
  (printf "  Backward: ~A allocs -> ~A buffers (~A% reduction)\n"
          ba bb (round (* 100 (- 1 (/ bb ba))))))

(printf "\nDone.\n")

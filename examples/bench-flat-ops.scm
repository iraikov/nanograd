;;; bench-flat-ops.scm -- profile element-wise and reduction ops in the replay plan.

(import scheme (chicken base) (chicken format) (chicken random) (chicken time))
(import (only srfi-1 fold iota))
(import srfi-4)
(import array-morphisms-core array-morphisms-context array-morphisms-realization)
(import array-morphisms-blas-exec array-morphisms-blas-egg-backend)
(import (prefix array-morphisms-grad am:))
(import nanograd-array-morphisms)

(register-blas-backend! (make-blas-egg-backend))

(define (make-random-f32 n)
  (let ((v (make-f32vector n 0.0)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (f32vector-set! v i (- (* 2.0 (pseudo-random-real)) 1.0)))))

(define (f32-morph data shape)
  (make-morphism data shape 'f32))

(define (time-ms thunk)
  (let ((t0 (cpu-time)))
    (thunk)
    (- (cpu-time) t0)))

(define (bench label iters thunk)
  (let ((ms (time-ms (lambda () (do ((i 0 (+ i 1))) ((= i iters)) (thunk))))))
    (printf "  ~A: ~A ms/iter\n" label (exact->inexact (/ ms iters)))))

(printf "=== Flat-op micro-benchmarks ===\n\n")

;;; ------------------------------------------------------------------
;;; 1. execute-flat-unary-compute  (relu)
;;; ------------------------------------------------------------------

(define relu-combiner (lambda (x) (if (> x 0.0) x 0.0)))

(printf "-- execute-flat-unary-compute (relu) --\n")

;; conv1 output size: 32*28*28*16 = 401408
(let* ((size (* 32 28 28 16))
       (data (make-random-f32 size))
       (out  (make-f32vector size 0.0)))
  (bench (format "relu [~A] = 32*28*28*16" size) 50
         (lambda () (execute-flat-unary-compute relu-combiner data out size 'f32))))

;; conv2 output size: 32*14*14*32 = 200704
(let* ((size (* 32 14 14 32))
       (data (make-random-f32 size))
       (out  (make-f32vector size 0.0)))
  (bench (format "relu [~A] = 32*14*14*32" size) 50
         (lambda () (execute-flat-unary-compute relu-combiner data out size 'f32))))

;; conv3 output size: 32*7*7*64 = 100352
(let* ((size (* 32 7 7 64))
       (data (make-random-f32 size))
       (out  (make-f32vector size 0.0)))
  (bench (format "relu [~A] = 32*7*7*64" size) 50
         (lambda () (execute-flat-unary-compute relu-combiner data out size 'f32))))

;; dense1 output: 32*128 = 4096
(let* ((size (* 32 128))
       (data (make-random-f32 size))
       (out  (make-f32vector size 0.0)))
  (bench (format "relu [~A] = 32*128" size) 50
         (lambda () (execute-flat-unary-compute relu-combiner data out size 'f32))))

;;; ------------------------------------------------------------------
;;; 2. execute-flat-bias-broadcast-compute  (bias add)
;;; ------------------------------------------------------------------

(printf "\n-- execute-flat-bias-broadcast-compute (bias add) --\n")

(define add-combiner +)

;; conv1 pre-act: [32*784, 16] = 401408, N=16 (bias size)
(let* ((size (* 32 784 16))
       (data (make-random-f32 size))
       (bias (make-random-f32 16))
       (out  (make-f32vector size 0.0)))
  (bench "bias-broadcast [32*784,16] N=16" 50
         (lambda () (execute-flat-bias-broadcast-compute add-combiner data bias out size 16 'f32))))

;; conv2 pre-act: [32*196, 32], N=32
(let* ((size (* 32 196 32))
       (data (make-random-f32 size))
       (bias (make-random-f32 32))
       (out  (make-f32vector size 0.0)))
  (bench "bias-broadcast [32*196,32] N=32" 50
         (lambda () (execute-flat-bias-broadcast-compute add-combiner data bias out size 32 'f32))))

;;; ------------------------------------------------------------------
;;; 3. execute-reduction-morphism (sum over axis 0 -- for db gradient)
;;; ------------------------------------------------------------------

(printf "\n-- reduce sum axis 0 (bias gradient) --\n")

;; Sum [25088, 16] -> [16]  (bias grad conv1)
(let* ((M (* 32 784)) (N 16)
       (a (am:make-var (f32-morph (make-random-f32 (* M N)) (list M N)) #f))
       (b (am:var-sum a '(0))))
  (bench "sum [25088,16]->[16]" 50
         (lambda () (am:realize b))))

;; Sum [6272, 32] -> [32]  (bias grad conv2)
(let* ((M (* 32 196)) (N 32)
       (a (am:make-var (f32-morph (make-random-f32 (* M N)) (list M N)) #f))
       (b (am:var-sum a '(0))))
  (bench "sum [6272,32]->[32]" 50
         (lambda () (am:realize b))))

;;; ------------------------------------------------------------------
;;; 4. execute-reduction-morphism (max over axis 1 -- for cross-entropy)
;;; ------------------------------------------------------------------

(printf "\n-- reduce max axis 1 (softmax stability) --\n")
(let* ((M 32) (N 4)
       (a (am:make-var (f32-morph (make-random-f32 (* M N)) (list M N)) #f))
       (b (am:var-reduce 'max a '(1) #t)))
  (bench "max [32,4]->[32,1]" 50
         (lambda () (am:realize b))))

;;; ------------------------------------------------------------------
;;; 5. Baseline: tight loop with f32vector-set! only (overhead floor)
;;; ------------------------------------------------------------------

(printf "\n-- floor: tight f32vector-set! loop --\n")
(let* ((size (* 32 784 16))
       (buf  (make-f32vector size 0.0)))
  (bench (format "f32vector-set! loop [~A]" size) 50
         (lambda ()
           (do ((i 0 (+ i 1))) ((= i size))
             (f32vector-set! buf i 1.0)))))

;;; ------------------------------------------------------------------
;;; 6. Full single-layer forward+backward via SSA (3 repeated replays)
;;; ------------------------------------------------------------------

(printf "\n-- single conv layer SSA (conv2: 16->32, 28->14) --\n")

(define single-conv
  (make-am-sequential
   (list (make-am-conv2d-layer 16 32 3 stride: 2 padding: 1
                               activation: (make-relu) dtype: 'f32))))

(define single-opt
  (make-adam (am-parameters single-conv) learning-rate: 0.001))

(define single-ctx (make-morphism-context))

(define single-input (make-random-f32 (* 32 16 28 28)))
(define single-target (make-f32vector (* 32 32 14 14) 0.0))

(define (single-step)
  (let* ((x (get-or-make-lazy
              (am:make-var (f32-morph single-input (list 32 16 28 28)) #f)))
         (tgt (get-or-make-lazy
               (am:make-var (f32-morph single-target (list 32 32 14 14)) #f)))
         (loss-fn (lambda (out)
                    (am-mse-loss out tgt))))
    (am-training-step/ssa single-ctx single-opt single-conv loss-fn x tgt)))

(printf "  trace: ")
(let ((ms (time-ms single-step)))
  (printf "~A ms\n" ms))

(printf "  replays (5):\n")
(do ((i 0 (+ i 1))) ((= i 5))
  (let ((ms (time-ms single-step)))
    (printf "    replay ~A: ~A ms\n" i ms)))

(printf "\nDone.\n")

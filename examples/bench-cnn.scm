;;; bench-cnn.scm -- profile one training batch of the CNN.
;;; Times full step (trace vs replay) and individual operations.

(import scheme (chicken base) (chicken format) (chicken random) (chicken time) (chicken sort))
(import (only srfi-1 fold iota take))
(import srfi-4)
(import array-morphisms-core array-morphisms-context array-morphisms-realization)
(import array-morphisms-blas-exec)
(import array-morphisms-blas-egg-backend)
(import (prefix array-morphisms-grad am:))
(import nanograd-autograd nanograd-layer nanograd-optimizer)
(import nanograd-array-morphisms)
(import (only array-morphisms-ssa replay-timing-reset! replay-timing-results))

(define (ms/iter ms iters) (exact->inexact (/ ms iters)))

(register-blas-backend! (make-blas-egg-backend))

;;; ---------------------------------------------------------------
;;; Helpers
;;; ---------------------------------------------------------------

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

;;; ---------------------------------------------------------------
;;; Model (same architecture as am-cnn-ssa.scm)
;;; ---------------------------------------------------------------

(set-pseudo-random-seed! "42")

(define batch-size 32)
(define image-size 28)

(define model
  (make-am-sequential
   (list
    (make-am-conv2d-layer 1 16 3  stride: 1 padding: 1
                          activation: (make-relu) dtype: 'f32)
    (make-am-conv2d-layer 16 32 3 stride: 2 padding: 1
                          activation: (make-relu) dtype: 'f32)
    (make-am-conv2d-layer 32 64 3 stride: 2 padding: 1
                          activation: (make-relu) dtype: 'f32)
    (make-am-flatten name: "Flatten")
    (make-am-dense-layer (* 64 7 7) 128
                         activation: (make-relu) dtype: 'f32)
    (make-am-dense-layer 128 4
                         activation: (make-identity) dtype: 'f32))))

;;; ---------------------------------------------------------------
;;; One synthetic batch
;;; ---------------------------------------------------------------

(define input-data  (make-random-f32 (* batch-size 1 image-size image-size)))
(define target-data (make-f32vector (* batch-size 4) 0.0))
(do ((i 0 (+ i 1))) ((= i batch-size))
  (f32vector-set! target-data (+ (* i 4) (modulo i 4)) 1.0))

(define (make-input-lt)
  (let ((mv (am:make-var (f32-morph input-data
                                    (list batch-size 1 image-size image-size))
                         #f)))
    (get-or-make-lazy mv)))

(define (make-target-lt)
  (let ((mv (am:make-var (f32-morph target-data (list batch-size 4)) #f)))
    (get-or-make-lazy mv)))

;;; ---------------------------------------------------------------
;;; Optimizer and SSA context
;;; ---------------------------------------------------------------

(define optimizer
  (make-adam (am-parameters model) learning-rate: 0.001 weight-decay: 0.0001))

(define ctx (make-morphism-context))

(define (do-step)
  (let* ((x-lt   (make-input-lt))
         (tgt-lt (make-target-lt))
         (loss-fn (lambda (logits-lt)
                    (am-cross-entropy-loss logits-lt tgt-lt))))
    (am-training-step/ssa ctx optimizer model loss-fn x-lt tgt-lt)))

;;; ---------------------------------------------------------------
;;; 1. Trace step (first call, compiles SSA replay plan)
;;; ---------------------------------------------------------------

(printf "=== CNN Batch Benchmark (batch=~A, img=~Ax~A) ===\n\n"
        batch-size image-size image-size)

(printf "-- Trace step (compiles SSA replay plan) --\n")
(let ((ms (time-ms do-step)))
  (printf "  trace step:  ~A ms\n\n" ms))

;;; ---------------------------------------------------------------
;;; 2. Replay timing
;;; ---------------------------------------------------------------

(printf "-- Replay steps (N=10) --\n")
(let ((times '()))
  (do ((i 0 (+ i 1))) ((= i 10))
    (set! times (cons (time-ms do-step) times)))
  (let* ((ts (reverse times))
         (total (fold + 0.0 ts))
         (avg   (/ total 10)))
    (for-each (lambda (i t)
                (printf "  replay ~A: ~A ms\n" i t))
              (iota 10)
              ts)
    (printf "  avg: ~A ms\n\n" avg)))

;;; ---------------------------------------------------------------
;;; 3. Forward-only pass (no grad)
;;; ---------------------------------------------------------------

(printf "-- Forward-only pass (no grad, no optimizer) --\n")
(let ((ms (time-ms (lambda () (forward model (make-input-lt))))))
  (printf "  forward only: ~A ms\n\n" ms))

;;; ---------------------------------------------------------------
;;; 4. Micro-benchmarks: im2col, col2im, implicit GEMM, BLAS matmul
;;;
;;; im2col/col2im are used only in the non-SSA path or as standalone
;;; ops.  The SSA training loop uses ri-conv-fwd / ri-conv-bwd-* which
;;; fuse im2col+GEMM+bias into a single pass with no col buffer.
;;; ---------------------------------------------------------------

(printf "-- execute-im2col-batched micro-benchmarks (~A iters each) --\n" 50)

;; conv1: [32, 1, 28, 28] -> col [32, 9, 784]
(let* ((N 32) (C 1) (H 28) (W 28) (KH 3) (KW 3) (SH 1) (SW 1) (PH 1) (PW 1)
       (OH 28) (OW 28) (iters 50)
       (src (make-random-f32 (* N C H W)))
       (out (make-f32vector (* N (* C KH KW) (* OH OW)) 0.0))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-im2col-batched out src N C H W KH KW SH SW PH PW OH OW 'f32))))))
  (printf "  conv1 [~Ax~Ax~Ax~A] -> [~A,~A,~A]: ~A ms/iter\n"
          N C H W N (* C KH KW) (* OH OW) (ms/iter ms iters)))

;; conv2: [32, 16, 28, 28] -> col [32, 144, 196]
(let* ((N 32) (C 16) (H 28) (W 28) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 14) (OW 14) (iters 50)
       (src (make-random-f32 (* N C H W)))
       (out (make-f32vector (* N (* C KH KW) (* OH OW)) 0.0))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-im2col-batched out src N C H W KH KW SH SW PH PW OH OW 'f32))))))
  (printf "  conv2 [~Ax~Ax~Ax~A] -> [~A,~A,~A]: ~A ms/iter\n"
          N C H W N (* C KH KW) (* OH OW) (ms/iter ms iters)))

;; conv3: [32, 32, 14, 14] -> col [32, 288, 49]
(let* ((N 32) (C 32) (H 14) (W 14) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 7) (OW 7) (iters 50)
       (src (make-random-f32 (* N C H W)))
       (out (make-f32vector (* N (* C KH KW) (* OH OW)) 0.0))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-im2col-batched out src N C H W KH KW SH SW PH PW OH OW 'f32))))))
  (printf "  conv3 [~Ax~Ax~Ax~A] -> [~A,~A,~A]: ~A ms/iter\n"
          N C H W N (* C KH KW) (* OH OW) (ms/iter ms iters)))

(printf "\n-- execute-col2im-batched micro-benchmarks (~A iters each) --\n" 50)

;; col2im conv1 backward
(let* ((N 32) (C 1) (H 28) (W 28) (KH 3) (KW 3) (SH 1) (SW 1) (PH 1) (PW 1)
       (OH 28) (OW 28) (iters 50)
       (col (make-random-f32 (* N (* C KH KW) (* OH OW))))
       (col-shape (vector N (* C KH KW) (* OH OW)))
       (out (make-f32vector (* N C H W) 0.0))
       (out-shape (vector N C H W))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-col2im-batched out out-shape col col-shape
                                                KH KW SH SW PH PW 'f32))))))
  (printf "  conv1 back [~Ax~Ax~Ax~A]: ~A ms/iter\n" N C H W (ms/iter ms iters)))

;; col2im conv2 backward
(let* ((N 32) (C 16) (H 28) (W 28) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 14) (OW 14) (iters 50)
       (col (make-random-f32 (* N (* C KH KW) (* OH OW))))
       (col-shape (vector N (* C KH KW) (* OH OW)))
       (out (make-f32vector (* N C H W) 0.0))
       (out-shape (vector N C H W))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-col2im-batched out out-shape col col-shape
                                                KH KW SH SW PH PW 'f32))))))
  (printf "  conv2 back [~Ax~Ax~Ax~A]: ~A ms/iter\n" N C H W (ms/iter ms iters)))

;; col2im conv3 backward
(let* ((N 32) (C 32) (H 14) (W 14) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 7) (OW 7) (iters 50)
       (col (make-random-f32 (* N (* C KH KW) (* OH OW))))
       (col-shape (vector N (* C KH KW) (* OH OW)))
       (out (make-f32vector (* N C H W) 0.0))
       (out-shape (vector N C H W))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-col2im-batched out out-shape col col-shape
                                                KH KW SH SW PH PW 'f32))))))
  (printf "  conv3 back [~Ax~Ax~Ax~A]: ~A ms/iter\n" N C H W (ms/iter ms iters)))

(printf "\n-- execute-conv-fwd-nchw micro-benchmarks (~A iters each) --\n" 50)
(printf "   (fused im2col+GEMM+bias, no col buffer -- same as ri-conv-fwd in SSA replay)\n")

;; conv1: N=32, C=1, H=W=28, KH=KW=3, S=1, P=1, OH=OW=28, out-ch=16
(let* ((N 32) (C 1) (H 28) (W 28) (KH 3) (KW 3) (SH 1) (SW 1) (PH 1) (PW 1)
       (OH 28) (OW 28) (out-ch 16) (fan-in (* C KH KW)) (iters 50)
       (src (make-random-f32 (* N C H W)))
       (wt  (make-random-f32 (* fan-in out-ch)))
       (b   (make-f32vector out-ch 0.0))
       (out (make-f32vector (* N OH OW out-ch) 0.0))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-fwd-nchw out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv1 [~Ax~Ax~Ax~A] -> [~A,~A]: ~A ms/iter\n"
          N C H W (* N OH OW) out-ch (ms/iter ms iters)))

;; conv2: N=32, C=16, H=W=28, KH=KW=3, S=2, P=1, OH=OW=14, out-ch=32
(let* ((N 32) (C 16) (H 28) (W 28) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 14) (OW 14) (out-ch 32) (fan-in (* C KH KW)) (iters 50)
       (src (make-random-f32 (* N C H W)))
       (wt  (make-random-f32 (* fan-in out-ch)))
       (b   (make-f32vector out-ch 0.0))
       (out (make-f32vector (* N OH OW out-ch) 0.0))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-fwd-nchw out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv2 [~Ax~Ax~Ax~A] -> [~A,~A]: ~A ms/iter\n"
          N C H W (* N OH OW) out-ch (ms/iter ms iters)))

;; conv3: N=32, C=32, H=W=14, KH=KW=3, S=2, P=1, OH=OW=7, out-ch=64
(let* ((N 32) (C 32) (H 14) (W 14) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 7) (OW 7) (out-ch 64) (fan-in (* C KH KW)) (iters 50)
       (src (make-random-f32 (* N C H W)))
       (wt  (make-random-f32 (* fan-in out-ch)))
       (b   (make-f32vector out-ch 0.0))
       (out (make-f32vector (* N OH OW out-ch) 0.0))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-fwd-nchw out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv3 [~Ax~Ax~Ax~A] -> [~A,~A]: ~A ms/iter\n"
          N C H W (* N OH OW) out-ch (ms/iter ms iters)))

(printf "\n-- execute-conv-bwd-data-nchw micro-benchmarks (~A iters each) --\n" 50)

(let* ((N 32) (C 1) (H 28) (W 28) (KH 3) (KW 3) (SH 1) (SW 1) (PH 1) (PW 1)
       (OH 28) (OW 28) (out-ch 16) (fan-in (* C KH KW)) (iters 50)
       (g    (make-random-f32 (* N OH OW out-ch)))
       (g-shape (vector (* N OH OW) out-ch))
       (wt   (make-random-f32 (* fan-in out-ch)))
       (dx   (make-f32vector (* N C H W) 0.0))
       (x-shape (vector N C H W))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-bwd-data-nchw dx x-shape g g-shape wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv1 bwd-data [~Ax~Ax~Ax~A]: ~A ms/iter\n" N C H W (ms/iter ms iters)))

(let* ((N 32) (C 16) (H 28) (W 28) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 14) (OW 14) (out-ch 32) (fan-in (* C KH KW)) (iters 50)
       (g    (make-random-f32 (* N OH OW out-ch)))
       (g-shape (vector (* N OH OW) out-ch))
       (wt   (make-random-f32 (* fan-in out-ch)))
       (dx   (make-f32vector (* N C H W) 0.0))
       (x-shape (vector N C H W))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-bwd-data-nchw dx x-shape g g-shape wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv2 bwd-data [~Ax~Ax~Ax~A]: ~A ms/iter\n" N C H W (ms/iter ms iters)))

(let* ((N 32) (C 32) (H 14) (W 14) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 7) (OW 7) (out-ch 64) (fan-in (* C KH KW)) (iters 50)
       (g    (make-random-f32 (* N OH OW out-ch)))
       (g-shape (vector (* N OH OW) out-ch))
       (wt   (make-random-f32 (* fan-in out-ch)))
       (dx   (make-f32vector (* N C H W) 0.0))
       (x-shape (vector N C H W))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-bwd-data-nchw dx x-shape g g-shape wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv3 bwd-data [~Ax~Ax~Ax~A]: ~A ms/iter\n" N C H W (ms/iter ms iters)))

(printf "\n-- execute-conv-bwd-weights-nchw micro-benchmarks (~A iters each) --\n" 50)

(let* ((N 32) (C 1) (H 28) (W 28) (KH 3) (KW 3) (SH 1) (SW 1) (PH 1) (PW 1)
       (OH 28) (OW 28) (out-ch 16) (fan-in (* C KH KW)) (iters 50)
       (g    (make-random-f32 (* N OH OW out-ch)))
       (g-shape (vector (* N OH OW) out-ch))
       (src  (make-random-f32 (* N C H W)))
       (dwt  (make-f32vector (* fan-in out-ch) 0.0))
       (wt-shape (vector fan-in out-ch))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-bwd-weights-nchw dwt wt-shape g g-shape src N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv1 bwd-wt [fan_in=~A, out_ch=~A]: ~A ms/iter\n" fan-in out-ch (ms/iter ms iters)))

(let* ((N 32) (C 16) (H 28) (W 28) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 14) (OW 14) (out-ch 32) (fan-in (* C KH KW)) (iters 50)
       (g    (make-random-f32 (* N OH OW out-ch)))
       (g-shape (vector (* N OH OW) out-ch))
       (src  (make-random-f32 (* N C H W)))
       (dwt  (make-f32vector (* fan-in out-ch) 0.0))
       (wt-shape (vector fan-in out-ch))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-bwd-weights-nchw dwt wt-shape g g-shape src N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv2 bwd-wt [fan_in=~A, out_ch=~A]: ~A ms/iter\n" fan-in out-ch (ms/iter ms iters)))

(let* ((N 32) (C 32) (H 14) (W 14) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 7) (OW 7) (out-ch 64) (fan-in (* C KH KW)) (iters 50)
       (g    (make-random-f32 (* N OH OW out-ch)))
       (g-shape (vector (* N OH OW) out-ch))
       (src  (make-random-f32 (* N C H W)))
       (dwt  (make-f32vector (* fan-in out-ch) 0.0))
       (wt-shape (vector fan-in out-ch))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-bwd-weights-nchw dwt wt-shape g g-shape src N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv3 bwd-wt [fan_in=~A, out_ch=~A]: ~A ms/iter\n" fan-in out-ch (ms/iter ms iters)))

;;; ---------------------------------------------------------------
;;; 4b. BLAS-backed conv micro-benchmarks
;;;
;;; Direct timing of execute-im2col-batched-mr and execute-conv-*-blas
;;; (the functions used in the SSA BLAS dispatch path), to explain the
;;; gap between standalone BLAS gemm timing and SSA per-step timing.
;;; ---------------------------------------------------------------

(printf "\n-- execute-im2col-batched-mr micro-benchmarks (~A iters each) --\n" 50)
(printf "   (MR layout: [N*OH_OW, fan_in] -- used by BLAS conv path)\n")

(let* ((N 32) (C 1) (H 28) (W 28) (KH 3) (KW 3) (SH 1) (SW 1) (PH 1) (PW 1)
       (OH 28) (OW 28) (fan-in (* C KH KW)) (M (* N OH OW)) (iters 50)
       (src (make-random-f32 (* N C H W)))
       (out (make-f32vector (* M fan-in) 0.0))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-im2col-batched-mr out src N C H W KH KW SH SW PH PW OH OW 'f32))))))
  (printf "  conv1 [~Ax~Ax~Ax~A] -> [~A,~A]: ~A ms/iter\n"
          N C H W M fan-in (ms/iter ms iters)))

(let* ((N 32) (C 16) (H 28) (W 28) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 14) (OW 14) (fan-in (* C KH KW)) (M (* N OH OW)) (iters 50)
       (src (make-random-f32 (* N C H W)))
       (out (make-f32vector (* M fan-in) 0.0))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-im2col-batched-mr out src N C H W KH KW SH SW PH PW OH OW 'f32))))))
  (printf "  conv2 [~Ax~Ax~Ax~A] -> [~A,~A]: ~A ms/iter\n"
          N C H W M fan-in (ms/iter ms iters)))

(let* ((N 32) (C 32) (H 14) (W 14) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 7) (OW 7) (fan-in (* C KH KW)) (M (* N OH OW)) (iters 50)
       (src (make-random-f32 (* N C H W)))
       (out (make-f32vector (* M fan-in) 0.0))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-im2col-batched-mr out src N C H W KH KW SH SW PH PW OH OW 'f32))))))
  (printf "  conv3 [~Ax~Ax~Ax~A] -> [~A,~A]: ~A ms/iter\n"
          N C H W M fan-in (ms/iter ms iters)))

(printf "\n-- execute-col2im-batched-mr micro-benchmarks (~A iters each) --\n" 50)

(let* ((N 32) (C 1) (H 28) (W 28) (KH 3) (KW 3) (SH 1) (SW 1) (PH 1) (PW 1)
       (OH 28) (OW 28) (fan-in (* C KH KW)) (M (* N OH OW)) (iters 50)
       (col (make-random-f32 (* M fan-in)))
       (out (make-f32vector (* N C H W) 0.0))
       (out-shape (vector N C H W))
       (col-shape (vector M fan-in))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-col2im-batched-mr out out-shape col col-shape KH KW SH SW PH PW 'f32))))))
  (printf "  conv1 back [~Ax~Ax~Ax~A]: ~A ms/iter\n" N C H W (ms/iter ms iters)))

(let* ((N 32) (C 16) (H 28) (W 28) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 14) (OW 14) (fan-in (* C KH KW)) (M (* N OH OW)) (iters 50)
       (col (make-random-f32 (* M fan-in)))
       (out (make-f32vector (* N C H W) 0.0))
       (out-shape (vector N C H W))
       (col-shape (vector M fan-in))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-col2im-batched-mr out out-shape col col-shape KH KW SH SW PH PW 'f32))))))
  (printf "  conv2 back [~Ax~Ax~Ax~A]: ~A ms/iter\n" N C H W (ms/iter ms iters)))

(let* ((N 32) (C 32) (H 14) (W 14) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 7) (OW 7) (fan-in (* C KH KW)) (M (* N OH OW)) (iters 50)
       (col (make-random-f32 (* M fan-in)))
       (out (make-f32vector (* N C H W) 0.0))
       (out-shape (vector N C H W))
       (col-shape (vector M fan-in))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-col2im-batched-mr out out-shape col col-shape KH KW SH SW PH PW 'f32))))))
  (printf "  conv3 back [~Ax~Ax~Ax~A]: ~A ms/iter\n" N C H W (ms/iter ms iters)))

(printf "\n-- execute-conv-fwd-blas micro-benchmarks (~A iters each) --\n" 50)
(printf "   (im2col-mr + BLAS gemm + bias -- actual BLAS SSA path)\n")

(let* ((N 32) (C 1) (H 28) (W 28) (KH 3) (KW 3) (SH 1) (SW 1) (PH 1) (PW 1)
       (OH 28) (OW 28) (out-ch 16) (fan-in (* C KH KW)) (M (* N OH OW)) (iters 50)
       (src (make-random-f32 (* N C H W)))
       (wt  (make-random-f32 (* fan-in out-ch)))
       (b   (make-f32vector out-ch 0.0))
       (out (make-f32vector (* M out-ch) 0.0))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-fwd-blas out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv1 [~Ax~Ax~Ax~A] -> [~A,~A]: ~A ms/iter\n"
          N C H W M out-ch (ms/iter ms iters)))

(let* ((N 32) (C 16) (H 28) (W 28) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 14) (OW 14) (out-ch 32) (fan-in (* C KH KW)) (M (* N OH OW)) (iters 50)
       (src (make-random-f32 (* N C H W)))
       (wt  (make-random-f32 (* fan-in out-ch)))
       (b   (make-f32vector out-ch 0.0))
       (out (make-f32vector (* M out-ch) 0.0))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-fwd-blas out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv2 [~Ax~Ax~Ax~A] -> [~A,~A]: ~A ms/iter\n"
          N C H W M out-ch (ms/iter ms iters)))

(let* ((N 32) (C 32) (H 14) (W 14) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 7) (OW 7) (out-ch 64) (fan-in (* C KH KW)) (M (* N OH OW)) (iters 50)
       (src (make-random-f32 (* N C H W)))
       (wt  (make-random-f32 (* fan-in out-ch)))
       (b   (make-f32vector out-ch 0.0))
       (out (make-f32vector (* M out-ch) 0.0))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-fwd-blas out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv3 [~Ax~Ax~Ax~A] -> [~A,~A]: ~A ms/iter\n"
          N C H W M out-ch (ms/iter ms iters)))

(printf "\n-- execute-conv-bwd-data-blas micro-benchmarks (~A iters each) --\n" 50)

(let* ((N 32) (C 1) (H 28) (W 28) (KH 3) (KW 3) (SH 1) (SW 1) (PH 1) (PW 1)
       (OH 28) (OW 28) (out-ch 16) (fan-in (* C KH KW)) (M (* N OH OW)) (iters 50)
       (g     (make-random-f32 (* M out-ch)))
       (g-shape (vector M out-ch))
       (wt    (make-random-f32 (* fan-in out-ch)))
       (dx    (make-f32vector (* N C H W) 0.0))
       (x-shape (vector N C H W))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-bwd-data-blas dx x-shape g g-shape wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv1 bwd-data [~Ax~Ax~Ax~A]: ~A ms/iter\n" N C H W (ms/iter ms iters)))

(let* ((N 32) (C 16) (H 28) (W 28) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 14) (OW 14) (out-ch 32) (fan-in (* C KH KW)) (M (* N OH OW)) (iters 50)
       (g     (make-random-f32 (* M out-ch)))
       (g-shape (vector M out-ch))
       (wt    (make-random-f32 (* fan-in out-ch)))
       (dx    (make-f32vector (* N C H W) 0.0))
       (x-shape (vector N C H W))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-bwd-data-blas dx x-shape g g-shape wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv2 bwd-data [~Ax~Ax~Ax~A]: ~A ms/iter\n" N C H W (ms/iter ms iters)))

(let* ((N 32) (C 32) (H 14) (W 14) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 7) (OW 7) (out-ch 64) (fan-in (* C KH KW)) (M (* N OH OW)) (iters 50)
       (g     (make-random-f32 (* M out-ch)))
       (g-shape (vector M out-ch))
       (wt    (make-random-f32 (* fan-in out-ch)))
       (dx    (make-f32vector (* N C H W) 0.0))
       (x-shape (vector N C H W))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-bwd-data-blas dx x-shape g g-shape wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv3 bwd-data [~Ax~Ax~Ax~A]: ~A ms/iter\n" N C H W (ms/iter ms iters)))

(printf "\n-- execute-conv-bwd-weights-blas micro-benchmarks (~A iters each) --\n" 50)

(let* ((N 32) (C 1) (H 28) (W 28) (KH 3) (KW 3) (SH 1) (SW 1) (PH 1) (PW 1)
       (OH 28) (OW 28) (out-ch 16) (fan-in (* C KH KW)) (M (* N OH OW)) (iters 50)
       (g     (make-random-f32 (* M out-ch)))
       (g-shape (vector M out-ch))
       (src   (make-random-f32 (* N C H W)))
       (dwt   (make-f32vector (* fan-in out-ch) 0.0))
       (wt-shape (vector fan-in out-ch))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-bwd-weights-blas dwt wt-shape g g-shape src N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv1 bwd-wt [fan_in=~A, out_ch=~A]: ~A ms/iter\n" fan-in out-ch (ms/iter ms iters)))

(let* ((N 32) (C 16) (H 28) (W 28) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 14) (OW 14) (out-ch 32) (fan-in (* C KH KW)) (M (* N OH OW)) (iters 50)
       (g     (make-random-f32 (* M out-ch)))
       (g-shape (vector M out-ch))
       (src   (make-random-f32 (* N C H W)))
       (dwt   (make-f32vector (* fan-in out-ch) 0.0))
       (wt-shape (vector fan-in out-ch))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-bwd-weights-blas dwt wt-shape g g-shape src N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv2 bwd-wt [fan_in=~A, out_ch=~A]: ~A ms/iter\n" fan-in out-ch (ms/iter ms iters)))

(let* ((N 32) (C 32) (H 14) (W 14) (KH 3) (KW 3) (SH 2) (SW 2) (PH 1) (PW 1)
       (OH 7) (OW 7) (out-ch 64) (fan-in (* C KH KW)) (M (* N OH OW)) (iters 50)
       (g     (make-random-f32 (* M out-ch)))
       (g-shape (vector M out-ch))
       (src   (make-random-f32 (* N C H W)))
       (dwt   (make-f32vector (* fan-in out-ch) 0.0))
       (wt-shape (vector fan-in out-ch))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (execute-conv-bwd-weights-blas dwt wt-shape g g-shape src N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv3 bwd-wt [fan_in=~A, out_ch=~A]: ~A ms/iter\n" fan-in out-ch (ms/iter ms iters)))

(printf "\n-- BLAS matmul micro-benchmarks (~A iters each) --\n" 50)

;; Matmul sizes for each conv (using realize to force execution)
(let* ((M (* 32 28 28)) (K (* 1 9)) (Nc 16) (iters 50)
       (a (am:make-var (f32-morph (make-random-f32 (* M K)) (list M K)) #f))
       (b (am:make-var (f32-morph (make-random-f32 (* K Nc)) (list K Nc)) #f))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (realize (am:var-value (am:var-matmul a b))))))))
  (printf "  conv1 matmul [~Ax~A]@[~Ax~A]: ~A ms/iter\n" M K K Nc (ms/iter ms iters)))

(let* ((M (* 32 14 14)) (K (* 16 9)) (Nc 32) (iters 50)
       (a (am:make-var (f32-morph (make-random-f32 (* M K)) (list M K)) #f))
       (b (am:make-var (f32-morph (make-random-f32 (* K Nc)) (list K Nc)) #f))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (realize (am:var-value (am:var-matmul a b))))))))
  (printf "  conv2 matmul [~Ax~A]@[~Ax~A]: ~A ms/iter\n" M K K Nc (ms/iter ms iters)))

(let* ((M (* 32 7 7)) (K (* 32 9)) (Nc 64) (iters 50)
       (a (am:make-var (f32-morph (make-random-f32 (* M K)) (list M K)) #f))
       (b (am:make-var (f32-morph (make-random-f32 (* K Nc)) (list K Nc)) #f))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (realize (am:var-value (am:var-matmul a b))))))))
  (printf "  conv3 matmul [~Ax~A]@[~Ax~A]: ~A ms/iter\n" M K K Nc (ms/iter ms iters)))

(let* ((M 32) (K (* 64 7 7)) (Nc 128) (iters 50)
       (a (am:make-var (f32-morph (make-random-f32 (* M K)) (list M K)) #f))
       (b (am:make-var (f32-morph (make-random-f32 (* K Nc)) (list K Nc)) #f))
       (ms (time-ms (lambda ()
                      (do ((i 0 (+ i 1))) ((= i iters))
                        (realize (am:var-value (am:var-matmul a b))))))))
  (printf "  dense1 matmul [~Ax~A]@[~Ax~A]: ~A ms/iter\n" M K K Nc (ms/iter ms iters)))

;;; ---------------------------------------------------------------
;;; 5. SSA context stats
;;; ---------------------------------------------------------------

(printf "\n-- SSA context stats --\n")
(let* ((s (context-stats ctx))
       (na (cdr (assq 'allocations s)))
       (nb (cdr (assq 'buffers s))))
  (printf "  allocations: ~A,  pool buffers: ~A,  reuse: ~A%\n"
          na nb (round (* 100 (- 1 (/ nb na))))))

(printf "\n-- Replay plan instruction counts --\n")
(let ((stats (am-replay-plan-stats ctx)))
  (if (null? stats)
      (printf "  (no replay plan compiled yet)\n")
      (let* ((counts       (cdr (assq 'counts stats)))
             (index-shapes (cdr (assq 'ri-index-shapes stats)))
             (reduce-specs (cdr (assq 'ri-reduce-specs stats))))
        (for-each (lambda (pair)
                    (printf "  ~A: ~A\n" (car pair) (cdr pair)))
                  (sort counts (lambda (a b) (> (cdr a) (cdr b)))))
        (when (pair? index-shapes)
          (printf "\n  ri-index shapes (index . shape):\n")
          (for-each (lambda (entry)
                      (printf "    [~A] ~A\n" (car entry) (cdr entry)))
                    index-shapes))
        (when (pair? reduce-specs)
          (printf "\n  ri-reduce specs (index axes keepdims? shape):\n")
          (for-each (lambda (spec)
                      (printf "    [~A] axes=~A keepdims?=~A shape=~A\n"
                              (car spec) (cadr spec) (caddr spec) (cadddr spec)))
                    reduce-specs)))))

;;; ---------------------------------------------------------------
;;; 6. Per-instruction type timing (compiled, 10 steps)
;;; ---------------------------------------------------------------

(printf "\n-- Per-instruction timing (10 steps) --\n")
(replay-timing-reset!)
(do ((i 0 (+ i 1))) ((= i 10)) (do-step))
(let* ((timing (replay-timing-results))
       (steps  (cdr (assq 'steps timing)))
       (tags   (cdr (assq 'per-tag timing))))
  (printf "  steps: ~A\n" steps)
  (for-each (lambda (pair)
              (printf "  ~A: ~A ms total, ~A ms/step\n"
                      (car pair)
                      (cdr pair)
                      (exact->inexact (/ (cdr pair) steps))))
            (sort tags (lambda (a b) (> (cdr a) (cdr b))))))

;;; ---------------------------------------------------------------
;;; 7. Phase timing breakdown
;;; ---------------------------------------------------------------

(printf "\n-- Phase timing breakdown (5 warm steps) --\n")

;; Run 5 more steps to get stable averages
(do ((i 0 (+ i 1))) ((= i 5)) (do-step))

;; Time just the optimizer step
(let ((opt-times '()))
  (do ((i 0 (+ i 1))) ((= i 5))
    (set! opt-times (cons (time-ms (lambda () (step! optimizer))) opt-times)))
  (let* ((total (fold + 0.0 opt-times))
         (avg   (/ total 5)))
    (printf "  optimizer step avg: ~A ms\n" avg)))

;; Total step time from the earlier benchmark
(let ((step-times '()))
  (do ((i 0 (+ i 1))) ((= i 5))
    (set! step-times (cons (time-ms do-step) step-times)))
  (let* ((total (fold + 0.0 step-times))
         (avg   (/ total 5)))
    (printf "  full step avg:      ~A ms\n" avg)))

(printf "\nDone.\n")

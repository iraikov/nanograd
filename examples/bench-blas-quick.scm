;;; bench-blas-quick.scm -- Quick BLAS conv micro-bench + SSA per-step timing.
;;; Skips the slow scalar micro-benchmarks from bench-cnn.scm.

(import scheme (chicken base) (chicken format) (chicken random) (chicken time) (chicken sort))
(import (only srfi-1 fold iota))
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

(define (make-random-f32 n)
  (let ((v (make-f32vector n 0.0)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (f32vector-set! v i (- (* 2.0 (pseudo-random-real)) 1.0)))))
(define (f32-morph data shape)
  (make-morphism data shape 'f32))
(define (time-ms thunk)
  (let ((t0 (cpu-time))) (thunk) (- (cpu-time) t0)))

(set-pseudo-random-seed! "42")
(define batch-size 32)
(define image-size 28)

(define model
  (make-am-sequential
   (list
    (make-am-conv2d-layer 1  16 3 stride: 1 padding: 1 activation: (make-relu) dtype: 'f32)
    (make-am-conv2d-layer 16 32 3 stride: 2 padding: 1 activation: (make-relu) dtype: 'f32)
    (make-am-conv2d-layer 32 64 3 stride: 2 padding: 1 activation: (make-relu) dtype: 'f32)
    (make-am-flatten name: "Flatten")
    (make-am-dense-layer (* 64 7 7) 128 activation: (make-relu) dtype: 'f32)
    (make-am-dense-layer 128 4 activation: (make-identity) dtype: 'f32))))

(define input-data  (make-random-f32 (* batch-size 1 image-size image-size)))
(define target-data (make-f32vector (* batch-size 4) 0.0))
(do ((i 0 (+ i 1))) ((= i batch-size))
  (f32vector-set! target-data (+ (* i 4) (modulo i 4)) 1.0))

(define optimizer
  (make-adam (am-parameters model) learning-rate: 0.001 weight-decay: 0.0001))
(define ctx (make-morphism-context))

(define (do-step)
  (let* ((x-lt  (let ((mv (am:make-var (f32-morph input-data  (list batch-size 1 image-size image-size)) #f))) (get-or-make-lazy mv)))
         (t-lt  (let ((mv (am:make-var (f32-morph target-data (list batch-size 4)) #f))) (get-or-make-lazy mv)))
         (loss-fn (lambda (logits-lt) (am-cross-entropy-loss logits-lt t-lt))))
    (am-training-step/ssa ctx optimizer model loss-fn x-lt t-lt)))

;; Warm-up: compile the SSA plan
(printf "Warming up (compiling SSA plan)...\n")
(do-step)
(printf "Done.\n\n")

;; ---------------------------------------------------------------
;; BLAS forward micro-benchmarks (50 iters each)
;; ---------------------------------------------------------------
(printf "-- execute-conv-fwd-blas (im2col-mr + BLAS + bias) --\n")
(let* ((N 32)(C 1)(H 28)(W 28)(KH 3)(KW 3)(SH 1)(SW 1)(PH 1)(PW 1)
       (OH 28)(OW 28)(out-ch 16)(fan-in (* C KH KW))(M (* N OH OW))(iters 50)
       (src (make-random-f32 (* N C H W)))
       (wt  (make-random-f32 (* fan-in out-ch)))
       (b   (make-f32vector out-ch 0.0))
       (out (make-f32vector (* M out-ch) 0.0))
       (ms  (time-ms (lambda () (do ((i 0 (+ i 1))) ((= i iters))
                                  (execute-conv-fwd-blas out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv1 [M=~A, out_ch=~A]: ~A ms/iter\n" M out-ch (ms/iter ms iters)))

(let* ((N 32)(C 16)(H 28)(W 28)(KH 3)(KW 3)(SH 2)(SW 2)(PH 1)(PW 1)
       (OH 14)(OW 14)(out-ch 32)(fan-in (* C KH KW))(M (* N OH OW))(iters 50)
       (src (make-random-f32 (* N C H W)))
       (wt  (make-random-f32 (* fan-in out-ch)))
       (b   (make-f32vector out-ch 0.0))
       (out (make-f32vector (* M out-ch) 0.0))
       (ms  (time-ms (lambda () (do ((i 0 (+ i 1))) ((= i iters))
                                  (execute-conv-fwd-blas out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv2 [M=~A, out_ch=~A]: ~A ms/iter\n" M out-ch (ms/iter ms iters)))

(let* ((N 32)(C 32)(H 14)(W 14)(KH 3)(KW 3)(SH 2)(SW 2)(PH 1)(PW 1)
       (OH 7)(OW 7)(out-ch 64)(fan-in (* C KH KW))(M (* N OH OW))(iters 50)
       (src (make-random-f32 (* N C H W)))
       (wt  (make-random-f32 (* fan-in out-ch)))
       (b   (make-f32vector out-ch 0.0))
       (out (make-f32vector (* M out-ch) 0.0))
       (ms  (time-ms (lambda () (do ((i 0 (+ i 1))) ((= i iters))
                                  (execute-conv-fwd-blas out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
  (printf "  conv3 [M=~A, out_ch=~A]: ~A ms/iter\n" M out-ch (ms/iter ms iters)))

;; ---------------------------------------------------------------
;; Per-instruction SSA timing (10 steps)
;; ---------------------------------------------------------------
(printf "\n-- Per-instruction timing (10 steps) --\n")
(replay-timing-reset!)
(do ((i 0 (+ i 1))) ((= i 10)) (do-step))
(let* ((timing (replay-timing-results))
       (steps  (cdr (assq 'steps timing)))
       (tags   (cdr (assq 'per-tag timing))))
  (printf "  steps: ~A\n" steps)
  (for-each (lambda (pair)
              (printf "  ~A: ~A ms/step\n"
                      (car pair) (exact->inexact (/ (cdr pair) steps))))
            (sort tags (lambda (a b) (> (cdr a) (cdr b))))))

;; ---------------------------------------------------------------
;; Full step timing (5 warm steps)
;; ---------------------------------------------------------------
(printf "\n-- Full step timing (5 steps) --\n")
(let ((step-times '()))
  (do ((i 0 (+ i 1))) ((= i 5))
    (set! step-times (cons (time-ms do-step) step-times)))
  (let* ((total (fold + 0.0 step-times))
         (avg   (/ total 5)))
    (printf "  full step avg: ~A ms\n" avg)))

(printf "\nDone.\n")

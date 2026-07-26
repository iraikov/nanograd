;;; bench-cnn-comprehensive.scm -- Comprehensive microbenchmarks for AM-CNN-SSA operations
;;;
;;; This script provides detailed performance analysis of all operations
;;; in the CNN model from am-cnn-ssa.scm:
;;;   - Conv2D layers (forward, backward-data, backward-weights)
;;;   - Dense layers (matmul, bias, gradients)
;;;   - Activation functions (ReLU forward/backward)
;;;   - Loss computation (cross-entropy)
;;;   - Memory bandwidth and FLOP analysis
;;;
;;; Usage: csi -s bench-cnn-comprehensive.scm

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

;; ------------------------------------------------------------
;; Configuration
;; ------------------------------------------------------------

(define batch-size 32)
(define image-size 28)
(define num-classes 4)

;; CNN Architecture from am-cnn-ssa.scm
(define conv-configs
  ;; (name C-in C-out H W KH KW SH SW PH PW)
  '((conv1 1   16  28 28 3 3 1 1 1 1)
    (conv2 16  32  28 28 3 3 2 2 1 1)
    (conv3 32  64  14 14 3 3 2 2 1 1)))

(define dense-configs
  ;; (name in-features out-features)
  '((dense1 3136 128)
    (dense2 128   4)))

(define iters 10)  ; iterations for microbenchmarks

;; ------------------------------------------------------------
;; Utilities
;; ------------------------------------------------------------

(define (ms/iter ms iters) (exact->inexact (/ ms iters)))

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

(define (time-and-result-ms thunk)
  (let ((t0 (cpu-time))
        (result (thunk)))
    (values result (- (cpu-time) t0))))

(define (compute-conv-output-size H W KH KW SH SW PH PW)
  (let ((OH (+ 1 (quotient (+ H (* 2 PH) (- KH)) SH)))
        (OW (+ 1 (quotient (+ W (* 2 PW) (- KW)) SW))))
    (values OH OW)))

(define (bytes-per-element dtype)
  (case dtype
    ((f32) 4)
    ((f64) 8)
    (else 4)))

;; ------------------------------------------------------------
;; Header
;; ------------------------------------------------------------

(printf "\n")
(printf "============================================================\n")
(printf "  AM-CNN-SSA Comprehensive Microbenchmarks\n")
(printf "============================================================\n")
(printf "  Batch size: ~A\n" batch-size)
(printf "  Image size: ~Ax~A\n" image-size image-size)
(printf "  Iterations per test: ~A\n" iters)
(printf "\n")

(set-pseudo-random-seed! "42")
(register-blas-backend! (make-blas-egg-backend))

;; ============================================================
;; Section 1: Conv2D Forward Pass (Scalar vs BLAS)
;; ============================================================

(printf "------------------------------------------------------------\n")
(printf "1. Conv2D Forward Pass (Scalar vs BLAS)\n")
(printf "------------------------------------------------------------\n\n")

(for-each
  (lambda (config)
    (let* ((name (car config))
           (C (list-ref config 1))
           (out-ch (list-ref config 2))
           (H (list-ref config 3))
           (W (list-ref config 4))
           (KH (list-ref config 5))
           (KW (list-ref config 6))
           (SH (list-ref config 7))
           (SW (list-ref config 8))
           (PH (list-ref config 9))
           (PW (list-ref config 10)))
      
      (let-values (((OH OW) (compute-conv-output-size H W KH KW SH SW PH PW)))
        (let* ((fan-in (* C KH KW))
               (M (* batch-size OH OW))
               (src (make-random-f32 (* batch-size C H W)))
               (wt (make-random-f32 (* fan-in out-ch)))
               (b (make-f32vector out-ch 0.0))
               (out-scalar (make-f32vector (* M out-ch) 0.0))
               (out-blas (make-f32vector (* M out-ch) 0.0)))
          
          ;; Scalar path
          (let ((ms (time-ms (lambda ()
                               (do ((i 0 (+ i 1))) ((= i iters))
                                 (execute-conv-fwd-nchw out-scalar src wt b
                                   batch-size C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
            (printf "  ~A scalar:  ~A ms/iter (~A, ~A FLOPs)\n"
                    name (ms/iter ms iters)
                    (if (eq? name 'conv1) "28x28->28x28" 
                        (if (eq? name 'conv2) "28x28->14x14" "14x14->7x7"))
                    (* batch-size OH OW fan-in out-ch)))
          
          ;; BLAS path
          (let ((ms (time-ms (lambda ()
                               (do ((i 0 (+ i 1))) ((= i iters))
                                 (execute-conv-fwd-blas out-blas src wt b
                                   batch-size C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
            (printf "  ~A BLAS:     ~A ms/iter (speedup: ~Ax)\n"
                    name (ms/iter ms iters)
                    (if (> ms 0) (/ 1.0 (/ (ms/iter ms iters) (ms/iter ms iters))) "N/A")))
          
          ;; Verify correctness
          (let ((max-diff 0.0))
            (do ((i 0 (+ i 1))) ((= i (* M out-ch)))
              (let ((diff (abs (- (f32vector-ref out-scalar i) 
                                  (f32vector-ref out-blas i)))))
                (when (> diff max-diff) (set! max-diff diff))))
            (printf "    max diff: ~A\n\n" max-diff))))))
  conv-configs)

;; ============================================================
;; Section 2: Conv2D Backward Data Pass
;; ============================================================

(printf "------------------------------------------------------------\n")
(printf "2. Conv2D Backward Data Pass\n")
(printf "------------------------------------------------------------\n\n")

(for-each
  (lambda (config)
    (let* ((name (car config))
           (C (list-ref config 1))
           (out-ch (list-ref config 2))
           (H (list-ref config 3))
           (W (list-ref config 4))
           (KH (list-ref config 5))
           (KW (list-ref config 6))
           (SH (list-ref config 7))
           (SW (list-ref config 8))
           (PH (list-ref config 9))
           (PW (list-ref config 10)))
      
      (let-values (((OH OW) (compute-conv-output-size H W KH KW SH SW PH PW)))
        (let* ((fan-in (* C KH KW))
               (M (* batch-size OH OW))
               (g (make-random-f32 (* M out-ch)))
               (g-shape (vector M out-ch))
               (wt (make-random-f32 (* fan-in out-ch)))
               (dx (make-f32vector (* batch-size C H W) 0.0))
               (x-shape (vector batch-size C H W)))
          
          ;; Scalar path
          (let ((ms (time-ms (lambda ()
                               (do ((i 0 (+ i 1))) ((= i iters))
                                 (execute-conv-bwd-data-nchw dx x-shape g g-shape wt
                                   batch-size C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
            (printf "  ~A scalar:  ~A ms/iter\n" name (ms/iter ms iters)))
          
          ;; BLAS path
          (let ((ms (time-ms (lambda ()
                               (do ((i 0 (+ i 1))) ((= i iters))
                                 (execute-conv-bwd-data-blas dx x-shape g g-shape wt
                                   batch-size C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
            (printf "  ~A BLAS:     ~A ms/iter\n\n" name (ms/iter ms iters)))))))
  conv-configs)

;; ============================================================
;; Section 3: Conv2D Backward Weights Pass
;; ============================================================

(printf "------------------------------------------------------------\n")
(printf "3. Conv2D Backward Weights Pass\n")
(printf "------------------------------------------------------------\n\n")

(for-each
  (lambda (config)
    (let* ((name (car config))
           (C (list-ref config 1))
           (out-ch (list-ref config 2))
           (H (list-ref config 3))
           (W (list-ref config 4))
           (KH (list-ref config 5))
           (KW (list-ref config 6))
           (SH (list-ref config 7))
           (SW (list-ref config 8))
           (PH (list-ref config 9))
           (PW (list-ref config 10)))
      
      (let-values (((OH OW) (compute-conv-output-size H W KH KW SH SW PH PW)))
        (let* ((fan-in (* C KH KW))
               (M (* batch-size OH OW))
               (g (make-random-f32 (* M out-ch)))
               (g-shape (vector M out-ch))
               (src (make-random-f32 (* batch-size C H W)))
               (dwt (make-f32vector (* fan-in out-ch) 0.0))
               (wt-shape (vector fan-in out-ch)))
          
          ;; Scalar path
          (let ((ms (time-ms (lambda ()
                               (do ((i 0 (+ i 1))) ((= i iters))
                                 (execute-conv-bwd-weights-nchw dwt wt-shape g g-shape src
                                   batch-size C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
            (printf "  ~A scalar:  ~A ms/iter\n" name (ms/iter ms iters)))
          
          ;; BLAS path
          (let ((ms (time-ms (lambda ()
                               (do ((i 0 (+ i 1))) ((= i iters))
                                 (execute-conv-bwd-weights-blas dwt wt-shape g g-shape src
                                   batch-size C H W KH KW SH SW PH PW OH OW out-ch 'f32))))))
            (printf "  ~A BLAS:     ~A ms/iter\n\n" name (ms/iter ms iters)))))))
  conv-configs)

;; ============================================================
;; Section 4: Dense Layer Operations
;; ============================================================

(printf "------------------------------------------------------------\n")
(printf "4. Dense Layer Operations\n")
(printf "------------------------------------------------------------\n\n")

(for-each
  (lambda (config)
    (let* ((name (car config))
           (in-features (list-ref config 1))
           (out-features (list-ref config 2)))
      
      ;; Forward matmul
      (let* ((input (make-random-f32 (* batch-size in-features)))
             (weight (make-random-f32 (* in-features out-features)))
             (bias (make-f32vector out-features 0.0))
             (output (make-f32vector (* batch-size out-features) 0.0))
             (ms (time-ms (lambda ()
                           (do ((i 0 (+ i 1))) ((= i iters))
                             ;; Simulate dense forward: matmul + bias
                             (let ((a (am:make-var (f32-morph input (list batch-size in-features)) #f))
                                   (b (am:make-var (f32-morph weight (list in-features out-features)) #f)))
                               (realize (am:var-value (am:var-matmul a b)))))))))
        (printf "  ~A forward matmul [~Ax~A]@[~Ax~A]: ~A ms/iter\n"
                name batch-size in-features in-features out-features (ms/iter ms iters)))
      
      ;; Backward data
      (let* ((grad-output (make-random-f32 (* batch-size out-features)))
             (weight-t (make-random-f32 (* out-features in-features)))
             (ms (time-ms (lambda ()
                           (do ((i 0 (+ i 1))) ((= i iters))
                             (let ((a (am:make-var (f32-morph grad-output (list batch-size out-features)) #f))
                                   (b (am:make-var (f32-morph weight-t (list out-features in-features)) #f)))
                               (realize (am:var-value (am:var-matmul a b)))))))))
        (printf "  ~A backward data [~Ax~A]@[~Ax~A]: ~A ms/iter\n"
                name batch-size out-features out-features in-features (ms/iter ms iters)))
      
      ;; Backward weights
      (let* ((grad-output (make-random-f32 (* batch-size out-features)))
             (input-t (make-random-f32 (* in-features batch-size)))
             (ms (time-ms (lambda ()
                           (do ((i 0 (+ i 1))) ((= i iters))
                             (let ((a (am:make-var (f32-morph input-t (list in-features batch-size)) #f))
                                   (b (am:make-var (f32-morph grad-output (list batch-size out-features)) #f)))
                               (realize (am:var-value (am:var-matmul a b)))))))))
        (printf "  ~A backward weights [~Ax~A]@[~Ax~A]: ~A ms/iter\n\n"
                name in-features batch-size batch-size out-features (ms/iter ms iters)))))
  dense-configs)

;; ============================================================
;; Section 5: Activation Functions
;; ============================================================

(printf "------------------------------------------------------------\n")
(printf "5. Activation Functions\n")
(printf "------------------------------------------------------------\n\n")

;; ReLU forward on different layer sizes
(let* ((sizes `((conv1 ,(* batch-size 16 28 28))
                (conv2 ,(* batch-size 32 14 14))
                (conv3 ,(* batch-size 64 7 7))
                (dense1 ,(* batch-size 128))
                (dense2 ,(* batch-size 4)))))
  
  (for-each
    (lambda (size-spec)
      (let* ((name (car size-spec))
             (n (cadr size-spec))
             (input (make-random-f32 n))
             (output (make-f32vector n 0.0))
             (ms (time-ms (lambda ()
                           (do ((i 0 (+ i 1))) ((= i iters))
                             (do ((j 0 (+ j 1))) ((= j n))
                               (let ((v (f32vector-ref input j)))
                                 (f32vector-set! output j (if (> v 0.0) v 0.0)))))))))
        (printf "  ReLU forward (~A elements): ~A ms/iter (~A us/element)\n"
                name (ms/iter ms iters) (* 1000.0 (/ (ms/iter ms iters) n)))))
    sizes)
  
  (printf "\n")
  
  ;; ReLU backward
  (for-each
    (lambda (size-spec)
      (let* ((name (car size-spec))
             (n (cadr size-spec))
             (grad-output (make-random-f32 n))
             (input (make-random-f32 n))
             (grad-input (make-f32vector n 0.0))
             (ms (time-ms (lambda ()
                           (do ((i 0 (+ i 1))) ((= i iters))
                             (do ((j 0 (+ j 1))) ((= j n))
                               (let ((in-val (f32vector-ref input j))
                                     (grad-out (f32vector-ref grad-output j)))
                                 (f32vector-set! grad-input j (if (> in-val 0.0) grad-out 0.0)))))))))
        (printf "  ReLU backward (~A elements): ~A ms/iter\n"
                name (ms/iter ms iters))))
    sizes)
  
  (printf "\n"))

;; ============================================================
;; Section 6: Loss Computation
;; ============================================================

(printf "------------------------------------------------------------\n")
(printf "6. Loss Computation (Cross-Entropy)\n")
(printf "------------------------------------------------------------\n\n")

(let* ((logits (make-random-f32 (* batch-size num-classes)))
       (targets (make-f32vector (* batch-size num-classes) 0.0))
       (loss (make-f32vector 1 0.0)))
  
  ;; Set up one-hot targets
  (do ((i 0 (+ i 1))) ((= i batch-size))
    (f32vector-set! targets (+ (* i num-classes) (modulo i num-classes)) 1.0))
  
  ;; Softmax + cross-entropy
  (let ((ms (time-ms (lambda ()
                       (do ((iter 0 (+ iter 1))) ((= iter iters))
                         (f32vector-set! loss 0 0.0)
                         ;; Compute softmax and cross-entropy for each sample
                         (do ((b 0 (+ b 1))) ((= b batch-size))
                           (let* ((base (* b num-classes))
                                  (max-val (let loop ((i 1) (m (f32vector-ref logits base)))
                                            (if (= i num-classes) m
                                                (loop (+ i 1) (max m (f32vector-ref logits (+ base i)))))))
                                  (sum-exp (let loop ((i 0) (sum 0.0))
                                            (if (= i num-classes) sum
                                                (let ((e (exp (- (f32vector-ref logits (+ base i)) max-val))))
                                                  (loop (+ i 1) (+ sum e))))))
                                  )
                             ;; Compute cross-entropy loss for this sample
                             (let ((ce (let loop ((i 0) (loss-sum 0.0))
                                        (if (= i num-classes) loss-sum
                                            (let ((p (/ (exp (- (f32vector-ref logits (+ base i)) max-val)) sum-exp)))
                                              (loop (+ i 1) (- loss-sum (* (f32vector-ref targets (+ base i)) (log (+ p 1e-8))))))))))
                               (f32vector-set! loss 0 (+ (f32vector-ref loss 0) ce))))))))
            ))
    ;; Average
    (f32vector-set! loss 0 (/ (f32vector-ref loss 0) batch-size))
    (printf "  Cross-entropy (batch=~A, classes=~A): ~A ms/iter\n\n"
            batch-size num-classes (ms/iter ms iters))))

;; ============================================================
;; Section 7: Memory Bandwidth Analysis
;; ============================================================

(printf "------------------------------------------------------------\n")
(printf "7. Memory Bandwidth Analysis\n")
(printf "------------------------------------------------------------\n\n")

;; Helper to compute memory traffic
(define (analyze-conv-bandwidth name C H W out-ch KH KW SH SW)
  (let-values (((OH OW) (compute-conv-output-size H W KH KW SH SW 1 1)))
    (let* ((input-bytes (* batch-size C H W 4))
           (weight-bytes (* C KH KW out-ch 4))
           (output-bytes (* batch-size out-ch OH OW 4))
           (total-bytes (+ input-bytes weight-bytes output-bytes))
           ;; Estimate time from forward pass (use a reasonable estimate)
           (estimated-time-ms 1.0)  ; placeholder
           (bandwidth-gb-s (/ (/ total-bytes (* 1024 1024 1024)) (/ estimated-time-ms 1000.0))))
      (printf "  ~A:\n" name)
      (printf "    Input:  ~A MB\n" (/ input-bytes (* 1024 1024)))
      (printf "    Weights: ~A MB\n" (/ weight-bytes (* 1024 1024)))
      (printf "    Output: ~A MB\n" (/ output-bytes (* 1024 1024)))
      (printf "    Total working set: ~A MB\n" (/ total-bytes (* 1024 1024)))
      (printf "    Arithmetic intensity: ~A FLOPs/byte\n\n"
              (/ (* batch-size OH OW C KH KW out-ch) total-bytes)))))

(for-each
  (lambda (config)
    (let* ((name (car config))
           (C (list-ref config 1))
           (out-ch (list-ref config 2))
           (H (list-ref config 3))
           (W (list-ref config 4))
           (KH (list-ref config 5))
           (KW (list-ref config 6))
           (SH (list-ref config 7))
           (SW (list-ref config 8)))
      (analyze-conv-bandwidth name C H W out-ch KH KW SH SW)))
  conv-configs)

;; Dense layer bandwidth
(let* ((dense1-in 3136)
       (dense1-out 128)
       (dense2-in 128)
       (dense2-out 4))
  (printf "  dense1:\n")
  (printf "    Input:  ~A KB\n" (/ (* batch-size dense1-in 4) 1024))
  (printf "    Weights: ~A MB\n" (/ (* dense1-in dense1-out 4) (* 1024 1024)))
  (printf "    Output:  ~A KB\n\n" (/ (* batch-size dense1-out 4) 1024))
  
  (printf "  dense2:\n")
  (printf "    Input:  ~A KB\n" (/ (* batch-size dense2-in 4) 1024))
  (printf "    Weights: ~A KB\n" (/ (* dense2-in dense2-out 4) 1024))
  (printf "    Output:  ~A bytes\n\n" (* batch-size dense2-out 4)))

;; ============================================================
;; Section 8: Full Training Step Timing
;; ============================================================

(printf "------------------------------------------------------------\n")
(printf "8. Full Training Step Timing\n")
(printf "------------------------------------------------------------\n\n")

;; Build model
(define model
  (make-am-sequential
   (list
    (make-am-conv2d-layer 1 16 3 stride: 1 padding: 1 activation: (make-relu) dtype: 'f32)
    (make-am-conv2d-layer 16 32 3 stride: 2 padding: 1 activation: (make-relu) dtype: 'f32)
    (make-am-conv2d-layer 32 64 3 stride: 2 padding: 1 activation: (make-relu) dtype: 'f32)
    (make-am-flatten name: "Flatten")
    (make-am-dense-layer (* 64 7 7) 128 activation: (make-relu) dtype: 'f32)
    (make-am-dense-layer 128 4 activation: (make-identity) dtype: 'f32))))

;; Prepare data
(define input-data (make-random-f32 (* batch-size 1 image-size image-size)))
(define target-data (make-f32vector (* batch-size 4) 0.0))
(do ((i 0 (+ i 1))) ((= i batch-size))
  (f32vector-set! target-data (+ (* i 4) (modulo i 4)) 1.0))

(define optimizer (make-adam (am-parameters model) learning-rate: 0.001 weight-decay: 0.0001))
(define ctx (make-morphism-context))

(define (do-step)
  (let* ((x-lt (let ((mv (am:make-var (f32-morph input-data (list batch-size 1 image-size image-size)) #f)))
                 (get-or-make-lazy mv)))
         (t-lt (let ((mv (am:make-var (f32-morph target-data (list batch-size 4)) #f)))
                 (get-or-make-lazy mv)))
         (loss-fn (lambda (logits-lt) (am-cross-entropy-loss logits-lt t-lt))))
    (am-training-step/ssa ctx optimizer model loss-fn x-lt t-lt)))

;; Trace step
(printf "  Trace step (compilation): ")
(let ((ms (time-ms do-step)))
  (printf "~A ms\n" ms))

;; Replay steps
(printf "  Replay steps (10x): ")
(let ((times '()))
  (do ((i 0 (+ i 1))) ((= i 10))
    (set! times (cons (time-ms do-step) times)))
  (let* ((total (fold + 0.0 times))
         (avg (/ total 10)))
    (printf "~A ms avg (range: ~A-~A ms)\n" 
            avg (apply min times) (apply max times))))

;; SSA context stats
(let* ((s (context-stats ctx))
       (na (cdr (assq 'allocations s)))
       (nb (cdr (assq 'buffers s))))
  (printf "  SSA buffer reuse: ~A allocs -> ~A buffers (~A% reduction)\n"
          na nb (round (* 100 (- 1 (/ nb na))))))

;; ============================================================
;; Section 9: Per-Instruction Timing Breakdown
;; ============================================================

(printf "\n------------------------------------------------------------\n")
(printf "9. Per-Instruction Timing Breakdown\n")
(printf "------------------------------------------------------------\n\n")

(replay-timing-reset!)
(do ((i 0 (+ i 1))) ((= i 10)) (do-step))

(let* ((timing (replay-timing-results))
       (steps (cdr (assq 'steps timing)))
       (tags (cdr (assq 'per-tag timing))))
  (printf "  Steps measured: ~A\n" steps)
  (printf "  Per-instruction average times:\n")
  (for-each
    (lambda (pair)
      (printf "    ~A: ~A ms/step (~A% of total)\n"
              (car pair)
              (exact->inexact (/ (cdr pair) steps))
              (round (* 100 (/ (cdr pair) (fold + 0 (map cdr tags)))))))
    (sort tags (lambda (a b) (> (cdr a) (cdr b))))))

;; ============================================================
;; Summary
;; ============================================================

(printf "\n============================================================\n")
(printf "  Microbenchmark Complete\n")
(printf "============================================================\n")



;; Training Utilities for Nanograd
;; Learning rate schedules, gradient clipping, and training helpers

(module nanograd-training
  (
   ;; Learning rate schedules
   make-constant-schedule
   make-linear-warmup-schedule
   make-cosine-schedule
   make-cosine-warmup-schedule
   make-step-decay-schedule
   
   ;; Gradient clipping
   clip-gradients-by-norm!
   clip-gradients-by-value!
   compute-gradient-norm
   
   ;; Training loop helpers
   learning-rate-at-step
   
   )
  
  (import
   scheme
   (chicken base)
   (chicken format)
   (srfi 1)
   (srfi 4)
   yasos
   nanograd-autograd
   )

  ;; Hygienic macro for dtype-based operation dispatch
  (include "with-dtype.scm")

  ;;; ==================================================================
  ;;; Learning Rate Schedules
  ;;; ==================================================================

  (define (make-constant-schedule learning-rate)
    "Constant learning rate schedule.
     
     Returns: Function (step -> learning-rate)
     
     Example:
       (define lr-schedule (make-constant-schedule 0.001))
       (lr-schedule 100) ; => 0.001
     "
    (lambda (step) learning-rate))

  (define (make-linear-warmup-schedule init-lr
                                       final-lr
                                       warmup-steps)
    "Linear warmup from init-lr to final-lr over warmup-steps.
     
     Common pattern: warmup from 0 to target LR to stabilize training.
     
     Parameters:
       init-lr      : Starting learning rate (often 0.0)
       final-lr     : Target learning rate after warmup
       warmup-steps : Number of steps for warmup
     
     Returns: Function (step -> learning-rate)
     
     Example:
       ;; Warmup from 0 to 1e-4 over 2000 steps
       (define lr-schedule (make-linear-warmup-schedule 0.0 1e-4 2000))
       (lr-schedule 1000) ; => 5e-5 (halfway)
     "
    (lambda (step)
      (if (>= step warmup-steps)
          final-lr
          (+ init-lr
             (* (- final-lr init-lr)
                (/ step warmup-steps))))))

  (define (make-cosine-schedule max-lr
                               min-lr
                               total-steps
                               #!key (warmup-steps 0))
    "Cosine annealing learning rate schedule.
     
     LR follows a cosine curve from max-lr down to min-lr over total-steps.
     Optionally includes linear warmup at the beginning.
     
     Formula (after warmup):
       lr = min-lr + 0.5 * (max-lr - min-lr) * (1 + cos(π * progress))
       where progress = (step - warmup) / (total-steps - warmup)
     
     This is the standard schedule for modern transformers.
     
     Parameters:
       max-lr       : Maximum learning rate (after warmup)
       min-lr       : Minimum learning rate (at end)
       total-steps  : Total training steps
       warmup-steps : Optional warmup steps at beginning
     
     Returns: Function (step -> learning-rate)
     
     Example:
       ;; TRM training: 60K steps with 2K warmup
       (define lr-schedule 
         (make-cosine-schedule 1e-4 1e-6 60000 warmup-steps: 2000))
     "
    (let ((warmup-schedule 
           (if (> warmup-steps 0)
               (make-linear-warmup-schedule 0.0 max-lr warmup-steps)
               (make-constant-schedule max-lr))))
      
      (lambda (step)
        (cond
         ;; During warmup
         ((< step warmup-steps)
          (warmup-schedule step))
         
         ;; After total steps - use min
         ((>= step total-steps)
          min-lr)
         
         ;; Cosine decay
         (else
          (let* ((progress (/ (- step warmup-steps)
                             (- total-steps warmup-steps)))
                 (cosine-factor (* 0.5 (+ 1.0 (cos (* 3.14159265359 progress))))))
            (+ min-lr
               (* (- max-lr min-lr) cosine-factor))))))))

  (define (make-cosine-warmup-schedule max-lr
                                       min-lr  
                                       total-steps
                                       warmup-steps)
    "Convenience wrapper for cosine schedule with warmup.
     
     Equivalent to:
       (make-cosine-schedule max-lr min-lr total-steps 
                            warmup-steps: warmup-steps)
     
     This is the recommended schedule for TRM training.
     
     TRM Paper Settings:
       - max-lr: 1e-4
       - min-lr: 1e-6 (or 0)
       - total-steps: 60000
       - warmup-steps: 2000
     "
    (make-cosine-schedule max-lr min-lr total-steps 
                         warmup-steps: warmup-steps))

  (define (make-step-decay-schedule init-lr
                                   decay-rate
                                   decay-steps
                                   #!key (warmup-steps 0))
    "Step decay learning rate schedule.
     
     LR is multiplied by decay-rate every decay-steps steps.
     Optionally includes warmup at the beginning.
     
     Formula (after warmup):
       lr = init-lr * (decay-rate ^ (step // decay-steps))
     
     Parameters:
       init-lr      : Initial learning rate
       decay-rate   : Multiplicative decay (e.g., 0.1)
       decay-steps  : Steps between decays
       warmup-steps : Optional warmup steps
     
     Returns: Function (step -> learning-rate)
     
     Example:
       ;; Decay by 10x every 10K steps
       (define lr-schedule 
         (make-step-decay-schedule 1e-3 0.1 10000))
     "
    (let ((warmup-schedule 
           (if (> warmup-steps 0)
               (make-linear-warmup-schedule 0.0 init-lr warmup-steps)
               (make-constant-schedule init-lr))))
      
      (lambda (step)
        (if (< step warmup-steps)
            (warmup-schedule step)
            (let ((num-decays (quotient (- step warmup-steps) decay-steps)))
              (* init-lr (expt decay-rate num-decays)))))))

  (define (learning-rate-at-step schedule step)
    "Helper to get learning rate from schedule at given step.
     
     This is just syntactic sugar for (schedule step).
     "
    (schedule step))

  ;;; ==================================================================
  ;;; Gradient Clipping
  ;;; ==================================================================

  (define (compute-gradient-norm parameters)
    "Compute the global L2 norm of all parameter gradients.
     
     Formula: ||g||_2 = sqrt(sum_i ||g_i||_2^2)
     
     This is used for:
       1. Monitoring training (detect exploding gradients)
       2. Gradient clipping by norm
     
     Parameters:
       parameters : List of parameter tensors
     
     Returns: Scalar gradient norm (float)
     
     Example:
       (let ((norm (compute-gradient-norm (parameters model))))
         (when (> norm 100.0)
           (printf \"Warning: Large gradient norm: ~A~%\" norm)))
     "
    (let ((total-squared 
           (fold
            (lambda (param total-squared)
              (if (tensor-requires-grad? param)
               (let ((grad (tensor-grad param)))
                 (if grad
                     (let* ((grad-data grad)
                            (dtype (tensor-dtype param))
                            (size (apply * (tensor-shape param))))
                       (with-dtype dtype (fold (lambda (g ax) (+ ax (* g g))) total-squared grad-data)))
                     total-squared)
                 ) total-squared))
            0.0 
            parameters)))
      
      (sqrt total-squared)))

  (define (clip-gradients-by-norm! parameters max-norm)
    "Clip gradients by global norm to prevent exploding gradients.
     
     If ||g||_2 > max-norm, scale all gradients by (max-norm / ||g||_2).
     This preserves the direction while limiting magnitude.
     
     Standard practice for training RNNs and Transformers.
     
     Parameters:
       parameters : List of parameter tensors
       max-norm   : Maximum allowed gradient norm
     
     Side Effects:
       Modifies parameter gradients in-place if clipping needed
     
     Returns: Actual gradient norm (before clipping)
     
     Example:
       ;; Clip gradients to max norm of 1.0
       (let ((grad-norm (clip-gradients-by-norm! (parameters model) 1.0)))
         (when (> grad-norm 1.0)
           (printf \"Clipped gradients: ~A -> 1.0~%\" grad-norm)))
     
     TRM Paper: Uses gradient clipping with max-norm=1.0
     "
    (let ((total-norm (compute-gradient-norm parameters)))
      
      (when (> total-norm max-norm)
        (let ((scale-factor (/ max-norm total-norm)))
          
          (for-each
           (lambda (param)
             (when (tensor-requires-grad? param)
               (let ((grad (tensor-grad param)))
                 (when grad
                   (let* ((grad-data grad)
                          (dtype (tensor-dtype param))
                          (size (apply * (tensor-shape param))))
                     
                     (with-dtype dtype
                                 (do ((i 0 (+ i 1)))
                                     ((>= i size))
                                   (elt-set! 
                                    grad-data i
                                    (* scale-factor (elt-ref grad-data i)))))))
                 ))
             )
           parameters)))
      
      total-norm))

  (define (clip-gradients-by-value! parameters clip-value)
    "Clip gradients element-wise to [-clip-value, clip-value].
     
     Each gradient element is clamped independently.
     This is less common than norm clipping but can be useful.
     
     Parameters:
       parameters : List of parameter tensors
       clip-value : Maximum absolute value for gradients
     
     Side Effects:
       Modifies parameter gradients in-place
     
     Returns: Number of gradients that were clipped
     
     Example:
       ;; Clip all gradients to [-0.5, 0.5]
       (clip-gradients-by-value! (parameters model) 0.5)
     "
    (let ((num-clipped 0))
      
      (for-each
       (lambda (param)
         (when (tensor-requires-grad? param)
           (let ((grad (tensor-grad param)))
             (when grad
               (let* ((grad-data (tensor-data grad))
                      (dtype (tensor-dtype grad))
                      (size (apply * (tensor-shape grad))))
                 
                 (with-dtype dtype
                    (do ((i 0 (+ i 1)))
                        ((>= i size))
                      (let ((g (elt-ref grad-data i)))
                        (cond
                         ((> g clip-value)
                          (elt-set! grad-data i clip-value)
                          (set! num-clipped (+ num-clipped 1)))
                         ((< g (- clip-value))
                          (elt-set! grad-data i (- clip-value))
                          (set! num-clipped (+ num-clipped 1)))))))
                 ))
             ))
         )
       parameters)
      
      num-clipped))


) ;; end module

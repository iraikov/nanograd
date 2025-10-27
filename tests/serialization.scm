;; Test file for layer serialization with dimension validation

(import scheme
        (chicken base)
        (chicken format)
        (chicken file)
        (chicken condition)
        (chicken random)
        (srfi 1)
        (srfi 4)
        nanograd-autograd
        nanograd-layer)

;;; ==================================================================
;;; Utility Functions
;;; ==================================================================

(define (vectors-close? vec1 vec2 tolerance)
  "Check if two f32vectors are close within tolerance"
  (let ((n1 (f32vector-length vec1))
        (n2 (f32vector-length vec2)))
    (and (= n1 n2)
         (let loop ((i 0) (max-diff 0.0))
           (if (= i n1)
               (<= max-diff tolerance)
               (let ((diff (abs (- (f32vector-ref vec1 i)
                                  (f32vector-ref vec2 i)))))
                 (loop (+ i 1) (max max-diff diff))))))))

(define (test-result name passed?)
  (printf "  ~A ~A\n" 
          (if passed? "O" "X")
          name))

(define test-counter 0)
(define test-passed 0)

(define (run-test name thunk)
  (set! test-counter (+ test-counter 1))
  (printf "\nTest ~A: ~A\n" test-counter name)
  (let ((result (thunk)))
    (when result (set! test-passed (+ test-passed 1)))
    result))

;;; ==================================================================
;;; Dense Layer Serialization
;;; ==================================================================

(define (test-dense-layer-basic)
  (run-test "Dense layer save/load with dimension checking"
    (lambda ()
      (printf "  Creating dense layer (3 → 2)...\n")
      
      ;; Create a dense layer
      (define layer (make-dense-layer 3 2 
                                      activation: (make-relu)
                                      dtype: 'f32
                                      name: "TestDense"))
      
      ;; Get original parameters
      (define original-params (parameters layer))
      (define original-weights (car original-params))
      (define original-biases (cadr original-params))
      
      (printf "    Input size: ~A\n" (layer-input-size layer))
      (printf "    Output size: ~A\n" (layer-output-size layer))
      
      ;; Save the layer
      (save-layer layer "test-dense-layer.s11n")
      (printf "  Layer saved to test-dense-layer.s11n\n")
      
      ;; Load the layer
      (define loaded-layer (load-layer "test-dense-layer.s11n"))
      (printf "  Layer loaded from test-dense-layer.s11n\n")
      
      ;; Verify loaded layer
      (define loaded-params (parameters loaded-layer))
      (define loaded-weights (car loaded-params))
      (define loaded-biases (cadr loaded-params))
      
      ;; Check metadata
      (define metadata-ok 
        (and (string=? (layer-name layer) (layer-name loaded-layer))
             (= (layer-input-size layer) (layer-input-size loaded-layer))
             (= (layer-output-size layer) (layer-output-size loaded-layer))
             (string=? (activation-name (layer-activation layer))
                      (activation-name (layer-activation loaded-layer)))))
      
      (test-result "Metadata preserved" metadata-ok)
      
      ;; Check if weights match
      (define weights-match 
        (vectors-close? (tensor-data original-weights) 
                       (tensor-data loaded-weights) 
                       1e-9))
      (test-result "Weights match" weights-match)
      
      ;; Check if biases match
      (define biases-match 
        (vectors-close? (tensor-data original-biases) 
                       (tensor-data loaded-biases) 
                       1e-9))
      (test-result "Biases match" biases-match)
      
      ;; Test forward pass
      (define test-input (make-tensor32 
                          (f32vector 1.0 2.0 3.0)
                          '(3)))
      
      (define original-output (forward layer test-input))
      (define loaded-output (forward loaded-layer test-input))
      
      (define forward-match 
        (vectors-close? (tensor-data original-output)
                       (tensor-data loaded-output) 
                       1e-6))
      (test-result "Forward pass matches" forward-match)
      
      ;; Cleanup
      (delete-file* "test-dense-layer.s11n")
      
      (and metadata-ok weights-match biases-match forward-match))))

;;; ==================================================================
;;; Sequential Model Serialization
;;; ==================================================================

(define (test-sequential-model)
  (run-test "Sequential model with multiple layers"
    (lambda ()
      (printf "  Creating sequential model (4 -> 8 -> 4 -> 2)...\n")
      
      ;; Create a sequential model
      (define model (make-sequential
                     (list
                      (make-dense-layer 4 8 activation: (make-relu) name: "Hidden1")
                      (make-dense-layer 8 4 activation: (make-relu) name: "Hidden2")
                      (make-dense-layer 4 2 activation: (make-identity) name: "Output"))
                     name: "TestNetwork"))
      
      (printf "    Input size: ~A\n" (layer-input-size model))
      (printf "    Output size: ~A\n" (layer-output-size model))
      (printf "    Total parameters: ~A\n" 
              (length (parameters model)))
      
      ;; Save the model
      (save-model model "test-model.s11n")
      (printf "  Model saved to test-model.s11n\n")
      
      ;; Load the model
      (define loaded-model (load-model "test-model.s11n"))
      (printf "  Model loaded from test-model.s11n\n")
      
      ;; Verify structure
      (define structure-ok
        (and (= (layer-input-size model) (layer-input-size loaded-model))
             (= (layer-output-size model) (layer-output-size loaded-model))
             (= (length (parameters model)) 
                (length (parameters loaded-model)))))
      
      (test-result "Model structure preserved" structure-ok)
      
      ;; Test forward pass
      (define test-input (make-tensor32 
                          (f32vector 1.0 2.0 3.0 4.0)
                          '(4)))
      
      (define original-output (forward model test-input))
      (define loaded-output (forward loaded-model test-input))
      
      (define forward-match
        (vectors-close? (tensor-data original-output)
                       (tensor-data loaded-output)
                       1e-6))
      
      (test-result "Sequential forward pass matches" forward-match)
      
      ;; Verify all parameters match
      (define params-match
        (let ((orig-params (parameters model))
              (load-params (parameters loaded-model)))
          (let loop ((op orig-params) (lp load-params) (all-match #t))
            (cond
             ((null? op) all-match)
             (else
              (let* ((orig-data (tensor-data (car op)))
                     (load-data (tensor-data (car lp)))
                     (match (vectors-close? orig-data load-data 1e-9)))
                (loop (cdr op) (cdr lp) (and all-match match))))))))
      
      (test-result "All parameters match" params-match)
      
      ;; Cleanup
      (delete-file* "test-model.s11n")
      
      (and structure-ok forward-match params-match))))

;;; ==================================================================
;;; Different Data Types
;;; ==================================================================

(define (test-different-dtypes)
  (run-test "Different data types (f32 vs f64)"
    (lambda ()
      (printf "  Testing f32 dtype...\n")
      (define f32-layer (make-dense-layer 2 3 
                                          dtype: 'f32
                                          name: "F32Dense"))
      
      (save-layer f32-layer "test-f32-layer.s11n")
      (define loaded-f32-layer (load-layer "test-f32-layer.s11n"))
      
      (define f32-ok
        (and (eq? (tensor-dtype (car (parameters f32-layer))) 'f32)
             (eq? (tensor-dtype (car (parameters loaded-f32-layer))) 'f32)))
      
      (test-result "F32 dtype preserved" f32-ok)
      
      (printf "  Testing f64 dtype...\n")
      (define f64-layer (make-dense-layer 2 3 
                                          dtype: 'f64
                                          name: "F64Dense"))
      
      (save-layer f64-layer "test-f64-layer.s11n")
      (define loaded-f64-layer (load-layer "test-f64-layer.s11n"))
      
      (define f64-ok
        (and (eq? (tensor-dtype (car (parameters f64-layer))) 'f64)
             (eq? (tensor-dtype (car (parameters loaded-f64-layer))) 'f64)))
      
      (test-result "F64 dtype preserved" f64-ok)
      
      ;; Cleanup
      (delete-file* "test-f32-layer.s11n")
      (delete-file* "test-f64-layer.s11n")
      
      (and f32-ok f64-ok))))

;;; ==================================================================
;;; Different Activation Functions
;;; ==================================================================

(define (test-activations)
  (run-test "Different activation functions"
    (lambda ()
      (define activations 
        (list (cons "ReLU" (make-relu))
              (cons "Tanh" (make-tanh))
              (cons "Sigmoid" (make-sigmoid))
              (cons "Identity" (make-identity))))
      
      (define results
        (map (lambda (act-pair)
               (let* ((act-name (car act-pair))
                      (activation (cdr act-pair))
                      (layer (make-dense-layer 2 2 
                                              activation: activation
                                              name: (string-append "Test-" act-name)))
                      (filepath (string-append "test-" act-name ".s11n")))
                 
                 (printf "  Testing ~A activation...\n" act-name)
                 (save-layer layer filepath)
                 (define loaded (load-layer filepath))
                 
                 (define ok
                   (string=? (activation-name (layer-activation layer))
                            (activation-name (layer-activation loaded))))
                 
                 (test-result (string-append act-name " activation preserved") ok)
                 (delete-file* filepath)
                 ok))
             activations))
      
      (every (lambda (x) x) results))))

;;; ==================================================================
;;; Dimension Validation
;;; ==================================================================

(define (test-dimension-validation)
  (run-test "Dimension mismatch detection"
    (lambda ()
      (printf "  Creating layer with specific dimensions...\n")
      
      ;; Create a layer and get its serializable form
      (define layer (make-dense-layer 5 3 name: "TestDim"))
      (define serializable (layer->serializable layer))
      
      ;; Manually corrupt the dimensions
      (printf "  Corrupting weight dimensions...\n")
      (define corrupted-serializable
        (let ((weights-ser (cdr (assq 'weights serializable))))
          ;; Change shape from (3, 5) to (3, 4) - dimension mismatch!
          (let ((corrupted-weights
                 (cons (cons 'dtype (cdr (assq 'dtype weights-ser)))
                       (cons (cons 'shape '(3 4))  ; Wrong!
                             (cdr (cdr weights-ser))))))
            (cons (cons 'type 'dense-layer)
                  (cons (cons 'weights corrupted-weights)
                        (cdr (cdr serializable)))))))
      
      ;; Try to deserialize - should fail
      (define caught-error #f)
      (handle-exceptions exn
        (begin
          (set! caught-error #t)
          (printf "  Caught expected error: ~A\n" 
                  (get-condition-property exn 'exn 'message)))
        (begin
          (serializable->layer corrupted-serializable)
          (printf "  ERROR: Should have caught dimension mismatch!\n")))
      
      (test-result "Dimension mismatch detected" caught-error)
      caught-error)))

;;; ==================================================================
;;; Conv2D Layer Serialization
;;; ==================================================================

(define (test-conv2d-layer)
  (run-test "Conv2D layer save/load"
    (lambda ()
      (printf "  Creating Conv2D layer (3 channels -> 16 channels, 3×3 kernel)...\n")
      
      ;; Create a convolutional layer
      (define conv-layer (make-conv2d-layer 3 16 3
                                            activation: (make-relu)
                                            dtype: 'f32
                                            name: "TestConv"))
      
      (printf "    Input channels: ~A\n" (layer-input-size conv-layer))
      (printf "    Output channels: ~A\n" (layer-output-size conv-layer))
      
      ;; Save
      (save-layer conv-layer "test-conv-layer.s11n")
      (printf "  Layer saved\n")
      
      ;; Load
      (define loaded-conv (load-layer "test-conv-layer.s11n"))
      (printf "  Layer loaded\n")
      
      ;; Verify structure
      (define structure-ok
        (and (conv2d-layer? loaded-conv)
             (= (layer-input-size conv-layer) 
                (layer-input-size loaded-conv))
             (= (layer-output-size conv-layer) 
                (layer-output-size loaded-conv))))
      
      (test-result "Conv2D structure preserved" structure-ok)
      
      ;; Test with a small image (3×8×8)
      (define img-size (* 3 8 8))
      (define img-input (make-tensor32 
                         (let ((v (make-f32vector img-size)))
                           (do ((i 0 (+ i 1)))
                               ((= i img-size) v)
                             (f32vector-set! v i 
                                            (/ (exact->inexact i) 
                                               (exact->inexact img-size)))))
                         '(3 8 8)))
      
      (define original-output (forward conv-layer img-input))
      (define loaded-output (forward loaded-conv img-input))
      
      (define forward-match
        (vectors-close? (tensor-data original-output)
                       (tensor-data loaded-output)
                       1e-6))
      
      (test-result "Conv2D forward pass matches" forward-match)
      
      ;; Cleanup
      (delete-file* "test-conv-layer.s11n")
      
      (and structure-ok forward-match))))

;;; ==================================================================
;;; Large Model Stress Test
;;; ==================================================================

(define (test-large-model)
  (run-test "Large sequential model stress test"
    (lambda ()
      (printf "  Creating large model with 5 layers...\n")
      
      ;; Create a larger model
      (define large-model 
        (make-sequential
         (list
          (make-dense-layer 100 80 activation: (make-relu) name: "L1")
          (make-dense-layer 80 60 activation: (make-relu) name: "L2")
          (make-dense-layer 60 40 activation: (make-tanh) name: "L3")
          (make-dense-layer 40 20 activation: (make-sigmoid) name: "L4")
          (make-dense-layer 20 10 activation: (make-identity) name: "L5"))
         name: "LargeNetwork"))
      
      (define param-count 
        (fold (lambda (p acc)
                (+ acc (f32vector-length (tensor-data p))))
              0
              (parameters large-model)))
      
      (printf "    Total parameters: ~A\n" param-count)
      
      ;; Save
      (save-model large-model "test-large-model.s11n")
      (printf "  Model saved\n")
      
      ;; Load
      (define loaded-large (load-model "test-large-model.s11n"))
      (printf "  Model loaded\n")
      
      ;; Test forward pass with random input
      (define test-input 
        (make-tensor32 
         (let ((v (make-f32vector 100)))
           (do ((i 0 (+ i 1)))
               ((= i 100) v)
             (f32vector-set! v i (pseudo-random-real))))
         '(100)))
      
      (define orig-out (forward large-model test-input))
      (define load-out (forward loaded-large test-input))
      
      (define match
        (vectors-close? (tensor-data orig-out)
                       (tensor-data load-out)
                       1e-6))
      
      (test-result "Large model forward pass matches" match)
      
      ;; Cleanup
      (delete-file* "test-large-model.s11n")
      
      match)))


(define (test-summary)
  (printf "\n")
  (printf "========================================\n")
  (printf "TEST SUMMARY\n")
  (printf "========================================\n")
  (printf "Total tests:  ~A\n" test-counter)
  (printf "Passed:       ~A\n" test-passed)
  (printf "Failed:       ~A\n" (- test-counter test-passed))
  (printf "Success rate: ~A%\n" 
          (if (> test-counter 0)
              (* 100.0 (/ test-passed test-counter))
              0))
  (printf "========================================\n\n"))

(define (run-tests)
  (printf "\n")
  (printf "========================================\n")
  (printf "Layer serialization tests\n")
  (printf "========================================\n")

  
  ;; Run all tests
  (test-dense-layer-basic)
  (test-sequential-model)
  (test-different-dtypes)
  (test-activations)
  (test-dimension-validation)
  (test-conv2d-layer)
  (test-large-model)
  (test-summary)
  )

;; Run the test suite
(run-tests)

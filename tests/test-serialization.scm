;; Test file for layer serialization with dimension validation

(import scheme
        (chicken base)
        (chicken format)
        (chicken file)
        (chicken condition)
        (chicken random)
        (srfi 1)
        (srfi 4)
        test
        nanograd-autograd
        nanograd-layer)

;;; ==================================================================
;;; Helper Functions
;;; ==================================================================

(define (approx-equal? actual expected tolerance)
  "Check if two numbers are approximately equal within tolerance"
  (<= (abs (- actual expected)) tolerance))

(define (vector-approx-equal? vec1 vec2 tolerance)
  "Check if two f32vectors are approximately equal within tolerance"
  (let ((n1 (f32vector-length vec1))
        (n2 (f32vector-length vec2)))
    (and (= n1 n2)
         (let loop ((i 0))
           (cond
            ((= i n1) #t)
            ((> (abs (- (f32vector-ref vec1 i)
                       (f32vector-ref vec2 i)))
                tolerance)
             #f)
            (else (loop (+ i 1))))))))

(define-syntax test-approximate
  (syntax-rules ()
    ((test-approximate name expected actual tolerance)
     (test-assert name (approx-equal? actual expected tolerance)))))

(define (test-vector-equal vec1 vec2 tolerance)
  "Test helper for vector equality with tolerance"
  (test-assert (vector-approx-equal? vec1 vec2 tolerance)))

;;; ==================================================================
;;; Dense Layer Serialization
;;; ==================================================================

(test-group "Dense Layer Serialization"
  
  (test-assert "Dense layer save/load with dimension checking"
    (begin
      ;; Create a dense layer
      (define layer (make-dense-layer 3 2 
                                      activation: (make-relu)
                                      dtype: 'f32
                                      name: "TestDense"))
      
      ;; Get original parameters
      (define original-params (parameters layer))
      (define original-weights (car original-params))
      (define original-biases (cadr original-params))
      
      ;; Save the layer
      (save-layer layer "test-dense-layer.s11n")
      
      ;; Load the layer
      (define loaded-layer (load-layer "test-dense-layer.s11n"))
      
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
      
      ;; Check if weights match
      (define weights-match 
        (vector-approx-equal? (tensor-data original-weights) 
                             (tensor-data loaded-weights) 
                             1e-9))
      
      ;; Check if biases match
      (define biases-match 
        (vector-approx-equal? (tensor-data original-biases) 
                             (tensor-data loaded-biases) 
                             1e-9))
      
      ;; Test forward pass
      (define test-input (make-tensor32 
                          (f32vector 1.0 2.0 3.0)
                          '(3)))
      
      (define original-output (forward layer test-input))
      (define loaded-output (forward loaded-layer test-input))
      
      (define forward-match 
        (vector-approx-equal? (tensor-data original-output)
                             (tensor-data loaded-output) 
                             1e-6))
      
      ;; Cleanup
      (delete-file* "test-dense-layer.s11n")
      
      (and metadata-ok weights-match biases-match forward-match))))

;;; ==================================================================
;;; Sequential Model Serialization
;;; ==================================================================

(test-group "Sequential Model Serialization"
  
  (test-assert "Sequential model with multiple layers"
    (begin
      ;; Create a sequential model
      (define model (make-sequential
                     (list
                      (make-dense-layer 4 8 activation: (make-relu) name: "Hidden1")
                      (make-dense-layer 8 4 activation: (make-relu) name: "Hidden2")
                      (make-dense-layer 4 2 activation: (make-identity) name: "Output"))
                     name: "TestNetwork"))
      
      ;; Save the model
      (save-model model "test-model.s11n")
      
      ;; Load the model
      (define loaded-model (load-model "test-model.s11n"))
      
      ;; Verify structure
      (define structure-ok
        (and (= (layer-input-size model) (layer-input-size loaded-model))
             (= (layer-output-size model) (layer-output-size loaded-model))
             (= (length (parameters model)) 
                (length (parameters loaded-model)))))
      
      ;; Test forward pass
      (define test-input (make-tensor32 
                          (f32vector 1.0 2.0 3.0 4.0)
                          '(4)))
      
      (define original-output (forward model test-input))
      (define loaded-output (forward loaded-model test-input))
      
      (define forward-match
        (vector-approx-equal? (tensor-data original-output)
                             (tensor-data loaded-output)
                             1e-6))
      
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
                     (match (vector-approx-equal? orig-data load-data 1e-9)))
                (loop (cdr op) (cdr lp) (and all-match match))))))))
      
      ;; Cleanup
      (delete-file* "test-model.s11n")
      
      (and structure-ok forward-match params-match))))

;;; ==================================================================
;;; Different Data Types
;;; ==================================================================

(test-group "Different Data Types"
  
  (test-assert "Float32 dtype preserved"
    (begin
      (define f32-layer (make-dense-layer 2 3 
                                          dtype: 'f32
                                          name: "F32Dense"))
      
      (save-layer f32-layer "test-f32-layer.s11n")
      (define loaded-f32-layer (load-layer "test-f32-layer.s11n"))
      
      (define result
        (and (eq? (tensor-dtype (car (parameters f32-layer))) 'f32)
             (eq? (tensor-dtype (car (parameters loaded-f32-layer))) 'f32)))
      
      (delete-file* "test-f32-layer.s11n")
      result))
  
  (test-assert "Float64 dtype preserved"
    (begin
      (define f64-layer (make-dense-layer 2 3 
                                          dtype: 'f64
                                          name: "F64Dense"))
      
      (save-layer f64-layer "test-f64-layer.s11n")
      (define loaded-f64-layer (load-layer "test-f64-layer.s11n"))
      
      (define result
        (and (eq? (tensor-dtype (car (parameters f64-layer))) 'f64)
             (eq? (tensor-dtype (car (parameters loaded-f64-layer))) 'f64)))
      
      (delete-file* "test-f64-layer.s11n")
      result)))

;;; ==================================================================
;;; Different Activation Functions
;;; ==================================================================

(test-group "Activation Functions"
  
  (define activations 
    (list (cons "ReLU" (make-relu))
          (cons "Tanh" (make-tanh))
          (cons "Sigmoid" (make-sigmoid))
          (cons "Identity" (make-identity))))
  
  (for-each
   (lambda (act-pair)
     (let* ((act-name (car act-pair))
            (activation (cdr act-pair)))
       (test-assert (string-append act-name " activation preserved")
         (begin
           (define layer (make-dense-layer 2 2 
                                          activation: activation
                                          name: (string-append "Test-" act-name)))
           (define filepath (string-append "test-" act-name ".s11n"))
           
           (save-layer layer filepath)
           (define loaded (load-layer filepath))
           
           (define result
             (string=? (activation-name (layer-activation layer))
                      (activation-name (layer-activation loaded))))
           
           (delete-file* filepath)
           result))))
   activations))

;;; ==================================================================
;;; Dimension Validation
;;; ==================================================================

(test-group "Dimension Validation"
  
  (test-assert "Dimension mismatch detection"
    (begin
      ;; Create a layer and get its serializable form
      (define layer (make-dense-layer 5 3 name: "TestDim"))
      (define serializable (layer->serializable layer))
      
      ;; Manually corrupt the dimensions
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
        (set! caught-error #t)
        (serializable->layer corrupted-serializable))
      
      caught-error)))

;;; ==================================================================
;;; Conv2D Layer Serialization
;;; ==================================================================

(test-group "Conv2D Layer Serialization"
  
  (test-assert "Conv2D layer save/load"
    (begin
      ;; Create a convolutional layer
      (define conv-layer (make-conv2d-layer 3 16 3
                                            activation: (make-relu)
                                            dtype: 'f32
                                            name: "TestConv"))
      
      ;; Save
      (save-layer conv-layer "test-conv-layer.s11n")
      
      ;; Load
      (define loaded-conv (load-layer "test-conv-layer.s11n"))
      
      ;; Verify structure
      (define structure-ok
        (and (conv2d-layer? loaded-conv)
             (= (layer-input-size conv-layer) 
                (layer-input-size loaded-conv))
             (= (layer-output-size conv-layer) 
                (layer-output-size loaded-conv))))
      
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
        (vector-approx-equal? (tensor-data original-output)
                             (tensor-data loaded-output)
                             1e-6))
      
      ;; Cleanup
      (delete-file* "test-conv-layer.s11n")
      
      (and structure-ok forward-match))))

;;; ==================================================================
;;; Large Model Stress Test
;;; ==================================================================

(test-group "Large Model Stress Test"
  
  (test-assert "Large sequential model"
    (begin
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
      
      ;; Save
      (save-model large-model "test-large-model.s11n")
      
      ;; Load
      (define loaded-large (load-model "test-large-model.s11n"))
      
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
      
      (define result
        (vector-approx-equal? (tensor-data orig-out)
                             (tensor-data load-out)
                             1e-6))
      
      ;; Cleanup
      (delete-file* "test-large-model.s11n")
      
      result)))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(test-exit)

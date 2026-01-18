;; Test file for GeLU and SiLU activation functions

(import scheme
        (chicken base)
        (chicken format)
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

(define-syntax test-approximate
  (syntax-rules ()
    ((test-approximate name expected actual tolerance)
     (test-assert name (approx-equal? actual expected tolerance)))))

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

;;; ==================================================================
;;; Unit Tests: GeLU Activation
;;; ==================================================================

(test-group "GeLU - Forward Pass"
  
  (let* ((x-data (f32vector -2.0 -1.0 0.0 1.0 2.0))
         (x (make-tensor32 x-data '(5) requires-grad: #t))
         (y (gelu x))
         (output (tensor-data y)))
    
    ;; Expected approximate values: (-0.046, -0.159, 0.0, 0.841, 1.954)
    (test-approximate "GeLU(-2.0) ~= -0.046"
      -0.046 (f32vector-ref output 0) 0.01)
    (test-approximate "GeLU(-1.0) ~= -0.159"
      -0.159 (f32vector-ref output 1) 0.01)
    (test-approximate "GeLU(0.0) = 0.0"
      0.0 (f32vector-ref output 2) 1e-5)
    (test-approximate "GeLU(1.0) ~= 0.841"
      0.841 (f32vector-ref output 3) 0.01)
    (test-approximate "GeLU(2.0) ~= 1.954"
      1.954 (f32vector-ref output 4) 0.01)))

(test-group "GeLU - Backward Pass"
  
  (let* ((x-data (f32vector 0.5 1.0 1.5))
         (x (make-tensor32 x-data '(3) requires-grad: #t))
         (y (gelu x)))
    
    ;; Backward pass: set gradient to 1.0 for all outputs
    (let ((grad (tensor-grad y)))
      (f32vector-set! grad 0 1.0)
      (f32vector-set! grad 1 1.0)
      (f32vector-set! grad 2 1.0))
    
    (backward! y)
    
    (let ((grad-x (tensor-grad x)))
      ;; GeLU derivative: d/dx GeLU(x) = Phi(x) + x . phi(x)
      ;; Expected approximate gradients: (0.867, 1.083, 1.128)
      (test-approximate "GeLU gradient at 0.5 ~= 0.867"
                        0.867 (f32vector-ref grad-x 0) 0.01)
      (test-approximate "GeLU gradient at 1.0 ~= 1.083"
                        1.083 (f32vector-ref grad-x 1) 0.01)
      (test-approximate "GeLU gradient at 1.5 ~= 1.128"
                        1.128 (f32vector-ref grad-x 2) 0.01))))

;;; ==================================================================
;;; Unit Tests: SiLU Activation
;;; ==================================================================

(test-group "SiLU - Forward Pass"
  
  (let* ((x-data (f32vector -2.0 -1.0 0.0 1.0 2.0))
         (x (make-tensor32 x-data '(5) requires-grad: #t))
         (y (silu x))
         (output (tensor-data y)))

    ;; Expected approximate values: (-0.238, -0.269, 0.0, 0.731, 1.762)
    (test-approximate "SiLU(-2.0) ~= -0.238"
                      -0.238 (f32vector-ref output 0) 0.01)
    (test-approximate "SiLU(-1.0) ~= -0.269"
                      -0.269 (f32vector-ref output 1) 0.01)
    (test-approximate "SiLU(0.0) = 0.0"
                      0.0 (f32vector-ref output 2) 1e-5)
    (test-approximate "SiLU(1.0) ~= 0.731"
                      0.731 (f32vector-ref output 3) 0.01)
    (test-approximate "SiLU(2.0) ~= 1.762"
                      1.762 (f32vector-ref output 4) 0.01)))

(test-group "SiLU - Backward Pass"
  
  (let* ((x-data (f32vector 0.5 1.0 1.5))
         (x (make-tensor32 x-data '(3) requires-grad: #t))
         (y (silu x)))
    
    ;; Backward pass
    (let ((grad (tensor-grad y)))
      (f32vector-set! grad 0 1.0)
      (f32vector-set! grad 1 1.0)
      (f32vector-set! grad 2 1.0))
    
    (backward! y)
    
    (let ((grad-x (tensor-grad x)))
      ;; SiLU derivative: d/dx SiLU(x) = sigma(x) + x . sigma(x) . (1-sigma(x))
      ;; Expected approximate gradients: (0.740, 0.928, 1.041)
      (test-approximate "SiLU gradient at 0.5 ~= 0.740"
                        0.740 (f32vector-ref grad-x 0) 0.01)
      (test-approximate "SiLU gradient at 1.0 ~= 0.928"
                        0.928 (f32vector-ref grad-x 1) 0.01)
      (test-approximate "SiLU gradient at 1.5 ~= 1.041"
                        1.041 (f32vector-ref grad-x 2) 0.01))))

;;; ==================================================================
;;; Unit Tests: Using Activations in Layers
;;; ==================================================================

(test-group "Dense Layer with GeLU Activation"
  
  (let* ((layer (make-dense-layer 3 2 
                                  dtype: 'f32
                                  activation: (make-gelu)
                                  name: "gelu-layer"))
         (x-data (f32vector 0.5 1.0 -0.5))
         (x (make-tensor32 x-data '(3) requires-grad: #t))
         (y (forward layer x)))
    
    (test "Layer name"
      "gelu-layer"
      (layer-name layer))
    
    (test "Activation name"
      "GeLU"
      (activation-name (layer-activation layer)))
    
    (test "Input shape"
      '(3)
      (tensor-shape x))
    
    (test "Output shape"
      '(2)
      (tensor-shape y))))

(test-group "Dense Layer with SiLU Activation"
  
  (let* ((layer (make-dense-layer 3 2 
                                  dtype: 'f32
                                  activation: (make-silu)
                                  name: "silu-layer"))
         (x-data (f32vector 0.5 1.0 -0.5))
         (x (make-tensor32 x-data '(3) requires-grad: #t))
         (y (forward layer x)))
    
    (test "Layer name"
      "silu-layer"
      (layer-name layer))
    
    (test "Activation name"
      "SiLU"
      (activation-name (layer-activation layer)))
    
    (test "Input shape"
      '(3)
      (tensor-shape x))
    
    (test "Output shape"
      '(2)
      (tensor-shape y))))

(test-group "Sequential Model with Mixed Activations"
  
  (let* ((model (make-sequential 
                 (list
                  (make-dense-layer 4 8 dtype: 'f32 
                                    activation: (make-gelu)
                                    name: "hidden1-gelu")
                  (make-dense-layer 8 4 dtype: 'f32 
                                    activation: (make-silu)
                                    name: "hidden2-silu")
                  (make-dense-layer 4 2 dtype: 'f32 
                                    activation: (make-identity)
                                    name: "output"))
                 name: "mixed-activations-model"))
         (x-data (f32vector 1.0 0.5 -0.5 -1.0))
         (x (make-tensor32 x-data '(4) requires-grad: #t))
         (y (forward model x)))
    
    (test "Model name"
      "mixed-activations-model"
      (layer-name model))
    
    (test "Input shape"
      '(4)
      (tensor-shape x))
    
    (test "Output shape"
      '(2)
      (tensor-shape y))))

;;; ==================================================================
;;; Unit Tests: Activation Comparison
;;; ==================================================================

(test-group "Comparing All Activation Functions"
  
  (let* ((x-data (f32vector -2.0 -1.0 0.0 1.0 2.0))
         (x (make-tensor32 x-data '(5))))
    
    ;; Just verify that all activations work
    (test-assert "ReLU produces output"
      (tensor-data (relu x)))
    
    (test-assert "LeakyReLU produces output"
      (tensor-data (leaky-relu x)))
    
    (test-assert "Sigmoid produces output"
      (tensor-data (sigmoid x)))
    
    (test-assert "Tanh produces output"
      (tensor-data (tanh-op x)))
    
    (test-assert "Softplus produces output"
      (tensor-data (softplus x)))
    
    (test-assert "GeLU produces output"
      (tensor-data (gelu x)))
    
    (test-assert "SiLU produces output"
      (tensor-data (silu x)))))

;;; ==================================================================
;;; Unit Tests: Serialization
;;; ==================================================================

(test-group "Layer Serialization with GeLU"
  
  (let* ((layer1 (make-dense-layer 3 2 
                                   dtype: 'f32
                                   activation: (make-gelu)
                                   name: "gelu-layer")))
    
    ;; Save and load
    (save-layer layer1 "test-gelu-layer.dat")
    (let ((layer2 (load-layer "test-gelu-layer.dat")))
      
      (test "Original layer name"
        "gelu-layer"
        (layer-name layer1))
      
      (test "Original activation"
        "GeLU"
        (activation-name (layer-activation layer1)))
      
      (test "Loaded layer name"
        "gelu-layer"
        (layer-name layer2))
      
      (test "Loaded activation"
        "GeLU"
        (activation-name (layer-activation layer2))))))

(test-group "Layer Serialization with SiLU"
  
  (let* ((layer1 (make-dense-layer 3 2 
                                   dtype: 'f32
                                   activation: (make-silu)
                                   name: "silu-layer")))
    
    ;; Save and load
    (save-layer layer1 "test-silu-layer.dat")
    (let ((layer2 (load-layer "test-silu-layer.dat")))
      
      (test "Original layer name"
        "silu-layer"
        (layer-name layer1))
      
      (test "Original activation"
        "SiLU"
        (activation-name (layer-activation layer1)))
      
      (test "Loaded layer name"
        "silu-layer"
        (layer-name layer2))
      
      (test "Loaded activation"
        "SiLU"
        (activation-name (layer-activation layer2))))))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(test-exit)

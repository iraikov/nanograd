;; Test file for GeLU and SiLU activation functions

(import scheme
        (chicken base)
        (chicken format)
        (srfi 4)
        nanograd-autograd
        nanograd-layer)

(print "\n=== Testing GeLU and SiLU Activation Functions ===\n")

;; ==================================================================
;; Test 1: Basic GeLU Forward Pass
;; ==================================================================
(print "Test 1: GeLU Forward Pass")
(print "----------------------------")

(let* ((x-data (f32vector -2.0 -1.0 0.0 1.0 2.0))
       (x (make-tensor32 x-data '(5) requires-grad: #t))
       (y (gelu x)))
  
  (print "Input:  " (tensor->list x))
  (print "GeLU:   " (tensor->list y))
  (print "Expected approximate: (-0.046, -0.159, 0.0, 0.841, 1.954)")
  (newline))

;; ==================================================================
;; Test 2: Basic SiLU Forward Pass
;; ==================================================================
(print "Test 2: SiLU Forward Pass")
(print "----------------------------")

(let* ((x-data (f32vector -2.0 -1.0 0.0 1.0 2.0))
       (x (make-tensor32 x-data '(5) requires-grad: #t))
       (y (silu x)))
  
  (print "Input:  " (tensor->list x))
  (print "SiLU:   " (tensor->list y))
  (print "Expected approximate: (-0.238, -0.269, 0.0, 0.731, 1.762)")
  (newline))

;; ==================================================================
;; Test 3: GeLU Gradient Computation
;; ==================================================================
(print "Test 3: GeLU Backward Pass")
(print "----------------------------")

(let* ((x-data (f32vector 0.5 1.0 1.5))
       (x (make-tensor32 x-data '(3) requires-grad: #t))
       (y (gelu x)))
  
  (print "Input:    " (tensor->list x))
  (print "GeLU:     " (tensor->list y))
  
  ;; Backward pass: set gradient to 1.0 for all outputs
  (let ((grad (tensor-grad y)))
    (f32vector-set! grad 0 1.0)
    (f32vector-set! grad 1 1.0)
    (f32vector-set! grad 2 1.0))
  
  (backward! y)
  
  (print "Gradient: " (f32vector->list (tensor-grad x)))
  (print "Expected approximate: (0.691, 0.920, 1.051)")
  (newline))

;; ==================================================================
;; Test 4: SiLU Gradient Computation
;; ==================================================================
(print "Test 4: SiLU Backward Pass")
(print "----------------------------")

(let* ((x-data (f32vector 0.5 1.0 1.5))
       (x (make-tensor32 x-data '(3) requires-grad: #t))
       (y (silu x)))
  
  (print "Input:    " (tensor->list x))
  (print "SiLU:     " (tensor->list y))
  
  ;; Backward pass
  (let ((grad (tensor-grad y)))
    (f32vector-set! grad 0 1.0)
    (f32vector-set! grad 1 1.0)
    (f32vector-set! grad 2 1.0))
  
  (backward! y)
  
  (print "Gradient: " (f32vector->list (tensor-grad x)))
  (print "Expected approximate: (0.770, 0.928, 1.025)")
  (newline))

;; ==================================================================
;; Test 5: Using GeLU in a Dense Layer
;; ==================================================================
(print "Test 5: Dense Layer with GeLU Activation")
(print "------------------------------------------")

(let* ((layer (make-dense-layer 3 2 
                                dtype: 'f32
                                activation: (make-gelu)
                                name: "gelu-layer"))
       (x-data (f32vector 0.5 1.0 -0.5))
       (x (make-tensor32 x-data '(3) requires-grad: #t))
       (y (forward layer x)))
  
  (print "Layer:  " (layer-name layer))
  (print "Activation: " (activation-name (layer-activation layer)))
  (print "Input shape:  " (tensor-shape x))
  (print "Output shape: " (tensor-shape y))
  (print "Output:       " (tensor->list y))
  (newline))

;; ==================================================================
;; Test 6: Using SiLU in a Dense Layer
;; ==================================================================
(print "Test 6: Dense Layer with SiLU Activation")
(print "------------------------------------------")

(let* ((layer (make-dense-layer 3 2 
                                dtype: 'f32
                                activation: (make-silu)
                                name: "silu-layer"))
       (x-data (f32vector 0.5 1.0 -0.5))
       (x (make-tensor32 x-data '(3) requires-grad: #t))
       (y (forward layer x)))
  
  (print "Layer:  " (layer-name layer))
  (print "Activation: " (activation-name (layer-activation layer)))
  (print "Input shape:  " (tensor-shape x))
  (print "Output shape: " (tensor-shape y))
  (print "Output:       " (tensor->list y))
  (newline))

;; ==================================================================
;; Test 7: Sequential Model with Multiple Activation Functions
;; ==================================================================
(print "Test 7: Sequential Model with Mixed Activations")
(print "-------------------------------------------------")

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
  
  (print "Model: " (layer-name model))
  (print "Input shape:  " (tensor-shape x))
  (print "Output shape: " (tensor-shape y))
  (print "Output:       " (tensor->list y))
  (newline))

;; ==================================================================
;; Test 8: Comparing All Activation Functions
;; ==================================================================
(print "Test 8: Comparing All Activation Functions")
(print "-------------------------------------------")

(let* ((x-data (f32vector -2.0 -1.0 0.0 1.0 2.0))
       (x (make-tensor32 x-data '(5))))
  
  (print "Input:       " (tensor->list x))
  (print "ReLU:        " (tensor->list (relu x)))
  (print "LeakyReLU:   " (tensor->list (leaky-relu x)))
  (print "Sigmoid:     " (tensor->list (sigmoid x)))
  (print "Tanh:        " (tensor->list (tanh-op x)))
  (print "Softplus:    " (tensor->list (softplus x)))
  (print "GeLU:        " (tensor->list (gelu x)))
  (print "SiLU:        " (tensor->list (silu x)))
  (newline))

;; ==================================================================
;; Test 9: Layer Serialization with GeLU and SiLU
;; ==================================================================
(print "Test 9: Serialization/Deserialization")
(print "--------------------------------------")

;; Create a layer with GeLU
(let* ((layer1 (make-dense-layer 3 2 
                                 dtype: 'f32
                                 activation: (make-gelu)
                                 name: "gelu-layer")))
  
  (print "Original layer: " (layer-name layer1) 
         " with activation: " (activation-name (layer-activation layer1)))
  
  ;; Save and load
  (save-layer layer1 "test-gelu-layer.dat")
  (let ((layer2 (load-layer "test-gelu-layer.dat")))
    (print "Loaded layer:   " (layer-name layer2)
           " with activation: " (activation-name (layer-activation layer2))))
  (newline))

;; Create a layer with SiLU
(let* ((layer1 (make-dense-layer 3 2 
                                 dtype: 'f32
                                 activation: (make-silu)
                                 name: "silu-layer")))
  
  (print "Original layer: " (layer-name layer1) 
         " with activation: " (activation-name (layer-activation layer1)))
  
  ;; Save and load
  (save-layer layer1 "test-silu-layer.dat")
  (let ((layer2 (load-layer "test-silu-layer.dat")))
    (print "Loaded layer:   " (layer-name layer2)
           " with activation: " (activation-name (layer-activation layer2))))
  (newline))

(print "=== All Tests Completed Successfully! ===\n")

# NanoGrad: Automatic Differentiation Framework for CHICKEN Scheme

A lightweight, YASOS-based automatic differentiation and neural
network framework for CHICKEN Scheme, featuring BLAS-accelerated
operations and a clean functional API.

## Features

- **Automatic Differentiation**: Reverse-mode autodiff with topological sorting for correct gradient computation
- **BLAS Integration**: High-performance linear algebra operations using CBLAS
- **YASOS Object System**: Clean, polymorphic object-oriented abstractions
- **Mixed Precision**: Support for both 32-bit (f32) and 64-bit (f64) floating-point
- **Neural Network Layers**: Dense layers, convolutional layers, batch normalization, and sequential containers
- **Activation Functions**: ReLU, Tanh, Sigmoid, Softmax, LeakyReLU, Softplus, SiLU, GeLU
- **Optimizers**: SGD (with momentum), Adam, RMSprop
- **Loss Functions**: MSE, Cross-Entropy
- **Advanced Operations**: Convolution, RMSNorm, Layer Normalization, Batch Normalization, Global Pooling
- **Tensor Operations**: Reduction operations, slicing, reshaping with full gradient support

## Installation

```bash
# Install dependencies
chicken-install yasos blas mathh srfi-1 srfi-4 srfi-42 srfi-69

# Clone the repository
git clone https://github.com/iraikov/nanograd.git
cd nanograd

chicken-install
```

## Quick Start

### Basic Tensor Operations

```scheme
(import nanograd-autograd)

;; Create tensors with automatic differentiation
(define x (make-tensor32 (f32vector 1.0 2.0 3.0) '(3) requires-grad?: #t))
(define y (make-tensor32 (f32vector 4.0 5.0 6.0) '(3) requires-grad?: #t))

;; Element-wise operations
(define z (add x y))  ; z = x + y
(define w (mul x y))  ; w = x * y

;; Matrix operations
(define A (make-tensor32 (f32vector 1.0 2.0 3.0 4.0) '(2 2)))
(define b (make-tensor32 (f32vector 1.0 2.0) '(2)))
(define result (matmul-op A b))  ; Matrix-vector multiplication

;; Compute gradients
(backward! result)
(print-tensor (tensor-grad A))
```

### Reduction Operations

```scheme
;; Sum all elements
(define total (sum-tensor x))

;; Compute mean
(define avg (mean-tensor x))

;; Compute product
(define prod (product-tensor x))

;; Custom reduction with gradient
(define custom-result
  (reduce-tensor x max
    compute-gradient: (lambda (grad-out idx val all-values)
                       ;; Custom gradient logic
                       (if (= val (apply max all-values))
                           grad-out
                           0.0))))
```

### Tensor Slicing

```scheme
;; Extract slice along first dimension
(define batch (make-tensor32 (make-f32vector 100) '(10 10)))
(define slice (slice-tensor batch 2 5))  ; Extract elements 2-6 along first dim

;; Gradients flow back correctly
(backward! (sum-tensor slice))
(print-tensor (tensor-grad batch))  ; Only positions 2-6 have gradients
```

### Building a Neural Network

```scheme
(import nanograd-layer nanograd-optimizer)

;; Define a simple classification network
(define model
  (make-sequential
   (list
    (make-dense-layer 784 128 activation: (make-relu) name: "Hidden1")
    (make-dense-layer 128 64 activation: (make-relu) name: "Hidden2")
    (make-dense-layer 64 10 activation: (make-identity) name: "Output"))
   name: "Classifier"))

;; Create optimizer
(define optimizer (make-adam (parameters model) learning-rate: 0.001))

;; Training loop
(do ((epoch 1 (+ epoch 1)))
    ((> epoch 10))
  
  ;; Training mode
  (set-training-mode! model #t)
  
  (for-each
   (lambda (batch)
     (let* ((x (car batch))
            (target (cdr batch))
            (pred (forward model x))
            (loss (cross-entropy-loss (softmax pred) target)))
       
       ;; Backward pass and optimize
       (backward! loss)
       (step! optimizer)
       (zero-grad-layer! model)))
   training-data)
  
  ;; Evaluation mode
  (set-eval-mode! model)
  (evaluate-model model validation-data))
```

### Convolutional Neural Network with Batch Normalization

```scheme
(define cnn
  (make-sequential
   (list
    (make-conv2d-layer 3 32 3 stride: 1 padding: 1 
                       activation: (make-relu) name: "Conv1")
    (make-batch-norm-2d 32 name: "BN1")
    (make-conv2d-layer 32 64 3 stride: 1 padding: 1 
                       activation: (make-relu) name: "Conv2")
    (make-batch-norm-2d 64 name: "BN2")
    ;; Global average pooling reduces spatial dimensions
    (make-dense-layer 64 128 activation: (make-relu) name: "FC1")
    (make-dense-layer 128 10 activation: (make-identity) name: "Output"))
   name: "CNN"))

;; Forward pass with global average pooling
(define (forward-with-pooling model input)
  (let* ((conv-output (forward (list-ref (get-layers model) 0) input))
         (bn-output (forward (list-ref (get-layers model) 1) conv-output))
         (pooled (global-avg-pool2d bn-output)))
    (forward (list-ref (get-layers model) 2) pooled)))
```

## Architecture

### Module Structure

- **`nanograd-autograd`**: Core automatic differentiation engine
  - Tensor abstraction with YASOS
  - Arithmetic operations (add, sub, mul, div)
  - BLAS operations (matmul, dot, scale)
  - Activation functions
  - Loss functions
  - Reduction operations (sum, mean, product, custom reductions)
  - Tensor manipulation (slice, reshape, flatten)
  - Gradient computation with cycle detection

- **`nanograd-layer`**: Neural network layer abstractions
  - Dense (fully connected) layers
  - Convolutional layers (2D)
  - Batch normalization (2D)
  - Global average pooling
  - Sequential containers
  - Activation function objects
  - Training/evaluation mode control

- **`nanograd-optimizer`**: Optimization algorithms
  - SGD with momentum and Nesterov
  - Adam with bias correction
  - RMSprop with momentum

### Design Principles

1. **Functional Programming**: Immutable tensors, pure operations where possible
2. **YASOS Objects**: Clean polymorphic dispatch for operations
3. **BLAS Efficiency**: Leverage optimized linear algebra for performance
4. **Explicit Gradient Management**: Manual control over backward passes
5. **Mixed Precision**: First-class support for both f32 and f64

## API Reference

### Tensor Operations

#### Constructors
```scheme
(make-tensor32 data shape #:key (requires-grad? #t))
(make-tensor64 data shape #:key (requires-grad? #t))
```

#### Accessors
```scheme
(tensor-data tensor)        ; Get underlying data vector
(tensor-grad tensor)        ; Get gradient vector
(tensor-shape tensor)       ; Get shape list
(tensor-dtype tensor)       ; Get dtype ('f32 or 'f64)
(tensor-requires-grad? t)   ; Check if gradients enabled
```

#### Arithmetic
```scheme
(add a b)                   ; Element-wise addition
(sub a b)                   ; Element-wise subtraction
(mul a b)                   ; Element-wise multiplication
(div a b)                   ; Element-wise division
(safe-div a b #:key (epsilon 1e-8))
```

#### Linear Algebra
```scheme
(matmul-op a b)            ; Matrix multiplication
(dot-op a b)               ; Dot product
(scale-op tensor scalar)   ; Scalar multiplication
```

#### Reduction Operations
```scheme
(reduce-tensor tensor reducer #:key (compute-gradient #f))
  ; Generic reduction with custom gradient
  ; reducer: (element accumulator) -> new-accumulator
  ; compute-gradient: (grad-out index value all-values) -> grad-in

(sum-tensor tensor)        ; Sum all elements (gradient: uniform)
(mean-tensor tensor)       ; Mean of all elements
(product-tensor tensor)    ; Product of all elements (gradient: product rule)
```

**Example: Custom Maximum Reduction**
```scheme
(define (max-tensor tensor)
  (reduce-tensor tensor max
    compute-gradient: (lambda (grad-out idx val all-values)
                       ;; Gradient flows only to maximum element
                       (if (= val (apply max all-values))
                           grad-out
                           0.0))))
```

#### Tensor Manipulation
```scheme
(slice-tensor tensor start length)
  ; Extract slice along first dimension
  ; tensor: Input tensor with shape (n, ...)
  ; start: Starting index
  ; length: Number of elements to extract
  ; Returns: Tensor with shape (length, ...)

(reshape tensor new-shape)  ; Reshape (must preserve total elements)
(flatten-tensor tensor)     ; Flatten to 1D
```

**Example: Batch Processing**
```scheme
;; Process mini-batches from a dataset
(define dataset (make-tensor32 (make-f32vector 1000) '(100 10)))

(do ((i 0 (+ i batch-size)))
    ((>= i 100))
  (let ((batch (slice-tensor dataset i batch-size)))
    (process-batch model batch)))
```

#### Activations
```scheme
(relu tensor)              ; ReLU activation
(tanh-op tensor)           ; Hyperbolic tangent
(sigmoid tensor)           ; Sigmoid (logistic)
(sigmoid-stable tensor)    ; Numerically stable sigmoid
(softmax tensor)           ; Softmax normalization
(log-softmax tensor)       ; Log-softmax
(silu tensor)              ; SiLU
(gelu tensor)              ; GeLU
(leaky-relu tensor #:key (alpha 0.01))
(softplus tensor #:key (beta 1.0))
```

#### Loss Functions
```scheme
(mse-loss pred target)              ; Mean squared error
(cross-entropy-loss pred target)    ; Cross-entropy loss
```

#### Gradient Operations
```scheme
(zero-grad! tensor)        ; Zero out gradients
(backward! tensor)         ; Compute gradients via backprop
(add-to-grad! tensor delta) ; Accumulate gradients
```

### Layer API

#### Layer Construction
```scheme
(make-dense-layer input-size output-size 
                  #:key (activation (make-identity))
                        (dtype 'f32)
                        (name "Dense"))

(make-conv2d-layer in-channels out-channels kernel-size
                   #:key (stride 1)
                         (padding 0)
                         (activation (make-identity))
                         (dtype 'f32)
                         (name "Conv2D"))

(make-batch-norm-2d num-features
                    #:key (epsilon 1e-5)
                          (momentum 0.1)
                          (dtype 'f32)
                          (name "BatchNorm2d"))

(make-sequential layers #:key (name "Sequential"))
```

#### Batch Normalization

Batch Normalization normalizes activations across the batch dimension, improving training stability and convergence:

```scheme
;; Create batch norm layer for 64 channels
(define bn (make-batch-norm-2d 64 epsilon: 1e-5 momentum: 0.1))

;; In training mode: uses batch statistics and updates running stats
(set-training-mode! bn #t)
(define normalized (forward bn input))  ; input shape: (64, H, W)

;; In eval mode: uses running statistics
(set-eval-mode! bn)
(define normalized (forward bn input))  ; Deterministic output
```

**Key Features:**
- Learnable scale (gamma) and shift (beta) parameters
- Running mean and variance for evaluation
- Training/eval mode switching
- Numerical stability with epsilon parameter

#### Global Average Pooling

```scheme
(global-avg-pool2d input)
  ; Global average pooling over spatial dimensions
  ; Input shape: (C, H, W)
  ; Output shape: (C,)
  ; Gradients distributed uniformly over spatial dimensions
```

**Example: Replace Fully Connected Layers**
```scheme
;; Traditional approach: flatten + dense
(define old-approach
  (make-sequential
   (list
    (make-conv2d-layer 64 128 3)
    ;; flatten: (128, 8, 8) -> (8192,)
    (make-dense-layer 8192 10))))

;; Modern approach: global average pooling + dense
(define new-approach
  (make-sequential
   (list
    (make-conv2d-layer 64 128 3)
    ;; global avg pool: (128, 8, 8) -> (128,)
    (make-dense-layer 128 10))))

;; Fewer parameters, better generalization!
```

#### Layer Operations
```scheme
(forward layer input)      ; Forward pass
(parameters layer)         ; Get trainable parameters
(zero-grad-layer! layer)   ; Zero all parameter gradients

;; Training/Evaluation Mode Control
(set-training-mode! layer training?)  ; Set training mode (boolean)
(set-eval-mode! layer)                ; Set evaluation mode (shorthand)
```

**Training vs Evaluation Mode:**
- **Training Mode**: 
  - Batch norm uses batch statistics
  - Dropout is active (if implemented)
  - Stochastic behavior enabled
  
- **Evaluation Mode**:
  - Batch norm uses running statistics
  - Dropout is disabled
  - Deterministic behavior

```scheme
;; Training
(set-training-mode! model #t)
(for-each train-step training-batches)

;; Evaluation
(set-eval-mode! model)
(define accuracy (evaluate model test-data))
```

#### Activation Objects
```scheme
(make-relu)                ; ReLU activation
(make-tanh)                ; Tanh activation
(make-sigmoid)             ; Sigmoid activation
(make-silu)                ; SiLU activation
(make-gelu)                ; GeLU activation
(make-identity)            ; No activation
```

### Optimizer API

#### Optimizer Construction
```scheme
(make-sgd parameters 
          #:key (learning-rate 0.01)
                (momentum 0.0)
                (weight-decay 0.0)
                (nesterov #f))

(make-adam parameters
           #:key (learning-rate 0.001)
                 (beta1 0.9)
                 (beta2 0.999)
                 (epsilon 1e-8)
                 (weight-decay 0.0))

(make-rmsprop parameters
              #:key (learning-rate 0.01)
                    (alpha 0.99)
                    (epsilon 1e-8)
                    (weight-decay 0.0)
                    (momentum 0.0))
```

#### Optimizer Operations
```scheme
(step! optimizer)                ; Apply parameter updates
(get-learning-rate optimizer)    ; Get current learning rate
(set-learning-rate! optimizer lr) ; Update learning rate
(optimizer-state optimizer)      ; Get optimizer configuration
```

## Examples

See the `examples/` directory for complete working examples:

- Linear regression
- Binary classification
- Multi-class classification
- Learning rate scheduling
- Batch training
- Convolutional networks with batch normalization

### Complete Training Example with Batch Norm

```scheme
(import nanograd-autograd nanograd-layer nanograd-optimizer)

;; Define ResNet-style block
(define (make-resnet-block in-channels out-channels)
  (make-sequential
   (list
    (make-conv2d-layer in-channels out-channels 3 
                       padding: 1 activation: (make-identity))
    (make-batch-norm-2d out-channels)
    ;; ReLU applied separately
    (make-conv2d-layer out-channels out-channels 3
                       padding: 1 activation: (make-identity))
    (make-batch-norm-2d out-channels))
   name: "ResNetBlock"))

;; Full model
(define model
  (make-sequential
   (list
    (make-conv2d-layer 3 64 7 stride: 2 padding: 3)
    (make-batch-norm-2d 64)
    (make-resnet-block 64 64)
    (make-resnet-block 64 128)
    ;; ... more blocks ...
    )
   name: "ResNet"))

;; Training loop with proper mode switching
(define (train-epoch model optimizer train-data)
  (set-training-mode! model #t)
  
  (for-each
   (lambda (batch)
     (let* ((x (car batch))
            (y (cdr batch))
            (pred (forward model x))
            (loss (cross-entropy-loss pred y)))
       
       (backward! loss)
       (step! optimizer)
       (zero-grad-layer! model)))
   train-data))

(define (evaluate-epoch model test-data)
  (set-eval-mode! model)
  
  (let ((total-correct 0)
        (total-samples 0))
    (for-each
     (lambda (batch)
       (let* ((x (car batch))
              (y (cdr batch))
              (pred (forward model x))
              (predicted-class (argmax (tensor->list pred)))
              (true-class (argmax (tensor->list y))))
         (when (= predicted-class true-class)
           (set! total-correct (+ total-correct 1)))
         (set! total-samples (+ total-samples 1))))
     test-data)
    
    (/ total-correct total-samples)))

;; Main training loop
(define optimizer (make-adam (parameters model) learning-rate: 0.001))

(do ((epoch 1 (+ epoch 1)))
    ((> epoch 100))
  (train-epoch model optimizer train-data)
  (let ((acc (evaluate-epoch model test-data)))
    (printf "Epoch ~A: Test Accuracy = ~A%\n" epoch (* 100 acc))))
```

### Shape Manipulation

```scheme
(define x (make-tensor32 (f32vector 1.0 2.0 3.0 4.0) '(2 2)))

;; Reshape (must preserve total elements)
(define x-flat (reshape x '(4)))

;; Transpose dimensions
(define x-t (transpose-tensor x '(1 0)))
```

### Custom Weight Initialization

```scheme
;; Xavier/Glorot initialization (built-in for layers)
(define init-scale (sqrt (/ 2.0 (+ input-size output-size))))

;; He initialization for ReLU networks
(define init-scale (sqrt (/ 2.0 fan-in)))
```

## Limitations

- No GPU support (CPU-only via BLAS)
- Limited to dense and convolutional operations
- No automatic batching (must be implemented manually)
- Single-threaded execution

## Dependencies

- **yasos**: Object system
- **blas**: BLAS bindings for CHICKEN
- **mathh**: Extended math functions
- **srfi-1**: List utilities
- **srfi-4**: Homogeneous numeric vectors
- **srfi-42**: Eager comprehensions
- **srfi-69**: Hash tables

## License

LPGLv3 License - see LICENSE file for details

## Acknowledgments

This framework is inspired by:
- **PyTorch**: Dynamic computation graphs and autograd design
- [micrograd](https://github.com/karpathy/micrograd): Minimalistic autograd engine by Andrej Karpathy
- [tinygrad](https://github.com/tinygrad/tinygrad): Small neural network framework

Built with CHICKEN Scheme and powered by YASOS and BLAS.

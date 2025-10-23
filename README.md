# NanoGrad: Automatic Differentiation Framework for CHICKEN Scheme

A lightweight, YASOS-based automatic differentiation and neural
network framework for CHICKEN Scheme, featuring BLAS-accelerated
operations and a clean functional API.

## Features

- **Automatic Differentiation**: Reverse-mode autodiff with topological sorting for correct gradient computation
- **BLAS Integration**: High-performance linear algebra operations using CBLAS
- **YASOS Object System**: Clean, polymorphic object-oriented abstractions
- **Mixed Precision**: Support for both 32-bit (f32) and 64-bit (f64) floating-point
- **Neural Network Layers**: Dense layers, convolutional layers, and sequential containers
- **Activation Functions**: ReLU, Tanh, Sigmoid, Softmax, LeakyReLU, Softplus
- **Optimizers**: SGD (with momentum), Adam, RMSprop
- **Loss Functions**: MSE, Cross-Entropy
- **Advanced Operations**: Convolution, RMSNorm, Layer Normalization

## Installation

```bash
# Install dependencies
chicken-install yasos blas mathh srfi-1 srfi-4 srfi-42 srfi-69

# Clone the repository
git clone https://github.com/iraikov/nanograd.git
cd nanograd

chicken-install

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
   training-data))
```

### Convolutional Neural Network

```scheme
(define cnn
  (make-sequential
   (list
    (make-conv2d-layer 3 32 3 stride: 1 padding: 1 
                       activation: (make-relu) name: "Conv1")
    (make-conv2d-layer 32 64 3 stride: 1 padding: 1 
                       activation: (make-relu) name: "Conv2")
    (make-dense-layer (* 64 8 8) 128 activation: (make-relu) name: "FC1")
    (make-dense-layer 128 10 activation: (make-identity) name: "Output"))
   name: "CNN"))
```

## Architecture

### Module Structure

- **`nanograd-autograd`**: Core automatic differentiation engine
  - Tensor abstraction with YASOS
  - Arithmetic operations (add, sub, mul, div)
  - BLAS operations (matmul, dot, scale)
  - Activation functions
  - Loss functions
  - Gradient computation with cycle detection

- **`nanograd-layer`**: Neural network layer abstractions
  - Dense (fully connected) layers
  - Convolutional layers (2D)
  - Sequential containers
  - Activation function objects

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

#### Activations
```scheme
(relu tensor)              ; ReLU activation
(tanh-op tensor)           ; Hyperbolic tangent
(sigmoid tensor)           ; Sigmoid (logistic)
(sigmoid-stable tensor)    ; Numerically stable sigmoid
(softmax tensor)           ; Softmax normalization
(log-softmax tensor)       ; Log-softmax
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

(make-sequential layers #:key (name "Sequential"))
```

#### Layer Operations
```scheme
(forward layer input)      ; Forward pass
(parameters layer)         ; Get trainable parameters
(zero-grad-layer! layer)   ; Zero all parameter gradients
```

#### Activation Objects
```scheme
(make-relu)                ; ReLU activation
(make-tanh)                ; Tanh activation
(make-sigmoid)             ; Sigmoid activation
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

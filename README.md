# NanoGrad: Automatic Differentiation Framework for CHICKEN Scheme

A lightweight, YASOS-based automatic differentiation and neural
network framework for CHICKEN Scheme, featuring BLAS-accelerated
operations, batch processing support, and a clean functional API.

## Features

- **Automatic Differentiation**: Reverse-mode autodiff with topological sorting for correct gradient computation
- **Batch Processing**: Native support for batched operations across layers and loss functions
- **BLAS Integration**: High-performance linear algebra operations using CBLAS
- **YASOS Object System**: Clean, polymorphic object-oriented abstractions
- **Mixed Precision**: Support for both 32-bit (f32) and 64-bit (f64) floating-point
- **Neural Network Layers**: Dense layers with batch support, convolutional layers (3D/4D), batch normalization, and sequential containers
- **Activation Functions**: ReLU, Tanh, Sigmoid, Softmax (with batch support), LeakyReLU, Softplus, SiLU, GeLU
- **Optimizers**: SGD (with momentum), Adam, RMSprop
- **Loss Functions**: MSE, Cross-Entropy (with batch support)
- **Advanced Operations**: Convolution, RMSNorm (1D/2D), Layer Normalization, Batch Normalization (3D/4D), Global Pooling
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

### Batch Processing

```scheme
;; Batch matrix multiplication
(define X (make-tensor32 (make-f32vector 60) '(10 2 3)))  ; 10 samples, 2x3 each
(define W (make-tensor32 (make-f32vector 12) '(3 4)))     ; Weight matrix 3x4

;; Each of the 10 samples is multiplied by W
(define Y (matmul-op X W))  ; Shape: (10, 2, 4)

;; Batch normalization
(define features (make-tensor32 (make-f32vector (* 32 64 8 8)) '(32 64 8 8)))
(define bn-layer (make-batch-norm-2d 64))

;; Training mode: uses batch statistics
(set-training-mode! bn-layer #t)
(define normalized (forward bn-layer features))  ; Normalized across batch

;; Evaluation mode: uses running statistics
(set-eval-mode! bn-layer)
(define test-normalized (forward bn-layer test-features))
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

### Building a Neural Network with Batch Support

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

;; Training loop with batches
(do ((epoch 1 (+ epoch 1)))
    ((> epoch 10))
  
  ;; Training mode
  (set-training-mode! model #t)
  
  (for-each
   (lambda (batch)
     (let* ((x (car batch))        ; Shape: (batch_size, 784)
            (target (cdr batch))   ; Shape: (batch_size, 10) one-hot
            (pred (forward model x))  ; Shape: (batch_size, 10)
            ;; Softmax and cross-entropy handle batches automatically
            (probs (softmax pred axis: -1))  ; Softmax along last axis
            (loss (cross-entropy-loss probs target reduction: 'mean)))
       
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
    ;; Handles both 3D (C,H,W) and 4D (N,C,H,W) inputs
    (make-conv2d-layer 3 32 3 stride: 1 padding: 1 
                       activation: (make-relu) name: "Conv1")
    (make-batch-norm-2d 32 name: "BN1")  ; Normalizes across batch
    (make-conv2d-layer 32 64 3 stride: 1 padding: 1 
                       activation: (make-relu) name: "Conv2")
    (make-batch-norm-2d 64 name: "BN2")
    ;; Global average pooling: (N,64,H,W) -> (N,64) or (64,H,W) -> (64,)
    (make-flatten name: "Flatten")
    (make-dense-layer 64 128 activation: (make-relu) name: "FC1")
    (make-dense-layer 128 10 activation: (make-identity) name: "Output"))
   name: "CNN"))

;; Forward pass with batched images
(define batch-images (make-tensor32 batch-data '(32 3 32 32)))  ; 32 RGB images
(define predictions (forward cnn batch-images))  ; Shape: (32, 10)
```

## Architecture

### Module Structure

- **`nanograd-autograd`**: Core automatic differentiation engine
  - Tensor abstraction with YASOS
  - Arithmetic operations (add, sub, mul, div)
  - BLAS operations (matmul, dot, scale) with batch support
  - Activation functions (including batched softmax/log-softmax)
  - Loss functions with batch reduction
  - Reduction operations (sum, mean, product, custom reductions)
  - Tensor manipulation (slice, reshape, flatten)
  - Gradient computation with cycle detection

- **`nanograd-layer`**: Neural network layer abstractions
  - Dense (fully connected) layers with 1D/2D input support
  - Convolutional layers (2D) with 3D/4D input support
  - Batch normalization (2D) with 3D/4D input support
  - Global average pooling with 3D/4D support
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
4. **Batch-First Design**: Native batch support throughout the stack
5. **Explicit Gradient Management**: Manual control over backward passes
6. **Mixed Precision**: First-class support for both f32 and f64

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
(matmul-op a b)            ; Matrix multiplication (batch-aware)
(dot-op a b)               ; Dot product
(scale-op tensor scalar)   ; Scalar multiplication
```

#### Reduction Operations
```scheme
(reduce-tensor tensor reducer #:key (compute-gradient #f))
  ; Generic reduction with custom gradient
  
(sum-tensor tensor)        ; Sum all elements
(mean-tensor tensor)       ; Mean of all elements
(product-tensor tensor)    ; Product of all elements
```

#### Tensor Manipulation
```scheme
(slice-tensor tensor start length)  ; Extract slice along first dimension
(reshape tensor new-shape)           ; Reshape tensor
(flatten-tensor tensor)              ; Flatten to 1D
```

#### Activations (Batch-Aware)
```scheme
(relu tensor)              ; ReLU activation
(tanh-op tensor)           ; Hyperbolic tangent
(sigmoid tensor)           ; Sigmoid (logistic)
(sigmoid-stable tensor)    ; Numerically stable sigmoid

;; Batch-aware softmax
(softmax tensor #:key (axis -1))
  ; 1D: (n_classes,) -> standard softmax
  ; 2D: (batch_size, n_classes) -> softmax along axis

(log-softmax tensor #:key (axis -1))
  ; More stable than log(softmax(x))

(silu tensor)              ; SiLU
(gelu tensor)              ; GeLU
(leaky-relu tensor #:key (alpha 0.01))
(softplus tensor #:key (beta 1.0))
```

#### Loss Functions (Batch-Aware)
```scheme
(mse-loss pred target #:key (reduction 'mean))
  ; reduction: 'mean (average over batch) or 'sum

(cross-entropy-loss pred target #:key (reduction 'mean) (from-logits #f))
  ; Supports both:
  ;   - 1D: (n_classes,) for single sample
  ;   - 2D: (batch_size, n_classes) for batches
  ; target can be one-hot or class indices
  ; from-logits: if true, applies log-softmax first
```

#### Normalization (Batch-Aware)
```scheme
(rmsnorm x weight #:key (epsilon 1e-5))
  ; 1D: (d_model,) -> standard RMSNorm
  ; 2D: (batch_size, d_model) -> RMSNorm per batch element

(l2-normalize tensor #:key (axis #f) (epsilon 1e-8))
  ; axis=#f: normalize entire tensor
  ; axis=n: normalize along specific axis (for 2D tensors)
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
  ; Supports:
  ;   1D input: (input_size,) -> (output_size,)
  ;   2D input: (batch_size, input_size) -> (batch_size, output_size)

(make-conv2d-layer in-channels out-channels kernel-size
                   #:key (stride 1) (padding 0)
                         (activation (make-identity))
                         (dtype 'f32)
                         (name "Conv2D"))
  ; Supports:
  ;   3D input: (C, H, W) -> (C_out, H_out, W_out)
  ;   4D input: (N, C, H, W) -> (N, C_out, H_out, W_out)

(make-batch-norm-2d num-features
                    #:key (epsilon 1e-5) (momentum 0.1)
                          (dtype 'f32)
                          (name "BatchNorm2d"))
  ; Supports:
  ;   3D input: (C, H, W) - treats as batch of 1
  ;   4D input: (N, C, H, W) - normalizes across batch dimension

(make-sequential layers #:key (name "Sequential"))
```

#### Global Average Pooling (Batch-Aware)
```scheme
(global-avg-pool2d input)
  ; 3D: (C, H, W) -> (C,)
  ; 4D: (N, C, H, W) -> (N, C)
  ; Averages over spatial dimensions
```

#### Layer Operations
```scheme
(forward layer input)      ; Forward pass (batch-aware)
(parameters layer)         ; Get trainable parameters
(zero-grad-layer! layer)   ; Zero all parameter gradients

;; Training/Evaluation Mode Control
(set-training-mode! layer training?)  ; Set training mode
(set-eval-mode! layer)                ; Set evaluation mode
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

### Batch Processing with Dense Layers

```scheme
(import nanograd-autograd nanograd-layer)

;; Create a batch of inputs
(define batch-size 32)
(define input-dim 784)
(define batch-data (make-f32vector (* batch-size input-dim)))

;; ... fill with data ...

(define batch-input (make-tensor32 batch-data (list batch-size input-dim)))

;; Dense layer automatically handles batches
(define layer (make-dense-layer input-dim 128 activation: (make-relu)))
(define output (forward layer batch-input))  ; Shape: (32, 128)

;; Batch normalization example
(define features (make-tensor32 (make-f32vector (* 32 128)) '(32 128)))
(define normalized (rmsnorm features gamma))  ; Normalized per batch element
```

### Batched Softmax and Cross-Entropy

```scheme
;; Batch of logits
(define logits (make-tensor32 (make-f32vector (* 32 10)) '(32 10)))
(define targets (make-tensor32 target-data '(32 10)))  ; One-hot encoded

;; Softmax along class dimension (axis 1)
(define probs (softmax logits axis: -1))  ; Shape: (32, 10), sums to 1 per row

;; Cross-entropy handles batches automatically
(define loss (cross-entropy-loss probs targets reduction: 'mean))

;; Alternative: use from-logits for numerical stability
(define loss-stable (cross-entropy-loss logits targets 
                                        from-logits: #t 
                                        reduction: 'mean))
```

### Complete Training Example with Batches

```scheme
(import nanograd-autograd nanograd-layer nanograd-optimizer)

;; Define model
(define model
  (make-sequential
   (list
    (make-dense-layer 784 256 activation: (make-relu))
    (make-dense-layer 256 128 activation: (make-relu))
    (make-dense-layer 128 10 activation: (make-identity)))
   name: "BatchMLP"))

(define optimizer (make-adam (parameters model) learning-rate: 0.001))

;; Training with batches
(define (train-epoch train-batches)
  (set-training-mode! model #t)
  
  (for-each
   (lambda (batch)
     (let* ((x (car batch))        ; Shape: (batch_size, 784)
            (y (cdr batch))        ; Shape: (batch_size, 10)
            (logits (forward model x))
            (loss (cross-entropy-loss logits y 
                                      from-logits: #t 
                                      reduction: 'mean)))
       
       (backward! loss)
       (step! optimizer)
       (zero-grad-layer! model)))
   train-batches))

;; Evaluation
(define (evaluate test-batches)
  (set-eval-mode! model)
  
  (let ((total-correct 0)
        (total-samples 0))
    (for-each
     (lambda (batch)
       (let* ((x (car batch))
              (y (cdr batch))
              (batch-size (car (tensor-shape x)))
              (logits (forward model x))
              (probs (softmax logits axis: -1)))
         
         ;; Count correct predictions per batch
         ;; (implementation details omitted)
         ))
     test-batches)
    
    (/ total-correct total-samples)))
```

### Convolutional Network with Batch Support

```scheme
(define cnn
  (make-sequential
   (list
    (make-conv2d-layer 3 32 3 padding: 1 activation: (make-relu))
    (make-batch-norm-2d 32)
    (make-conv2d-layer 32 64 3 padding: 1 activation: (make-relu))
    (make-batch-norm-2d 64)
    (make-flatten)
    (make-dense-layer (* 64 32 32) 10))
   name: "CNN"))

;; Process batch of images
(define batch-images (make-tensor32 image-data '(16 3 32 32)))  ; 16 images
(set-training-mode! cnn #t)
(define predictions (forward cnn batch-images))  ; Shape: (16, 10)
```

## Performance Notes

- NanoGrad uses BLAS for matrix operations, including batched GEMM
- Batch operations are significantly more efficient than processing samples individually
- Use f32 (32-bit) tensors when 64-bit precision is not required
- The framework detects computation graph cycles
- Batch normalization adds minimal overhead and significantly improves training
- Global average pooling reduces parameters without sacrificing performance

## Batch Processing Best Practices

1. Always use batches during training for better performance and stable gradients
2. Set appropriate batch sizes (typically 16-256 depending on memory)
3. Use batch normalization for deeper networks (>10 layers)
4. Switch to eval mode during validation/testing to use running statistics
5. Prefer global average pooling over large fully-connected layers

## Limitations

- CPU-only (no GPU support)
- No automatic batching (must manually create batches)
- Limited built-in layer types
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

LGPLv3 License - see LICENSE file for details

## Acknowledgments

This framework is inspired by:
- **PyTorch**: Dynamic computation graphs, autograd design, and batch-first conventions
- [micrograd](https://github.com/karpathy/micrograd): Minimalistic autograd engine by Andrej Karpathy
- [tinygrad](https://github.com/tinygrad/tinygrad): Small neural network framework

Built with CHICKEN Scheme and powered by YASOS and BLAS.

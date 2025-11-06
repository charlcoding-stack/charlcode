# Charl Programming Language

A statically-typed programming language for machine learning research and development, implemented in Rust with native tensor operations and automatic differentiation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)

---

## Overview

Charl is a domain-specific language designed for machine learning applications. It provides:

- **Native tensor operations** with zero-copy semantics
- **Automatic differentiation** for gradient computation
- **Static type system** with type inference
- **Rust-based implementation** for memory safety and performance
- **34 built-in ML functions** covering tensors, neural networks, and optimization
- **Minimal dependencies** - Core functionality requires no external libraries

The language is designed for researchers and practitioners who need predictable performance and low-level control over ML operations.

---

## Performance Characteristics

Benchmark results for MNIST classifier training (1,000 samples, 5 epochs):

| Implementation | Total Time | Throughput | Memory Usage |
|----------------|-----------|------------|--------------|
| Charl | 414ms | 12,064 samples/s | Minimal |
| PyTorch (CPU) | 9,255ms | 540 samples/s | Standard |

**Performance factors:**
- Compiled ahead-of-time via Rust
- No interpreter overhead
- Zero-copy tensor operations
- Direct LLVM-optimized machine code

Full benchmark methodology and results available in [`benchmarks/`](benchmarks/).

---

## Installation

### Prerequisites
- Rust 1.70 or higher ([rustup.rs](https://rustup.rs/))
- Git

### Build from Source

```bash
git clone https://github.com/charlcoding-stack/charlcode.git
cd charlcode
cargo build --release
```

### Install Binary

**Linux/macOS:**
```bash
sudo cp target/release/charl /usr/local/bin/
charl --version
```

**Windows:**
```powershell
copy target\release\charl.exe C:\Windows\System32\
charl --version
```

---

## Quick Start

### Basic Neural Network Training

Create a file `example.ch`:

```charl
// XOR problem: 2-layer neural network
let x = tensor([1.0, 0.0])
let y = tensor([1.0])

// Network architecture: 2 -> 4 -> 1
let w1 = tensor_randn([2, 4])
let b1 = tensor_zeros([4])
let w2 = tensor_randn([4, 1])
let b2 = tensor_zeros([1])

// Training parameters
let lr = 0.5
let epochs = 100
let epoch = 0

while epoch < epochs {
    // Forward pass
    let z1 = nn_linear(x, w1, b1)
    let h1 = nn_relu(z1)
    let z2 = nn_linear(h1, w2, b2)
    let pred = nn_sigmoid(z2)
    let loss = loss_mse(pred, y)

    // Backward pass
    let grad_pred = autograd_compute_mse_grad(pred, y)
    let grad_z2 = autograd_compute_sigmoid_grad(pred, grad_pred)
    let grads_2 = autograd_compute_linear_grad(h1, w2, b2, grad_z2)
    let grad_z1 = autograd_compute_relu_grad(z1, grads_2.0)
    let grads_1 = autograd_compute_linear_grad(x, w1, b1, grad_z1)

    // Parameter updates
    w1 = optim_sgd_step(w1, grads_1.1, lr)
    b1 = optim_sgd_step(b1, grads_1.2, lr)
    w2 = optim_sgd_step(w2, grads_2.1, lr)
    b2 = optim_sgd_step(b2, grads_2.2, lr)

    epoch = epoch + 1
}

print("Training complete")
```

Run:
```bash
charl run example.ch
```

---

## Language Reference

### Type System

Charl uses static typing with inference:

```charl
let x: int = 42
let y: float = 3.14
let s: string = "text"
let b: bool = true
let arr: [float] = [1.0, 2.0, 3.0]
let tup: (int, float) = (1, 2.0)
```

### Control Flow

```charl
// Conditional
if x > 0 {
    print("positive")
} else {
    print("non-positive")
}

// Loops
while i < 10 {
    i = i + 1
}

// Pattern matching
let result = match x {
    0 => "zero"
    1 => "one"
    _ => "other"
}
```

### Functions

```charl
fn linear(x: float, m: float, b: float) -> float {
    return m * x + b
}

let y = linear(2.0, 3.0, 1.0)  // 7.0
```

---

## ML Function Library

### Tensor Operations (13 functions)

**Creation:**
```charl
tensor([1.0, 2.0, 3.0])           // From array literal
tensor_zeros([3, 3])              // Zero-initialized matrix
tensor_ones([2, 4])               // One-initialized matrix
tensor_randn([10, 10])            // Random normal distribution
```

**Arithmetic:**
```charl
tensor_add(a, b)                  // Element-wise addition
tensor_sub(a, b)                  // Element-wise subtraction
tensor_mul(a, b)                  // Element-wise/scalar multiplication
tensor_div(a, b)                  // Element-wise/scalar division
tensor_matmul(A, B)               // Matrix multiplication
```

**Manipulation:**
```charl
tensor_reshape(t, [2, 3])         // Reshape dimensions
tensor_transpose(t)               // Transpose (2D only)
tensor_sum(t)                     // Sum reduction
tensor_mean(t)                    // Mean reduction
```

### Autograd Functions (4 functions)

```charl
tensor_requires_grad(t, true)     // Enable gradient tracking
tensor_grad(t)                    // Access computed gradients
tensor_set_grad(t, grad_values)   // Manually set gradients
tensor_zero_grad(t)               // Reset gradients to zero
```

### Neural Network Layers (5 functions)

**Linear Layer:**
```charl
nn_linear(input, weight, bias)    // Fully connected: y = xW + b
```

**Activation Functions:**
```charl
nn_relu(x)                        // ReLU: max(0, x)
nn_sigmoid(x)                     // Sigmoid: 1/(1+exp(-x))
nn_tanh(x)                        // Hyperbolic tangent
nn_softmax(x)                     // Softmax normalization
```

### Loss Functions (2 functions)

```charl
loss_mse(pred, target)            // Mean squared error
loss_cross_entropy(pred, target)  // Cross-entropy loss
```

### Optimization (4 functions)

**Stochastic Gradient Descent:**
```charl
optim_sgd_step(params, grads, lr)
```

**SGD with Momentum:**
```charl
let result = optim_sgd_momentum_step(params, grads, velocity, lr, momentum)
params = result.0
velocity = result.1
```

**Adam Optimizer:**
```charl
let result = optim_adam_step(params, grads, m, v, t, lr, beta1, beta2)
params = result.0
m = result.1
v = result.2
```

**Gradient Clipping:**
```charl
tensor_clip_grad(grads, max_norm)
```

### Backpropagation Helpers (4 functions)

```charl
// Linear layer gradients
let grads = autograd_compute_linear_grad(input, weight, bias, output_grad)
let grad_input = grads.0
let grad_weight = grads.1
let grad_bias = grads.2

// Activation gradients
autograd_compute_relu_grad(input, output_grad)
autograd_compute_sigmoid_grad(output, output_grad)

// Loss gradients
autograd_compute_mse_grad(pred, target)
```

---

## Complete Examples

### Example 1: Parameter Optimization

Minimize f(x) = (x - 5)² using gradient descent:

```charl
let x = tensor([0.0])
let lr = 0.1
let epochs = 50
let epoch = 0

while epoch < epochs {
    // Forward
    let x_val = tensor_sum(x)
    let loss = (x_val - 5.0) * (x_val - 5.0)

    // Gradient: df/dx = 2(x - 5)
    let grad = 2.0 * (x_val - 5.0)

    // Update
    x = optim_sgd_step(x, [grad], lr)
    epoch = epoch + 1
}
// Result: x ≈ 5.0
```

### Example 2: Adam Optimizer

Same optimization problem with Adam:

```charl
let x = tensor([0.0])
let m = [0.0]
let v = [0.0]
let t = 0

let lr = 0.5
let beta1 = 0.9
let beta2 = 0.999
let epochs = 50
let epoch = 0

while epoch < epochs {
    t = t + 1

    let x_val = tensor_sum(x)
    let grad = 2.0 * (x_val - 5.0)

    let result = optim_adam_step(x, [grad], m, v, t, lr, beta1, beta2)
    x = result.0
    m = result.1
    v = result.2

    epoch = epoch + 1
}
```

### Example 3: Multi-Layer Network

Full backpropagation through 2-layer network:

```charl
// Network: 2 -> 4 -> 1
let w1 = tensor_randn([2, 4])
let b1 = tensor_zeros([4])
let w2 = tensor_randn([4, 1])
let b2 = tensor_zeros([1])

// Training sample
let x = tensor([1.0, 0.0])
let y = tensor([1.0])

let lr = 0.5
let epochs = 100
let epoch = 0

while epoch < epochs {
    // Forward
    let z1 = nn_linear(x, w1, b1)
    let h1 = nn_relu(z1)
    let z2 = nn_linear(h1, w2, b2)
    let pred = nn_sigmoid(z2)
    let loss = loss_mse(pred, y)

    // Backward
    let grad_pred = autograd_compute_mse_grad(pred, y)
    let grad_z2 = autograd_compute_sigmoid_grad(pred, grad_pred)
    let grads_2 = autograd_compute_linear_grad(h1, w2, b2, grad_z2)
    let grad_z1 = autograd_compute_relu_grad(z1, grads_2.0)
    let grads_1 = autograd_compute_linear_grad(x, w1, b1, grad_z1)

    // Update
    w1 = optim_sgd_step(w1, grads_1.1, lr)
    b1 = optim_sgd_step(b1, grads_1.2, lr)
    w2 = optim_sgd_step(w2, grads_2.1, lr)
    b2 = optim_sgd_step(b2, grads_2.2, lr)

    epoch = epoch + 1
}
```

Additional examples available in [`examples/`](examples/):
- `tensor_basic.ch` - Tensor operations
- `tensor_matmul.ch` - Linear algebra
- `tensor_autograd.ch` - Gradient tracking
- `neural_network.ch` - Network construction
- `training_simple.ch` - Basic optimization
- `training_backprop.ch` - Full training loop

---

## Architecture

```
┌─────────────────────────────────────────┐
│         Charl Language Stack            │
├─────────────────────────────────────────┤
│  Frontend                               │
│  ├─ Lexer (Tokenization)               │
│  ├─ Parser (AST construction)          │
│  └─ Type system (Static checking)      │
├─────────────────────────────────────────┤
│  Interpreter                            │
│  ├─ Tree-walking execution             │
│  ├─ Environment management             │
│  └─ Builtin dispatch (34 functions)    │
├─────────────────────────────────────────┤
│  Backend (Rust)                         │
│  ├─ Tensor operations                  │
│  ├─ Autograd system                    │
│  ├─ Neural network primitives          │
│  ├─ Optimizers (SGD, Momentum, Adam)   │
│  └─ GPU support (wgpu, optional)       │
└─────────────────────────────────────────┘
```

---

## Project Structure

```
charlcode/
├── src/
│   ├── lexer/              # Tokenization
│   ├── parser/             # Syntax analysis
│   ├── ast/                # Abstract syntax tree
│   ├── types/              # Type system
│   ├── interpreter/        # Execution engine
│   ├── autograd/           # Automatic differentiation
│   ├── tensor_builtins.rs  # ML function implementations
│   ├── nn/                 # Neural network layers
│   ├── optim/              # Optimization algorithms
│   ├── gpu/                # GPU acceleration
│   └── main.rs             # CLI interface
├── examples/               # Example programs
├── benchmarks/             # Performance benchmarks
├── vscode-charl/           # VS Code language extension
└── README.md
```

---

## CLI Usage

```bash
# Execute a script
charl run script.ch

# Interactive REPL
charl repl

# Display version
charl version

# Show help
charl --help
```

---

## Development

### Building

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Lint
cargo clippy

# Format
cargo fmt
```

### Running Tests

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test '*'

# Specific test
cargo test test_tensor_ops
```

---

## Contributing

Contributions are welcome. Please see:
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community standards

### Contribution Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Implement changes with tests
4. Run test suite: `cargo test`
5. Run linter: `cargo clippy`
6. Format code: `cargo fmt`
7. Commit changes: `git commit -m 'Description'`
8. Push branch: `git push origin feature-name`
9. Open pull request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## References

- **Repository**: [github.com/charlcoding-stack/charlcode](https://github.com/charlcoding-stack/charlcode)
- **Issues**: [github.com/charlcoding-stack/charlcode/issues](https://github.com/charlcoding-stack/charlcode/issues)
- **Discussions**: [github.com/charlcoding-stack/charlcode/discussions](https://github.com/charlcoding-stack/charlcode/discussions)

---

## Technical Details

### Tensor Implementation

Tensors are implemented as contiguous memory blocks with shape metadata. Operations use:
- **BLAS-like routines** for matrix operations
- **SIMD vectorization** where applicable
- **Zero-copy views** for reshaping/transposing when possible

### Gradient Computation

The autograd system uses:
- **Computation graph** tracking (optional)
- **Manual gradient functions** for each operation
- **Reverse-mode differentiation** for backpropagation

### Memory Management

- **Stack allocation** for small tensors
- **Heap allocation** with reference counting for large tensors
- **No garbage collection** - deterministic cleanup via Rust's ownership

### Type System

- **Static typing** with Hindley-Milner style inference
- **No implicit conversions** - explicit casts required
- **Parametric polymorphism** for generic functions

---

**Built by the Charl community**

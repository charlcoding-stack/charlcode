# Charl

A programming language for AI and machine learning development.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.91.0-orange.svg)](https://www.rust-lang.org/)

## Overview

Charl is designed for building AI and machine learning systems with a focus on:

- **Automatic Differentiation** - Reverse-mode automatic differentiation built into the language
- **Neural Network Layers** - Dense, Conv2D, RNN, LSTM, and custom layers
- **GPU Acceleration** - WGPU backend for compute operations
- **Knowledge Graphs** - Built-in support for graph-based reasoning
- **Type System** - Static typing with inference

## Installation

### Option 1: Pre-compiled Binaries (Recommended)

Download the latest release for your platform from the [releases page](https://github.com/charlcoding-stack/charlcode/releases).

**Linux/macOS:**
```bash
# Download and extract
tar -xzf charl-*.tar.gz

# Move to PATH
sudo mv charl /usr/local/bin/

# Verify installation
charl --version
```

**Windows:**
```powershell
# Extract the zip file
# Move charl.exe to a directory in your PATH
# Verify installation
charl.exe --version
```

### Option 2: Build from Source

```bash
# Clone repository
git clone https://github.com/charlcoding-stack/charlcode.git
cd charlcode

# Build
cargo build --release

# Install (optional)
sudo cp target/release/charl /usr/local/bin/

# Verify
charl --version
```

## Quick Start

### Example

```rust
// hello.ch - Neural network for classification
model = Sequential([
    Dense(784, 512),
    ReLU(),
    Dense(512, 10),
    Softmax()
])

optimizer = Adam(lr=0.001)
loss = CrossEntropy()

for epoch in 1..10 {
    for (x, y) in train_loader {
        output = model.forward(x)
        loss_val = loss(output, y)
        loss_val.backward()
        optimizer.step()
    }
}
```

## Features

### Tensor Operations

```rust
// Automatic differentiation
x = Tensor([2.0, 3.0], requires_grad=true)
y = x * 2 + 1
z = y.sum()

z.backward()  // Compute gradients
print(x.grad())  // [2.0, 2.0]
```

### Neural Network Layers

- Dense (fully connected)
- Conv2D (convolutional)
- RNN, LSTM (recurrent)
- Dropout, BatchNorm
- Activation functions (ReLU, Sigmoid, Tanh, Softmax)

### Optimizers

- SGD with momentum
- Adam
- RMSprop
- AdaGrad

## Documentation

- **Website**: [charlbase.org](https://charlbase.org)
- **Getting Started**: [charlbase.org/docs/getting-started.html](https://charlbase.org/docs/getting-started.html)
- **Language Reference**: [charlbase.org/docs/language-reference.html](https://charlbase.org/docs/language-reference.html)
- **API Reference**: [charlbase.org/docs/api-reference.html](https://charlbase.org/docs/api-reference.html)
- **Examples**: [charlbase.org/examples.html](https://charlbase.org/examples.html)

## Building from Source

### Prerequisites

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**macOS:**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Windows:**
```powershell
# Install Rust from: https://rustup.rs/
# Install Visual Studio Build Tools
```

### Build Commands

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test

# Run specific test
cargo test test_tensor_operations

# Check code
cargo clippy

# Format code
cargo fmt
```

## CLI Commands

```bash
# Run a Charl script
charl run script.ch

# Start REPL
charl repl

# Build standalone executable
charl build script.ch -o output

# Show version
charl --version

# Show help
charl --help
```

## Project Structure

```
charlcode/
├── src/
│   ├── lexer/          # Tokenization
│   ├── parser/         # Syntax analysis
│   ├── ast/            # Abstract syntax tree
│   ├── types/          # Type system
│   ├── interpreter/    # Execution engine
│   ├── autograd/       # Automatic differentiation
│   ├── nn/             # Neural network layers
│   ├── optim/          # Optimizers
│   └── gpu/            # Hardware acceleration
├── examples/           # Example programs
├── benches/           # Benchmarks
└── tests/             # Integration tests
```

## Contributing

We welcome contributions! Please see:

- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community standards
- [CLA.md](CLA.md) - Contributor License Agreement

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes and add tests
4. Run tests (`cargo test`)
5. Run linter (`cargo clippy`)
6. Format code (`cargo fmt`)
7. Commit your changes
8. Push to your fork
9. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Community

- **GitHub Issues**: [Bug reports and feature requests](https://github.com/charlcoding-stack/charlcode/issues)
- **Discussions**: [General questions and discussions](https://github.com/charlcoding-stack/charlcode/discussions)
- **Website**: [charlbase.org](https://charlbase.org)

## Acknowledgments

- Rust community for excellent tooling
- LLVM project for optimization infrastructure
- wgpu team for cross-platform GPU compute
- PyTorch and TensorFlow for inspiration

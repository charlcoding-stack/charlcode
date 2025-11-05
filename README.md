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

### Quick Install (Recommended)

**Linux/macOS:**
```bash
curl -sSf https://charlbase.org/install.sh | sh
```

**Windows (PowerShell as Administrator):**
```powershell
irm https://charlbase.org/install.ps1 | iex
```

The installer automatically:
- Detects your system and architecture
- Installs Rust if needed (Linux/macOS only)
- Compiles and installs Charl
- Adds Charl to your PATH
- Installs VS Code extension (if VS Code is detected)

### Option 2: Manual Installation

**Linux/macOS:**
```bash
# Download and extract
curl -L https://github.com/charlcoding-stack/charlcode/releases/latest/download/charl-linux-x86_64.tar.gz | tar xz

# Move to PATH
sudo mv charl /usr/local/bin/

# Verify installation
charl --version
```

**Windows:**
```powershell
# Download from GitHub Releases
Invoke-WebRequest -Uri "https://github.com/charlcoding-stack/charlcode/releases/latest/download/charl-windows-x86_64.zip" -OutFile charl.zip

# Extract
Expand-Archive charl.zip -DestinationPath "$env:USERPROFILE\.charl\bin"

# Add to PATH (run as Administrator)
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
[Environment]::SetEnvironmentVariable("Path", "$userPath;$env:USERPROFILE\.charl\bin", "User")

# Verify (restart terminal first)
charl --version
```

### Option 3: Build from Source

**Requirements:**
- Rust 1.70+ ([rustup.rs](https://rustup.rs/))
- Git

**Linux/macOS:**
```bash
# Clone repository
git clone https://github.com/charlcoding-stack/charlcode.git
cd charlcode

# Build release
cargo build --release

# Install system-wide
sudo cp target/release/charl /usr/local/bin/

# Verify
charl --version
```

**Windows:**
```powershell
# Clone repository
git clone https://github.com/charlcoding-stack/charlcode.git
cd charlcode

# Build release
cargo build --release

# Binary location
.\target\release\charl.exe --version

# Install to user directory
Copy-Item target\release\charl.exe "$env:USERPROFILE\.charl\bin\"
# Add to PATH as shown in Option 2
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

## Editor Support

### VS Code Extension

Charl includes an official VS Code extension with full language support:

**Features:**
- 🎨 Syntax highlighting for all Charl keywords and types
- 📝 22+ code snippets (fn, match, for, tuple, etc.)
- ⚙️ Auto-indentation and bracket matching
- 🔧 Auto-closing pairs for brackets and quotes
- 📂 Code folding regions

**Installation:**

The extension is automatically installed when using `install.sh`. If you need to install it manually:

```bash
# Copy extension to VS Code extensions directory
cp -r vscode-charl ~/.vscode/extensions/charl-lang.charl-1.0.0

# Restart VS Code
```

**Usage:**

Once installed, all `.ch` files will have:
- Colorized syntax highlighting
- IntelliSense code snippets (type `fn` + Tab, `match` + Tab, etc.)
- Automatic indentation
- Bracket matching and auto-closing

For more details, see: [vscode-charl/README.md](vscode-charl/README.md)

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
├── vscode-charl/       # VS Code extension
│   ├── syntaxes/       # TextMate grammar
│   ├── snippets/       # Code templates
│   └── package.json    # Extension manifest
├── examples/           # Example programs
├── benches/            # Benchmarks
├── tests/              # Integration tests
└── install.sh          # Installation script
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

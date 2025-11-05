#!/bin/bash
# Charl Language Installer
#
# Usage:
#   curl -sSf https://charlbase.org/install.sh | sh
#   OR
#   wget -qO- https://charlbase.org/install.sh | sh

set -e

CHARL_VERSION="0.1.0"
INSTALL_DIR="$HOME/.charl"
BIN_DIR="$INSTALL_DIR/bin"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           Charl Language Installer v${CHARL_VERSION}              ║"
echo "║   Revolutionary AI/ML Programming Language                ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Detect OS and Architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

echo "🔍 Detected system:"
echo "   OS: $OS"
echo "   Architecture: $ARCH"
echo ""

# Check if Rust is installed (needed for from-source installation)
if ! command -v cargo &> /dev/null; then
    echo "⚠️  Rust not found. Installing Rust first..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "✅ Rust installed successfully"
else
    echo "✅ Rust found: $(rustc --version)"
fi
echo ""

# Create installation directory
echo "📁 Creating installation directory..."
mkdir -p "$BIN_DIR"

# For now, build from source (in production, download pre-built binaries)
echo "🔨 Building Charl from source..."
echo "   (In production, this would download pre-built binaries)"
echo ""

# Clone repository (or in production, download release tarball)
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "📥 Downloading Charl source code..."
# In production:
# curl -L "https://github.com/YOUR_USERNAME/charl/archive/refs/tags/v${CHARL_VERSION}.tar.gz" | tar xz

# For now, copy from current directory (if running locally)
if [ -f "../Cargo.toml" ]; then
    echo "   Using local source"
    cp -r ../* .
else
    echo "❌ Error: No Cargo.toml found. Cannot build."
    exit 1
fi

echo "🔧 Compiling Charl (this may take a few minutes)..."
cargo build --release --bin charl

echo "📦 Installing binary..."
cp target/release/charl "$BIN_DIR/"

# Add to PATH
echo ""
echo "🔗 Setting up PATH..."

SHELL_RC=""
case "$SHELL" in
    */bash)
        SHELL_RC="$HOME/.bashrc"
        ;;
    */zsh)
        SHELL_RC="$HOME/.zshrc"
        ;;
    */fish)
        SHELL_RC="$HOME/.config/fish/config.fish"
        ;;
    *)
        SHELL_RC="$HOME/.profile"
        ;;
esac

# Check if already in PATH
if ! grep -q "export PATH=\"\$HOME/.charl/bin:\$PATH\"" "$SHELL_RC" 2>/dev/null; then
    echo "export PATH=\"\$HOME/.charl/bin:\$PATH\"" >> "$SHELL_RC"
    echo "✅ Added Charl to PATH in $SHELL_RC"
else
    echo "✅ Charl already in PATH"
fi

# Install VS Code extension (if VS Code is installed)
echo ""
echo "🎨 Checking for VS Code..."
if command -v code &> /dev/null; then
    echo "✅ VS Code found"
    echo "📦 Installing Charl language extension for VS Code..."

    VSCODE_EXT_DIR="$HOME/.vscode/extensions/charl-lang.charl-1.0.0"

    # Check if vscode-charl directory exists in the installation source
    if [ -d "vscode-charl" ]; then
        mkdir -p "$HOME/.vscode/extensions"
        cp -r vscode-charl "$VSCODE_EXT_DIR"
        echo "✅ VS Code extension installed"
        echo "   • Syntax highlighting for .ch files"
        echo "   • Auto-indentation and bracket matching"
        echo "   • 22 code snippets ready to use"
        echo "   • Restart VS Code to activate the extension"
    else
        echo "⚠️  VS Code extension not found in installation package"
        echo "   You can download it from: https://charlbase.org/downloads"
    fi
else
    echo "ℹ️  VS Code not found - skipping extension installation"
    echo "   Install VS Code from: https://code.visualstudio.com/"
    echo "   Then install the Charl extension from: https://charlbase.org/downloads"
fi

# Cleanup
cd ..
rm -rf "$TEMP_DIR"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║               Installation Complete! 🎉                   ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Charl has been installed to: $BIN_DIR/charl"
echo ""
echo "📚 To get started:"
echo "   1. Restart your terminal (or run: source $SHELL_RC)"
echo "   2. Verify installation: charl --version"
echo "   3. Try the examples: charl run examples/hello.ch"
echo ""
if command -v code &> /dev/null && [ -d "$HOME/.vscode/extensions/charl-lang.charl-1.0.0" ]; then
echo "🎨 VS Code Extension:"
echo "   • Restart VS Code to activate syntax highlighting"
echo "   • Open any .ch file to see colorized code"
echo "   • Use snippets: type 'fn', 'match', 'for' and press Tab"
echo ""
fi
echo "📖 Learn more:"
echo "   • Documentation: https://charlbase.org/docs"
echo "   • Website: https://charlbase.org"
echo "   • GitHub: https://github.com/YOUR_USERNAME/charl"
echo "   • Examples: https://charlbase.org/examples"
echo ""
echo "🚀 Happy coding with Charl!"
echo ""

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
GITHUB_REPO="charlcoding-stack/charlcode"

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

# Map architecture names
case "$ARCH" in
    x86_64|amd64)
        ARCH="x86_64"
        ;;
    aarch64|arm64)
        ARCH="arm64"
        ;;
    *)
        echo "❌ Unsupported architecture: $ARCH"
        echo "   Supported: x86_64, aarch64/arm64"
        exit 1
        ;;
esac

# Determine binary URL based on platform
case "$OS" in
    linux)
        BINARY_FILE="charl-linux-${ARCH}.tar.gz"
        ;;
    darwin)
        BINARY_FILE="charl-macos-${ARCH}.tar.gz"
        ;;
    *)
        echo "❌ Unsupported OS: $OS"
        echo "   Supported: Linux, macOS"
        echo "   For Windows, use: irm https://charlbase.org/install.ps1 | iex"
        exit 1
        ;;
esac

BINARY_URL="https://github.com/${GITHUB_REPO}/releases/download/v${CHARL_VERSION}/${BINARY_FILE}"

# Create installation directory
echo "📁 Creating installation directory..."
mkdir -p "$BIN_DIR"

# Try to download pre-built binary
echo "📥 Downloading Charl binary..."
echo "   URL: $BINARY_URL"
echo ""

TEMP_DIR=$(mktemp -d)
DOWNLOAD_PATH="$TEMP_DIR/$BINARY_FILE"

if curl -fL "$BINARY_URL" -o "$DOWNLOAD_PATH" 2>/dev/null; then
    echo "✅ Download successful"
    echo ""

    # Extract binary
    echo "📦 Extracting binary..."
    tar -xzf "$DOWNLOAD_PATH" -C "$BIN_DIR/"

    if [ -f "$BIN_DIR/charl" ]; then
        chmod +x "$BIN_DIR/charl"
        echo "✅ Binary installed successfully"
    else
        echo "❌ Error: Binary not found after extraction"
        exit 1
    fi

    # Cleanup
    rm -rf "$TEMP_DIR"
else
    echo "⚠️  Pre-built binary not available for your platform"
    echo "   Falling back to building from source..."
    echo ""

    # Check if Rust is installed
    if ! command -v cargo &> /dev/null; then
        echo "📦 Rust is required to build from source"
        echo "   Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
        echo "✅ Rust installed successfully"
    else
        echo "✅ Rust found: $(rustc --version)"
    fi
    echo ""

    # Clone repository
    cd "$TEMP_DIR"
    echo "📥 Cloning Charl repository..."
    if command -v git &> /dev/null; then
        git clone "https://github.com/${GITHUB_REPO}.git" .
    else
        echo "❌ Error: git is required to build from source"
        echo "   Please install git and try again"
        exit 1
    fi

    echo "🔧 Compiling Charl (this may take a few minutes)..."
    cargo build --release --bin charl

    echo "📦 Installing binary..."
    cp target/release/charl "$BIN_DIR/"
    chmod +x "$BIN_DIR/charl"

    # Cleanup
    cd ..
    rm -rf "$TEMP_DIR"
fi

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

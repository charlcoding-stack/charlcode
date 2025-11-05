#!/bin/bash
# Setup script para Charl Language en Ubuntu
# Este script instala todas las dependencias necesarias

set -e  # Exit on error

echo "🚀 Charl Language - Setup en Ubuntu"
echo "===================================="
echo ""

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Función para imprimir con color
print_step() {
    echo -e "${BLUE}[PASO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# 1. Actualizar sistema
print_step "Actualizando sistema..."
sudo apt update
sudo apt upgrade -y
print_success "Sistema actualizado"

# 2. Instalar herramientas básicas
print_step "Instalando herramientas básicas..."
sudo apt install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    curl \
    git \
    cmake \
    clang \
    lldb \
    lld
print_success "Herramientas básicas instaladas"

# 3. Instalar LLVM 16 (CRÍTICO para Phase 7)
print_step "Instalando LLVM 16 para Phase 7..."
sudo apt install -y \
    llvm-16 \
    llvm-16-dev \
    llvm-16-runtime \
    libclang-16-dev \
    clang-16 \
    lld-16
print_success "LLVM 16 instalado"

# 4. Instalar soporte GPU/Vulkan (CRÍTICO para Phase 8)
print_step "Instalando soporte GPU/Vulkan para Phase 8..."
sudo apt install -y \
    vulkan-tools \
    mesa-vulkan-drivers \
    libvulkan-dev \
    vulkan-validationlayers
print_success "Vulkan instalado"

# 5. Instalar Rust (si no está)
if ! command -v rustc &> /dev/null; then
    print_step "Instalando Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    print_success "Rust instalado"
else
    print_success "Rust ya está instalado"
fi

# Asegurar que Rust está en PATH
source $HOME/.cargo/env

# 6. Instalar componentes adicionales de Rust
print_step "Instalando componentes de Rust..."
rustup component add rustfmt clippy
print_success "Componentes de Rust instalados"

echo ""
echo "=================================="
echo "✅ VERIFICACIÓN DE INSTALACIONES"
echo "=================================="
echo ""

# Verificar Rust
echo -n "🦀 Rust: "
rustc --version || print_error "Rust NO instalado"

echo -n "📦 Cargo: "
cargo --version || print_error "Cargo NO instalado"

# Verificar LLVM
echo -n "⚡ LLVM: "
llvm-config-16 --version || print_error "LLVM-16 NO instalado"

echo -n "🔧 Clang: "
clang-16 --version | head -1 || print_error "Clang-16 NO instalado"

# Verificar Vulkan
echo ""
echo "🎮 Vulkan:"
if command -v vulkaninfo &> /dev/null; then
    vulkaninfo --summary 2>/dev/null | grep -E "(GPU|Device Name)" | head -5 || echo "   No GPU detectada (OK para desarrollo CPU)"
    print_success "Vulkan instalado correctamente"
else
    print_error "Vulkan NO instalado"
fi

echo ""
echo "=================================="
echo "🎯 SIGUIENTE PASO"
echo "=================================="
echo ""
echo "Ahora ejecuta:"
echo "  cd /home/vboxuser/Desktop/Projects/AI/charlcode"
echo "  cargo build --release"
echo "  cargo test"
echo ""
echo "¡Listo para desarrollar Charl Language! 🚀"
echo ""

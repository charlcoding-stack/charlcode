#!/bin/bash
# build-releases.sh - Generate all release packages for Charl v0.1.0
#
# This script builds pre-compiled binaries for all supported platforms
# and creates distribution packages ready for GitHub Releases.
#
# Usage:
#   ./build-releases.sh [--all|--current|--linux|--macos|--windows]
#
# Options:
#   --all      Build for all platforms (default)
#   --current  Build only for current platform
#   --linux    Build for Linux platforms only
#   --macos    Build for macOS platforms only
#   --windows  Build for Windows platforms only

set -e

VERSION="0.1.0"
RELEASE_DIR="releases"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Charl Release Builder v${VERSION}                    ║${NC}"
echo -e "${BLUE}║   Building multi-platform distribution packages          ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Parse command line arguments
BUILD_MODE="all"
if [ $# -gt 0 ]; then
    BUILD_MODE="$1"
    BUILD_MODE="${BUILD_MODE#--}"  # Remove leading --
fi

# Create release directory
mkdir -p "$RELEASE_DIR"

# Define platform configurations
# Format: "rust-target:output-filename:binary-name"
declare -A LINUX_PLATFORMS=(
    ["x86_64-unknown-linux-gnu"]="charl-linux-x86_64.tar.gz:charl"
    ["aarch64-unknown-linux-gnu"]="charl-linux-arm64.tar.gz:charl"
)

declare -A MACOS_PLATFORMS=(
    ["x86_64-apple-darwin"]="charl-macos-x86_64.tar.gz:charl"
    ["aarch64-apple-darwin"]="charl-macos-arm64.tar.gz:charl"
)

declare -A WINDOWS_PLATFORMS=(
    ["x86_64-pc-windows-gnu"]="charl-windows-x86_64.zip:charl.exe"
)

# Function to check if target is installed
check_target() {
    local target=$1
    if rustup target list --installed | grep -q "^$target$"; then
        return 0
    else
        return 1
    fi
}

# Function to build for a specific target
build_target() {
    local target=$1
    local package=$2
    local binary=$3

    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Building for: ${target}${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Check if target is installed
    if ! check_target "$target"; then
        echo -e "${YELLOW}⚙️  Installing target: ${target}${NC}"
        rustup target add "$target" || {
            echo -e "${RED}❌ Failed to install target: ${target}${NC}"
            echo -e "${YELLOW}⚠️  Skipping this target${NC}"
            echo ""
            return 1
        }
    else
        echo -e "${GREEN}✅ Target already installed: ${target}${NC}"
    fi

    # Build release binary
    echo -e "${BLUE}🔧 Compiling (this may take a few minutes)...${NC}"
    if cargo build --release --target "$target" --bin charl 2>&1 | grep -E "(Compiling|Finished|error)"; then
        echo -e "${GREEN}✅ Build successful${NC}"
    else
        echo -e "${RED}❌ Build failed for ${target}${NC}"
        echo ""
        return 1
    fi

    # Verify binary exists
    local binary_path="target/${target}/release/${binary}"
    if [ ! -f "$binary_path" ]; then
        echo -e "${RED}❌ Binary not found: ${binary_path}${NC}"
        echo ""
        return 1
    fi

    # Get binary size
    local size=$(ls -lh "$binary_path" | awk '{print $5}')
    echo -e "${GREEN}📦 Binary size: ${size}${NC}"

    # Package based on format
    local package_path="${RELEASE_DIR}/${package}"
    echo -e "${BLUE}📦 Creating package: ${package}${NC}"

    if [[ $package == *.tar.gz ]]; then
        # Create tar.gz archive
        tar -czf "$package_path" -C "target/${target}/release" "$binary"
    elif [[ $package == *.zip ]]; then
        # Create zip archive
        (cd "target/${target}/release" && zip -q "../../../${package_path}" "$binary")
    fi

    if [ -f "$package_path" ]; then
        local pkg_size=$(ls -lh "$package_path" | awk '{print $5}')
        echo -e "${GREEN}✅ Package created: ${package} (${pkg_size})${NC}"

        # Generate SHA256 checksum
        if command -v sha256sum &> /dev/null; then
            local checksum=$(sha256sum "$package_path" | awk '{print $1}')
            echo "$checksum  $package" >> "${RELEASE_DIR}/SHA256SUMS"
            echo -e "${GREEN}🔒 SHA256: ${checksum}${NC}"
        fi
    else
        echo -e "${RED}❌ Package creation failed${NC}"
        return 1
    fi

    echo ""
    return 0
}

# Function to build all platforms in an associative array
build_platforms() {
    local -n platforms=$1
    local built=0
    local failed=0

    for target in "${!platforms[@]}"; do
        local config="${platforms[$target]}"
        local package="${config%%:*}"
        local binary="${config##*:}"

        if build_target "$target" "$package" "$binary"; then
            ((built++))
        else
            ((failed++))
        fi
    done

    return $failed
}

# Clear previous checksums
rm -f "${RELEASE_DIR}/SHA256SUMS"

# Build based on mode
total_built=0
total_failed=0

case "$BUILD_MODE" in
    all)
        echo -e "${BLUE}🌍 Building for all platforms...${NC}"
        echo ""

        echo -e "${GREEN}═══ Linux Platforms ═══${NC}"
        build_platforms LINUX_PLATFORMS

        echo -e "${GREEN}═══ macOS Platforms ═══${NC}"
        build_platforms MACOS_PLATFORMS

        echo -e "${GREEN}═══ Windows Platforms ═══${NC}"
        build_platforms WINDOWS_PLATFORMS
        ;;

    current)
        echo -e "${BLUE}🖥️  Building for current platform only...${NC}"
        echo ""

        OS=$(uname -s | tr '[:upper:]' '[:lower:]')
        ARCH=$(uname -m)

        case "$OS-$ARCH" in
            linux-x86_64)
                build_target "x86_64-unknown-linux-gnu" "charl-linux-x86_64.tar.gz" "charl"
                ;;
            linux-aarch64)
                build_target "aarch64-unknown-linux-gnu" "charl-linux-arm64.tar.gz" "charl"
                ;;
            darwin-x86_64)
                build_target "x86_64-apple-darwin" "charl-macos-x86_64.tar.gz" "charl"
                ;;
            darwin-arm64)
                build_target "aarch64-apple-darwin" "charl-macos-arm64.tar.gz" "charl"
                ;;
            *)
                echo -e "${RED}❌ Unsupported platform: $OS-$ARCH${NC}"
                exit 1
                ;;
        esac
        ;;

    linux)
        echo -e "${BLUE}🐧 Building for Linux platforms...${NC}"
        echo ""
        build_platforms LINUX_PLATFORMS
        ;;

    macos)
        echo -e "${BLUE}🍎 Building for macOS platforms...${NC}"
        echo ""
        build_platforms MACOS_PLATFORMS
        ;;

    windows)
        echo -e "${BLUE}🪟 Building for Windows platforms...${NC}"
        echo ""
        build_platforms WINDOWS_PLATFORMS
        ;;

    *)
        echo -e "${RED}❌ Unknown build mode: $BUILD_MODE${NC}"
        echo ""
        echo "Usage: $0 [--all|--current|--linux|--macos|--windows]"
        exit 1
        ;;
esac

# Summary
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Build Summary                          ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ -d "$RELEASE_DIR" ] && [ "$(ls -A $RELEASE_DIR)" ]; then
    echo -e "${GREEN}✅ Release packages created in: ${RELEASE_DIR}/${NC}"
    echo ""
    ls -lh "$RELEASE_DIR"
    echo ""

    if [ -f "${RELEASE_DIR}/SHA256SUMS" ]; then
        echo -e "${GREEN}🔒 SHA256 Checksums:${NC}"
        cat "${RELEASE_DIR}/SHA256SUMS"
        echo ""
    fi

    echo -e "${YELLOW}📤 Next steps:${NC}"
    echo -e "   1. Test the installers with these packages"
    echo -e "   2. Upload packages to GitHub Releases:"
    echo -e "      ${BLUE}https://github.com/charlcoding-stack/charlcode/releases/tag/v${VERSION}${NC}"
    echo -e "   3. Update install.sh and install.ps1 to point to the release"
    echo ""
else
    echo -e "${RED}❌ No packages were created${NC}"
    exit 1
fi

echo -e "${GREEN}🚀 Build complete!${NC}"
echo ""

# Charl Language Installer for Windows
# Usage: irm https://charlbase.org/install.ps1 | iex

$ErrorActionPreference = "Stop"

$CHARL_VERSION = "0.1.0"
$INSTALL_DIR = "$env:USERPROFILE\.charl"
$BIN_DIR = "$INSTALL_DIR\bin"

Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║           Charl Language Installer v$CHARL_VERSION              ║" -ForegroundColor Cyan
Write-Host "║   Revolutionary AI/ML Programming Language                ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Detect OS and Architecture
$OS = "Windows"
$ARCH = $env:PROCESSOR_ARCHITECTURE
if ($ARCH -eq "AMD64") {
    $ARCH = "x86_64"
}

Write-Host "🔍 Detected system:" -ForegroundColor Yellow
Write-Host "   OS: $OS"
Write-Host "   Architecture: $ARCH"
Write-Host ""

# Check if Rust is installed
Write-Host "🔍 Checking for Rust..." -ForegroundColor Yellow
$rustInstalled = Get-Command rustc -ErrorAction SilentlyContinue

if (-not $rustInstalled) {
    Write-Host "⚠️  Rust not found. Installing Rust first..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please install Rust from: https://rustup.rs/" -ForegroundColor Cyan
    Write-Host "After installing Rust, run this installer again." -ForegroundColor Cyan
    Write-Host ""
    # Open browser to rustup.rs
    Start-Process "https://rustup.rs/"
    exit 1
} else {
    $rustVersion = & rustc --version
    Write-Host "✅ Rust found: $rustVersion" -ForegroundColor Green
}
Write-Host ""

# Create installation directory
Write-Host "📁 Creating installation directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $BIN_DIR | Out-Null

# Option 1: Download pre-built binary (production)
# Option 2: Build from source (development)

$downloadUrl = "https://github.com/charlcoding-stack/charlcode/releases/download/v$CHARL_VERSION/charl-windows-x86_64.zip"
$zipPath = "$env:TEMP\charl-windows-x86_64.zip"

Write-Host "📥 Downloading Charl binary..." -ForegroundColor Yellow
Write-Host "   URL: $downloadUrl"
Write-Host ""

try {
    # Try to download pre-built binary
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -ErrorAction Stop

    Write-Host "📦 Extracting binary..." -ForegroundColor Yellow
    Expand-Archive -Path $zipPath -DestinationPath $BIN_DIR -Force
    Remove-Item $zipPath

    Write-Host "✅ Binary installed successfully" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Pre-built binary not available. Building from source..." -ForegroundColor Yellow
    Write-Host ""

    # Build from source
    $TEMP_DIR = "$env:TEMP\charl-build"
    New-Item -ItemType Directory -Force -Path $TEMP_DIR | Out-Null
    Set-Location $TEMP_DIR

    Write-Host "📥 Cloning Charl repository..." -ForegroundColor Yellow
    git clone https://github.com/charlcoding-stack/charlcode.git .

    Write-Host "🔧 Compiling Charl (this may take a few minutes)..." -ForegroundColor Yellow
    cargo build --release --bin charl

    Write-Host "📦 Installing binary..." -ForegroundColor Yellow
    Copy-Item "target\release\charl.exe" "$BIN_DIR\charl.exe"

    # Cleanup
    Set-Location $env:USERPROFILE
    Remove-Item -Recurse -Force $TEMP_DIR
}

# Add to PATH
Write-Host ""
Write-Host "🔗 Setting up PATH..." -ForegroundColor Yellow

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$BIN_DIR*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$BIN_DIR", "User")
    Write-Host "✅ Added Charl to PATH" -ForegroundColor Green
    Write-Host "   Note: Restart your terminal to use 'charl' command" -ForegroundColor Cyan
} else {
    Write-Host "✅ Charl already in PATH" -ForegroundColor Green
}

# Install VS Code extension (if VS Code is installed)
Write-Host ""
Write-Host "🎨 Checking for VS Code..." -ForegroundColor Yellow

$vscodeInstalled = Get-Command code -ErrorAction SilentlyContinue

if ($vscodeInstalled) {
    Write-Host "✅ VS Code found" -ForegroundColor Green
    Write-Host "📦 Installing Charl language extension for VS Code..." -ForegroundColor Yellow

    $VSCODE_EXT_DIR = "$env:USERPROFILE\.vscode\extensions\charl-lang.charl-1.0.0"

    # Check if vscode-charl directory exists in the installation source
    $vscodeExtSource = "$TEMP_DIR\vscode-charl"

    if (Test-Path $vscodeExtSource) {
        New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.vscode\extensions" | Out-Null
        Copy-Item -Recurse -Force $vscodeExtSource $VSCODE_EXT_DIR

        Write-Host "✅ VS Code extension installed" -ForegroundColor Green
        Write-Host "   • Syntax highlighting for .ch files" -ForegroundColor Cyan
        Write-Host "   • Auto-indentation and bracket matching" -ForegroundColor Cyan
        Write-Host "   • 22 code snippets ready to use" -ForegroundColor Cyan
        Write-Host "   • Restart VS Code to activate the extension" -ForegroundColor Yellow
    } else {
        Write-Host "⚠️  VS Code extension not found in installation package" -ForegroundColor Yellow
        Write-Host "   You can download it from: https://charlbase.org/downloads" -ForegroundColor Cyan
    }
} else {
    Write-Host "ℹ️  VS Code not found - skipping extension installation" -ForegroundColor Cyan
    Write-Host "   Install VS Code from: https://code.visualstudio.com/" -ForegroundColor Cyan
    Write-Host "   Then install the Charl extension from: https://charlbase.org/downloads" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║               Installation Complete! 🎉                   ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Charl has been installed to: $BIN_DIR\charl.exe" -ForegroundColor Cyan
Write-Host ""
Write-Host "📚 To get started:" -ForegroundColor Yellow
Write-Host "   1. Restart your terminal (to reload PATH)" -ForegroundColor White
Write-Host "   2. Verify installation: charl --version" -ForegroundColor White
Write-Host "   3. Try the examples: charl run examples\hello.ch" -ForegroundColor White
Write-Host ""

if ($vscodeInstalled -and (Test-Path $VSCODE_EXT_DIR)) {
    Write-Host "🎨 VS Code Extension:" -ForegroundColor Yellow
    Write-Host "   • Restart VS Code to activate syntax highlighting" -ForegroundColor White
    Write-Host "   • Open any .ch file to see colorized code" -ForegroundColor White
    Write-Host "   • Use snippets: type 'fn', 'match', 'for' and press Tab" -ForegroundColor White
    Write-Host ""
}

Write-Host "📖 Learn more:" -ForegroundColor Yellow
Write-Host "   • Documentation: https://charlbase.org/docs" -ForegroundColor Cyan
Write-Host "   • Website: https://charlbase.org" -ForegroundColor Cyan
Write-Host "   • GitHub: https://github.com/charlcoding-stack/charlcode" -ForegroundColor Cyan
Write-Host "   • Examples: https://charlbase.org/examples" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 Happy coding with Charl!" -ForegroundColor Green
Write-Host ""

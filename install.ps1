# Charl Language Installer for Windows
# Usage: irm https://charlbase.org/install.ps1 | iex

$ErrorActionPreference = "Stop"

$CHARL_VERSION = "0.1.0"
$INSTALL_DIR = "$env:USERPROFILE\.charl"
$BIN_DIR = "$INSTALL_DIR\bin"
$GITHUB_REPO = "charlcoding-stack/charlcode"

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
} elseif ($ARCH -eq "ARM64") {
    $ARCH = "arm64"
} else {
    Write-Host "❌ Unsupported architecture: $ARCH" -ForegroundColor Red
    Write-Host "   Supported: AMD64 (x86_64)" -ForegroundColor Yellow
    exit 1
}

Write-Host "🔍 Detected system:" -ForegroundColor Yellow
Write-Host "   OS: $OS"
Write-Host "   Architecture: $ARCH"
Write-Host ""

# Create installation directory
Write-Host "📁 Creating installation directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $BIN_DIR | Out-Null

# Try to download pre-built binary
$downloadUrl = "https://github.com/$GITHUB_REPO/releases/download/v$CHARL_VERSION/charl-windows-$ARCH.zip"
$zipPath = "$env:TEMP\charl-windows-$ARCH.zip"

Write-Host "📥 Downloading Charl binary..." -ForegroundColor Yellow
Write-Host "   URL: $downloadUrl" -ForegroundColor Gray
Write-Host ""

try {
    # Try to download pre-built binary
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -ErrorAction Stop

    Write-Host "✅ Download successful" -ForegroundColor Green
    Write-Host ""

    Write-Host "📦 Extracting binary..." -ForegroundColor Yellow
    Expand-Archive -Path $zipPath -DestinationPath $BIN_DIR -Force
    Remove-Item $zipPath

    # Verify binary exists
    if (Test-Path "$BIN_DIR\charl.exe") {
        Write-Host "✅ Binary installed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Error: Binary not found after extraction" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "⚠️  Pre-built binary not available for your platform" -ForegroundColor Yellow
    Write-Host "   Falling back to building from source..." -ForegroundColor Yellow
    Write-Host ""

    # Check if Rust is installed
    Write-Host "🔍 Checking for Rust..." -ForegroundColor Yellow
    $rustInstalled = Get-Command rustc -ErrorAction SilentlyContinue

    if (-not $rustInstalled) {
        Write-Host "📦 Rust is required to build from source" -ForegroundColor Yellow
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

    # Check if Git is installed
    $gitInstalled = Get-Command git -ErrorAction SilentlyContinue
    if (-not $gitInstalled) {
        Write-Host "❌ Error: git is required to build from source" -ForegroundColor Red
        Write-Host "   Please install git from: https://git-scm.com/" -ForegroundColor Cyan
        exit 1
    }

    # Build from source
    $TEMP_DIR = "$env:TEMP\charl-build-$(Get-Random)"
    New-Item -ItemType Directory -Force -Path $TEMP_DIR | Out-Null

    try {
        Set-Location $TEMP_DIR

        Write-Host "📥 Cloning Charl repository..." -ForegroundColor Yellow
        git clone "https://github.com/$GITHUB_REPO.git" . 2>&1 | Out-Null

        # Find VS Developer Command Prompt
        $vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
        $vsDevCmd = "$vsPath\Common7\Tools\VsDevCmd.bat"

        if (Test-Path $vsDevCmd) {
            Write-Host "✅ Found Visual Studio Build Tools" -ForegroundColor Green
            Write-Host "🔧 Compiling Charl with MSVC environment (5-10 minutes)..." -ForegroundColor Yellow
            Write-Host "   This resolves link.exe conflicts with Git" -ForegroundColor Gray
            Write-Host ""

            # Use Developer Command Prompt to compile
            $buildScript = @"
call "$vsDevCmd" >nul 2>&1
cd "$TEMP_DIR"
cargo build --release --bin charl 2>&1
"@
            $buildScript | Out-File -FilePath "$TEMP_DIR\build.bat" -Encoding ASCII

            $proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "$TEMP_DIR\build.bat" -Wait -PassThru -NoNewWindow -RedirectStandardOutput "$TEMP_DIR\build.log" -RedirectStandardError "$TEMP_DIR\build.err"

            if ($proc.ExitCode -eq 0 -and (Test-Path "$TEMP_DIR\target\release\charl.exe")) {
                Write-Host "✅ Compilation successful" -ForegroundColor Green
            } else {
                Write-Host "❌ Compilation failed" -ForegroundColor Red
                if (Test-Path "$TEMP_DIR\build.log") {
                    Get-Content "$TEMP_DIR\build.log" | Select-String -Pattern "error" | ForEach-Object { Write-Host $_ -ForegroundColor Red }
                }
                throw "Compilation failed"
            }
        } else {
            Write-Host "⚠️  VS Build Tools not found, trying standard compilation..." -ForegroundColor Yellow
            cargo build --release --bin charl 2>&1 | Select-String -Pattern "(Compiling|Finished|error)" | Write-Host
        }

        Write-Host "📦 Installing binary..." -ForegroundColor Yellow
        Copy-Item "target\release\charl.exe" "$BIN_DIR\charl.exe" -Force

        Write-Host "✅ Binary compiled and installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "❌ Build failed: $_" -ForegroundColor Red
        Write-Host "" -ForegroundColor Red
        Write-Host "Troubleshooting:" -ForegroundColor Yellow
        Write-Host "1. Ensure Visual Studio Build Tools is installed" -ForegroundColor White
        Write-Host "2. Try running build-windows.bat manually from the repo" -ForegroundColor White
        Write-Host "3. See: https://github.com/$GITHUB_REPO/issues" -ForegroundColor White
        exit 1
    } finally {
        # Cleanup
        Set-Location $env:USERPROFILE
        if (Test-Path $TEMP_DIR) {
            Remove-Item -Recurse -Force $TEMP_DIR -ErrorAction SilentlyContinue
        }
    }
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

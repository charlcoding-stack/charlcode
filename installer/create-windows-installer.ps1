# Create Windows Installer for Charl
#
# This script creates a simple self-extracting installer for Windows
# that doesn't require compilation or external tools.
#
# Usage:
#   1. First compile charl.exe: cargo build --release
#   2. Run this script: .\create-windows-installer.ps1
#   3. Output: charl-installer-0.1.0.exe

param(
    [string]$Version = "0.1.0",
    [string]$OutputDir = "..\releases"
)

Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       Charl Windows Installer Creator v$Version          ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check if binary exists
$binaryPath = "..\target\release\charl.exe"
if (-not (Test-Path $binaryPath)) {
    Write-Host "❌ Error: charl.exe not found at $binaryPath" -ForegroundColor Red
    Write-Host "   Please compile first: cargo build --release" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Found charl.exe" -ForegroundColor Green

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Create a simple ZIP package
$zipPath = "$OutputDir\charl-$Version-windows-x64-portable.zip"
Write-Host "📦 Creating portable ZIP package..." -ForegroundColor Yellow

$tempDir = "$env:TEMP\charl-package"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

try {
    # Copy files to temp directory
    Copy-Item $binaryPath "$tempDir\charl.exe"
    Copy-Item "..\README.md" "$tempDir\" -ErrorAction SilentlyContinue
    Copy-Item "..\LICENSE" "$tempDir\" -ErrorAction SilentlyContinue

    # Create install script
    $installScript = @'
@echo off
echo ╔═══════════════════════════════════════════════════════════╗
echo ║         Charl Language Portable Installer                ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.

set INSTALL_DIR=%USERPROFILE%\.charl\bin

echo Installing Charl to: %INSTALL_DIR%
echo.

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

copy /Y charl.exe "%INSTALL_DIR%\charl.exe" >nul

echo ✅ Charl installed successfully!
echo.
echo Adding to PATH...

rem Check if already in PATH
echo %PATH% | findstr /C:"%INSTALL_DIR%" >nul
if errorlevel 1 (
    rem Add to user PATH
    for /f "usebackq tokens=2,*" %%A in (`reg query HKCU\Environment /v PATH 2^>nul`) do set UserPath=%%B
    setx PATH "%UserPath%;%INSTALL_DIR%" >nul
    echo ✅ Added to PATH
    echo.
    echo ⚠️  Please restart your terminal for PATH changes to take effect
) else (
    echo ℹ️  Already in PATH
)

echo.
echo Installation complete!
echo.
echo To verify: charl --version
echo (restart your terminal first)
echo.
pause
'@
    $installScript | Out-File -FilePath "$tempDir\install.bat" -Encoding ASCII

    # Create uninstall script
    $uninstallScript = @'
@echo off
echo Uninstalling Charl...
echo.

set INSTALL_DIR=%USERPROFILE%\.charl\bin

if exist "%INSTALL_DIR%\charl.exe" (
    del "%INSTALL_DIR%\charl.exe"
    echo ✅ Removed charl.exe
) else (
    echo ℹ️  charl.exe not found
)

echo.
echo ⚠️  Note: PATH may still contain %INSTALL_DIR%
echo    You can remove it manually from Environment Variables
echo.
pause
'@
    $uninstallScript | Out-File -FilePath "$tempDir\uninstall.bat" -Encoding ASCII

    # Create README for the package
    $packageReadme = @"
# Charl v$Version - Portable Windows Installation

## Installation

1. Run `install.bat`
   - Installs Charl to `%USERPROFILE%\.charl\bin`
   - Adds to PATH automatically

2. Restart your terminal

3. Verify: `charl --version`

## Manual Installation

If you prefer manual installation:

1. Copy `charl.exe` to a directory of your choice
2. Add that directory to your PATH:
   - Search for "Environment Variables" in Start Menu
   - Edit "Path" under "User variables"
   - Add the directory containing charl.exe

## Usage

```
charl --help        Show help
charl --version     Show version
charl run file.ch   Run a Charl program
charl repl          Start interactive REPL
```

## Documentation

- Website: https://charlbase.org
- Getting Started: https://charlbase.org/docs/getting-started.html
- GitHub: https://github.com/charlcoding-stack/charlcode

## Uninstallation

Run `uninstall.bat` to remove Charl

## Support

- Issues: https://github.com/charlcoding-stack/charlcode/issues
- Discussions: https://github.com/charlcoding-stack/charlcode/discussions
"@
    $packageReadme | Out-File -FilePath "$tempDir\README-INSTALL.txt" -Encoding UTF8

    # Create the ZIP
    Compress-Archive -Path "$tempDir\*" -DestinationPath $zipPath -Force

    Write-Host "✅ Created: $zipPath" -ForegroundColor Green

    $zipSize = (Get-Item $zipPath).Length / 1KB
    Write-Host "   Size: $([math]::Round($zipSize, 2)) KB" -ForegroundColor Gray

} finally {
    # Cleanup
    Remove-Item -Recurse -Force $tempDir -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║              Package Created Successfully! 🎉             ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "📦 Portable Package: $zipPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Test the package on a clean Windows machine" -ForegroundColor White
Write-Host "  2. Upload to GitHub Releases" -ForegroundColor White
Write-Host "  3. Users can download and run install.bat" -ForegroundColor White
Write-Host ""

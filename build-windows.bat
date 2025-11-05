@echo off
REM Charl Windows Build Script
REM This script resolves the link.exe conflict between Git and MSVC
REM and compiles Charl with the correct toolchain.

echo ╔═══════════════════════════════════════════════════════════╗
echo ║         Charl Windows Build Script v0.1.0                ║
echo ║   Compiles Charl with correct MSVC environment           ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.

REM Check if VS Build Tools is installed
set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
if not exist "%VS_PATH%" (
    echo ❌ Visual Studio Build Tools not found
    echo.
    echo Please install Visual Studio Build Tools from:
    echo https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
    echo.
    echo Select "Desktop development with C++" during installation.
    exit /b 1
)

REM Check if Rust is installed
where cargo >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Rust not found
    echo.
    echo Please install Rust from: https://rustup.rs/
    exit /b 1
)

echo ✅ Visual Studio Build Tools found
echo ✅ Rust found
echo.

REM Find vcvarsall.bat
set "VCVARSALL=%VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat"
if not exist "%VCVARSALL%" (
    echo ❌ vcvarsall.bat not found
    exit /b 1
)

echo 🔧 Setting up MSVC environment...
echo.

REM Call vcvarsall.bat to set up environment
call "%VCVARSALL%" x64 >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to initialize MSVC environment
    exit /b 1
)

echo ✅ MSVC environment configured
echo.

REM Remove Git's link.exe from PATH to avoid conflicts
echo 🔍 Checking for link.exe conflicts...
set "NEW_PATH="
for %%P in ("%PATH:;=";"%") do (
    echo %%~P | findstr /I /C:"Git" >nul
    if errorlevel 1 (
        if defined NEW_PATH (
            set "NEW_PATH=!NEW_PATH!;%%~P"
        ) else (
            set "NEW_PATH=%%~P"
        )
    )
)
set "PATH=%NEW_PATH%"

REM Verify we're using MSVC link.exe
where link.exe >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ MSVC link.exe not found in PATH
    exit /b 1
)

echo ✅ Using MSVC link.exe
echo.

REM Clean previous build
echo 🧹 Cleaning previous build...
cargo clean >nul 2>&1

REM Build in release mode
echo 🔨 Compiling Charl in release mode...
echo    (This may take 5-10 minutes)
echo.

cargo build --release --bin charl

if %ERRORLEVEL% equ 0 (
    echo.
    echo ╔═══════════════════════════════════════════════════════════╗
    echo ║              Build Successful! 🎉                         ║
    echo ╚═══════════════════════════════════════════════════════════╝
    echo.
    echo Binary location: target\release\charl.exe
    echo.
    echo To run: .\target\release\charl.exe --version
    echo.
) else (
    echo.
    echo ╔═══════════════════════════════════════════════════════════╗
    echo ║              Build Failed ❌                              ║
    echo ╚═══════════════════════════════════════════════════════════╝
    echo.
    echo Check the errors above for details.
    exit /b 1
)

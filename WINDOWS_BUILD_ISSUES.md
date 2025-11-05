# Windows Build Issues and Solutions

## Known Issue: link.exe Conflict with Git

### Problem

When compiling Charl on Windows, you may encounter this error:

```
error: linking with `link.exe` failed: exit code: 1
link: extra operand 'C:\\Users\\...\\...rcgu.o'
```

### Root Cause

Git for Windows includes its own `link.exe` command (Unix symlink tool) in its PATH. When Rust tries to compile, it finds Git's `link.exe` instead of Microsoft Visual C++'s `link.exe` (the linker), causing the compilation to fail.

### Solution 1: Use build-windows.bat (Recommended)

We provide a script that automatically resolves this conflict:

```bash
# Clone the repository
git clone https://github.com/charlcoding-stack/charlcode.git
cd charlcode

# Run the build script
build-windows.bat
```

The script:
- ✅ Configures MSVC environment correctly
- ✅ Removes Git paths from PATH temporarily
- ✅ Uses the correct link.exe
- ✅ Compiles in release mode

### Solution 2: Use the Installer (Fixed)

The updated installer now handles this automatically:

```powershell
irm https://charlbase.org/install.ps1 | iex
```

**Requirements:**
- Rust: https://rustup.rs/
- Git: https://git-scm.com/
- Visual Studio Build Tools 2022: https://visualstudio.microsoft.com/downloads/
  - Select "Desktop development with C++"

### Solution 3: Manual Compilation with Developer Command Prompt

1. Open "Developer Command Prompt for VS 2022" from Start Menu
2. Clone and build:

```cmd
git clone https://github.com/charlcoding-stack/charlcode.git
cd charlcode
cargo build --release
```

The Developer Command Prompt automatically sets up the correct PATH.

### Solution 4: Temporary PATH Fix

You can temporarily remove Git from PATH:

```powershell
# In PowerShell
$env:Path = ($env:Path -split ';' | Where-Object { $_ -notlike '*Git*' }) -join ';'

# Then compile
cargo build --release
```

## Other Common Windows Issues

### Issue: "MSVC linker not found"

**Error:**
```
error: linker `link.exe` not found
```

**Solution:**
Install Visual Studio Build Tools with "Desktop development with C++" workload.

```powershell
# Download and install
Start-Process "https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022"
```

### Issue: "cargo: command not found"

**Error:**
```
cargo : The term 'cargo' is not recognized
```

**Solution:**
1. Install Rust from https://rustup.rs/
2. Restart your terminal/PowerShell
3. Verify: `cargo --version`

### Issue: Long Compilation Time

**Observation:**
Compilation takes 5-15 minutes on first build.

**Explanation:**
This is normal. Charl has many dependencies (~200 crates). Subsequent builds are much faster due to caching.

**Workaround:**
Build in debug mode first (faster):
```cmd
cargo build
.\target\debug\charl.exe --version
```

Then build release version:
```cmd
cargo build --release
```

### Issue: "out of memory" during compilation

**Error:**
```
error: could not compile ...
LINK : fatal error LNK1248: image size exceeds maximum allowable size
```

**Solution:**
1. Close other applications
2. Build with fewer parallel jobs:
```cmd
cargo build --release -j 2
```

## Verifying Your Installation

After successful build:

```powershell
# Check binary exists
Test-Path target\release\charl.exe

# Run version check
.\target\release\charl.exe --version

# Should output: charl 0.1.0
```

## Getting Help

If you still have issues:

1. **Check existing issues:** https://github.com/charlcoding-stack/charlcode/issues
2. **Create new issue:** Include:
   - Windows version
   - Rust version (`rustc --version`)
   - VS Build Tools version
   - Full error output

3. **Community:**
   - GitHub Discussions: https://github.com/charlcoding-stack/charlcode/discussions
   - Documentation: https://charlbase.org/docs

## Pre-compiled Binary (Coming Soon)

We're working on providing pre-compiled Windows binaries so you won't need to compile from source. Check releases:

https://github.com/charlcoding-stack/charlcode/releases

---

**Last Updated:** 2025-11-05
**Applies to:** Charl v0.1.0
**Platform:** Windows 10/11

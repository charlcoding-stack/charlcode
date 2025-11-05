# Charl Installation Guide for Windows

Complete guide for installing Charl on Windows 10/11.

## Prerequisites

- Windows 10 or Windows 11 (64-bit)
- PowerShell 5.1 or later
- Internet connection

## Method 1: Automated Installation (Recommended)

### Step 1: Install Rust

Charl requires Rust to compile from source. Download and install Rust:

1. Visit [https://rustup.rs/](https://rustup.rs/)
2. Download `rustup-init.exe`
3. Run the installer and follow the prompts
4. Restart your terminal after installation

Verify Rust installation:
```powershell
rustc --version
cargo --version
```

### Step 2: Run Charl Installer

Open PowerShell **as Administrator** and run:

```powershell
irm https://charlbase.org/install.ps1 | iex
```

Or download and run locally:

```powershell
# Download installer
Invoke-WebRequest -Uri "https://charlbase.org/install.ps1" -OutFile "$env:TEMP\install.ps1"

# Run installer
& "$env:TEMP\install.ps1"
```

### What the installer does:

1. ✅ Checks for Rust installation
2. ✅ Downloads Charl source code
3. ✅ Compiles Charl in release mode
4. ✅ Installs to `%USERPROFILE%\.charl\bin\`
5. ✅ Adds Charl to your PATH
6. ✅ Installs VS Code extension (if VS Code is detected)

### Step 3: Verify Installation

Restart your terminal and run:

```powershell
charl --version
```

You should see:
```
charl 0.1.0
```

## Method 2: Manual Installation

### Option A: Pre-compiled Binary

1. **Download the binary:**

   Go to [GitHub Releases](https://github.com/charlcoding-stack/charlcode/releases/latest) and download:
   - `charl-windows-x86_64.zip`

2. **Extract the binary:**

   ```powershell
   # Create installation directory
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.charl\bin"

   # Extract (assuming zip is in Downloads)
   Expand-Archive "$env:USERPROFILE\Downloads\charl-windows-x86_64.zip" -DestinationPath "$env:USERPROFILE\.charl\bin"
   ```

3. **Add to PATH:**

   ```powershell
   # Get current user PATH
   $userPath = [Environment]::GetEnvironmentVariable("Path", "User")

   # Add Charl to PATH
   $charlBin = "$env:USERPROFILE\.charl\bin"
   [Environment]::SetEnvironmentVariable("Path", "$userPath;$charlBin", "User")
   ```

4. **Restart terminal and verify:**

   ```powershell
   charl --version
   ```

### Option B: Build from Source

1. **Install prerequisites:**
   - Rust (from [rustup.rs](https://rustup.rs/))
   - Git (from [git-scm.com](https://git-scm.com/))

2. **Clone repository:**

   ```powershell
   git clone https://github.com/charlcoding-stack/charlcode.git
   cd charlcode
   ```

3. **Build in release mode:**

   ```powershell
   cargo build --release
   ```

   This takes 3-5 minutes depending on your hardware.

4. **Install binary:**

   ```powershell
   # Create installation directory
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.charl\bin"

   # Copy binary
   Copy-Item "target\release\charl.exe" "$env:USERPROFILE\.charl\bin\"
   ```

5. **Add to PATH (as shown in Option A, step 3)**

## Installing VS Code Extension

The VS Code extension provides syntax highlighting and IDE support for `.ch` files.

### Automatic (included with installer)

If you used Method 1, the extension is installed automatically.

### Manual Installation

1. **Locate the extension:**

   If you built from source, the extension is in `charlcode\vscode-charl\`

2. **Copy to VS Code extensions directory:**

   ```powershell
   # Create extensions directory if it doesn't exist
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.vscode\extensions"

   # Copy extension
   Copy-Item -Recurse ".\vscode-charl" "$env:USERPROFILE\.vscode\extensions\charl-lang.charl-1.0.0"
   ```

3. **Restart VS Code**

4. **Verify installation:**
   - Open any `.ch` file
   - Check bottom-right corner - it should say "Charl"
   - Code should have syntax highlighting

## Testing Your Installation

1. **Create a test file:**

   ```powershell
   New-Item -ItemType File -Path "test.ch" -Force
   ```

2. **Add this code to `test.ch`:**

   ```charl
   fn greet(name: string) -> string {
       return "Hello, " + name + "!";
   }

   let message: string = greet("World");
   print(message);
   ```

3. **Run the file:**

   ```powershell
   charl run test.ch
   ```

   Expected output:
   ```
   Hello, World!
   ```

## Troubleshooting

### Issue: "charl is not recognized as a command"

**Solution:** PATH not updated properly.

1. Verify binary location:
   ```powershell
   Test-Path "$env:USERPROFILE\.charl\bin\charl.exe"
   ```

2. If true, restart your terminal completely (close all PowerShell windows)

3. If still not working, manually add to PATH:
   ```powershell
   $env:Path += ";$env:USERPROFILE\.charl\bin"
   ```

### Issue: "Rust not found" during installation

**Solution:** Install Rust from [rustup.rs](https://rustup.rs/)

1. Download and run `rustup-init.exe`
2. Follow the installation prompts
3. Restart your terminal
4. Run the Charl installer again

### Issue: VS Code extension not activating

**Solution:**

1. Check if extension is installed:
   ```powershell
   Test-Path "$env:USERPROFILE\.vscode\extensions\charl-lang.charl-1.0.0"
   ```

2. If false, reinstall manually (see "Installing VS Code Extension" section)

3. Restart VS Code completely (close all windows)

4. Open a `.ch` file and check language mode (bottom-right)

### Issue: Build fails with "linker error"

**Solution:** Install Visual Studio Build Tools

1. Download from: [https://visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/)
2. Select "Build Tools for Visual Studio"
3. In installer, select "Desktop development with C++"
4. Restart your computer
5. Try building again

## Uninstallation

To remove Charl:

```powershell
# Remove binary
Remove-Item -Recurse -Force "$env:USERPROFILE\.charl"

# Remove from PATH
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$newPath = ($userPath -split ';' | Where-Object { $_ -notlike "*\.charl\bin*" }) -join ';'
[Environment]::SetEnvironmentVariable("Path", $newPath, "User")

# Remove VS Code extension
Remove-Item -Recurse -Force "$env:USERPROFILE\.vscode\extensions\charl-lang.charl-1.0.0"
```

## Additional Resources

- **Documentation:** [https://charlbase.org/docs](https://charlbase.org/docs)
- **GitHub:** [https://github.com/charlcoding-stack/charlcode](https://github.com/charlcoding-stack/charlcode)
- **Examples:** [https://charlbase.org/examples](https://charlbase.org/examples)
- **Issue Tracker:** [https://github.com/charlcoding-stack/charlcode/issues](https://github.com/charlcoding-stack/charlcode/issues)

## Next Steps

After successful installation:

1. Read the [Getting Started Guide](https://charlbase.org/docs/getting-started.html)
2. Try the [Examples](https://charlbase.org/examples)
3. Learn the [Language Reference](https://charlbase.org/docs/language-reference.html)
4. Join the [Discussions](https://github.com/charlcoding-stack/charlcode/discussions)

---

**Last Updated:** 2025-11-05
**Version:** 0.1.0
**Platform:** Windows 10/11

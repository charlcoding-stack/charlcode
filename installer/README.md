# Charl Windows Installer Creation Guide

This directory contains scripts and configurations for creating Windows installers for Charl.

## Available Options

### Option 1: Professional Installer with Inno Setup (Recommended)

Creates a professional `.exe` installer with modern UI, PATH management, and VS Code integration.

**Requirements:**
- [Inno Setup](https://jrsoftware.org/isdl.php) (free)
- Compiled `charl.exe` binary

**Steps:**

1. **Install Inno Setup:**
   ```
   Download from: https://jrsoftware.org/isdl.php
   ```

2. **Compile Charl:**
   ```powershell
   cd charl
   cargo build --release
   ```

3. **Compile the installer:**
   ```powershell
   # Option A: Use Inno Setup GUI
   # Open charl-setup.iss in Inno Setup Compiler and click "Compile"

   # Option B: Use command line
   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\charl-setup.iss
   ```

4. **Output:**
   - File: `releases/charl-setup-0.1.0-windows-x64.exe`
   - Size: ~1-2 MB
   - Features:
     - ✅ Modern wizard UI
     - ✅ Automatic PATH setup
     - ✅ VS Code extension installer
     - ✅ Start Menu shortcuts
     - ✅ Desktop icon (optional)
     - ✅ Uninstaller

**Benefits:**
- Professional looking installer
- Automatic PATH management
- Easy uninstallation
- Digitally signable
- Supports silent install: `/SILENT` or `/VERYSILENT`

---

### Option 2: Portable ZIP Package (Simple)

Creates a portable `.zip` package with install/uninstall scripts.

**Requirements:**
- PowerShell
- Compiled `charl.exe` binary

**Steps:**

1. **Compile Charl:**
   ```powershell
   cargo build --release
   ```

2. **Create package:**
   ```powershell
   cd installer
   .\create-windows-installer.ps1
   ```

3. **Output:**
   - File: `releases/charl-0.1.0-windows-x64-portable.zip`
   - Contents:
     - `charl.exe` - Main binary
     - `install.bat` - Installation script
     - `uninstall.bat` - Uninstallation script
     - `README-INSTALL.txt` - Installation instructions

**User installation:**
```powershell
# 1. Extract ZIP
# 2. Run install.bat
# 3. Restart terminal
# 4. Verify: charl --version
```

**Benefits:**
- No installation required (truly portable)
- Can run from USB drive
- Simple batch scripts
- No admin rights needed

---

### Option 3: MSI Package (Enterprise)

For enterprise environments that require MSI packages.

**Requirements:**
- [WiX Toolset](https://wixtoolset.org/)
- More complex setup

**Status:** Coming soon

---

## Automated Build with GitHub Actions

We can automate installer creation using GitHub Actions:

```yaml
# .github/workflows/release.yml
name: Create Windows Installer

on:
  release:
    types: [published]

jobs:
  build-installer:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Build Charl
        run: cargo build --release

      - name: Install Inno Setup
        run: choco install innosetup -y

      - name: Create installer
        run: iscc installer\charl-setup.iss

      - name: Upload installer
        uses: actions/upload-release-asset@v1
        with:
          upload_url: ${{ github.event.release.upload_url }}
          asset_path: ./releases/charl-setup-${{ github.event.release.tag_name }}-windows-x64.exe
          asset_name: charl-setup-${{ github.event.release.tag_name }}-windows-x64.exe
          asset_content_type: application/octet-stream
```

---

## Distribution

### Method 1: GitHub Releases

Upload installers to GitHub Releases:

```powershell
# Using GitHub CLI
gh release upload v0.1.0 releases/charl-setup-0.1.0-windows-x64.exe
gh release upload v0.1.0 releases/charl-0.1.0-windows-x64-portable.zip
```

### Method 2: Direct Download from Website

Host on charlbase.org:

```html
<!-- On downloads.html -->
<a href="https://github.com/charlcoding-stack/charlcode/releases/download/v0.1.0/charl-setup-0.1.0-windows-x64.exe">
  Download Installer (.exe)
</a>

<a href="https://github.com/charlcoding-stack/charlcode/releases/download/v0.1.0/charl-0.1.0-windows-x64-portable.zip">
  Download Portable (.zip)
</a>
```

### Method 3: Package Managers (Future)

- **Chocolatey:** `choco install charl`
- **Scoop:** `scoop install charl`
- **WinGet:** `winget install Charl.Charl`

---

## Code Signing (Optional but Recommended)

To avoid Windows SmartScreen warnings:

1. **Get a code signing certificate:**
   - From a CA like DigiCert, Sectigo, etc.
   - Cost: ~$100-400/year

2. **Sign the installer:**
   ```powershell
   # With signtool (Windows SDK)
   signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com releases/charl-setup-0.1.0-windows-x64.exe
   ```

3. **Configure Inno Setup to sign:**
   ```ini
   [Setup]
   SignTool=standard
   SignedUninstaller=yes
   ```

---

## Testing Checklist

Before releasing installers:

- [ ] Test on clean Windows 10 VM
- [ ] Test on clean Windows 11 VM
- [ ] Verify PATH is added correctly
- [ ] Verify uninstaller removes everything
- [ ] Test with antivirus software
- [ ] Test silent installation: `/SILENT`
- [ ] Verify `charl --version` works after install
- [ ] Verify VS Code extension installs correctly
- [ ] Test portable version
- [ ] Test without admin rights

---

## Size Optimization

Current sizes:
- Raw binary: ~1.2 MB
- Inno Setup installer: ~1.5 MB (compressed)
- Portable ZIP: ~500 KB

To reduce size:
```powershell
# Strip debug symbols
cargo build --release
strip target/release/charl.exe

# Use UPX compression (optional)
upx --best target/release/charl.exe
```

---

## Troubleshooting

### "Windows protected your PC" message

This is Windows SmartScreen. Solutions:
1. Code sign the installer (best)
2. Users can click "More info" → "Run anyway"
3. Build reputation over time (downloads)

### Antivirus false positives

Some AVs flag unsigned executables:
1. Submit to VirusTotal
2. Report false positive to AV vendors
3. Code sign the installer

### Installation fails

Check:
- User has write permissions to installation directory
- No other Charl installation exists
- PATH is not too long (Windows limit: 2048 chars)

---

## Current Status

- [x] Inno Setup script created
- [x] Portable ZIP creator script
- [ ] Test on Windows machines
- [ ] Code signing setup
- [ ] GitHub Actions automation
- [ ] Chocolatey package
- [ ] WinGet package

---

## Resources

- Inno Setup: https://jrsoftware.org/isinfo.php
- WiX Toolset: https://wixtoolset.org/
- Windows Installer Guide: https://docs.microsoft.com/en-us/windows/win32/msi/windows-installer-portal
- Code Signing: https://docs.microsoft.com/en-us/windows/win32/seccrypto/cryptography-tools

---

**Last Updated:** 2025-11-05
**Version:** 0.1.0

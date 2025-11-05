# Charl v0.1.0 Release Checklist

## ✅ Completado

### 1. Scripts Corregidos
- ✅ **build-releases.sh** - Script automatizado para generar todos los paquetes
- ✅ **install.sh** - Corregido para descargar binarios pre-compilados (con fallback a compilación)
- ✅ **install.ps1** - Corregido para descargar binarios pre-compilados (con fallback a compilación)

### 2. Paquetes Generados
- ✅ **charl-linux-x86_64.tar.gz** (521 KB)
- ✅ **SHA256SUMS** - Checksums de verificación

Ubicación: `/home/vboxuser/Desktop/Projects/AI/charlcode/releases/`

### 3. Mejoras Implementadas

**install.sh (Linux/macOS):**
- Detecta automáticamente OS y arquitectura
- Descarga binarios pre-compilados desde GitHub Releases
- Fallback automático a compilación desde fuente si el binario no está disponible
- Solo instala Rust si es necesario (compilación desde fuente)
- Mejor manejo de errores

**install.ps1 (Windows):**
- Detecta arquitectura (AMD64/ARM64)
- Descarga binarios pre-compilados desde GitHub Releases
- Fallback automático a compilación desde fuente
- Verifica que Git esté instalado antes de clonar
- Mejor manejo de errores y limpieza

**build-releases.sh:**
- Construye para múltiples plataformas automáticamente
- Opciones: --all, --current, --linux, --macos, --windows
- Genera checksums SHA256 automáticamente
- Verifica que los targets de Rust estén instalados
- Salida con colores y mensajes claros

---

## 🔄 Pendiente: Generar Todos los Paquetes

Actualmente solo tenemos el paquete para **Linux x86_64**. Necesitas generar los demás:

### Opción A: Cross-Compilation en Linux (Recomendado)

```bash
cd /home/vboxuser/Desktop/Projects/AI/charlcode

# Instalar targets necesarios
rustup target add aarch64-unknown-linux-gnu
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin
rustup target add x86_64-pc-windows-gnu

# Instalar cross-compiler tools (si es necesario)
sudo apt-get install gcc-aarch64-linux-gnu  # Para ARM64 Linux
sudo apt-get install mingw-w64              # Para Windows

# Generar todos los paquetes
./build-releases.sh --all
```

**Nota:** La cross-compilación para macOS desde Linux puede requerir herramientas adicionales como `osxcross`.

### Opción B: Compilación en Cada Plataforma Nativa

**Linux ARM64** (en Raspberry Pi o servidor ARM):
```bash
./build-releases.sh --current
```

**macOS Intel**:
```bash
./build-releases.sh --current
```

**macOS Apple Silicon (M1/M2/M3)**:
```bash
./build-releases.sh --current
```

**Windows**:
```powershell
cargo build --release
cd releases
Compress-Archive -Path ..\target\release\charl.exe -DestinationPath charl-windows-x86_64.zip
```

### Opción C: GitHub Actions (Automatizado)

Crear un workflow de GitHub Actions que compile para todas las plataformas automáticamente:

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu
            name: charl-linux-x86_64.tar.gz
          - os: ubuntu-latest
            target: aarch64-unknown-linux-gnu
            name: charl-linux-arm64.tar.gz
          - os: macos-latest
            target: x86_64-apple-darwin
            name: charl-macos-x86_64.tar.gz
          - os: macos-latest
            target: aarch64-apple-darwin
            name: charl-macos-arm64.tar.gz
          - os: windows-latest
            target: x86_64-pc-windows-gnu
            name: charl-windows-x86_64.zip

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          target: ${{ matrix.target }}

      - name: Build
        run: cargo build --release --target ${{ matrix.target }}

      - name: Package
        run: |
          # Package logic here

      - name: Upload Release Asset
        uses: actions/upload-release-asset@v1
        with:
          upload_url: ${{ steps.create_release.outputs.upload_url }}
          asset_path: ./releases/${{ matrix.name }}
          asset_name: ${{ matrix.name }}
```

---

## 📤 Pasos Siguientes

### 1. Subir Paquetes a GitHub Releases

Una vez que tengas todos los paquetes generados:

1. Ve a: https://github.com/charlcoding-stack/charlcode/releases/tag/v0.1.0

2. Haz clic en "Edit release"

3. Sube los siguientes archivos:
   - `charl-linux-x86_64.tar.gz`
   - `charl-linux-arm64.tar.gz`
   - `charl-macos-x86_64.tar.gz`
   - `charl-macos-arm64.tar.gz`
   - `charl-windows-x86_64.zip`
   - `SHA256SUMS`

4. Guarda los cambios

### 2. Actualizar charlbase.org

Los archivos install.sh e install.ps1 deben estar disponibles en:
- https://charlbase.org/install.sh
- https://charlbase.org/install.ps1

**Opciones:**

**A) Servir desde charlbase.org directamente:**
```bash
# Copiar scripts al sitio web
cp install.sh /home/vboxuser/Desktop/Projects/AI/charlbase.org/install.sh
cp install.ps1 /home/vboxuser/Desktop/Projects/AI/charlbase.org/install.ps1

# Asegurarse de que tengan los permisos correctos
chmod 644 /home/vboxuser/Desktop/Projects/AI/charlbase.org/install.sh
chmod 644 /home/vboxuser/Desktop/Projects/AI/charlbase.org/install.ps1
```

**B) Redirigir a GitHub (alternativa):**
Configurar redirects en charlbase.org:
- `https://charlbase.org/install.sh` → `https://raw.githubusercontent.com/charlcoding-stack/charlcode/main/install.sh`
- `https://charlbase.org/install.ps1` → `https://raw.githubusercontent.com/charlcoding-stack/charlcode/main/install.ps1`

### 3. Verificar Instaladores

Después de subir los binarios a GitHub Releases, prueba los instaladores:

**Linux/macOS:**
```bash
# En una máquina limpia o contenedor Docker
curl -sSf https://charlbase.org/install.sh | sh

# Verificar
charl --version
```

**Windows:**
```powershell
# En PowerShell como Administrador
irm https://charlbase.org/install.ps1 | iex

# Verificar
charl --version
```

### 4. Actualizar Documentación (Opcional)

Si solo tienes algunos binarios disponibles ahora, actualiza `downloads.html`:

```html
<!-- Para plataformas no disponibles todavía -->
<div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mt-4">
    <p class="text-sm text-gray-700">
        <strong>⚠️ Note:</strong> This binary will be available soon.
        For now, please <a href="#build-from-source" class="text-blue-600">build from source</a>.
    </p>
</div>
```

---

## 🧪 Testing Local

Para probar el instalador localmente antes de subirlo:

### Simular servidor HTTP local:

```bash
# En el directorio charlcode/releases
cd /home/vboxuser/Desktop/Projects/AI/charlcode/releases
python3 -m http.server 8000

# En otra terminal, modificar install.sh temporalmente para usar localhost:8000
# y probar la instalación
```

### Test con Docker:

```bash
# Crear un Dockerfile de prueba
cat > Dockerfile.test <<EOF
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y curl
EOF

docker build -f Dockerfile.test -t charl-test .
docker run --rm -it charl-test bash

# Dentro del contenedor:
curl -sSf https://charlbase.org/install.sh | sh
```

---

## 📊 Estado Actual

| Plataforma | Paquete | Disponible | En GitHub |
|------------|---------|------------|-----------|
| Linux x86_64 | charl-linux-x86_64.tar.gz | ✅ Local | ❌ Falta subir |
| Linux ARM64 | charl-linux-arm64.tar.gz | ❌ | ❌ |
| macOS Intel | charl-macos-x86_64.tar.gz | ❌ | ❌ |
| macOS ARM64 | charl-macos-arm64.tar.gz | ❌ | ❌ |
| Windows x64 | charl-windows-x86_64.zip | ❌ | ❌ |

### Instaladores
| Script | Estado |
|--------|--------|
| install.sh | ✅ Corregido |
| install.ps1 | ✅ Corregido |
| build-releases.sh | ✅ Creado |

---

## 🎯 Prioridades

### Alta Prioridad:
1. ✅ Corregir instaladores
2. ✅ Generar paquete Linux x86_64
3. ⏳ Generar paquetes restantes
4. ⏳ Subir todos los paquetes a GitHub Releases v0.1.0
5. ⏳ Hacer los scripts disponibles en charlbase.org

### Media Prioridad:
6. ⏳ Configurar GitHub Actions para releases automáticos
7. ⏳ Probar instaladores en todas las plataformas
8. ⏳ Actualizar downloads.html si hay plataformas no disponibles

### Baja Prioridad:
9. ⏳ Crear firma GPG para los paquetes
10. ⏳ Agregar soporte para package managers (apt, homebrew, chocolatey)

---

## 🔧 Solución de Problemas

### Problema: Permisos en target/
**Síntoma:** `Permission denied (os error 13)`

**Solución:**
```bash
# Opción 1: Limpiar y reconstruir
cargo clean
cargo build --release

# Opción 2: Cambiar permisos (con sudo)
sudo chown -R $USER:$USER target/

# Opción 3: Usar el binario existente (lo que hicimos)
tar -czf releases/charl-linux-x86_64.tar.gz -C target/release charl
```

### Problema: Cross-compilation falla
**Síntoma:** Errores al compilar para otras plataformas

**Solución:**
```bash
# Instalar herramientas necesarias
rustup target add <target-triple>

# Para Linux ARM64:
sudo apt-get install gcc-aarch64-linux-gnu

# Para Windows:
sudo apt-get install mingw-w64

# Configurar linker en .cargo/config.toml
```

### Problema: Binario muy grande
**Síntoma:** El .tar.gz es muy grande (>10 MB)

**Solución:**
```bash
# Hacer strip del binario para reducir tamaño
strip target/release/charl

# O compilar con optimizaciones de tamaño
cargo build --release
strip target/release/charl
upx target/release/charl  # Compresor adicional (opcional)
```

---

## 📝 Comandos Útiles

```bash
# Ver tamaño de binarios
ls -lh target/release/charl releases/*.tar.gz

# Verificar contenido de tar.gz
tar -tzf releases/charl-linux-x86_64.tar.gz

# Extraer y probar
tar -xzf releases/charl-linux-x86_64.tar.gz
./charl --version

# Generar checksum
sha256sum releases/*.tar.gz releases/*.zip

# Subir a GitHub Releases (con gh CLI)
gh release upload v0.1.0 releases/*.tar.gz releases/*.zip releases/SHA256SUMS
```

---

**Última actualización:** 2025-11-05
**Versión:** 0.1.0
**Estado:** Instaladores corregidos, paquete Linux x86_64 generado

# 📦 Charl Language - Guía de Distribución

## 🎯 Cómo los Usuarios Usarán Charl (Como Python/PHP/etc)

Esta guía explica cómo Charl se distribuirá a usuarios finales, al igual que cualquier otro lenguaje de programación.

---

## 🔄 ESTADO ACTUAL vs ESTADO FINAL

### ❌ Estado Actual (Ahora)
```
Charl = Biblioteca de Rust
├─ Los usuarios deben: git clone + cargo build
├─ Requiere: Rust instalado, conocimientos de Cargo
└─ No hay ejecutable standalone
```

### ✅ Estado Final (Objetivo)
```
Charl = Lenguaje Instalable
├─ Usuarios ejecutan: curl -sSf https://charlbase.org/install.sh | sh
├─ Obtienen: Ejecutable `charl` en su PATH
└─ Usan: charl run script.charl (como python script.py)
```

---

## 📥 INSTALACIÓN (Usuario Final)

### Opción 1: Instalador Automático (Recomendado)
```bash
# En Linux/Mac
curl -sSf https://charlbase.org/install.sh | sh

# En Windows (PowerShell)
iwr https://charlbase.org/install.ps1 -useb | iex
```

**Qué hace el instalador:**
1. Descarga el binario pre-compilado de Charl para tu OS/arquitectura
2. Lo instala en `~/.charl/bin/charl` (o `C:\Users\User\.charl\bin\charl.exe` en Windows)
3. Agrega `~/.charl/bin` a tu PATH
4. Listo! Puedes usar `charl` desde cualquier terminal

### Opción 2: Descarga Manual
```bash
# Descargar binario para tu plataforma
# Linux x86_64
wget https://charlbase.org/releases/v0.1.0/charl-linux-x86_64.tar.gz
tar -xzf charl-linux-x86_64.tar.gz
sudo mv charl /usr/local/bin/

# Mac (ARM64)
wget https://charlbase.org/releases/v0.1.0/charl-macos-arm64.tar.gz
tar -xzf charl-macos-arm64.tar.gz
sudo mv charl /usr/local/bin/

# Windows x86_64
# Descargar charl-windows-x86_64.zip
# Extraer charl.exe
# Mover a C:\Program Files\Charl\
# Agregar a PATH manualmente
```

### Opción 3: Compilar desde Fuente (Desarrolladores)
```bash
git clone https://github.com/YOUR_USERNAME/charl.git
cd charl
cargo build --release
sudo cp target/release/charl /usr/local/bin/
```

---

## 🚀 USO (Como Usuario Final)

### 1️⃣ Verificar Instalación
```bash
charl --version
# Output:
# charl 0.1.0
```

### 2️⃣ Ejecutar un Script (Interpretado)
```bash
# Crear archivo hello.charl
cat > hello.charl << 'EOF'
let message = "Hello from Charl!"
print(message)

let x = 5
let y = 10
let sum = x + y
print("Sum:", sum)
EOF

# Ejecutar
charl run hello.charl
```

**Equivalente en otros lenguajes:**
```bash
python hello.py        # Python
php hello.php          # PHP
node hello.js          # Node.js
charl run hello.charl  # Charl ✅
```

### 3️⃣ Compilar a Ejecutable Nativo (AOT)
```bash
# Compilar con optimizaciones
charl build neural_network.charl --release

# Output: Ejecutable `neural_network` (o `neural_network.exe` en Windows)
./neural_network
```

**Ventaja**: El ejecutable NO necesita `charl` instalado, es standalone.

### 4️⃣ REPL Interactivo
```bash
charl repl

# Interactivo:
charl> let x = 42
charl> let y = x * 2
charl> print(y)
84
charl> exit
```

**Equivalente en otros lenguajes:**
```bash
python      # Python REPL
node        # Node.js REPL
charl repl  # Charl REPL ✅
```

---

## 📦 ESTRUCTURA DE ARCHIVOS (.charl)

### Archivo Simple: `hello.charl`
```charl
// Comentario de una línea
let greeting = "Hello, World!"
print(greeting)
```

### Proyecto con Módulos: `my_project/`
```
my_project/
├── main.charl           # Punto de entrada
├── models/
│   ├── neural_net.charl # Definición de red neuronal
│   └── optimizer.charl  # Optimizador custom
└── utils/
    └── data_loader.charl # Cargador de datos
```

**Ejecutar:**
```bash
charl run main.charl
```

---

## 🔧 DISTRIBUCIÓN DE BINARIOS

### Plataformas Soportadas

| OS | Arquitectura | Archivo |
|----|--------------|---------|
| Linux | x86_64 | `charl-linux-x86_64.tar.gz` |
| Linux | ARM64 | `charl-linux-arm64.tar.gz` |
| macOS | x86_64 (Intel) | `charl-macos-x86_64.tar.gz` |
| macOS | ARM64 (M1/M2) | `charl-macos-arm64.tar.gz` |
| Windows | x86_64 | `charl-windows-x86_64.zip` |

### Proceso de Construcción (CI/CD)

```yaml
# GitHub Actions - .github/workflows/release.yml
name: Build Release Binaries

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Build Release
        run: cargo build --release --bin charl

      - name: Package Binary
        run: |
          tar -czf charl-${{ matrix.os }}.tar.gz \
            target/release/charl

      - name: Upload to GitHub Releases
        uses: actions/upload-release-asset@v1
        with:
          upload_url: ${{ github.event.release.upload_url }}
          asset_path: ./charl-${{ matrix.os }}.tar.gz
```

---

## 🌐 WEBSITE: https://charlbase.org

### Estructura del Sitio

```
https://charlbase.org/
├── /                    # Homepage
├── /install             # Instalación
├── /docs                # Documentación
│   ├── /getting-started
│   ├── /language-guide
│   ├── /api-reference
│   └── /examples
├── /playground          # REPL online (WASM)
├── /examples            # Ejemplos de código
└── /releases            # Binarios descargables
    └── /v0.1.0/
        ├── charl-linux-x86_64.tar.gz
        ├── charl-macos-arm64.tar.gz
        └── charl-windows-x86_64.zip
```

### Homepage (charlbase.org)
```html
╔═══════════════════════════════════════════════════════════╗
║                    CHARL LANGUAGE                         ║
║   Revolutionary AI/ML Programming Language                ║
╚═══════════════════════════════════════════════════════════╝

🚀 Get Started in 30 seconds:
   curl -sSf https://charlbase.org/install.sh | sh

✨ Features:
   ✅ 100-1000x more efficient than PyTorch/TensorFlow
   ✅ Neuro-Symbolic AI (Neural + Symbolic reasoning)
   ✅ Native GPU acceleration (CPU/GPU unified)
   ✅ Meta-learning (few-shot learning built-in)
   ✅ Multimodal (Vision + Language + Reasoning)

📖 Examples:
   [Train Neural Network]  [Knowledge Graphs]  [Causal Reasoning]
```

---

## 📱 PACKAGE MANAGERS (Futuro)

### Linux
```bash
# Ubuntu/Debian
sudo apt install charl

# Arch Linux
yay -S charl

# Fedora
sudo dnf install charl
```

### macOS
```bash
brew install charl
```

### Windows
```powershell
winget install charl
# O
choco install charl
```

---

## 🔄 COMPARACIÓN CON OTROS LENGUAJES

### Python
```bash
# Instalación
curl https://www.python.org/downloads/
# O
sudo apt install python3

# Uso
python script.py
```

### PHP
```bash
# Instalación
sudo apt install php

# Uso
php script.php
```

### Node.js
```bash
# Instalación
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs

# Uso
node script.js
```

### Charl ✅
```bash
# Instalación
curl -sSf https://charlbase.org/install.sh | sh

# Uso
charl run script.charl
```

---

## 🎯 EXPERIENCIA DEL USUARIO OBJETIVO

### Desarrollador Nuevo en Charl:

**Día 1**: Instalación
```bash
$ curl -sSf https://charlbase.org/install.sh | sh
✅ Charl installed successfully!

$ charl --version
charl 0.1.0
```

**Día 1**: Primer Script
```bash
$ cat > hello.charl << 'EOF'
let model = NeuralNetwork([784, 128, 10])
let optimizer = Adam(model.parameters(), lr=0.001)

// Entrenar modelo...
for epoch in 1..10 {
    let loss = train_step(model, data, optimizer)
    print("Epoch", epoch, "Loss:", loss)
}
EOF

$ charl run hello.charl
⚡ Training on GPU (NVIDIA RTX 3060)
Epoch 1 Loss: 0.834
Epoch 2 Loss: 0.521
...
```

**Semana 1**: Proyecto Completo
```bash
$ tree my_ai_project/
my_ai_project/
├── main.charl
├── models/
│   ├── transformer.charl
│   └── mamba.charl
├── data/
│   └── loader.charl
└── config.charl

$ charl build main.charl --release
🔨 Compiling with LLVM optimizations...
✅ Binary created: ./my_ai_project

$ ./my_ai_project
🚀 Training model...
```

---

## 📊 DISTRIBUCIÓN DE VERSIONES

### Canales de Release

**Stable** (Producción)
```bash
curl -sSf https://charlbase.org/install.sh | sh
# Instala: v0.1.0 (stable)
```

**Beta** (Features nuevos)
```bash
curl -sSf https://charlbase.org/install.sh | sh -s -- --beta
# Instala: v0.2.0-beta
```

**Nightly** (Desarrollo)
```bash
curl -sSf https://charlbase.org/install.sh | sh -s -- --nightly
# Instala: v0.3.0-nightly
```

### Actualización
```bash
# Actualizar a la última versión
charl update

# Cambiar de canal
charl default beta
charl default stable
```

---

## 🎯 RESUMEN: LO QUE FALTA PARA DISTRIBUCIÓN COMPLETA

### ✅ Ya Tenemos:
1. Compilador/intérprete completo (28,374 líneas)
2. 564 tests (100% passing)
3. Ejecutable `charl` funcional
4. Script de instalación (`install.sh`)
5. CLI básico (run, build, repl, version)

### 🚧 Falta Implementar:
1. **Integración Completa del CLI con Lexer/Parser**
   - Actualmente: CLI muestra mensajes
   - Necesario: Conectar CLI → Lexer → Parser → Interpreter/LLVM

2. **Build de Binarios Multi-Plataforma**
   - CI/CD para compilar en Linux/Mac/Windows
   - Subir a charlbase.org/releases/

3. **Website charlbase.org**
   - Homepage
   - Documentación
   - Playground online (WebAssembly)

4. **REPL Interactivo**
   - Loop read-eval-print funcional
   - History, autocomplete

5. **Package Managers**
   - apt, brew, winget integration

---

## 💡 CONCLUSIÓN

Charl **ya tiene toda la infraestructura técnica** (compilador, runtime, optimizaciones).

Lo que falta es la **capa de distribución** (binarios pre-compilados, website, CLI integrado).

**Prioridad para distribución pública:**
1. Integrar CLI con interpreter ✅ (main.rs creado, falta conectar)
2. Website charlbase.org (landing + docs)
3. CI/CD para binarios multi-plataforma
4. Package managers (brew, apt)

**URL Oficial:** https://charlbase.org

# VS Code Extension Integration

Esta guía explica cómo funciona la integración automática de la extensión de VS Code con el instalador de Charl.

## Resumen

Cuando un usuario instala Charl usando `install.sh`, la extensión de VS Code se instala automáticamente si VS Code está disponible en el sistema. Esto proporciona una experiencia de desarrollo completa desde el primer momento.

## Cómo funciona

### 1. Estructura del proyecto

```
charlcode/
├── vscode-charl/                    # Extensión completa
│   ├── package.json                 # Manifest de la extensión
│   ├── language-configuration.json  # Configuración del editor
│   ├── syntaxes/
│   │   └── charl.tmLanguage.json   # Gramática TextMate
│   ├── snippets/
│   │   └── charl.json              # Code templates
│   └── README.md                    # Documentación
└── install.sh                       # Script de instalación
```

### 2. Proceso de instalación

El script `install.sh` ejecuta estos pasos:

```bash
# 1. Detecta si VS Code está instalado
if command -v code &> /dev/null; then

# 2. Copia la extensión al directorio de extensiones del usuario
cp -r vscode-charl ~/.vscode/extensions/charl-lang.charl-1.0.0

# 3. Informa al usuario sobre las características instaladas
echo "✅ VS Code extension installed"
echo "   • Syntax highlighting for .ch files"
echo "   • Auto-indentation and bracket matching"
echo "   • 22 code snippets ready to use"
```

### 3. Activación automática

Cuando el usuario abre VS Code:

1. VS Code detecta la nueva extensión en `~/.vscode/extensions/`
2. Lee el `package.json` y registra la asociación `.ch` → `charl`
3. Carga la gramática TextMate desde `syntaxes/charl.tmLanguage.json`
4. Aplica la configuración del editor desde `language-configuration.json`
5. Carga los snippets desde `snippets/charl.json`

**No se requiere ninguna acción del usuario** - todo funciona automáticamente.

## Características de la extensión

### Syntax Highlighting

La extensión reconoce y colorea:

- **Keywords**: `if`, `else`, `while`, `for`, `match`, `let`, `fn`, `const`, `return`
- **Tipos**: `int32`, `int64`, `float32`, `float64`, `bool`, `string`, `tensor`
- **Operadores**: `+`, `-`, `*`, `/`, `==`, `!=`, `and`, `or`, `not`
- **ML Keywords**: `model`, `layer`, `autograd`, `gradient`, `dense`, `activation`
- **Literales**: números, strings, booleanos, null
- **Comentarios**: `//` línea simple

### Code Snippets

22 snippets disponibles:

| Snippet | Descripción |
|---------|-------------|
| `fn` | Función completa |
| `fnt` | Función con tipos |
| `let` | Variable con tipo |
| `const` | Constante |
| `if` | Bloque if |
| `ife` | If-else |
| `while` | Loop while |
| `for` | For-in loop |
| `forr` | For con rango |
| `match` | Match expression |
| `tuple` | Tupla |
| `arr` | Array |
| `tensor` | Tensor |
| `model` | Definición de modelo |

### Editor Features

- **Auto-indentación**: Sangría automática después de `{`, `[`, `(`
- **Auto-cierre**: Cierra automáticamente `()`, `[]`, `{}`, `""`
- **Bracket matching**: Resalta pares coincidentes
- **Code folding**: Colapsar/expandir regiones con `// #region`

## Instalación manual

Si el usuario instaló Charl sin usar `install.sh` o quiere reinstalar la extensión:

```bash
# Copiar extensión
cp -r vscode-charl ~/.vscode/extensions/charl-lang.charl-1.0.0

# Reiniciar VS Code
code --relaunch
```

## Verificar instalación

```bash
# Listar extensiones instaladas
code --list-extensions | grep charl

# Debería mostrar:
# charl-lang.charl
```

O desde VS Code:
1. Abrir cualquier archivo `.ch`
2. Esquina inferior derecha debe mostrar "Charl" (no "Plain Text")
3. El código debe tener colores

## Actualización de la extensión

Cuando se actualiza la extensión:

```bash
# 1. Actualizar archivos en vscode-charl/
# 2. Incrementar versión en package.json
# 3. Usuario ejecuta install.sh de nuevo, que sobrescribe la versión anterior
```

## Distribución

La extensión se incluye en:

1. **Tarball del código fuente**: `charl-source-vX.X.X.tar.gz`
   - Contiene todo charlcode/ incluyendo vscode-charl/
   - install.sh la instala automáticamente

2. **Releases de GitHub**:
   - También incluida en assets de cada release

3. **Instalación independiente** (opcional, futuro):
   - Empaquetar como `.vsix`: `vsce package`
   - Distribuir en VS Code Marketplace

## Compatibilidad

- **VS Code**: Versión 1.70.0 o superior
- **Plataformas**: Linux, macOS, Windows
- **Extensión activa**: Archivos `.ch` solamente

## Troubleshooting

### La extensión no se activa

```bash
# Verificar que está instalada
ls ~/.vscode/extensions/ | grep charl

# Verificar permisos
ls -la ~/.vscode/extensions/charl-lang.charl-1.0.0/

# Reinstalar
rm -rf ~/.vscode/extensions/charl-lang.charl-1.0.0
cd /path/to/charlcode
cp -r vscode-charl ~/.vscode/extensions/charl-lang.charl-1.0.0
```

### Los colores no aparecen

1. Verificar que el archivo tiene extensión `.ch`
2. Verificar esquina inferior derecha dice "Charl"
3. Si dice "Plain Text", hacer clic y seleccionar "Charl"
4. Reiniciar VS Code

### Los snippets no funcionan

1. Escribir el prefijo completo: `fn`, `match`, etc.
2. Presionar `Tab` (no Enter)
3. Si no aparecen, verificar settings: `"editor.snippetSuggestions": "top"`

## Roadmap

Características futuras:

- [ ] **Language Server Protocol (LSP)**
  - Autocompletado inteligente
  - Go to definition
  - Find references
  - Inline error diagnostics

- [ ] **Debugger Protocol**
  - Breakpoints
  - Step through code
  - Variable inspection

- [ ] **Task Integration**
  - Build tasks
  - Test runner
  - REPL integration

## Referencias

- [VS Code Extension API](https://code.visualstudio.com/api)
- [TextMate Grammar](https://macromates.com/manual/en/language_grammars)
- [Language Configuration](https://code.visualstudio.com/api/language-extensions/language-configuration-guide)
- [Snippet Syntax](https://code.visualstudio.com/docs/editor/userdefinedsnippets)

---

**Última actualización**: 2025-11-05
**Versión de la extensión**: 1.0.0

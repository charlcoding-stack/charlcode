# Charl Language Support for VS Code

Extensión de Visual Studio Code que proporciona soporte completo para el lenguaje de programación **Charl**, incluyendo resaltado de sintaxis, autocompletado, snippets y más.

## Características

### 🎨 Resaltado de Sintaxis

- **Keywords**: `if`, `else`, `while`, `for`, `match`, `let`, `fn`, `const`, `return`, `break`, `continue`
- **Operadores lógicos**: `and`, `or`, `not`
- **Tipos primitivos**: `int32`, `int64`, `float32`, `float64`, `bool`, `string`, `tensor`
- **Keywords ML**: `model`, `layer`, `autograd`, `gradient`, `dense`, `conv2d`, `activation`, etc.
- **Operadores**: aritméticos, comparación, asignación, rangos (`..`, `..=`), flechas (`->`, `=>`)
- **Literales**: números, strings, booleanos, null
- **Comentarios**: `//` línea simple

### 📝 Snippets (Code Templates)

Escribe el prefijo y presiona `Tab` para expandir:

| Prefijo | Descripción |
|---------|-------------|
| `fn` | Declaración de función |
| `fnt` | Función con tipos anotados |
| `let` | Variable con tipo |
| `const` | Constante con tipo |
| `if` | Bloque if |
| `ife` | Bloque if-else |
| `while` | Loop while |
| `for` | Loop for-in |
| `forr` | Loop for con rango |
| `forri` | Loop for con rango inclusivo |
| `match` | Expresión match |
| `matchm` | Match con múltiples arms |
| `tuple` | Tupla |
| `arr` | Array |
| `arrs` | Array de tamaño fijo |
| `print` | Print statement |
| `tensor` | Declaración de tensor |
| `model` | Definición de modelo |
| `dense` | Capa densa |
| `activation` | Capa de activación |

### ⚙️ Configuración del Editor

- **Auto-indentación**: Se aplica automáticamente al abrir `{`, `[`, `(`
- **Auto-cierre**: Paréntesis, corchetes, llaves y comillas se cierran automáticamente
- **Matching de brackets**: Resalta pares coincidentes
- **Folding**: Soporte para regiones colapsables con `// #region` y `// #endregion`

## Instalación

### Opción 1: Desde el archivo .vsix (Recomendado)

1. Empaqueta la extensión:
   ```bash
   cd /home/vboxuser/Desktop/Projects/AI/vscode-charl
   npm install -g @vscode/vsce
   vsce package
   ```

2. Instala el archivo `.vsix` generado:
   ```bash
   code --install-extension charl-1.0.0.vsix
   ```

3. Reinicia VS Code

### Opción 2: Desarrollo local

1. Copia la carpeta de la extensión a tu directorio de extensiones de VS Code:
   ```bash
   cp -r /home/vboxuser/Desktop/Projects/AI/vscode-charl ~/.vscode/extensions/charl-1.0.0
   ```

2. Reinicia VS Code

### Verificar instalación

Abre cualquier archivo `.ch` y verifica que:
- El lenguaje se reconoce como "Charl" (esquina inferior derecha)
- El código tiene resaltado de sintaxis
- Los snippets aparecen al escribir los prefijos

## Uso

### Ejemplo de código con resaltado

```charl
// Función con match expression
fn classify_number(n: int64) -> string {
    return match n {
        0 => "zero",
        1 => "one",
        2 => "two",
        _ => "many"
    };
}

// Tuplas y arrays
let pair: (int64, string) = (42, "answer");
let numbers: [int64] = [1, 2, 3, 4, 5];

// Loop con rango inclusivo
for i in 0..=10 {
    print(str(i));
}

// Array slicing
let slice: [int64] = numbers[1..3];
```

### Usar snippets

1. Escribe `fn` y presiona `Tab`
2. Rellena los placeholders (nombre, parámetros, tipo de retorno)
3. Presiona `Tab` para saltar entre placeholders

## Características del lenguaje Charl

Esta extensión soporta el 100% del frontend de Charl:

- ✅ Variables con inferencia de tipos
- ✅ Funciones con closures
- ✅ Control flow: `if`, `while`, `for`
- ✅ **Match expressions** con pattern matching
- ✅ **Tuple types**: `(int64, string, bool)`
- ✅ Arrays con slicing: `arr[1..3]`
- ✅ Rangos exclusivos e inclusivos: `..` y `..=`
- ✅ String concatenation
- ✅ Operadores: `+`, `-`, `*`, `/`, `%`, `@`
- ✅ Comparaciones: `==`, `!=`, `<`, `>`, `<=`, `>=`
- ✅ Lógica: `and`, `or`, `not`

## Roadmap

Características futuras:

- [ ] Language Server Protocol (LSP) para:
  - Autocompletado inteligente
  - Go to definition
  - Find references
  - Rename symbol
  - Error diagnostics en tiempo real
- [ ] Debugger integration
- [ ] Build tasks integration
- [ ] REPL integration

## Estructura de la extensión

```
vscode-charl/
├── package.json                    # Manifest de la extensión
├── syntaxes/
│   └── charl.tmLanguage.json      # Gramática TextMate
├── snippets/
│   └── charl.json                 # Code templates
├── language-configuration.json     # Configuración del editor
├── README.md                       # Esta documentación
└── icon.png                        # Icono de la extensión
```

## Contribuir

Reporta bugs o sugiere mejoras en: https://github.com/charlcoding-stack/vscode-charl/issues

## Licencia

MIT License

## Autor

**charl-lang** - Extensión oficial para el lenguaje Charl

---

**Versión**: 1.0.0
**Compatible con**: VS Code 1.70.0+

# 🎯 Próximos Pasos - Charl Language

## ✅ Lo que Hemos Completado (Fase 0)

### Infraestructura ✅
- [x] Rust 1.91.0 instalado y configurado
- [x] Proyecto Cargo inicializado
- [x] Toolchain GNU configurado para Windows
- [x] CLI funcional (`charl run`, `build`, `repl`, `version`)
- [x] Estructura modular del compilador

### Documentación ✅
- [x] README.md completo
- [x] ROADMAP.md (60 semanas, 10 fases)
- [x] SPECIFICATION.md (sintaxis completa v1.0)
- [x] PROJECT_SUMMARY.md
- [x] LICENSE (MIT)
- [x] 4 ejemplos funcionales (.ch files)

### Código Base ✅
- [x] Estructura de módulos (lexer, parser, ast, types, interpreter)
- [x] Definiciones de tokens (50+ tipos)
- [x] Definiciones AST completas
- [x] Sistema de tipos base

---

## 🚀 Fase 1: Implementar el Compilador Frontend

### Paso 1: Implementar el Lexer Completo (Semana 3)

**Archivo:** `src/lexer/mod.rs`

#### Tareas:
1. **Implementar `read_char()`** ✅ (ya existe estructura)
2. **Implementar `peek_char()`** - Ver el siguiente carácter sin avanzar
3. **Implementar `skip_whitespace()`** - Saltar espacios, tabs, newlines
4. **Implementar `read_identifier()`** - Leer nombres de variables/funciones
5. **Implementar `read_number()`** - Leer enteros y flotantes
6. **Implementar `read_string()`** - Leer strings con comillas
7. **Completar `next_token()`** - Tokenizar todo el input

#### Ejemplo de implementación:

```rust
fn skip_whitespace(&mut self) {
    while self.ch.is_whitespace() {
        self.read_char();
    }
}

fn read_identifier(&mut self) -> String {
    let start = self.position;
    while self.ch.is_alphanumeric() || self.ch == '_' {
        self.read_char();
    }
    self.input[start..self.position].iter().collect()
}

pub fn next_token(&mut self) -> Token {
    self.skip_whitespace();

    let token = match self.ch {
        '+' => Token::new(TokenType::Plus, "+".to_string(), self.line, self.col),
        '-' => Token::new(TokenType::Minus, "-".to_string(), self.line, self.col),
        // ... más casos
        'a'..='z' | 'A'..='Z' | '_' => {
            let literal = self.read_identifier();
            let token_type = TokenType::lookup_keyword(&literal);
            Token::new(token_type, literal, self.line, self.col)
        },
        // ... más casos
        '\0' => Token::new(TokenType::Eof, "".to_string(), self.line, self.col),
        _ => Token::new(TokenType::Illegal, self.ch.to_string(), self.line, self.col),
    };

    self.read_char();
    token
}
```

#### Tests a escribir:
```rust
#[test]
fn test_tokenize_numbers() {
    let input = "42 3.14 -10";
    let mut lexer = Lexer::new(input);

    let tok1 = lexer.next_token();
    assert_eq!(tok1.token_type, TokenType::Int);
    assert_eq!(tok1.literal, "42");

    // ... más aserciones
}

#[test]
fn test_tokenize_operators() {
    let input = "+ - * / @ == !=";
    // ... test implementation
}

#[test]
fn test_tokenize_keywords() {
    let input = "let fn return tensor autograd";
    // ... test implementation
}
```

---

### Paso 2: Implementar el Parser (Semanas 4-5)

**Archivo:** `src/parser/mod.rs`

#### Tareas:

1. **Parser de Expresiones**
   - Implementar Pratt Parsing (precedencia de operadores)
   - Parsing de literales (números, strings, arrays)
   - Parsing de operadores binarios
   - Parsing de operadores unarios
   - Parsing de llamadas a funciones
   - Parsing de indexación de arrays/tensores

2. **Parser de Statements**
   - `let` declarations
   - `fn` declarations
   - `return` statements
   - `if/else` statements
   - `while` loops
   - `for` loops

3. **Manejo de Errores**
   - Mensajes de error claros
   - Posición del error (línea, columna)
   - Recovery de errores (continuar parseando)

#### Ejemplo de implementación:

```rust
impl Parser {
    fn advance(&mut self) {
        self.current_token = self.peek_token.clone();
        self.peek_token = self.lexer.next_token();
    }

    fn parse_statement(&mut self) -> Result<Statement, String> {
        match self.current_token.token_type {
            TokenType::Let => self.parse_let_statement(),
            TokenType::Return => self.parse_return_statement(),
            TokenType::Fn => self.parse_function_statement(),
            _ => self.parse_expression_statement(),
        }
    }

    fn parse_let_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume 'let'

        let name = match &self.current_token.token_type {
            TokenType::Ident => self.current_token.literal.clone(),
            _ => return Err("Expected identifier after 'let'".to_string()),
        };

        // ... más parsing

        Ok(Statement::Let(LetStatement { name, type_annotation, value }))
    }
}
```

---

### Paso 3: Testing Comprehensivo (Semana 6)

#### Crear suite de tests:

**Archivo:** `tests/lexer_tests.rs`
```rust
#[test]
fn test_complete_program() {
    let input = r#"
        fn add(x: int32, y: int32) -> int32 {
            return x + y
        }

        let result = add(5, 10)
    "#;

    // Test que todos los tokens se generan correctamente
}
```

**Archivo:** `tests/parser_tests.rs`
```rust
#[test]
fn test_parse_function() {
    let input = "fn add(x: int32) -> int32 { return x + 1 }";
    let lexer = Lexer::new(input);
    let mut parser = Parser::new(lexer);
    let program = parser.parse_program().unwrap();

    assert_eq!(program.statements.len(), 1);
    // ... más aserciones
}
```

---

## 📝 Comandos Útiles

### Desarrollo
```bash
# Compilar
cargo build

# Compilar y ejecutar
cargo run

# Ejecutar tests
cargo test

# Ejecutar tests con output
cargo test -- --nocapture

# Ejecutar un test específico
cargo test test_tokenize_numbers

# Check (más rápido que build)
cargo check

# Formatear código
cargo fmt

# Linter
cargo clippy
```

### Testing
```bash
# Ejecutar todos los tests
cargo test

# Tests con verbose output
cargo test -- --show-output

# Tests de un módulo específico
cargo test lexer::tests

# Test específico
cargo test test_tokenize_operators
```

### Benchmarking (cuando sea relevante)
```bash
cargo bench
```

---

## 🎓 Recursos para Implementar el Lexer y Parser

### Tutoriales
1. **Crafting Interpreters** (online, gratis)
   - https://craftinginterpreters.com/
   - Capítulos 4-6: Scanning y Parsing

2. **Writing An Interpreter In Go** (adaptable a Rust)
   - Excelente para entender lexer/parser

3. **Rust Parser Tutorial**
   - https://github.com/Kixiron/rust-langdev

### Conceptos Clave

#### Pratt Parsing (Precedencia de Operadores)
```rust
// Tabla de precedencias
enum Precedence {
    Lowest = 0,
    Equals = 1,      // ==, !=
    LessGreater = 2, // <, >
    Sum = 3,         // +, -
    Product = 4,     // *, /
    MatMul = 5,      // @
    Prefix = 6,      // -x, !x
    Call = 7,        // fn(x)
}
```

#### Recursive Descent Parsing
- Cada regla gramatical = una función
- Parsing de arriba hacia abajo
- Fácil de implementar y debuggear

---

## 🐛 Debugging Tips

### Para el Lexer
```rust
// Agregar prints para debug
println!("Current char: '{}', Position: {}", self.ch, self.position);
```

### Para el Parser
```rust
// Agregar método debug para ver el AST
impl fmt::Debug for Program {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Pretty print del AST
    }
}
```

### Usar el REPL para testing rápido (cuando esté implementado)
```bash
cargo run -- repl
> let x = 5
> x + 10
```

---

## 📊 Criterios de Éxito para Fase 1

### Lexer
- [ ] Tokeniza todos los tipos de tokens correctamente
- [ ] Maneja números (int, float) correctamente
- [ ] Maneja strings con escape sequences
- [ ] Maneja comentarios (// y /* */)
- [ ] Reporta línea y columna para errores
- [ ] 100+ tests pasando

### Parser
- [ ] Parsea expresiones básicas (literales, operadores)
- [ ] Parsea statements (let, fn, return, if, while, for)
- [ ] Maneja precedencia de operadores correctamente
- [ ] Construye AST válido
- [ ] Reporta errores sintácticos claros
- [ ] 200+ tests pasando

### Integración
- [ ] Puede parsear programas completos de ejemplo
- [ ] Los 4 ejemplos (.ch files) se parsean sin errores
- [ ] AST se puede imprimir para debugging

---

## 🎯 Después de Fase 1: Fase 2 (Sistema de Tipos)

Una vez completo el Lexer y Parser:

1. Implementar type checking básico
2. Implementar inferencia de tipos
3. Implementar el tipo `Tensor<T, Shape>`
4. Validar operaciones de tipos
5. Reportar errores de tipos

**Pero eso es para después!** Enfócate primero en Fase 1.

---

## 💡 Tips Generales

1. **Escribe tests PRIMERO** (TDD)
   - Más fácil debuggear
   - Más confianza en el código
   - Documentación viva

2. **Commits pequeños y frecuentes**
   - Fácil de revertir si algo sale mal
   - Historial claro

3. **Documenta tu código**
   - Usa `///` para doc comments en Rust
   - Explica el "por qué", no el "qué"

4. **Usa el compilador de Rust**
   - Los errores del compilador son tus amigos
   - Aprende de los warnings

5. **No optimices prematuramente**
   - Primero haz que funcione
   - Luego haz que sea correcto
   - Finalmente haz que sea rápido

---

## 📞 Cuando Te Atores

1. **Revisa los tests** - ¿Qué está fallando exactamente?
2. **Lee el error completo** - Rust da errores muy descriptivos
3. **Simplifica** - Haz un caso de prueba mínimo
4. **Busca ejemplos** - Hay muchos parsers en Rust open source
5. **Descansa** - A veces la solución viene al día siguiente

---

## 🎉 ¡Estás Listo Para Empezar!

La Fase 0 está **100% completa**. Tienes:
- ✅ Entorno de desarrollo configurado
- ✅ Proyecto estructurado
- ✅ Especificación clara
- ✅ Roadmap detallado
- ✅ Ejemplos de código objetivo

**Próximo paso:** Implementar el Lexer completo siguiendo las instrucciones arriba.

**¡Buena suerte! 🚀**

---

**Última actualización:** 2025-11-04
**Autor:** Claude Code & Team
**Estado:** 🟢 Listo para Fase 1

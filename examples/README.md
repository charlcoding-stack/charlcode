# Charl Example Programs

This directory contains example programs demonstrating Charl's features.

## Basic Examples

### hello.charl - Variables and Basic Operations
```bash
charl run examples/hello.charl
```
Demonstrates:
- Variable declarations
- String literals
- Integer operations

### math.charl - Mathematical Operations
```bash
charl run examples/math.charl
```
Demonstrates:
- Arithmetic operations (+, *, etc.)
- Variable composition

### function.charl - Function Definitions
```bash
charl run examples/function.charl --verbose
```
Demonstrates:
- Function declarations with type annotations
- Function calls
- Return statements

### arrays.charl - Array Operations
```bash
charl run examples/arrays.charl
```
Demonstrates:
- Array literals
- Array indexing
- Array element access

## Running Examples

### Quick Run (Silent)
```bash
charl run examples/hello.charl
```

### Verbose Mode (Show Details)
```bash
charl run examples/hello.charl --verbose
```

Verbose mode shows:
- Source code
- Lexing and parsing stages
- Number of statements parsed
- Execution result

## Example Output

```bash
$ charl run examples/function.charl --verbose
🚀 Running Charl script: examples/function.charl
📝 Source code (142 bytes):
--------------------------------------------------
// Function definition and call with type annotations
fn square(x: int32) -> int32 {
    return x * x
}

let num = 5
let result = square(num)
--------------------------------------------------

🔤 Lexing...
🌳 Parsing...
✅ Parsed 3 statements
⚡ Executing...

✅ Execution completed successfully
📊 Result: Integer(25)
```

## Coming Soon

- `neural_net.charl` - Neural network training
- `autograd.charl` - Automatic differentiation
- `gpu_demo.charl` - GPU acceleration
- `symbolic.charl` - Symbolic reasoning

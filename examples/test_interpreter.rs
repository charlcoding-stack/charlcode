// Test program to demonstrate the interpreter in action
// Run with: cargo run --example test_interpreter

fn main() {
    use charl::lexer::Lexer;
    use charl::parser::Parser;
    use charl::interpreter::Interpreter;

    let examples = vec![
        ("Arithmetic", "5 + 3 * 2"),
        ("Variables", "let x = 10\nlet y = 20\nx + y"),
        ("Functions", r#"
            fn double(x: int32) -> int32 {
                return x * 2
            }
            fn quadruple(x: int32) -> int32 {
                return double(double(x))
            }
            quadruple(5)
        "#),
        ("Arrays", "let arr = [1, 2, 3, 4, 5]\narr[2]"),
        ("Closures", r#"
            let multiplier = 5
            fn multiply_by(x: int32) -> int32 {
                return x * multiplier
            }
            multiply_by(7)
        "#),
        ("Complex Expression", r#"
            fn add(a: int32, b: int32) -> int32 { return a + b }
            fn sub(a: int32, b: int32) -> int32 { return a - b }
            fn mul(a: int32, b: int32) -> int32 { return a * b }

            let x = 10
            let y = 5
            let result = mul(add(x, y), sub(x, y))
            result
        "#),
    ];

    println!("🚀 Charl Interpreter Demonstration\n");
    println!("{}",  "=".repeat(70));

    for (name, code) in examples {
        println!("\n📝 Example: {}", name);
        println!("{}", "-".repeat(70));
        println!("Code:\n{}\n", code.trim());

        let lexer = Lexer::new(code);
        let mut parser = Parser::new(lexer);

        match parser.parse_program() {
            Ok(program) => {
                let mut interpreter = Interpreter::new();
                match interpreter.eval(program) {
                    Ok(value) => {
                        println!("✅ Result: {:?}", value);
                    }
                    Err(e) => {
                        println!("❌ Runtime Error: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("❌ Parse Error: {}", e);
            }
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("\n✅ All examples completed!");
}

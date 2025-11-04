// Test program to demonstrate the lexer in action
// Run with: cargo run --example test_lexer

fn main() {
    use charl::lexer::Lexer;

    let code = r#"
    // Simple function with tensor
    fn calculate(x: tensor<float32, [3]>) -> float32 {
        let sum = x[0] + x[1] + x[2]
        return sum * 2.0
    }

    let data: tensor<float32, [3]> = [1.0, 2.0, 3.0]
    let result = calculate(data)
    "#;

    println!("🔍 Charl Lexer Demonstration\n");
    println!("Input code:");
    println!("{}", code);
    println!("\n{}", "=".repeat(60));
    println!("\nTokens generated:\n");

    let mut lexer = Lexer::new(code);
    let mut count = 0;

    loop {
        let token = lexer.next_token();
        count += 1;

        if token.token_type == charl::lexer::TokenType::Eof {
            break;
        }

        println!(
            "{:3}. {:20} {:20} (line {}, col {})",
            count,
            format!("{:?}", token.token_type),
            format!("\"{}\"", token.literal),
            token.line,
            token.column
        );
    }

    println!("\n{}", "=".repeat(60));
    println!("\n✅ Total tokens: {}", count);
    println!("✅ Lexer completed successfully!");
}

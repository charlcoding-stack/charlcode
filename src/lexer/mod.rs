// Lexer module - Tokenization of Charl source code
// Phase 1.1: Convert source code into tokens

pub mod token;

pub use token::{Token, TokenType};

/// The Lexer converts raw source code into a stream of tokens
pub struct Lexer {
    input: Vec<char>,
    position: usize,      // current position in input
    read_position: usize, // current reading position (after current char)
    ch: char,             // current char under examination
    line: usize,          // current line number (for error reporting)
    column: usize,        // current column number (for error reporting)
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        let mut lexer = Lexer {
            input: input.chars().collect(),
            position: 0,
            read_position: 0,
            ch: '\0',
            line: 1,
            column: 0,
        };
        lexer.read_char();
        lexer
    }

    fn read_char(&mut self) {
        if self.read_position >= self.input.len() {
            self.ch = '\0';
        } else {
            self.ch = self.input[self.read_position];
        }
        self.position = self.read_position;
        self.read_position += 1;
        self.column += 1;

        if self.ch == '\n' {
            self.line += 1;
            self.column = 0;
        }
    }

    fn peek_char(&self) -> char {
        if self.read_position >= self.input.len() {
            '\0'
        } else {
            self.input[self.read_position]
        }
    }

    fn skip_whitespace(&mut self) {
        while self.ch.is_whitespace() {
            self.read_char();
        }
    }

    fn skip_comment(&mut self) {
        if self.ch == '/' && self.peek_char() == '/' {
            // Single-line comment: skip until end of line
            while self.ch != '\n' && self.ch != '\0' {
                self.read_char();
            }
        } else if self.ch == '/' && self.peek_char() == '*' {
            // Multi-line comment: skip until */
            self.read_char(); // skip '/'
            self.read_char(); // skip '*'

            while !(self.ch == '*' && self.peek_char() == '/') && self.ch != '\0' {
                self.read_char();
            }

            if self.ch != '\0' {
                self.read_char(); // skip '*'
                self.read_char(); // skip '/'
            }
        }
    }

    fn read_identifier(&mut self) -> String {
        let start = self.position;
        while self.ch.is_alphanumeric() || self.ch == '_' {
            self.read_char();
        }
        self.input[start..self.position].iter().collect()
    }

    fn read_number(&mut self) -> (String, TokenType) {
        let start = self.position;
        let mut is_float = false;

        // Read digits
        while self.ch.is_numeric() {
            self.read_char();
        }

        // Check for decimal point
        if self.ch == '.' && self.peek_char().is_numeric() {
            is_float = true;
            self.read_char(); // consume '.'

            while self.ch.is_numeric() {
                self.read_char();
            }
        }

        // Check for scientific notation (e.g., 1.5e-10)
        if self.ch == 'e' || self.ch == 'E' {
            is_float = true;
            self.read_char(); // consume 'e'

            if self.ch == '+' || self.ch == '-' {
                self.read_char(); // consume sign
            }

            while self.ch.is_numeric() {
                self.read_char();
            }
        }

        let literal: String = self.input[start..self.position].iter().collect();
        let token_type = if is_float {
            TokenType::Float
        } else {
            TokenType::Int
        };

        (literal, token_type)
    }

    fn read_string(&mut self) -> String {
        self.read_char(); // skip opening quote
        let start = self.position;

        while self.ch != '"' && self.ch != '\0' {
            if self.ch == '\\' && self.peek_char() == '"' {
                // Handle escaped quote
                self.read_char(); // skip backslash
            }
            self.read_char();
        }

        let literal: String = self.input[start..self.position].iter().collect();

        if self.ch == '"' {
            self.read_char(); // skip closing quote
        }

        literal
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        // Skip comments
        while self.ch == '/' && (self.peek_char() == '/' || self.peek_char() == '*') {
            self.skip_comment();
            self.skip_whitespace();
        }

        let line = self.line;
        let column = self.column;

        match self.ch {
            // Operators
            '+' => {
                self.read_char();
                Token::new(TokenType::Plus, "+".to_string(), line, column)
            }
            '-' => {
                if self.peek_char() == '>' {
                    self.read_char();
                    self.read_char();
                    Token::new(TokenType::Arrow, "->".to_string(), line, column)
                } else {
                    self.read_char();
                    Token::new(TokenType::Minus, "-".to_string(), line, column)
                }
            }
            '*' => {
                self.read_char();
                Token::new(TokenType::Multiply, "*".to_string(), line, column)
            }
            '/' => {
                self.read_char();
                Token::new(TokenType::Divide, "/".to_string(), line, column)
            }
            '%' => {
                self.read_char();
                Token::new(TokenType::Modulo, "%".to_string(), line, column)
            }
            '@' => {
                self.read_char();
                Token::new(TokenType::MatMul, "@".to_string(), line, column)
            }

            // Comparison operators
            '=' => {
                if self.peek_char() == '=' {
                    self.read_char();
                    self.read_char();
                    Token::new(TokenType::Equal, "==".to_string(), line, column)
                } else if self.peek_char() == '>' {
                    self.read_char();
                    self.read_char();
                    Token::new(TokenType::FatArrow, "=>".to_string(), line, column)
                } else {
                    self.read_char();
                    Token::new(TokenType::Assign, "=".to_string(), line, column)
                }
            }
            '!' => {
                if self.peek_char() == '=' {
                    self.read_char();
                    self.read_char();
                    Token::new(TokenType::NotEqual, "!=".to_string(), line, column)
                } else {
                    self.read_char();
                    Token::new(TokenType::Not, "!".to_string(), line, column)
                }
            }
            '<' => {
                if self.peek_char() == '=' {
                    self.read_char();
                    self.read_char();
                    Token::new(TokenType::LessEqual, "<=".to_string(), line, column)
                } else {
                    self.read_char();
                    Token::new(TokenType::LessThan, "<".to_string(), line, column)
                }
            }
            '>' => {
                if self.peek_char() == '=' {
                    self.read_char();
                    self.read_char();
                    Token::new(TokenType::GreaterEqual, ">=".to_string(), line, column)
                } else {
                    self.read_char();
                    Token::new(TokenType::GreaterThan, ">".to_string(), line, column)
                }
            }

            // Logical operators
            '&' => {
                if self.peek_char() == '&' {
                    self.read_char();
                    self.read_char();
                    Token::new(TokenType::And, "&&".to_string(), line, column)
                } else {
                    self.read_char();
                    Token::new(TokenType::Illegal, "&".to_string(), line, column)
                }
            }
            '|' => {
                if self.peek_char() == '|' {
                    self.read_char();
                    self.read_char();
                    Token::new(TokenType::Or, "||".to_string(), line, column)
                } else {
                    self.read_char();
                    Token::new(TokenType::Illegal, "|".to_string(), line, column)
                }
            }

            // Delimiters
            ',' => {
                self.read_char();
                Token::new(TokenType::Comma, ",".to_string(), line, column)
            }
            ';' => {
                self.read_char();
                Token::new(TokenType::Semicolon, ";".to_string(), line, column)
            }
            ':' => {
                self.read_char();
                Token::new(TokenType::Colon, ":".to_string(), line, column)
            }
            '.' => {
                if self.peek_char() == '.' {
                    self.read_char(); // consume first '.'
                    self.read_char(); // consume second '.'

                    // Check if followed by '=' for inclusive range
                    if self.ch == '=' {
                        self.read_char(); // consume '='
                        Token::new(TokenType::DotDotEqual, "..=".to_string(), line, column)
                    } else {
                        Token::new(TokenType::DotDot, "..".to_string(), line, column)
                    }
                } else {
                    self.read_char();
                    Token::new(TokenType::Dot, ".".to_string(), line, column)
                }
            }

            // Brackets
            '(' => {
                self.read_char();
                Token::new(TokenType::LParen, "(".to_string(), line, column)
            }
            ')' => {
                self.read_char();
                Token::new(TokenType::RParen, ")".to_string(), line, column)
            }
            '{' => {
                self.read_char();
                Token::new(TokenType::LBrace, "{".to_string(), line, column)
            }
            '}' => {
                self.read_char();
                Token::new(TokenType::RBrace, "}".to_string(), line, column)
            }
            '[' => {
                self.read_char();
                Token::new(TokenType::LBracket, "[".to_string(), line, column)
            }
            ']' => {
                self.read_char();
                Token::new(TokenType::RBracket, "]".to_string(), line, column)
            }

            // Strings
            '"' => {
                let literal = self.read_string();
                Token::new(TokenType::String, literal, line, column)
            }

            // End of file
            '\0' => Token::new(TokenType::Eof, String::new(), line, column),

            // Default: identifiers, keywords, or numbers
            _ => {
                if self.ch.is_alphabetic() || self.ch == '_' {
                    let literal = self.read_identifier();
                    let token_type = TokenType::lookup_keyword(&literal);
                    Token::new(token_type, literal, line, column)
                } else if self.ch.is_numeric() {
                    let (literal, token_type) = self.read_number();
                    Token::new(token_type, literal, line, column)
                } else {
                    let ch = self.ch;
                    self.read_char();
                    Token::new(TokenType::Illegal, ch.to_string(), line, column)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_creation() {
        let lexer = Lexer::new("let x = 5;");
        assert_eq!(lexer.line, 1);
    }

    #[test]
    fn test_tokenize_simple_let_statement() {
        let input = "let x = 5;";
        let mut lexer = Lexer::new(input);

        let tests = vec![
            (TokenType::Let, "let"),
            (TokenType::Ident, "x"),
            (TokenType::Assign, "="),
            (TokenType::Int, "5"),
            (TokenType::Semicolon, ";"),
            (TokenType::Eof, ""),
        ];

        for (expected_type, expected_literal) in tests {
            let token = lexer.next_token();
            assert_eq!(token.token_type, expected_type);
            assert_eq!(token.literal, expected_literal);
        }
    }

    #[test]
    fn test_tokenize_operators() {
        let input = "+ - * / % @ == != < <= > >= = ->";
        let mut lexer = Lexer::new(input);

        let tests = vec![
            TokenType::Plus,
            TokenType::Minus,
            TokenType::Multiply,
            TokenType::Divide,
            TokenType::Modulo,
            TokenType::MatMul,
            TokenType::Equal,
            TokenType::NotEqual,
            TokenType::LessThan,
            TokenType::LessEqual,
            TokenType::GreaterThan,
            TokenType::GreaterEqual,
            TokenType::Assign,
            TokenType::Arrow,
        ];

        for expected_type in tests {
            let token = lexer.next_token();
            assert_eq!(token.token_type, expected_type);
        }
    }

    #[test]
    fn test_tokenize_delimiters() {
        let input = "(){}[],;:";
        let mut lexer = Lexer::new(input);

        let tests = vec![
            TokenType::LParen,
            TokenType::RParen,
            TokenType::LBrace,
            TokenType::RBrace,
            TokenType::LBracket,
            TokenType::RBracket,
            TokenType::Comma,
            TokenType::Semicolon,
            TokenType::Colon,
        ];

        for expected_type in tests {
            let token = lexer.next_token();
            assert_eq!(token.token_type, expected_type);
        }
    }

    #[test]
    fn test_tokenize_keywords() {
        let input = "let fn return if else for while tensor model autograd";
        let mut lexer = Lexer::new(input);

        let tests = vec![
            TokenType::Let,
            TokenType::Fn,
            TokenType::Return,
            TokenType::If,
            TokenType::Else,
            TokenType::For,
            TokenType::While,
            TokenType::Tensor,
            TokenType::Model,
            TokenType::Autograd,
        ];

        for expected_type in tests {
            let token = lexer.next_token();
            assert_eq!(token.token_type, expected_type);
        }
    }

    #[test]
    fn test_tokenize_type_keywords() {
        let input = "int32 int64 float32 float64 bool";
        let mut lexer = Lexer::new(input);

        let tests = vec![
            TokenType::Int32,
            TokenType::Int64,
            TokenType::Float32,
            TokenType::Float64,
            TokenType::Bool,
        ];

        for expected_type in tests {
            let token = lexer.next_token();
            assert_eq!(token.token_type, expected_type);
        }
    }

    #[test]
    fn test_tokenize_ml_keywords() {
        let input = "dense conv2d dropout activation relu sigmoid tanh softmax";
        let mut lexer = Lexer::new(input);

        let tests = vec![
            TokenType::Dense,
            TokenType::Conv2D,
            TokenType::Dropout,
            TokenType::Activation,
            TokenType::Relu,
            TokenType::Sigmoid,
            TokenType::Tanh,
            TokenType::Softmax,
        ];

        for expected_type in tests {
            let token = lexer.next_token();
            assert_eq!(token.token_type, expected_type);
        }
    }

    #[test]
    fn test_tokenize_integers() {
        let input = "0 42 100 999";
        let mut lexer = Lexer::new(input);

        let expected_values = vec!["0", "42", "100", "999"];

        for expected_value in expected_values {
            let token = lexer.next_token();
            assert_eq!(token.token_type, TokenType::Int);
            assert_eq!(token.literal, expected_value);
        }
    }

    #[test]
    fn test_tokenize_floats() {
        let input = "3.14 2.718 0.5 1.0";
        let mut lexer = Lexer::new(input);

        let expected_values = vec!["3.14", "2.718", "0.5", "1.0"];

        for expected_value in expected_values {
            let token = lexer.next_token();
            assert_eq!(token.token_type, TokenType::Float);
            assert_eq!(token.literal, expected_value);
        }
    }

    #[test]
    fn test_tokenize_scientific_notation() {
        let input = "1e5 1.5e-10 2.0E+3";
        let mut lexer = Lexer::new(input);

        let expected_values = vec!["1e5", "1.5e-10", "2.0E+3"];

        for expected_value in expected_values {
            let token = lexer.next_token();
            assert_eq!(token.token_type, TokenType::Float);
            assert_eq!(token.literal, expected_value);
        }
    }

    #[test]
    fn test_tokenize_strings() {
        let input = r#""hello" "world" "neural network""#;
        let mut lexer = Lexer::new(input);

        let expected_values = vec!["hello", "world", "neural network"];

        for expected_value in expected_values {
            let token = lexer.next_token();
            assert_eq!(token.token_type, TokenType::String);
            assert_eq!(token.literal, expected_value);
        }
    }

    #[test]
    fn test_tokenize_identifiers() {
        let input = "x y variable_name _private myFunc123";
        let mut lexer = Lexer::new(input);

        let expected_values = vec!["x", "y", "variable_name", "_private", "myFunc123"];

        for expected_value in expected_values {
            let token = lexer.next_token();
            assert_eq!(token.token_type, TokenType::Ident);
            assert_eq!(token.literal, expected_value);
        }
    }

    #[test]
    fn test_skip_single_line_comment() {
        let input = "let x = 5 // this is a comment\nlet y = 10";
        let mut lexer = Lexer::new(input);

        let tests = vec![
            (TokenType::Let, "let"),
            (TokenType::Ident, "x"),
            (TokenType::Assign, "="),
            (TokenType::Int, "5"),
            (TokenType::Let, "let"),
            (TokenType::Ident, "y"),
            (TokenType::Assign, "="),
            (TokenType::Int, "10"),
        ];

        for (expected_type, expected_literal) in tests {
            let token = lexer.next_token();
            assert_eq!(token.token_type, expected_type);
            assert_eq!(token.literal, expected_literal);
        }
    }

    #[test]
    fn test_skip_multi_line_comment() {
        let input = "let x = 5 /* this is a\nmulti-line comment */ let y = 10";
        let mut lexer = Lexer::new(input);

        let tests = vec![
            (TokenType::Let, "let"),
            (TokenType::Ident, "x"),
            (TokenType::Assign, "="),
            (TokenType::Int, "5"),
            (TokenType::Let, "let"),
            (TokenType::Ident, "y"),
            (TokenType::Assign, "="),
            (TokenType::Int, "10"),
        ];

        for (expected_type, expected_literal) in tests {
            let token = lexer.next_token();
            assert_eq!(token.token_type, expected_type);
            assert_eq!(token.literal, expected_literal);
        }
    }

    #[test]
    fn test_tokenize_function_declaration() {
        let input = "fn add(x: int32, y: int32) -> int32 { return x + y }";
        let mut lexer = Lexer::new(input);

        let tests = vec![
            TokenType::Fn,
            TokenType::Ident, // add
            TokenType::LParen,
            TokenType::Ident, // x
            TokenType::Colon,
            TokenType::Int32,
            TokenType::Comma,
            TokenType::Ident, // y
            TokenType::Colon,
            TokenType::Int32,
            TokenType::RParen,
            TokenType::Arrow,
            TokenType::Int32,
            TokenType::LBrace,
            TokenType::Return,
            TokenType::Ident, // x
            TokenType::Plus,
            TokenType::Ident, // y
            TokenType::RBrace,
        ];

        for expected_type in tests {
            let token = lexer.next_token();
            assert_eq!(token.token_type, expected_type);
        }
    }

    #[test]
    fn test_tokenize_tensor_declaration() {
        let input = "let matrix: tensor<float32, [2, 3]> = [[1, 2, 3], [4, 5, 6]]";
        let mut lexer = Lexer::new(input);

        let tests = vec![
            TokenType::Let,
            TokenType::Ident, // matrix
            TokenType::Colon,
            TokenType::Tensor,
            TokenType::LessThan,
            TokenType::Float32,
            TokenType::Comma,
            TokenType::LBracket,
            TokenType::Int, // 2
            TokenType::Comma,
            TokenType::Int, // 3
            TokenType::RBracket,
            TokenType::GreaterThan,
            TokenType::Assign,
            TokenType::LBracket,
        ];

        for expected_type in tests.iter().take(15) {
            let token = lexer.next_token();
            assert_eq!(token.token_type, *expected_type);
        }
    }

    #[test]
    fn test_tokenize_model_dsl() {
        let input = r#"
model NeuralNet {
    layers {
        dense(784, 128, activation: relu)
    }
}
"#;
        let mut lexer = Lexer::new(input);

        let tests = vec![
            TokenType::Model,
            TokenType::Ident, // NeuralNet
            TokenType::LBrace,
            TokenType::Layers,
            TokenType::LBrace,
            TokenType::Dense,
            TokenType::LParen,
            TokenType::Int, // 784
            TokenType::Comma,
            TokenType::Int, // 128
            TokenType::Comma,
            TokenType::Activation,
            TokenType::Colon,
            TokenType::Relu,
            TokenType::RParen,
            TokenType::RBrace,
            TokenType::RBrace,
        ];

        for expected_type in tests {
            let token = lexer.next_token();
            assert_eq!(token.token_type, expected_type);
        }
    }

    #[test]
    fn test_line_and_column_tracking() {
        let input = "let x = 5\nlet y = 10";
        let mut lexer = Lexer::new(input);

        let token1 = lexer.next_token(); // let
        assert_eq!(token1.line, 1);

        lexer.next_token(); // x
        lexer.next_token(); // =
        lexer.next_token(); // 5

        let token5 = lexer.next_token(); // let (second line)
        assert_eq!(token5.line, 2);
    }

    #[test]
    fn test_tokenize_logical_operators() {
        let input = "and or not true false";
        let mut lexer = Lexer::new(input);

        let tests = vec![
            TokenType::And,
            TokenType::Or,
            TokenType::Not,
            TokenType::True,
            TokenType::False,
        ];

        for expected_type in tests {
            let token = lexer.next_token();
            assert_eq!(token.token_type, expected_type);
        }
    }

    #[test]
    fn test_tokenize_complete_program() {
        let input = r#"
fn quadratic(x: float32) -> float32 {
    return x * x + 2.0 * x + 1.0
}

let result = quadratic(3.0)
"#;
        let mut lexer = Lexer::new(input);

        // Just verify it doesn't crash and produces valid tokens
        let mut token_count = 0;
        loop {
            let token = lexer.next_token();
            token_count += 1;
            if token.token_type == TokenType::Eof {
                break;
            }
            // Ensure no illegal tokens
            assert_ne!(token.token_type, TokenType::Illegal);
        }

        assert!(token_count > 20); // Should have many tokens
    }
}

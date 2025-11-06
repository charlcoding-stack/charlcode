// Token definitions for Charl language

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub token_type: TokenType,
    pub literal: String,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    // Special tokens
    Illegal,
    Eof,

    // Identifiers and literals
    Ident,  // variable names, function names
    Int,    // integer literals: 42, 100
    Float,  // float literals: 3.14, 2.718
    String, // string literals: "hello"

    // Operators
    Assign,   // =
    Plus,     // +
    Minus,    // -
    Multiply, // *
    Divide,   // /
    Modulo,   // %
    MatMul,   // @ (matrix multiplication)

    // Comparison operators
    Equal,        // ==
    NotEqual,     // !=
    LessThan,     // <
    LessEqual,    // <=
    GreaterThan,  // >
    GreaterEqual, // >=

    // Logical operators
    And, // and
    Or,  // or
    Not, // not

    // Delimiters
    Comma,     // ,
    Semicolon, // ;
    Colon,     // :
    Arrow,     // ->
    FatArrow,  // =>
    Dot,          // . (tuple indexing)
    DotDot,       // .. (range)
    DotDotEqual,  // ..= (inclusive range)

    // Brackets
    LParen,   // (
    RParen,   // )
    LBrace,   // {
    RBrace,   // }
    LBracket, // [
    RBracket, // ]
    LAngle,   // < (also used for generics)
    RAngle,   // > (also used for generics)

    // Keywords
    Let,      // let
    Fn,       // fn
    Return,   // return
    If,       // if
    Else,     // else
    For,      // for
    While,    // while
    Match,    // match
    Break,    // break
    Continue, // continue
    True,     // true
    False,    // false

    // Type keywords
    Int32,   // int32
    Int64,   // int64
    Float32, // float32
    Float64, // float64
    Bool,    // bool
    StringType, // string (type annotation)
    Tensor,  // tensor

    // AI/ML specific keywords
    Model,      // model
    Layer,      // layer
    Layers,     // layers
    Autograd,   // autograd
    Gradient,   // gradient
    Dense,      // dense
    Conv2D,     // conv2d
    Activation, // activation
    Dropout,    // dropout

    // Activation functions
    Relu,    // relu
    Sigmoid, // sigmoid
    Tanh,    // tanh
    Softmax, // softmax
}

impl TokenType {
    /// Convert a keyword string to its corresponding TokenType
    pub fn lookup_keyword(ident: &str) -> TokenType {
        match ident {
            "let" => TokenType::Let,
            "fn" => TokenType::Fn,
            "return" => TokenType::Return,
            "if" => TokenType::If,
            "else" => TokenType::Else,
            "for" => TokenType::For,
            "while" => TokenType::While,
            "match" => TokenType::Match,
            "break" => TokenType::Break,
            "continue" => TokenType::Continue,
            "true" => TokenType::True,
            "false" => TokenType::False,
            "and" => TokenType::And,
            "or" => TokenType::Or,
            "not" => TokenType::Not,

            // Types
            "int32" => TokenType::Int32,
            "int64" => TokenType::Int64,
            "float32" => TokenType::Float32,
            "float64" => TokenType::Float64,
            "bool" => TokenType::Bool,
            "string" => TokenType::StringType,
            "tensor" => TokenType::Tensor,

            // AI/ML keywords
            "model" => TokenType::Model,
            "layer" => TokenType::Layer,
            "layers" => TokenType::Layers,
            "autograd" => TokenType::Autograd,
            "gradient" => TokenType::Gradient,
            "dense" => TokenType::Dense,
            // "conv2d" => TokenType::Conv2D,  // Removed: using as builtin function instead
            "activation" => TokenType::Activation,
            // "dropout" => TokenType::Dropout,  // Removed: using as builtin function instead

            // Activations
            "relu" => TokenType::Relu,
            "sigmoid" => TokenType::Sigmoid,
            "tanh" => TokenType::Tanh,
            "softmax" => TokenType::Softmax,

            _ => TokenType::Ident,
        }
    }
}

impl Token {
    pub fn new(token_type: TokenType, literal: String, line: usize, column: usize) -> Self {
        Token {
            token_type,
            literal,
            line,
            column,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_lookup() {
        assert_eq!(TokenType::lookup_keyword("let"), TokenType::Let);
        assert_eq!(TokenType::lookup_keyword("tensor"), TokenType::Tensor);
        assert_eq!(TokenType::lookup_keyword("autograd"), TokenType::Autograd);
        assert_eq!(TokenType::lookup_keyword("unknown"), TokenType::Ident);
    }
}

// Abstract Syntax Tree (AST) definitions
// Phase 1.2: Data structures representing parsed Charl code

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Let(LetStatement),
    Return(ReturnStatement),
    Expression(ExpressionStatement),
    Function(FunctionStatement),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetStatement {
    pub name: String,
    pub type_annotation: Option<TypeAnnotation>,
    pub value: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReturnStatement {
    pub value: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpressionStatement {
    pub expression: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionStatement {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<TypeAnnotation>,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub type_annotation: TypeAnnotation,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Identifier(String),
    IntegerLiteral(i64),
    FloatLiteral(f64),
    BooleanLiteral(bool),
    StringLiteral(String),
    ArrayLiteral(Vec<Expression>),
    TensorLiteral(Vec<Expression>),

    // Binary operations
    Binary {
        left: Box<Expression>,
        operator: BinaryOperator,
        right: Box<Expression>,
    },

    // Unary operations
    Unary {
        operator: UnaryOperator,
        operand: Box<Expression>,
    },

    // Function call
    Call {
        function: Box<Expression>,
        arguments: Vec<Expression>,
    },

    // Indexing
    Index {
        object: Box<Expression>,
        index: Box<Expression>,
    },

    // Autograd expression
    Autograd {
        expression: Box<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    MatMul,        // @ for matrix multiplication
    Equal,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Negate,
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeAnnotation {
    Int32,
    Int64,
    Float32,
    Float64,
    Bool,
    Tensor {
        dtype: Box<TypeAnnotation>,
        shape: Vec<usize>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_creation() {
        let program = Program { statements: vec![] };
        assert_eq!(program.statements.len(), 0);
    }
}

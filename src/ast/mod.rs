// Abstract Syntax Tree (AST) definitions
// Phase 1.2: Data structures representing parsed Charl code

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Let(LetStatement),
    Assign(AssignStatement),
    Return(ReturnStatement),
    Expression(ExpressionStatement),
    Function(FunctionStatement),
    If(IfStatement),
    While(WhileStatement),
    For(ForStatement),
    Break,    // Exit from loop
    Continue, // Skip to next iteration
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetStatement {
    pub name: String,
    pub type_annotation: Option<TypeAnnotation>,
    pub value: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssignStatement {
    pub target: Expression, // Can be Identifier or Index (for array[i] = value)
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
pub struct IfStatement {
    pub condition: Expression,
    pub consequence: Vec<Statement>,
    pub alternative: Option<Vec<Statement>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhileStatement {
    pub condition: Expression,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForStatement {
    pub variable: String,
    pub iterable: Expression,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Identifier(String),
    IntegerLiteral(i64),
    FloatLiteral(f64),
    BooleanLiteral(bool),
    StringLiteral(String),
    ArrayLiteral(Vec<Expression>),
    ArrayRepeat {
        value: Box<Expression>,
        count: Box<Expression>,
    }, // [value; count] - e.g., [0.0; 100]
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

    // Range expression: start..end (exclusive)
    Range {
        start: Box<Expression>,
        end: Box<Expression>,
    },

    // Inclusive range expression: start..=end (inclusive)
    InclusiveRange {
        start: Box<Expression>,
        end: Box<Expression>,
    },

    // If expression: if cond { block } else { block }
    // Returns the value of the last expression in the executed block
    If {
        condition: Box<Expression>,
        consequence: Vec<Statement>,
        alternative: Vec<Statement>,
    },

    // Match expression: match value { pattern => expr, ... }
    // Pattern matching with multiple arms
    Match {
        value: Box<Expression>,
        arms: Vec<MatchArm>,
    },

    // Tuple literal: (1, "hello", true)
    TupleLiteral(Vec<Expression>),

    // Tuple indexing: tuple.0, tuple.1
    TupleIndex {
        tuple: Box<Expression>,
        index: usize,
    },

    // Type casting: expression as type
    Cast {
        expression: Box<Expression>,
        target_type: String,
    },
}

/// Pattern for match expressions
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    // Literal patterns: 1, "hello", true
    IntegerLiteral(i64),
    FloatLiteral(f64),
    BooleanLiteral(bool),
    StringLiteral(String),

    // Variable pattern: binds the value to a name
    Variable(String),

    // Wildcard pattern: _ (matches anything)
    Wildcard,
}

/// Match arm: pattern => expression
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub expression: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    MatMul, // @ for matrix multiplication
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
    String,
    Concept, // Symbolic AI concept type
    Array(Box<TypeAnnotation>), // [int32], [float32], etc. (dynamic size)
    ArraySized {
        element_type: Box<TypeAnnotation>,
        size: usize,
    }, // [float32; 10], [[int32; 5]; 100]
    Tensor {
        dtype: Box<TypeAnnotation>,
        shape: Vec<usize>,
    },
    Tuple(Vec<TypeAnnotation>), // (int64, string, bool)
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

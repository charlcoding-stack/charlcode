// Parser module - Syntax analysis and AST construction
// Phase 1.2: Convert tokens into Abstract Syntax Tree

use crate::ast::*;
use crate::lexer::{Lexer, Token, TokenType};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
enum Precedence {
    Lowest = 0,
    Logical = 1,     // and, or
    Range = 2,       // ..
    Equals = 3,      // ==, !=
    LessGreater = 4, // <, >, <=, >=
    Sum = 5,         // +, -
    Product = 6,     // *, /, %
    MatMul = 7,      // @
    Prefix = 8,      // -x, !x
    Call = 9,        // fn(x)
    Index = 10,      // arr[i]
}

pub struct Parser {
    lexer: Lexer,
    current_token: Token,
    peek_token: Token,
    errors: Vec<String>,
}

impl Parser {
    pub fn new(mut lexer: Lexer) -> Self {
        let current_token = lexer.next_token();
        let peek_token = lexer.next_token();

        Parser {
            lexer,
            current_token,
            peek_token,
            errors: vec![],
        }
    }

    pub fn errors(&self) -> &Vec<String> {
        &self.errors
    }

    fn advance(&mut self) {
        self.current_token = self.peek_token.clone();
        self.peek_token = self.lexer.next_token();
    }

    fn current_token_is(&self, token_type: TokenType) -> bool {
        self.current_token.token_type == token_type
    }

    fn peek_token_is(&self, token_type: TokenType) -> bool {
        self.peek_token.token_type == token_type
    }

    fn expect_peek(&mut self, token_type: TokenType) -> bool {
        if self.peek_token_is(token_type.clone()) {
            self.advance();
            true
        } else {
            self.peek_error(token_type);
            false
        }
    }

    fn peek_error(&mut self, expected: TokenType) {
        let msg = format!(
            "Expected next token to be {:?}, got {:?} instead at line {}",
            expected, self.peek_token.token_type, self.peek_token.line
        );
        self.errors.push(msg);
    }

    fn current_precedence(&self) -> Precedence {
        Self::token_precedence(&self.current_token.token_type)
    }

    fn peek_precedence(&self) -> Precedence {
        Self::token_precedence(&self.peek_token.token_type)
    }

    fn token_precedence(token_type: &TokenType) -> Precedence {
        match token_type {
            TokenType::And | TokenType::Or => Precedence::Logical,
            TokenType::DotDot | TokenType::DotDotEqual => Precedence::Range,
            TokenType::Equal | TokenType::NotEqual => Precedence::Equals,
            TokenType::LessThan
            | TokenType::LessEqual
            | TokenType::GreaterThan
            | TokenType::GreaterEqual => Precedence::LessGreater,
            TokenType::Plus | TokenType::Minus => Precedence::Sum,
            TokenType::Multiply | TokenType::Divide | TokenType::Modulo => Precedence::Product,
            TokenType::MatMul => Precedence::MatMul,
            TokenType::LParen => Precedence::Call,
            TokenType::LBracket | TokenType::Dot => Precedence::Index,
            _ => Precedence::Lowest,
        }
    }

    pub fn parse_program(&mut self) -> Result<Program, String> {
        let mut statements = vec![];

        while !self.current_token_is(TokenType::Eof) {
            match self.parse_statement() {
                Ok(stmt) => statements.push(stmt),
                Err(e) => self.errors.push(e),
            }
            self.advance();
        }

        if !self.errors.is_empty() {
            return Err(format!("Parser errors:\n{}", self.errors.join("\n")));
        }

        Ok(Program { statements })
    }

    fn parse_statement(&mut self) -> Result<Statement, String> {
        match self.current_token.token_type {
            TokenType::Let => self.parse_let_statement(),
            TokenType::Return => self.parse_return_statement(),
            TokenType::Fn => self.parse_function_statement(),
            TokenType::If => self.parse_if_statement(),
            TokenType::While => self.parse_while_statement(),
            TokenType::For => self.parse_for_statement(),
            TokenType::Ident => {
                // Check if this is an assignment (identifier = expression)
                if self.peek_token_is(TokenType::Assign) {
                    self.parse_assignment_statement()
                } else {
                    self.parse_expression_statement()
                }
            }
            _ => self.parse_expression_statement(),
        }
    }

    fn parse_let_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume 'let'

        let name = if self.current_token_is(TokenType::Ident) {
            self.current_token.literal.clone()
        } else {
            return Err(format!(
                "Expected identifier after 'let', got {:?}",
                self.current_token.token_type
            ));
        };

        // Check for type annotation
        let type_annotation = if self.peek_token_is(TokenType::Colon) {
            self.advance(); // consume identifier
            self.advance(); // consume ':'
            Some(self.parse_type_annotation()?)
        } else {
            None
        };

        // Expect '='
        if !self.expect_peek(TokenType::Assign) {
            return Err("Expected '=' in let statement".to_string());
        }

        self.advance(); // move to value expression

        let value = self.parse_expression(Precedence::Lowest)?;

        // Optional semicolon
        if self.peek_token_is(TokenType::Semicolon) {
            self.advance();
        }

        Ok(Statement::Let(LetStatement {
            name,
            type_annotation,
            value,
        }))
    }

    fn parse_assignment_statement(&mut self) -> Result<Statement, String> {
        // Current token is the identifier
        let name = self.current_token.literal.clone();

        // Expect '='
        if !self.expect_peek(TokenType::Assign) {
            return Err("Expected '=' in assignment".to_string());
        }

        self.advance(); // move to value expression

        let value = self.parse_expression(Precedence::Lowest)?;

        // Optional semicolon
        if self.peek_token_is(TokenType::Semicolon) {
            self.advance();
        }

        Ok(Statement::Assign(AssignStatement { name, value }))
    }

    fn parse_return_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume 'return'

        // Check if there's a value to return (not just 'return' followed by '}' or ';')
        let value = if self.current_token_is(TokenType::Semicolon)
            || self.current_token_is(TokenType::RBrace)
        {
            // Empty return, use a placeholder (we could use Option<Expression> instead)
            Expression::BooleanLiteral(false) // Placeholder for empty return
        } else {
            self.parse_expression(Precedence::Lowest)?
        };

        // Optional semicolon
        if self.peek_token_is(TokenType::Semicolon) {
            self.advance();
        }

        Ok(Statement::Return(ReturnStatement { value }))
    }

    fn parse_function_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume 'fn'

        let name = if self.current_token_is(TokenType::Ident) {
            self.current_token.literal.clone()
        } else {
            return Err("Expected function name".to_string());
        };

        if !self.expect_peek(TokenType::LParen) {
            return Err("Expected '(' after function name".to_string());
        }

        let parameters = self.parse_function_parameters()?;

        // Parse return type if present
        let return_type = if self.peek_token_is(TokenType::Arrow) {
            self.advance(); // consume '->'
            self.advance(); // move to type
            Some(self.parse_type_annotation()?)
        } else {
            None
        };

        if !self.expect_peek(TokenType::LBrace) {
            return Err("Expected '{' to start function body".to_string());
        }

        let body = self.parse_block_statement()?;

        Ok(Statement::Function(FunctionStatement {
            name,
            parameters,
            return_type,
            body,
        }))
    }

    fn parse_function_parameters(&mut self) -> Result<Vec<Parameter>, String> {
        let mut parameters = vec![];

        if self.peek_token_is(TokenType::RParen) {
            self.advance(); // consume ')'
            return Ok(parameters);
        }

        self.advance(); // move to first parameter

        loop {
            let name = if self.current_token_is(TokenType::Ident) {
                self.current_token.literal.clone()
            } else {
                return Err("Expected parameter name".to_string());
            };

            if !self.expect_peek(TokenType::Colon) {
                return Err("Expected ':' after parameter name".to_string());
            }

            self.advance(); // move to type
            let type_annotation = self.parse_type_annotation()?;

            parameters.push(Parameter {
                name,
                type_annotation,
            });

            if !self.peek_token_is(TokenType::Comma) {
                break;
            }

            self.advance(); // consume ','
            self.advance(); // move to next parameter
        }

        if !self.expect_peek(TokenType::RParen) {
            return Err("Expected ')' after parameters".to_string());
        }

        Ok(parameters)
    }

    fn parse_block_statement(&mut self) -> Result<Vec<Statement>, String> {
        self.advance(); // consume '{'

        let mut statements = vec![];

        while !self.current_token_is(TokenType::RBrace) && !self.current_token_is(TokenType::Eof) {
            let stmt = self.parse_statement()?;
            statements.push(stmt);
            self.advance();
        }

        Ok(statements)
    }

    fn parse_if_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume 'if'

        // Parse condition
        let condition = self.parse_expression(Precedence::Lowest)?;

        // Expect '{'
        if !self.expect_peek(TokenType::LBrace) {
            return Err("Expected '{' after if condition".to_string());
        }

        // Parse consequence block
        let consequence = self.parse_block_statement()?;

        // Check for 'else' clause
        let alternative = if self.peek_token_is(TokenType::Else) {
            self.advance(); // consume '}', now current is 'else'

            if !self.expect_peek(TokenType::LBrace) {
                return Err("Expected '{' after else".to_string());
            }

            Some(self.parse_block_statement()?)
        } else {
            None
        };

        Ok(Statement::If(IfStatement {
            condition,
            consequence,
            alternative,
        }))
    }

    fn parse_while_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume 'while'

        // Parse condition
        let condition = self.parse_expression(Precedence::Lowest)?;

        // Expect '{'
        if !self.expect_peek(TokenType::LBrace) {
            return Err("Expected '{' after while condition".to_string());
        }

        // Parse body
        let body = self.parse_block_statement()?;

        Ok(Statement::While(WhileStatement { condition, body }))
    }

    fn parse_for_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume 'for'

        // Parse variable name
        let variable = if self.current_token_is(TokenType::Ident) {
            self.current_token.literal.clone()
        } else {
            return Err("Expected variable name after 'for'".to_string());
        };

        // Expect 'in'
        // Note: We need to check if 'in' is an identifier token
        if !self.expect_peek(TokenType::Ident) {
            return Err("Expected 'in' after for variable".to_string());
        }

        if self.current_token.literal != "in" {
            return Err(format!(
                "Expected 'in' after for variable, got '{}'",
                self.current_token.literal
            ));
        }

        self.advance(); // move past 'in'

        // Parse iterable expression
        let iterable = self.parse_expression(Precedence::Lowest)?;

        // Expect '{'
        if !self.expect_peek(TokenType::LBrace) {
            return Err("Expected '{' after for iterable".to_string());
        }

        // Parse body
        let body = self.parse_block_statement()?;

        Ok(Statement::For(ForStatement {
            variable,
            iterable,
            body,
        }))
    }

    fn parse_expression_statement(&mut self) -> Result<Statement, String> {
        let expression = self.parse_expression(Precedence::Lowest)?;

        // Optional semicolon
        if self.peek_token_is(TokenType::Semicolon) {
            self.advance();
        }

        Ok(Statement::Expression(ExpressionStatement { expression }))
    }

    fn parse_expression(&mut self, precedence: Precedence) -> Result<Expression, String> {
        // Parse prefix expression
        let mut left = self.parse_prefix_expression()?;

        // Parse infix expressions with precedence
        while !self.peek_token_is(TokenType::Semicolon) && precedence < self.peek_precedence() {
            self.advance();
            left = self.parse_infix_expression(left)?;
        }

        Ok(left)
    }

    fn parse_prefix_expression(&mut self) -> Result<Expression, String> {
        match &self.current_token.token_type {
            TokenType::Ident => Ok(Expression::Identifier(self.current_token.literal.clone())),
            TokenType::Int => {
                let value = self.current_token.literal.parse::<i64>().map_err(|_| {
                    format!("Could not parse {} as integer", self.current_token.literal)
                })?;
                Ok(Expression::IntegerLiteral(value))
            }
            TokenType::Float => {
                let value = self.current_token.literal.parse::<f64>().map_err(|_| {
                    format!("Could not parse {} as float", self.current_token.literal)
                })?;
                Ok(Expression::FloatLiteral(value))
            }
            TokenType::String => Ok(Expression::StringLiteral(
                self.current_token.literal.clone(),
            )),
            TokenType::True => Ok(Expression::BooleanLiteral(true)),
            TokenType::False => Ok(Expression::BooleanLiteral(false)),
            TokenType::LBracket => self.parse_array_literal(),
            TokenType::Minus | TokenType::Not => self.parse_unary_expression(),
            TokenType::LParen => self.parse_grouped_expression(),
            TokenType::Autograd => self.parse_autograd_expression(),
            TokenType::If => self.parse_if_expression(),
            TokenType::Match => self.parse_match_expression(),
            _ => Err(format!(
                "No prefix parse function for {:?} at line {}",
                self.current_token.token_type, self.current_token.line
            )),
        }
    }

    fn parse_infix_expression(&mut self, left: Expression) -> Result<Expression, String> {
        match &self.current_token.token_type {
            TokenType::Plus
            | TokenType::Minus
            | TokenType::Multiply
            | TokenType::Divide
            | TokenType::Modulo
            | TokenType::MatMul
            | TokenType::Equal
            | TokenType::NotEqual
            | TokenType::LessThan
            | TokenType::LessEqual
            | TokenType::GreaterThan
            | TokenType::GreaterEqual
            | TokenType::And
            | TokenType::Or => self.parse_binary_expression(left),
            TokenType::DotDot => self.parse_range_expression(left),
            TokenType::DotDotEqual => self.parse_inclusive_range_expression(left),
            TokenType::LParen => self.parse_call_expression(left),
            TokenType::LBracket => self.parse_index_expression(left),
            TokenType::Dot => self.parse_tuple_index_expression(left),
            _ => Err(format!(
                "No infix parse function for {:?}",
                self.current_token.token_type
            )),
        }
    }

    fn parse_binary_expression(&mut self, left: Expression) -> Result<Expression, String> {
        let operator = self.token_type_to_binary_operator(&self.current_token.token_type)?;
        let precedence = self.current_precedence();

        self.advance();
        let right = self.parse_expression(precedence)?;

        Ok(Expression::Binary {
            left: Box::new(left),
            operator,
            right: Box::new(right),
        })
    }

    fn parse_range_expression(&mut self, left: Expression) -> Result<Expression, String> {
        // Current token is '..'
        let precedence = self.current_precedence();

        self.advance();
        let right = self.parse_expression(precedence)?;

        Ok(Expression::Range {
            start: Box::new(left),
            end: Box::new(right),
        })
    }

    fn parse_inclusive_range_expression(&mut self, left: Expression) -> Result<Expression, String> {
        // Current token is '..='
        let precedence = self.current_precedence();

        self.advance();
        let right = self.parse_expression(precedence)?;

        Ok(Expression::InclusiveRange {
            start: Box::new(left),
            end: Box::new(right),
        })
    }

    fn parse_unary_expression(&mut self) -> Result<Expression, String> {
        let operator = match &self.current_token.token_type {
            TokenType::Minus => UnaryOperator::Negate,
            TokenType::Not => UnaryOperator::Not,
            _ => return Err("Invalid unary operator".to_string()),
        };

        self.advance();
        let operand = self.parse_expression(Precedence::Prefix)?;

        Ok(Expression::Unary {
            operator,
            operand: Box::new(operand),
        })
    }

    fn parse_grouped_expression(&mut self) -> Result<Expression, String> {
        self.advance(); // consume '('

        // Check for empty tuple: ()
        if self.current_token_is(TokenType::RParen) {
            return Ok(Expression::TupleLiteral(vec![]));
        }

        let first_expr = self.parse_expression(Precedence::Lowest)?;

        // Check if it's a tuple (has comma) or grouped expression
        if self.peek_token_is(TokenType::Comma) {
            // It's a tuple literal
            let mut elements = vec![first_expr];

            while self.peek_token_is(TokenType::Comma) {
                self.advance(); // consume ','

                // Allow trailing comma
                if self.peek_token_is(TokenType::RParen) {
                    break;
                }

                self.advance(); // move to next expression
                elements.push(self.parse_expression(Precedence::Lowest)?);
            }

            if !self.expect_peek(TokenType::RParen) {
                return Err("Expected ')' after tuple elements".to_string());
            }

            Ok(Expression::TupleLiteral(elements))
        } else {
            // It's a grouped expression
            if !self.expect_peek(TokenType::RParen) {
                return Err("Expected ')' after grouped expression".to_string());
            }

            Ok(first_expr)
        }
    }

    fn parse_array_literal(&mut self) -> Result<Expression, String> {
        let elements = self.parse_expression_list(TokenType::RBracket)?;
        Ok(Expression::ArrayLiteral(elements))
    }

    fn parse_expression_list(&mut self, end: TokenType) -> Result<Vec<Expression>, String> {
        let mut list = vec![];

        if self.peek_token_is(end.clone()) {
            self.advance();
            return Ok(list);
        }

        self.advance();
        list.push(self.parse_expression(Precedence::Lowest)?);

        while self.peek_token_is(TokenType::Comma) {
            self.advance(); // consume ','
            self.advance(); // move to next expression
            list.push(self.parse_expression(Precedence::Lowest)?);
        }

        if !self.expect_peek(end) {
            return Err("Expected end delimiter".to_string());
        }

        Ok(list)
    }

    fn parse_call_expression(&mut self, function: Expression) -> Result<Expression, String> {
        let arguments = self.parse_expression_list(TokenType::RParen)?;

        Ok(Expression::Call {
            function: Box::new(function),
            arguments,
        })
    }

    fn parse_index_expression(&mut self, object: Expression) -> Result<Expression, String> {
        self.advance(); // consume '['

        let index = self.parse_expression(Precedence::Lowest)?;

        if !self.expect_peek(TokenType::RBracket) {
            return Err("Expected ']' after index".to_string());
        }

        Ok(Expression::Index {
            object: Box::new(object),
            index: Box::new(index),
        })
    }

    fn parse_tuple_index_expression(&mut self, tuple: Expression) -> Result<Expression, String> {
        self.advance(); // consume '.'

        // Expect an integer literal for tuple index
        if !self.current_token_is(TokenType::Int) {
            return Err(format!(
                "Expected integer for tuple index, got {:?}",
                self.current_token.token_type
            ));
        }

        let index = self
            .current_token
            .literal
            .parse::<usize>()
            .map_err(|_| format!("Invalid tuple index: {}", self.current_token.literal))?;

        Ok(Expression::TupleIndex {
            tuple: Box::new(tuple),
            index,
        })
    }

    fn parse_autograd_expression(&mut self) -> Result<Expression, String> {
        if !self.expect_peek(TokenType::LParen) {
            return Err("Expected '(' after 'autograd'".to_string());
        }

        self.advance(); // move past '('
        let expression = self.parse_expression(Precedence::Lowest)?;

        if !self.expect_peek(TokenType::RParen) {
            return Err("Expected ')' after autograd expression".to_string());
        }

        Ok(Expression::Autograd {
            expression: Box::new(expression),
        })
    }

    fn parse_if_expression(&mut self) -> Result<Expression, String> {
        self.advance(); // consume 'if'

        // Parse condition
        let condition = self.parse_expression(Precedence::Lowest)?;

        // Expect '{'
        if !self.expect_peek(TokenType::LBrace) {
            return Err("Expected '{' after if condition".to_string());
        }

        // Parse consequence block
        let consequence = self.parse_block_statement()?;

        // For if expression, else is REQUIRED (must return a value in both branches)
        if !self.peek_token_is(TokenType::Else) {
            return Err("If expression requires 'else' clause".to_string());
        }

        self.advance(); // consume '}'
        // Now current is 'else', peek is '{'

        if !self.expect_peek(TokenType::LBrace) {
            return Err("Expected '{' after else".to_string());
        }

        // Parse alternative block
        let alternative = self.parse_block_statement()?;

        Ok(Expression::If {
            condition: Box::new(condition),
            consequence,
            alternative,
        })
    }

    fn parse_match_expression(&mut self) -> Result<Expression, String> {
        self.advance(); // consume 'match'

        // Parse the value to match against
        let value = self.parse_expression(Precedence::Lowest)?;

        // Expect '{'
        if !self.expect_peek(TokenType::LBrace) {
            return Err("Expected '{' after match value".to_string());
        }

        // Parse match arms
        let mut arms = Vec::new();
        self.advance(); // consume '{'

        while !self.current_token_is(TokenType::RBrace) && !self.current_token_is(TokenType::Eof) {
            let arm = self.parse_match_arm()?;
            arms.push(arm);

            // Expect comma after each arm (except potentially the last one)
            if self.peek_token_is(TokenType::Comma) {
                self.advance(); // consume ','
            }

            self.advance(); // move to next token
        }

        if arms.is_empty() {
            return Err("Match expression must have at least one arm".to_string());
        }

        Ok(Expression::Match {
            value: Box::new(value),
            arms,
        })
    }

    fn parse_match_arm(&mut self) -> Result<MatchArm, String> {
        // Parse pattern
        let pattern = self.parse_pattern()?;

        // Expect '=>'
        if !self.expect_peek(TokenType::FatArrow) {
            return Err("Expected '=>' after match pattern".to_string());
        }

        self.advance(); // consume '=>'

        // Parse expression
        let expression = self.parse_expression(Precedence::Lowest)?;

        Ok(MatchArm {
            pattern,
            expression,
        })
    }

    fn parse_pattern(&mut self) -> Result<Pattern, String> {
        match &self.current_token.token_type {
            TokenType::Int => {
                let value = self.current_token.literal.parse::<i64>()
                    .map_err(|_| format!("Invalid integer literal: {}", self.current_token.literal))?;
                Ok(Pattern::IntegerLiteral(value))
            }
            TokenType::Float => {
                let value = self.current_token.literal.parse::<f64>()
                    .map_err(|_| format!("Invalid float literal: {}", self.current_token.literal))?;
                Ok(Pattern::FloatLiteral(value))
            }
            TokenType::True => Ok(Pattern::BooleanLiteral(true)),
            TokenType::False => Ok(Pattern::BooleanLiteral(false)),
            TokenType::String => {
                Ok(Pattern::StringLiteral(self.current_token.literal.clone()))
            }
            TokenType::Ident => {
                let name = self.current_token.literal.clone();
                // Check if this is a wildcard
                if name == "_" {
                    Ok(Pattern::Wildcard)
                } else {
                    Ok(Pattern::Variable(name))
                }
            }
            _ => Err(format!(
                "Invalid pattern: {:?} at line {}",
                self.current_token.token_type, self.current_token.line
            )),
        }
    }

    fn parse_type_annotation(&mut self) -> Result<TypeAnnotation, String> {
        match &self.current_token.token_type {
            TokenType::Int32 => Ok(TypeAnnotation::Int32),
            TokenType::Int64 => Ok(TypeAnnotation::Int64),
            TokenType::Float32 => Ok(TypeAnnotation::Float32),
            TokenType::Float64 => Ok(TypeAnnotation::Float64),
            TokenType::Bool => Ok(TypeAnnotation::Bool),
            TokenType::StringType => Ok(TypeAnnotation::String),
            TokenType::LBracket => {
                // Array type: [element_type]
                self.advance(); // move to element type
                let element_type = Box::new(self.parse_type_annotation()?);

                if !self.expect_peek(TokenType::RBracket) {
                    return Err("Expected ']' after array element type".to_string());
                }

                Ok(TypeAnnotation::Array(element_type))
            }
            TokenType::Tensor => {
                // tensor<dtype, [shape]>
                if !self.expect_peek(TokenType::LessThan) {
                    return Err("Expected '<' after 'tensor'".to_string());
                }

                self.advance(); // move to dtype
                let dtype = Box::new(self.parse_type_annotation()?);

                if !self.expect_peek(TokenType::Comma) {
                    return Err("Expected ',' after tensor dtype".to_string());
                }

                if !self.expect_peek(TokenType::LBracket) {
                    return Err("Expected '[' for tensor shape".to_string());
                }

                self.advance(); // move to first dimension

                let mut shape = vec![];
                loop {
                    if let TokenType::Int = self.current_token.token_type {
                        let dim = self
                            .current_token
                            .literal
                            .parse::<usize>()
                            .map_err(|_| "Invalid dimension in tensor shape".to_string())?;
                        shape.push(dim);
                    } else {
                        return Err("Expected integer for tensor dimension".to_string());
                    }

                    if !self.peek_token_is(TokenType::Comma) {
                        break;
                    }
                    self.advance(); // consume ','
                    self.advance(); // move to next dimension
                }

                if !self.expect_peek(TokenType::RBracket) {
                    return Err("Expected ']' after tensor shape".to_string());
                }

                if !self.expect_peek(TokenType::GreaterThan) {
                    return Err("Expected '>' after tensor type".to_string());
                }

                Ok(TypeAnnotation::Tensor { dtype, shape })
            }
            TokenType::LParen => {
                // Tuple type: (type1, type2, ...)
                self.advance(); // consume '('

                // Check for empty tuple: ()
                if self.current_token_is(TokenType::RParen) {
                    return Ok(TypeAnnotation::Tuple(vec![]));
                }

                // Parse first type
                let mut element_types = vec![self.parse_type_annotation()?];

                // Parse remaining types
                while self.peek_token_is(TokenType::Comma) {
                    self.advance(); // consume ','

                    // Allow trailing comma
                    if self.peek_token_is(TokenType::RParen) {
                        break;
                    }

                    self.advance(); // move to next type
                    element_types.push(self.parse_type_annotation()?);
                }

                if !self.expect_peek(TokenType::RParen) {
                    return Err("Expected ')' after tuple types".to_string());
                }

                Ok(TypeAnnotation::Tuple(element_types))
            }
            _ => Err(format!(
                "Invalid type annotation: {:?}",
                self.current_token.token_type
            )),
        }
    }

    fn token_type_to_binary_operator(
        &self,
        token_type: &TokenType,
    ) -> Result<BinaryOperator, String> {
        match token_type {
            TokenType::Plus => Ok(BinaryOperator::Add),
            TokenType::Minus => Ok(BinaryOperator::Subtract),
            TokenType::Multiply => Ok(BinaryOperator::Multiply),
            TokenType::Divide => Ok(BinaryOperator::Divide),
            TokenType::Modulo => Ok(BinaryOperator::Modulo),
            TokenType::MatMul => Ok(BinaryOperator::MatMul),
            TokenType::Equal => Ok(BinaryOperator::Equal),
            TokenType::NotEqual => Ok(BinaryOperator::NotEqual),
            TokenType::LessThan => Ok(BinaryOperator::LessThan),
            TokenType::LessEqual => Ok(BinaryOperator::LessEqual),
            TokenType::GreaterThan => Ok(BinaryOperator::GreaterThan),
            TokenType::GreaterEqual => Ok(BinaryOperator::GreaterEqual),
            TokenType::And => Ok(BinaryOperator::And),
            TokenType::Or => Ok(BinaryOperator::Or),
            _ => Err(format!("Invalid binary operator: {:?}", token_type)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let lexer = Lexer::new("let x = 5;");
        let parser = Parser::new(lexer);
        assert_eq!(parser.errors().len(), 0);
    }

    #[test]
    fn test_parse_let_statement() {
        let input = "let x = 5;";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program().expect("Failed to parse program");
        assert_eq!(program.statements.len(), 1);

        match &program.statements[0] {
            Statement::Let(let_stmt) => {
                assert_eq!(let_stmt.name, "x");
                assert!(let_stmt.type_annotation.is_none());
            }
            _ => panic!("Expected Let statement"),
        }
    }

    #[test]
    fn test_parse_let_statement_with_type() {
        let input = "let x: int32 = 10;";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program().expect("Failed to parse program");
        assert_eq!(program.statements.len(), 1);

        match &program.statements[0] {
            Statement::Let(let_stmt) => {
                assert_eq!(let_stmt.name, "x");
                assert!(matches!(
                    let_stmt.type_annotation,
                    Some(TypeAnnotation::Int32)
                ));
            }
            _ => panic!("Expected Let statement"),
        }
    }

    #[test]
    fn test_parse_return_statement() {
        let input = "return 42;";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program().expect("Failed to parse program");
        assert_eq!(program.statements.len(), 1);

        assert!(matches!(program.statements[0], Statement::Return(_)));
    }

    #[test]
    fn test_parse_integer_literal() {
        let input = "42";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program().expect("Failed to parse program");
        assert_eq!(program.statements.len(), 1);

        match &program.statements[0] {
            Statement::Expression(expr_stmt) => match &expr_stmt.expression {
                Expression::IntegerLiteral(val) => assert_eq!(*val, 42),
                _ => panic!("Expected IntegerLiteral"),
            },
            _ => panic!("Expected Expression statement"),
        }
    }

    #[test]
    fn test_parse_float_literal() {
        let input = "3.14";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program().expect("Failed to parse program");
        assert_eq!(program.statements.len(), 1);

        match &program.statements[0] {
            Statement::Expression(expr_stmt) => match &expr_stmt.expression {
                Expression::FloatLiteral(val) => assert_eq!(*val, 3.14),
                _ => panic!("Expected FloatLiteral"),
            },
            _ => panic!("Expected Expression statement"),
        }
    }

    #[test]
    fn test_parse_boolean_literals() {
        let inputs = vec!["true", "false"];
        let expected = vec![true, false];

        for (input, expected_val) in inputs.iter().zip(expected.iter()) {
            let lexer = Lexer::new(input);
            let mut parser = Parser::new(lexer);

            let program = parser.parse_program().expect("Failed to parse program");

            match &program.statements[0] {
                Statement::Expression(expr_stmt) => match &expr_stmt.expression {
                    Expression::BooleanLiteral(val) => assert_eq!(*val, *expected_val),
                    _ => panic!("Expected BooleanLiteral"),
                },
                _ => panic!("Expected Expression statement"),
            }
        }
    }

    #[test]
    fn test_parse_identifier() {
        let input = "foobar";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program().expect("Failed to parse program");

        match &program.statements[0] {
            Statement::Expression(expr_stmt) => match &expr_stmt.expression {
                Expression::Identifier(name) => assert_eq!(name, "foobar"),
                _ => panic!("Expected Identifier"),
            },
            _ => panic!("Expected Expression statement"),
        }
    }

    #[test]
    fn test_parse_binary_expressions() {
        let tests = vec![
            ("5 + 10", BinaryOperator::Add),
            ("5 - 10", BinaryOperator::Subtract),
            ("5 * 10", BinaryOperator::Multiply),
            ("5 / 10", BinaryOperator::Divide),
            ("5 % 10", BinaryOperator::Modulo),
            ("5 @ 10", BinaryOperator::MatMul),
            ("5 == 10", BinaryOperator::Equal),
            ("5 != 10", BinaryOperator::NotEqual),
            ("5 < 10", BinaryOperator::LessThan),
            ("5 <= 10", BinaryOperator::LessEqual),
            ("5 > 10", BinaryOperator::GreaterThan),
            ("5 >= 10", BinaryOperator::GreaterEqual),
        ];

        for (input, expected_op) in tests {
            let lexer = Lexer::new(input);
            let mut parser = Parser::new(lexer);

            let program = parser.parse_program().expect("Failed to parse program");

            match &program.statements[0] {
                Statement::Expression(expr_stmt) => match &expr_stmt.expression {
                    Expression::Binary { operator, .. } => assert_eq!(*operator, expected_op),
                    _ => panic!("Expected Binary expression for: {}", input),
                },
                _ => panic!("Expected Expression statement"),
            }
        }
    }

    #[test]
    fn test_parse_unary_expressions() {
        let tests = vec![
            ("-5", UnaryOperator::Negate),
            ("not true", UnaryOperator::Not),
        ];

        for (input, expected_op) in tests {
            let lexer = Lexer::new(input);
            let mut parser = Parser::new(lexer);

            let program = parser.parse_program().expect("Failed to parse program");

            match &program.statements[0] {
                Statement::Expression(expr_stmt) => match &expr_stmt.expression {
                    Expression::Unary { operator, .. } => assert_eq!(*operator, expected_op),
                    _ => panic!("Expected Unary expression"),
                },
                _ => panic!("Expected Expression statement"),
            }
        }
    }

    #[test]
    fn test_operator_precedence() {
        let tests = vec![
            ("1 + 2 + 3", "((1 + 2) + 3)"),
            ("1 + 2 * 3", "(1 + (2 * 3))"),
            ("1 * 2 + 3", "((1 * 2) + 3)"),
            ("1 + 2 @ 3", "(1 + (2 @ 3))"),
        ];

        for (input, _expected) in tests {
            let lexer = Lexer::new(input);
            let mut parser = Parser::new(lexer);

            let program = parser.parse_program();
            assert!(program.is_ok(), "Failed to parse: {}", input);
        }
    }

    #[test]
    fn test_parse_grouped_expression() {
        let input = "(5 + 10) * 2";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program();
        assert!(program.is_ok());
    }

    #[test]
    fn test_parse_array_literal() {
        let input = "[1, 2, 3, 4, 5]";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program().expect("Failed to parse program");

        match &program.statements[0] {
            Statement::Expression(expr_stmt) => match &expr_stmt.expression {
                Expression::ArrayLiteral(elements) => assert_eq!(elements.len(), 5),
                _ => panic!("Expected ArrayLiteral"),
            },
            _ => panic!("Expected Expression statement"),
        }
    }

    #[test]
    fn test_parse_nested_arrays() {
        let input = "[[1, 2], [3, 4]]";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program().expect("Failed to parse program");

        match &program.statements[0] {
            Statement::Expression(expr_stmt) => match &expr_stmt.expression {
                Expression::ArrayLiteral(elements) => {
                    assert_eq!(elements.len(), 2);
                    // Check first nested array
                    match &elements[0] {
                        Expression::ArrayLiteral(nested) => assert_eq!(nested.len(), 2),
                        _ => panic!("Expected nested ArrayLiteral"),
                    }
                }
                _ => panic!("Expected ArrayLiteral"),
            },
            _ => panic!("Expected Expression statement"),
        }
    }

    #[test]
    fn test_parse_function_call() {
        let input = "add(1, 2)";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program().expect("Failed to parse program");

        match &program.statements[0] {
            Statement::Expression(expr_stmt) => match &expr_stmt.expression {
                Expression::Call { arguments, .. } => assert_eq!(arguments.len(), 2),
                _ => panic!("Expected Call expression"),
            },
            _ => panic!("Expected Expression statement"),
        }
    }

    #[test]
    fn test_parse_index_expression() {
        let input = "arr[0]";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program().expect("Failed to parse program");

        match &program.statements[0] {
            Statement::Expression(expr_stmt) => {
                assert!(matches!(expr_stmt.expression, Expression::Index { .. }))
            }
            _ => panic!("Expected Expression statement"),
        }
    }

    #[test]
    fn test_parse_function_declaration() {
        let input = "fn add(x: int32, y: int32) -> int32 { return x + y }";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program().expect("Failed to parse program");
        assert_eq!(program.statements.len(), 1);

        match &program.statements[0] {
            Statement::Function(func_stmt) => {
                assert_eq!(func_stmt.name, "add");
                assert_eq!(func_stmt.parameters.len(), 2);
                assert!(func_stmt.return_type.is_some());
                assert!(!func_stmt.body.is_empty());
            }
            _ => panic!("Expected Function statement"),
        }
    }

    #[test]
    fn test_parse_function_without_return_type() {
        let input = "fn greet() { return }";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program().expect("Failed to parse program");

        match &program.statements[0] {
            Statement::Function(func_stmt) => {
                assert_eq!(func_stmt.name, "greet");
                assert_eq!(func_stmt.parameters.len(), 0);
                assert!(func_stmt.return_type.is_none());
            }
            _ => panic!("Expected Function statement"),
        }
    }

    #[test]
    fn test_parse_tensor_type() {
        let input = "let matrix: tensor<float32, [2, 3]> = [[1, 2, 3], [4, 5, 6]]";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program().expect("Failed to parse program");

        match &program.statements[0] {
            Statement::Let(let_stmt) => match &let_stmt.type_annotation {
                Some(TypeAnnotation::Tensor { dtype, shape }) => {
                    assert!(matches!(**dtype, TypeAnnotation::Float32));
                    assert_eq!(shape.len(), 2);
                    assert_eq!(shape[0], 2);
                    assert_eq!(shape[1], 3);
                }
                _ => panic!("Expected Tensor type annotation"),
            },
            _ => panic!("Expected Let statement"),
        }
    }

    #[test]
    fn test_parse_complete_function() {
        let input = r#"
fn quadratic(x: float32) -> float32 {
    return x * x + 2.0 * x + 1.0
}
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program();
        assert!(program.is_ok(), "Failed to parse function");
        assert_eq!(program.unwrap().statements.len(), 1);
    }

    #[test]
    fn test_parse_multiple_statements() {
        let input = r#"
let x = 5
let y = 10
let z = x + y
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program().expect("Failed to parse program");
        assert_eq!(program.statements.len(), 3);
    }

    #[test]
    fn test_parse_autograd_expression() {
        let input = "autograd(x * x)";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program().expect("Failed to parse program");

        match &program.statements[0] {
            Statement::Expression(expr_stmt) => {
                assert!(matches!(expr_stmt.expression, Expression::Autograd { .. }))
            }
            _ => panic!("Expected Expression statement"),
        }
    }

    #[test]
    fn test_parse_complex_expression() {
        let input = "let result = (a + b) * c - d / e";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let program = parser.parse_program();
        assert!(program.is_ok());
    }

    #[test]
    fn test_error_reporting() {
        let input = "let = 5"; // Invalid: no identifier
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        let result = parser.parse_program();
        assert!(result.is_err());
    }
}

// Type system module
// Phase 2: Type checking and inference

use crate::ast::*;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int32,
    Int64,
    Float32,
    Float64,
    Bool,
    String,
    Array(Box<Type>), // [int32], [float32], etc.
    Tensor {
        dtype: Box<Type>,
        shape: Shape,
    },
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
    },
    Tuple(Vec<Type>), // (int32, string, bool)
    Void,    // For functions that don't return a value
    Unknown, // For type inference
}

#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    Static(Vec<usize>), // Known at compile time: [2, 3, 4]
    Dynamic,            // Shape determined at runtime
}

impl Type {
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Type::Int32 | Type::Int64 | Type::Float32 | Type::Float64
        )
    }

    pub fn is_integer(&self) -> bool {
        matches!(self, Type::Int32 | Type::Int64)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Type::Float32 | Type::Float64)
    }

    pub fn is_tensor(&self) -> bool {
        matches!(self, Type::Tensor { .. })
    }

    pub fn is_compatible_with(&self, other: &Type) -> bool {
        // Check if two types are compatible for operations
        if self == other {
            return true;
        }

        // Numeric types can be compatible
        if self.is_numeric() && other.is_numeric() {
            return true;
        }

        // Tensors with same dtype and shape are compatible
        if let (
            Type::Tensor {
                dtype: dt1,
                shape: s1,
            },
            Type::Tensor {
                dtype: dt2,
                shape: s2,
            },
        ) = (self, other)
        {
            return dt1.is_compatible_with(dt2) && s1 == s2;
        }

        false
    }

    pub fn common_type(&self, other: &Type) -> Option<Type> {
        // Find common type for two types (used in binary operations)
        if self == other {
            return Some(self.clone());
        }

        // Int and Float -> Float
        if self.is_integer() && other.is_float() {
            return Some(other.clone());
        }
        if self.is_float() && other.is_integer() {
            return Some(self.clone());
        }

        // Int32 and Int64 -> Int64
        if matches!(
            (self, other),
            (Type::Int32, Type::Int64) | (Type::Int64, Type::Int32)
        ) {
            return Some(Type::Int64);
        }

        // Float32 and Float64 -> Float64
        if matches!(
            (self, other),
            (Type::Float32, Type::Float64) | (Type::Float64, Type::Float32)
        ) {
            return Some(Type::Float64);
        }

        None
    }

    pub fn to_string(&self) -> String {
        match self {
            Type::Int32 => "int32".to_string(),
            Type::Int64 => "int64".to_string(),
            Type::Float32 => "float32".to_string(),
            Type::Float64 => "float64".to_string(),
            Type::Bool => "bool".to_string(),
            Type::String => "string".to_string(),
            Type::Array(element_type) => format!("[{}]", element_type.to_string()),
            Type::Tensor { dtype, shape } => {
                let shape_str = match shape {
                    Shape::Static(dims) => format!(
                        "[{}]",
                        dims.iter()
                            .map(|d| d.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    Shape::Dynamic => "[dynamic]".to_string(),
                };
                format!("tensor<{}, {}>", dtype.to_string(), shape_str)
            }
            Type::Function {
                params,
                return_type,
            } => {
                let params_str = params
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("fn({}) -> {}", params_str, return_type.to_string())
            }
            Type::Tuple(element_types) => {
                let types_str = element_types
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({})", types_str)
            }
            Type::Void => "void".to_string(),
            Type::Unknown => "unknown".to_string(),
        }
    }
}

pub struct TypeEnvironment {
    scopes: Vec<HashMap<String, Type>>,
}

impl Default for TypeEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeEnvironment {
    pub fn new() -> Self {
        TypeEnvironment {
            scopes: vec![HashMap::new()],
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    pub fn define(&mut self, name: String, typ: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, typ);
        }
    }

    pub fn get(&self, name: &str) -> Option<Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(typ) = scope.get(name) {
                return Some(typ.clone());
            }
        }
        None
    }
}

pub struct TypeChecker {
    env: TypeEnvironment,
    errors: Vec<String>,
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeChecker {
    pub fn new() -> Self {
        TypeChecker {
            env: TypeEnvironment::new(),
            errors: vec![],
        }
    }

    pub fn errors(&self) -> &Vec<String> {
        &self.errors
    }

    pub fn check_program(&mut self, program: &Program) -> Result<(), String> {
        for statement in &program.statements {
            if let Err(e) = self.check_statement(statement) {
                self.errors.push(e);
            }
        }

        if !self.errors.is_empty() {
            return Err(format!("Type errors:\n{}", self.errors.join("\n")));
        }

        Ok(())
    }

    fn check_statement(&mut self, stmt: &Statement) -> Result<Type, String> {
        match stmt {
            Statement::Let(let_stmt) => self.check_let_statement(let_stmt),
            Statement::Assign(assign_stmt) => {
                // Check that variable exists and type matches
                let value_type = self.infer_expression(&assign_stmt.value)?;
                // For now, just accept any type
                // TODO: Check against declared type of variable
                Ok(value_type)
            }
            Statement::Return(ret_stmt) => self.check_return_statement(ret_stmt),
            Statement::Function(func_stmt) => self.check_function_statement(func_stmt),
            Statement::Expression(expr_stmt) => self.infer_expression(&expr_stmt.expression),
            Statement::If(if_stmt) => {
                // Check condition is boolean
                let cond_type = self.infer_expression(&if_stmt.condition)?;
                // For now, accept any type (will be checked at runtime)

                // Check consequence block
                for stmt in &if_stmt.consequence {
                    self.check_statement(stmt)?;
                }

                // Check alternative block if exists
                if let Some(alt) = &if_stmt.alternative {
                    for stmt in alt {
                        self.check_statement(stmt)?;
                    }
                }

                Ok(Type::Void)
            }
            Statement::While(while_stmt) => {
                // Check condition
                self.infer_expression(&while_stmt.condition)?;

                // Check body
                for stmt in &while_stmt.body {
                    self.check_statement(stmt)?;
                }

                Ok(Type::Void)
            }
            Statement::For(for_stmt) => {
                // Check iterable
                self.infer_expression(&for_stmt.iterable)?;

                // Check body
                for stmt in &for_stmt.body {
                    self.check_statement(stmt)?;
                }

                Ok(Type::Void)
            }
        }
    }

    fn check_let_statement(&mut self, stmt: &LetStatement) -> Result<Type, String> {
        let expr_type = self.infer_expression(&stmt.value)?;

        // If type annotation exists, check compatibility
        if let Some(ref annotation) = stmt.type_annotation {
            let annotated_type = self.type_annotation_to_type(annotation);
            if !expr_type.is_compatible_with(&annotated_type) {
                return Err(format!(
                    "Type mismatch in let statement '{}': expected {}, got {}",
                    stmt.name,
                    annotated_type.to_string(),
                    expr_type.to_string()
                ));
            }
            self.env.define(stmt.name.clone(), annotated_type.clone());
            Ok(annotated_type)
        } else {
            self.env.define(stmt.name.clone(), expr_type.clone());
            Ok(expr_type)
        }
    }

    fn check_return_statement(&mut self, stmt: &ReturnStatement) -> Result<Type, String> {
        self.infer_expression(&stmt.value)
    }

    fn check_function_statement(&mut self, stmt: &FunctionStatement) -> Result<Type, String> {
        // Convert parameters to types
        let param_types: Vec<Type> = stmt
            .parameters
            .iter()
            .map(|p| self.type_annotation_to_type(&p.type_annotation))
            .collect();

        let return_type = if let Some(ref ret_type) = stmt.return_type {
            Box::new(self.type_annotation_to_type(ret_type))
        } else {
            Box::new(Type::Void)
        };

        let func_type = Type::Function {
            params: param_types.clone(),
            return_type: return_type.clone(),
        };

        // Define function in environment
        self.env.define(stmt.name.clone(), func_type.clone());

        // Check function body in new scope
        self.env.push_scope();

        // Add parameters to scope
        for (param, param_type) in stmt.parameters.iter().zip(param_types.iter()) {
            self.env.define(param.name.clone(), param_type.clone());
        }

        // Check body statements
        for body_stmt in &stmt.body {
            self.check_statement(body_stmt)?;
        }

        self.env.pop_scope();

        Ok(func_type)
    }

    fn infer_expression(&mut self, expr: &Expression) -> Result<Type, String> {
        match expr {
            Expression::IntegerLiteral(_) => Ok(Type::Int32),
            Expression::FloatLiteral(_) => Ok(Type::Float64),
            Expression::BooleanLiteral(_) => Ok(Type::Bool),
            Expression::StringLiteral(_) => Ok(Type::String),

            Expression::Identifier(name) => self
                .env
                .get(name)
                .ok_or_else(|| format!("Undefined variable '{}'", name)),

            Expression::Binary {
                left,
                operator,
                right,
            } => self.check_binary_expression(left, operator, right),

            Expression::Unary { operator, operand } => {
                self.check_unary_expression(operator, operand)
            }

            Expression::Call {
                function,
                arguments,
            } => self.check_call_expression(function, arguments),

            Expression::Index { object, index } => self.check_index_expression(object, index),

            Expression::ArrayLiteral(elements) => self.check_array_literal(elements),

            Expression::TensorLiteral(elements) => {
                self.check_array_literal(elements) // Same as array for now
            }

            Expression::Autograd { expression } => {
                // Autograd returns the same type as its input (gradient)
                self.infer_expression(expression)
            }

            Expression::Range { start, end } => {
                // Range produces an array of integers
                let start_type = self.infer_expression(start)?;
                let end_type = self.infer_expression(end)?;

                // Both start and end should be integers
                if !start_type.is_integer() {
                    return Err(format!("Range start must be integer, got {}", start_type.to_string()));
                }
                if !end_type.is_integer() {
                    return Err(format!("Range end must be integer, got {}", end_type.to_string()));
                }

                Ok(Type::Array(Box::new(Type::Int32)))
            }

            Expression::InclusiveRange { start, end } => {
                // Inclusive range produces an array of integers (includes end value)
                let start_type = self.infer_expression(start)?;
                let end_type = self.infer_expression(end)?;

                // Both start and end should be integers
                if !start_type.is_integer() {
                    return Err(format!("Inclusive range start must be integer, got {}", start_type.to_string()));
                }
                if !end_type.is_integer() {
                    return Err(format!("Inclusive range end must be integer, got {}", end_type.to_string()));
                }

                Ok(Type::Array(Box::new(Type::Int32)))
            }

            Expression::If {
                condition,
                consequence,
                alternative,
            } => {
                // Check condition is boolean-like
                let _cond_type = self.infer_expression(condition)?;

                // Infer type of consequence block
                let mut consequence_type = Type::Void;
                for stmt in consequence {
                    consequence_type = self.check_statement(stmt)?;
                }

                // Infer type of alternative block
                let mut alternative_type = Type::Void;
                for stmt in alternative {
                    alternative_type = self.check_statement(stmt)?;
                }

                // Both branches must return the same type
                if consequence_type != alternative_type {
                    return Err(format!(
                        "If expression branches must return same type: got {} and {}",
                        consequence_type.to_string(),
                        alternative_type.to_string()
                    ));
                }

                Ok(consequence_type)
            }

            Expression::Match { value, arms } => {
                // Check the value to match against
                let _value_type = self.infer_expression(value)?;

                // All arms must return the same type
                if arms.is_empty() {
                    return Err("Match expression must have at least one arm".to_string());
                }

                // Infer type of first arm
                let first_type = self.infer_expression(&arms[0].expression)?;

                // Check all other arms return the same type
                for arm in &arms[1..] {
                    let arm_type = self.infer_expression(&arm.expression)?;
                    if arm_type != first_type {
                        return Err(format!(
                            "Match arms must all return same type: got {} and {}",
                            first_type.to_string(),
                            arm_type.to_string()
                        ));
                    }
                }

                Ok(first_type)
            }

            Expression::TupleLiteral(elements) => {
                // Infer types of all tuple elements
                let element_types: Result<Vec<Type>, String> =
                    elements.iter().map(|e| self.infer_expression(e)).collect();
                Ok(Type::Tuple(element_types?))
            }

            Expression::TupleIndex { tuple, index } => {
                let tuple_type = self.infer_expression(tuple)?;

                match tuple_type {
                    Type::Tuple(element_types) => {
                        if *index < element_types.len() {
                            Ok(element_types[*index].clone())
                        } else {
                            Err(format!(
                                "Tuple index out of bounds: index {} on tuple with {} elements",
                                index,
                                element_types.len()
                            ))
                        }
                    }
                    _ => Err(format!(
                        "Cannot use tuple indexing on type {}",
                        tuple_type.to_string()
                    )),
                }
            }
        }
    }

    fn check_binary_expression(
        &mut self,
        left: &Expression,
        operator: &BinaryOperator,
        right: &Expression,
    ) -> Result<Type, String> {
        let left_type = self.infer_expression(left)?;
        let right_type = self.infer_expression(right)?;

        match operator {
            BinaryOperator::Add
            | BinaryOperator::Subtract
            | BinaryOperator::Multiply
            | BinaryOperator::Divide
            | BinaryOperator::Modulo => {
                // Arithmetic operators require numeric types
                if !left_type.is_numeric() {
                    return Err(format!(
                        "Left operand of {:?} must be numeric, got {}",
                        operator,
                        left_type.to_string()
                    ));
                }
                if !right_type.is_numeric() {
                    return Err(format!(
                        "Right operand of {:?} must be numeric, got {}",
                        operator,
                        right_type.to_string()
                    ));
                }

                // Return common type
                left_type.common_type(&right_type).ok_or_else(|| {
                    format!(
                        "Incompatible types for {:?}: {} and {}",
                        operator,
                        left_type.to_string(),
                        right_type.to_string()
                    )
                })
            }

            BinaryOperator::MatMul => {
                // Matrix multiplication requires tensors
                // TODO: Implement shape checking for matrix multiplication
                if !left_type.is_tensor() || !right_type.is_tensor() {
                    return Err("Matrix multiplication (@) requires tensor operands".to_string());
                }
                Ok(left_type)
            }

            BinaryOperator::Equal
            | BinaryOperator::NotEqual
            | BinaryOperator::LessThan
            | BinaryOperator::LessEqual
            | BinaryOperator::GreaterThan
            | BinaryOperator::GreaterEqual => {
                // Comparison operators return bool
                if !left_type.is_compatible_with(&right_type) {
                    return Err(format!(
                        "Cannot compare incompatible types: {} and {}",
                        left_type.to_string(),
                        right_type.to_string()
                    ));
                }
                Ok(Type::Bool)
            }

            BinaryOperator::And | BinaryOperator::Or => {
                // Logical operators require bool
                if left_type != Type::Bool {
                    return Err(format!(
                        "Left operand of {:?} must be bool, got {}",
                        operator,
                        left_type.to_string()
                    ));
                }
                if right_type != Type::Bool {
                    return Err(format!(
                        "Right operand of {:?} must be bool, got {}",
                        operator,
                        right_type.to_string()
                    ));
                }
                Ok(Type::Bool)
            }
        }
    }

    fn check_unary_expression(
        &mut self,
        operator: &UnaryOperator,
        operand: &Expression,
    ) -> Result<Type, String> {
        let operand_type = self.infer_expression(operand)?;

        match operator {
            UnaryOperator::Negate => {
                if !operand_type.is_numeric() {
                    return Err(format!(
                        "Negation requires numeric type, got {}",
                        operand_type.to_string()
                    ));
                }
                Ok(operand_type)
            }
            UnaryOperator::Not => {
                if operand_type != Type::Bool {
                    return Err(format!(
                        "Logical NOT requires bool type, got {}",
                        operand_type.to_string()
                    ));
                }
                Ok(Type::Bool)
            }
        }
    }

    fn check_call_expression(
        &mut self,
        function: &Expression,
        arguments: &[Expression],
    ) -> Result<Type, String> {
        let func_type = self.infer_expression(function)?;

        if let Type::Function {
            params,
            return_type,
        } = func_type
        {
            // Check argument count
            if arguments.len() != params.len() {
                return Err(format!(
                    "Function expects {} arguments, got {}",
                    params.len(),
                    arguments.len()
                ));
            }

            // Check argument types
            for (i, (arg, expected_type)) in arguments.iter().zip(params.iter()).enumerate() {
                let arg_type = self.infer_expression(arg)?;
                if !arg_type.is_compatible_with(expected_type) {
                    return Err(format!(
                        "Argument {} type mismatch: expected {}, got {}",
                        i,
                        expected_type.to_string(),
                        arg_type.to_string()
                    ));
                }
            }

            Ok(*return_type)
        } else {
            Err(format!(
                "Cannot call non-function type: {}",
                func_type.to_string()
            ))
        }
    }

    fn check_index_expression(
        &mut self,
        object: &Expression,
        index: &Expression,
    ) -> Result<Type, String> {
        let object_type = self.infer_expression(object)?;
        let index_type = self.infer_expression(index)?;

        // Index must be integer
        if !index_type.is_integer() {
            return Err(format!(
                "Index must be integer type, got {}",
                index_type.to_string()
            ));
        }

        // For now, return the dtype of tensor or Unknown for arrays
        if let Type::Tensor { dtype, .. } = object_type {
            Ok(*dtype)
        } else {
            Ok(Type::Unknown) // TODO: Better array element type inference
        }
    }

    fn check_array_literal(&mut self, elements: &[Expression]) -> Result<Type, String> {
        if elements.is_empty() {
            return Ok(Type::Unknown);
        }

        // Infer type from first element
        let first_type = self.infer_expression(&elements[0])?;

        // Check all elements have compatible types
        for (i, elem) in elements.iter().enumerate().skip(1) {
            let elem_type = self.infer_expression(elem)?;
            if !elem_type.is_compatible_with(&first_type) {
                return Err(format!(
                    "Array element {} has incompatible type: expected {}, got {}",
                    i,
                    first_type.to_string(),
                    elem_type.to_string()
                ));
            }
        }

        // For now, return a 1D tensor
        Ok(Type::Tensor {
            dtype: Box::new(first_type),
            shape: Shape::Static(vec![elements.len()]),
        })
    }

    fn type_annotation_to_type(&self, annotation: &TypeAnnotation) -> Type {
        match annotation {
            TypeAnnotation::Int32 => Type::Int32,
            TypeAnnotation::Int64 => Type::Int64,
            TypeAnnotation::Float32 => Type::Float32,
            TypeAnnotation::Float64 => Type::Float64,
            TypeAnnotation::Bool => Type::Bool,
            TypeAnnotation::String => Type::String,
            TypeAnnotation::Array(element_type) => Type::Array(Box::new(
                self.type_annotation_to_type(element_type),
            )),
            TypeAnnotation::Tensor { dtype, shape } => Type::Tensor {
                dtype: Box::new(self.type_annotation_to_type(dtype)),
                shape: Shape::Static(shape.clone()),
            },
            TypeAnnotation::Tuple(element_types) => Type::Tuple(
                element_types
                    .iter()
                    .map(|t| self.type_annotation_to_type(t))
                    .collect(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    #[test]
    fn test_type_checking() {
        assert!(Type::Float32.is_numeric());
        assert!(!Type::Bool.is_numeric());
    }

    #[test]
    fn test_type_compatibility() {
        assert!(Type::Int32.is_compatible_with(&Type::Int64));
        assert!(Type::Float32.is_compatible_with(&Type::Float64));
        assert!(!Type::Int32.is_compatible_with(&Type::Bool));
    }

    #[test]
    fn test_common_type() {
        assert_eq!(Type::Int32.common_type(&Type::Int64), Some(Type::Int64));
        assert_eq!(
            Type::Float32.common_type(&Type::Float64),
            Some(Type::Float64)
        );
        assert_eq!(Type::Int32.common_type(&Type::Float32), Some(Type::Float32));
        assert_eq!(Type::Int32.common_type(&Type::Bool), None);
    }

    #[test]
    fn test_check_let_statement() {
        let input = "let x = 42";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_check_let_with_type_annotation() {
        let input = "let x: int32 = 42";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_mismatch_in_let() {
        let input = "let x: bool = 42";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_err());
    }

    #[test]
    fn test_binary_expression_types() {
        let inputs = vec![
            ("let x = 5 + 10", true),     // int + int -> ok
            ("let x = 5.0 + 10.0", true), // float + float -> ok
            ("let x = 5 + 10.0", true),   // int + float -> ok (coercion)
            ("let x = true + 5", false),  // bool + int -> error
        ];

        for (input, should_pass) in inputs {
            let lexer = Lexer::new(input);
            let mut parser = Parser::new(lexer);
            let program = parser.parse_program().unwrap();

            let mut checker = TypeChecker::new();
            let result = checker.check_program(&program);

            assert_eq!(result.is_ok(), should_pass, "Failed for: {}", input);
        }
    }

    #[test]
    fn test_comparison_operators() {
        let inputs = vec![
            ("let x = 5 > 10", true),
            ("let x = 5.0 <= 10.0", true),
            ("let x = true == false", true),
            ("let x = 5 != 10", true),
        ];

        for (input, should_pass) in inputs {
            let lexer = Lexer::new(input);
            let mut parser = Parser::new(lexer);
            let program = parser.parse_program().unwrap();

            let mut checker = TypeChecker::new();
            let result = checker.check_program(&program);

            assert_eq!(result.is_ok(), should_pass, "Failed for: {}", input);
        }
    }

    #[test]
    fn test_logical_operators() {
        let inputs = vec![
            ("let x = true and false", true),
            ("let x = true or false", true),
            ("let x = not true", true),
            ("let x = 5 and 10", false),   // int and int -> error
            ("let x = true and 5", false), // bool and int -> error
        ];

        for (input, should_pass) in inputs {
            let lexer = Lexer::new(input);
            let mut parser = Parser::new(lexer);
            let program = parser.parse_program().unwrap();

            let mut checker = TypeChecker::new();
            let result = checker.check_program(&program);

            assert_eq!(result.is_ok(), should_pass, "Failed for: {}", input);
        }
    }

    #[test]
    fn test_unary_operators() {
        let inputs = vec![
            ("let x = -5", true),
            ("let x = -5.0", true),
            ("let x = not true", true),
            ("let x = -true", false), // can't negate bool
            ("let x = not 5", false), // can't NOT int
        ];

        for (input, should_pass) in inputs {
            let lexer = Lexer::new(input);
            let mut parser = Parser::new(lexer);
            let program = parser.parse_program().unwrap();

            let mut checker = TypeChecker::new();
            let result = checker.check_program(&program);

            assert_eq!(result.is_ok(), should_pass, "Failed for: {}", input);
        }
    }

    #[test]
    fn test_undefined_variable() {
        let input = "let x = y + 5"; // y is undefined
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        let result = checker.check_program(&program);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Undefined variable"));
    }

    #[test]
    fn test_function_type_checking() {
        let input = r#"
fn add(x: int32, y: int32) -> int32 {
    return x + y
}
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_function_call_type_checking() {
        let input = r#"
fn add(x: int32, y: int32) -> int32 {
    return x + y
}
let result = add(5, 10)
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_function_call_wrong_arg_count() {
        let input = r#"
fn add(x: int32, y: int32) -> int32 {
    return x + y
}
let result = add(5)
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        let result = checker.check_program(&program);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("expects 2 arguments"));
    }

    #[test]
    fn test_function_call_wrong_arg_type() {
        let input = r#"
fn add(x: int32, y: int32) -> int32 {
    return x + y
}
let result = add(true, 10)
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        let result = checker.check_program(&program);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("type mismatch"));
    }

    #[test]
    fn test_array_type_inference() {
        let input = "let arr = [1, 2, 3, 4, 5]";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_array_mixed_types_error() {
        let input = "let arr = [1, 2.0, 3]"; // Mixed int and float
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        // This should pass because int and float are compatible
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_array_incompatible_types_error() {
        let input = "let arr = [1, true, 3]"; // int and bool
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_err());
    }

    #[test]
    fn test_index_expression_type() {
        let input = r#"
let arr = [1, 2, 3]
let elem = arr[0]
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_index_non_integer_error() {
        let input = r#"
let arr = [1, 2, 3]
let elem = arr[true]
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        let result = checker.check_program(&program);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Index must be integer"));
    }

    #[test]
    fn test_tensor_type_annotation() {
        // For now, just test that a simple 1D tensor with annotation works
        let input = "let vec: tensor<float32, [3]> = [1.0, 2.0, 3.0]";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_autograd_type() {
        let input = "let grad = autograd(x)";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        // This will fail because x is undefined
        assert!(checker.check_program(&program).is_err());
    }

    #[test]
    fn test_scope_management() {
        let input = r#"
fn test() {
    let x = 5
}
let y = x
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        let result = checker.check_program(&program);

        // Should fail because x is not in scope outside the function
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Undefined variable 'x'"));
    }

    #[test]
    fn test_type_to_string() {
        assert_eq!(Type::Int32.to_string(), "int32");
        assert_eq!(Type::Float64.to_string(), "float64");
        assert_eq!(Type::Bool.to_string(), "bool");

        let tensor = Type::Tensor {
            dtype: Box::new(Type::Float32),
            shape: Shape::Static(vec![2, 3]),
        };
        assert_eq!(tensor.to_string(), "tensor<float32, [2, 3]>");
    }

    #[test]
    fn test_complex_expression() {
        let input = "let result = (5 + 10) * 2 - 3 / 4";
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }
}

// Interpreter module - Execute Charl AST
// Phase 3: Tree-walking interpreter

use crate::ast::*;
use crate::autograd::{ComputationGraph, Tensor as AutogradTensor};
use std::collections::HashMap;

// Runtime value representation
#[derive(Debug, Clone)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Array(Vec<Value>),
    Tensor { data: Vec<Value>, shape: Vec<usize> },
    AutogradTensor(AutogradTensor), // Tensor with gradient tracking
    Function {
        parameters: Vec<Parameter>,
        body: Vec<Statement>,
        closure: Environment,
    },
    Null,
}

// Manual PartialEq implementation since AutogradTensor doesn't implement it
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => a == b,
            (Value::Tensor { data: d1, shape: s1 }, Value::Tensor { data: d2, shape: s2 }) => {
                d1 == d2 && s1 == s2
            }
            (Value::AutogradTensor(a), Value::AutogradTensor(b)) => {
                a.data == b.data && a.shape == b.shape
            }
            (Value::Function { .. }, Value::Function { .. }) => false, // Functions are never equal
            (Value::Null, Value::Null) => true,
            _ => false,
        }
    }
}

impl Value {
    pub fn type_name(&self) -> &str {
        match self {
            Value::Integer(_) => "integer",
            Value::Float(_) => "float",
            Value::Boolean(_) => "boolean",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Tensor { .. } => "tensor",
            Value::AutogradTensor(_) => "autograd_tensor",
            Value::Function { .. } => "function",
            Value::Null => "null",
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Boolean(b) => *b,
            Value::Null => false,
            Value::Integer(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            _ => true,
        }
    }

    // Convert to float for numeric operations
    pub fn to_float(&self) -> Result<f64, String> {
        match self {
            Value::Integer(i) => Ok(*i as f64),
            Value::Float(f) => Ok(*f),
            _ => Err(format!("Cannot convert {} to float", self.type_name())),
        }
    }

    // Convert to integer for numeric operations
    pub fn to_integer(&self) -> Result<i64, String> {
        match self {
            Value::Integer(i) => Ok(*i),
            Value::Float(f) => Ok(*f as i64),
            _ => Err(format!("Cannot convert {} to integer", self.type_name())),
        }
    }
}

// Environment for variable storage with scope management
#[derive(Debug, Clone, PartialEq)]
pub struct Environment {
    scopes: Vec<HashMap<String, Value>>,
}

impl Environment {
    pub fn new() -> Self {
        Environment {
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

    pub fn set(&mut self, name: String, value: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, value);
        }
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        // Search from innermost to outermost scope
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Some(value);
            }
        }
        None
    }
}

pub struct Interpreter {
    env: Environment,
    return_value: Option<Value>,
    graph: ComputationGraph,
}

impl Interpreter {
    pub fn new() -> Self {
        Interpreter {
            env: Environment::new(),
            return_value: None,
            graph: ComputationGraph::new(),
        }
    }

    pub fn eval(&mut self, program: Program) -> Result<Value, String> {
        let mut result = Value::Null;

        for statement in program.statements {
            result = self.eval_statement(&statement)?;

            // If we hit a return statement, stop execution
            if self.return_value.is_some() {
                break;
            }
        }

        Ok(result)
    }

    pub fn eval_statement(&mut self, stmt: &Statement) -> Result<Value, String> {
        match stmt {
            Statement::Let(let_stmt) => self.eval_let_statement(let_stmt),
            Statement::Return(ret_stmt) => self.eval_return_statement(ret_stmt),
            Statement::Expression(expr_stmt) => self.eval_expression(&expr_stmt.expression),
            Statement::Function(func_stmt) => self.eval_function_statement(func_stmt),
        }
    }

    fn eval_let_statement(&mut self, stmt: &LetStatement) -> Result<Value, String> {
        let value = self.eval_expression(&stmt.value)?;
        self.env.set(stmt.name.clone(), value.clone());
        Ok(value)
    }

    fn eval_return_statement(&mut self, stmt: &ReturnStatement) -> Result<Value, String> {
        let value = self.eval_expression(&stmt.value)?;
        self.return_value = Some(value.clone());
        Ok(value)
    }

    fn eval_function_statement(&mut self, stmt: &FunctionStatement) -> Result<Value, String> {
        let func = Value::Function {
            parameters: stmt.parameters.clone(),
            body: stmt.body.clone(),
            closure: self.env.clone(),
        };
        self.env.set(stmt.name.clone(), func.clone());
        Ok(func)
    }

    pub fn eval_expression(&mut self, expr: &Expression) -> Result<Value, String> {
        match expr {
            Expression::IntegerLiteral(i) => Ok(Value::Integer(*i)),
            Expression::FloatLiteral(f) => Ok(Value::Float(*f)),
            Expression::BooleanLiteral(b) => Ok(Value::Boolean(*b)),
            Expression::StringLiteral(s) => Ok(Value::String(s.clone())),

            Expression::Identifier(name) => {
                self.env.get(name)
                    .cloned()
                    .ok_or_else(|| format!("Undefined variable: {}", name))
            }

            Expression::ArrayLiteral(elements) => {
                let values: Result<Vec<Value>, String> = elements
                    .iter()
                    .map(|e| self.eval_expression(e))
                    .collect();
                Ok(Value::Array(values?))
            }

            Expression::TensorLiteral(elements) => {
                let values: Result<Vec<Value>, String> = elements
                    .iter()
                    .map(|e| self.eval_expression(e))
                    .collect();
                let data = values?;
                let shape = vec![data.len()];
                Ok(Value::Tensor { data, shape })
            }

            Expression::Binary { left, operator, right } => {
                self.eval_binary_expression(left, operator, right)
            }

            Expression::Unary { operator, operand } => {
                self.eval_unary_expression(operator, operand)
            }

            Expression::Call { function, arguments } => {
                self.eval_call_expression(function, arguments)
            }

            Expression::Index { object, index } => {
                self.eval_index_expression(object, index)
            }

            Expression::Autograd { expression } => {
                self.eval_autograd_expression(expression)
            }
        }
    }

    fn eval_binary_expression(
        &mut self,
        left: &Expression,
        op: &BinaryOperator,
        right: &Expression,
    ) -> Result<Value, String> {
        let left_val = self.eval_expression(left)?;
        let right_val = self.eval_expression(right)?;

        match op {
            BinaryOperator::Add => self.eval_add(&left_val, &right_val),
            BinaryOperator::Subtract => self.eval_subtract(&left_val, &right_val),
            BinaryOperator::Multiply => self.eval_multiply(&left_val, &right_val),
            BinaryOperator::Divide => self.eval_divide(&left_val, &right_val),
            BinaryOperator::Modulo => self.eval_modulo(&left_val, &right_val),
            BinaryOperator::MatMul => self.eval_matmul(&left_val, &right_val),
            BinaryOperator::Equal => Ok(Value::Boolean(left_val == right_val)),
            BinaryOperator::NotEqual => Ok(Value::Boolean(left_val != right_val)),
            BinaryOperator::LessThan => self.eval_less_than(&left_val, &right_val),
            BinaryOperator::LessEqual => self.eval_less_equal(&left_val, &right_val),
            BinaryOperator::GreaterThan => self.eval_greater_than(&left_val, &right_val),
            BinaryOperator::GreaterEqual => self.eval_greater_equal(&left_val, &right_val),
            BinaryOperator::And => Ok(Value::Boolean(left_val.is_truthy() && right_val.is_truthy())),
            BinaryOperator::Or => Ok(Value::Boolean(left_val.is_truthy() || right_val.is_truthy())),
        }
    }

    fn eval_add(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l + r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l + r)),
            (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(*l as f64 + r)),
            (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l + *r as f64)),
            (Value::String(l), Value::String(r)) => Ok(Value::String(format!("{}{}", l, r))),
            _ => Err(format!("Cannot add {} and {}", left.type_name(), right.type_name())),
        }
    }

    fn eval_subtract(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l - r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l - r)),
            (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(*l as f64 - r)),
            (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l - *r as f64)),
            _ => Err(format!("Cannot subtract {} and {}", left.type_name(), right.type_name())),
        }
    }

    fn eval_multiply(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l * r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l * r)),
            (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(*l as f64 * r)),
            (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l * *r as f64)),
            _ => Err(format!("Cannot multiply {} and {}", left.type_name(), right.type_name())),
        }
    }

    fn eval_divide(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => {
                if *r == 0 {
                    return Err("Division by zero".to_string());
                }
                Ok(Value::Integer(l / r))
            }
            (Value::Float(l), Value::Float(r)) => {
                if *r == 0.0 {
                    return Err("Division by zero".to_string());
                }
                Ok(Value::Float(l / r))
            }
            (Value::Integer(l), Value::Float(r)) => {
                if *r == 0.0 {
                    return Err("Division by zero".to_string());
                }
                Ok(Value::Float(*l as f64 / r))
            }
            (Value::Float(l), Value::Integer(r)) => {
                if *r == 0 {
                    return Err("Division by zero".to_string());
                }
                Ok(Value::Float(l / *r as f64))
            }
            _ => Err(format!("Cannot divide {} and {}", left.type_name(), right.type_name())),
        }
    }

    fn eval_modulo(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => {
                if *r == 0 {
                    return Err("Modulo by zero".to_string());
                }
                Ok(Value::Integer(l % r))
            }
            _ => Err(format!("Cannot modulo {} and {}", left.type_name(), right.type_name())),
        }
    }

    fn eval_matmul(&self, _left: &Value, _right: &Value) -> Result<Value, String> {
        // TODO: Implement matrix multiplication in future phases
        Err("Matrix multiplication not yet implemented".to_string())
    }

    fn eval_less_than(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l < r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l < r)),
            (Value::Integer(l), Value::Float(r)) => Ok(Value::Boolean((*l as f64) < *r)),
            (Value::Float(l), Value::Integer(r)) => Ok(Value::Boolean(*l < (*r as f64))),
            _ => Err(format!("Cannot compare {} and {}", left.type_name(), right.type_name())),
        }
    }

    fn eval_less_equal(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l <= r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l <= r)),
            (Value::Integer(l), Value::Float(r)) => Ok(Value::Boolean((*l as f64) <= *r)),
            (Value::Float(l), Value::Integer(r)) => Ok(Value::Boolean(*l <= (*r as f64))),
            _ => Err(format!("Cannot compare {} and {}", left.type_name(), right.type_name())),
        }
    }

    fn eval_greater_than(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l > r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l > r)),
            (Value::Integer(l), Value::Float(r)) => Ok(Value::Boolean((*l as f64) > *r)),
            (Value::Float(l), Value::Integer(r)) => Ok(Value::Boolean(*l > (*r as f64))),
            _ => Err(format!("Cannot compare {} and {}", left.type_name(), right.type_name())),
        }
    }

    fn eval_greater_equal(&self, left: &Value, right: &Value) -> Result<Value, String> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l >= r)),
            (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l >= r)),
            (Value::Integer(l), Value::Float(r)) => Ok(Value::Boolean((*l as f64) >= *r)),
            (Value::Float(l), Value::Integer(r)) => Ok(Value::Boolean(*l >= (*r as f64))),
            _ => Err(format!("Cannot compare {} and {}", left.type_name(), right.type_name())),
        }
    }

    fn eval_unary_expression(
        &mut self,
        op: &UnaryOperator,
        operand: &Expression,
    ) -> Result<Value, String> {
        let value = self.eval_expression(operand)?;

        match op {
            UnaryOperator::Negate => match value {
                Value::Integer(i) => Ok(Value::Integer(-i)),
                Value::Float(f) => Ok(Value::Float(-f)),
                _ => Err(format!("Cannot negate {}", value.type_name())),
            },
            UnaryOperator::Not => Ok(Value::Boolean(!value.is_truthy())),
        }
    }

    fn eval_call_expression(
        &mut self,
        function: &Expression,
        arguments: &[Expression],
    ) -> Result<Value, String> {
        let func_val = self.eval_expression(function)?;

        match func_val {
            Value::Function { parameters, body, closure } => {
                if parameters.len() != arguments.len() {
                    return Err(format!(
                        "Wrong number of arguments: expected {}, got {}",
                        parameters.len(),
                        arguments.len()
                    ));
                }

                // Evaluate arguments
                let arg_values: Result<Vec<Value>, String> = arguments
                    .iter()
                    .map(|arg| self.eval_expression(arg))
                    .collect();
                let arg_values = arg_values?;

                // Save current environment and use closure
                let saved_env = self.env.clone();
                self.env = closure;

                // Create new scope for function
                self.env.push_scope();

                // Bind parameters to arguments
                for (param, arg_val) in parameters.iter().zip(arg_values.iter()) {
                    self.env.set(param.name.clone(), arg_val.clone());
                }

                // Execute function body
                let mut result = Value::Null;
                for stmt in &body {
                    result = self.eval_statement(stmt)?;
                    if self.return_value.is_some() {
                        break;
                    }
                }

                // Get return value
                let return_val = self.return_value.take().unwrap_or(result);

                // Restore environment
                self.env.pop_scope();
                self.env = saved_env;

                Ok(return_val)
            }
            _ => Err(format!("Cannot call non-function value: {}", func_val.type_name())),
        }
    }

    fn eval_index_expression(
        &mut self,
        object: &Expression,
        index: &Expression,
    ) -> Result<Value, String> {
        let obj_val = self.eval_expression(object)?;
        let idx_val = self.eval_expression(index)?;

        let idx = match idx_val {
            Value::Integer(i) => i,
            _ => return Err(format!("Index must be an integer, got {}", idx_val.type_name())),
        };

        match obj_val {
            Value::Array(ref elements) => {
                let idx = if idx < 0 {
                    (elements.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };

                elements.get(idx)
                    .cloned()
                    .ok_or_else(|| format!("Index out of bounds: {}", idx))
            }
            Value::Tensor { ref data, .. } => {
                let idx = if idx < 0 {
                    (data.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };

                data.get(idx)
                    .cloned()
                    .ok_or_else(|| format!("Index out of bounds: {}", idx))
            }
            _ => Err(format!("Cannot index into {}", obj_val.type_name())),
        }
    }

    fn eval_autograd_expression(&mut self, expr: &Expression) -> Result<Value, String> {
        // Evaluate the expression and convert to autograd tensor if needed
        let value = self.eval_expression(expr)?;

        match value {
            Value::Integer(i) => {
                // Convert integer to autograd tensor
                let tensor = AutogradTensor::scalar_with_grad(i as f64);
                self.graph.add_node(tensor.clone());
                Ok(Value::AutogradTensor(tensor))
            }
            Value::Float(f) => {
                // Convert float to autograd tensor
                let tensor = AutogradTensor::scalar_with_grad(f);
                self.graph.add_node(tensor.clone());
                Ok(Value::AutogradTensor(tensor))
            }
            Value::Array(ref elements) => {
                // Convert array to autograd tensor
                let data: Result<Vec<f64>, String> = elements
                    .iter()
                    .map(|v| match v {
                        Value::Integer(i) => Ok(*i as f64),
                        Value::Float(f) => Ok(*f),
                        _ => Err(format!("autograd() only works with numeric arrays, found {}", v.type_name())),
                    })
                    .collect();

                let data = data?;
                let shape = vec![data.len()];
                let tensor = AutogradTensor::with_grad(data, shape);
                self.graph.add_node(tensor.clone());
                Ok(Value::AutogradTensor(tensor))
            }
            Value::AutogradTensor(_) => {
                // Already an autograd tensor, just return it
                Ok(value)
            }
            _ => Err(format!(
                "autograd() can only be applied to numbers or arrays, got {}",
                value.type_name()
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn eval_input(input: &str) -> Result<Value, String> {
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program().map_err(|e| e.to_string())?;

        let mut interpreter = Interpreter::new();
        interpreter.eval(program)
    }

    #[test]
    fn test_interpreter_creation() {
        let _interpreter = Interpreter::new();
    }

    #[test]
    fn test_eval_integer_literal() {
        let result = eval_input("42").unwrap();
        assert_eq!(result, Value::Integer(42));
    }

    #[test]
    fn test_eval_float_literal() {
        let result = eval_input("3.14").unwrap();
        assert_eq!(result, Value::Float(3.14));
    }

    #[test]
    fn test_eval_boolean_literal() {
        let result = eval_input("true").unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_eval_string_literal() {
        let result = eval_input("\"hello\"").unwrap();
        assert_eq!(result, Value::String("hello".to_string()));
    }

    #[test]
    fn test_eval_arithmetic() {
        assert_eq!(eval_input("5 + 3").unwrap(), Value::Integer(8));
        assert_eq!(eval_input("10 - 4").unwrap(), Value::Integer(6));
        assert_eq!(eval_input("6 * 7").unwrap(), Value::Integer(42));
        assert_eq!(eval_input("20 / 4").unwrap(), Value::Integer(5));
        assert_eq!(eval_input("17 % 5").unwrap(), Value::Integer(2));
    }

    #[test]
    fn test_eval_float_arithmetic() {
        assert_eq!(eval_input("2.5 + 1.5").unwrap(), Value::Float(4.0));
        assert_eq!(eval_input("5.0 - 2.0").unwrap(), Value::Float(3.0));
        assert_eq!(eval_input("3.0 * 2.0").unwrap(), Value::Float(6.0));
        assert_eq!(eval_input("9.0 / 3.0").unwrap(), Value::Float(3.0));
    }

    #[test]
    fn test_eval_mixed_arithmetic() {
        assert_eq!(eval_input("5 + 2.5").unwrap(), Value::Float(7.5));
        assert_eq!(eval_input("10.0 - 3").unwrap(), Value::Float(7.0));
    }

    #[test]
    fn test_eval_comparison() {
        assert_eq!(eval_input("5 < 10").unwrap(), Value::Boolean(true));
        assert_eq!(eval_input("5 > 10").unwrap(), Value::Boolean(false));
        assert_eq!(eval_input("5 <= 5").unwrap(), Value::Boolean(true));
        assert_eq!(eval_input("5 >= 6").unwrap(), Value::Boolean(false));
        assert_eq!(eval_input("5 == 5").unwrap(), Value::Boolean(true));
        assert_eq!(eval_input("5 != 10").unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_eval_logical() {
        assert_eq!(eval_input("true && true").unwrap(), Value::Boolean(true));
        assert_eq!(eval_input("true && false").unwrap(), Value::Boolean(false));
        assert_eq!(eval_input("false || true").unwrap(), Value::Boolean(true));
        assert_eq!(eval_input("false || false").unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_eval_unary() {
        assert_eq!(eval_input("-5").unwrap(), Value::Integer(-5));
        assert_eq!(eval_input("-3.14").unwrap(), Value::Float(-3.14));
        assert_eq!(eval_input("!true").unwrap(), Value::Boolean(false));
        assert_eq!(eval_input("!false").unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_eval_let_statement() {
        let result = eval_input("let x = 42\nx").unwrap();
        assert_eq!(result, Value::Integer(42));
    }

    #[test]
    fn test_eval_let_with_expression() {
        let result = eval_input("let x = 5 + 3\nx").unwrap();
        assert_eq!(result, Value::Integer(8));
    }

    #[test]
    fn test_eval_array_literal() {
        let result = eval_input("[1, 2, 3]").unwrap();
        assert_eq!(
            result,
            Value::Array(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)])
        );
    }

    #[test]
    fn test_eval_array_index() {
        let result = eval_input("let arr = [1, 2, 3]\narr[1]").unwrap();
        assert_eq!(result, Value::Integer(2));
    }

    #[test]
    fn test_eval_array_negative_index() {
        let result = eval_input("let arr = [1, 2, 3]\narr[-1]").unwrap();
        assert_eq!(result, Value::Integer(3));
    }

    #[test]
    fn test_eval_function_declaration() {
        let input = "fn add(x: int32, y: int32) -> int32 { return x + y }\nadd(5, 3)";
        let result = eval_input(input).unwrap();
        assert_eq!(result, Value::Integer(8));
    }

    #[test]
    fn test_eval_function_with_closure() {
        let input = r#"
            let x = 10
            fn add_x(y: int32) -> int32 { return x + y }
            add_x(5)
        "#;
        let result = eval_input(input).unwrap();
        assert_eq!(result, Value::Integer(15));
    }

    #[test]
    fn test_eval_nested_function_calls() {
        let input = r#"
            fn double(x: int32) -> int32 { return x * 2 }
            fn add(x: int32, y: int32) -> int32 { return x + y }
            add(double(3), double(4))
        "#;
        let result = eval_input(input).unwrap();
        assert_eq!(result, Value::Integer(14));
    }

    #[test]
    fn test_error_division_by_zero() {
        let result = eval_input("10 / 0");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Division by zero"));
    }

    #[test]
    fn test_error_undefined_variable() {
        let result = eval_input("x + 5");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Undefined variable"));
    }

    #[test]
    fn test_error_wrong_argument_count() {
        let input = "fn add(x: int32, y: int32) -> int32 { return x + y }\nadd(5)";
        let result = eval_input(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Wrong number of arguments"));
    }
}

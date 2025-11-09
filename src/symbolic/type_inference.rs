// Type Inference System
// Implements Hindley-Milner type inference for symbolic reasoning
//
// This module provides type inference capabilities for:
// - Static type checking
// - Type reconstruction
// - Polymorphic type inference
// - Integration with knowledge graphs for software analysis
//
// Usage:
// ```rust
// use charl::symbolic::TypeInference;
//
// let mut inference = TypeInference::new();
// inference.infer_program(&ast)?;
// let violations = inference.check_types()?;
// ```

use super::rule_engine::{Action, Severity};
use crate::ast::{BinaryOperator, Expression, Program, Statement, TypeAnnotation, UnaryOperator};
use crate::knowledge_graph::{EntityType, KnowledgeGraph, RelationType};
use std::collections::HashMap;

/// Type variable for polymorphic types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeVar(pub String);

/// Type representation for inference
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InferredType {
    /// Primitive types
    Int,
    Float,
    Bool,
    String,
    Concept, // Symbolic AI concept type
    Void,

    /// Array type: [element_type]
    Array(Box<InferredType>),

    /// Tensor type with optional shape
    Tensor(Option<Vec<usize>>),

    /// Function type: (params) -> return_type
    Function {
        params: Vec<InferredType>,
        return_type: Box<InferredType>,
    },

    /// Type variable (for polymorphism)
    Var(TypeVar),

    /// Generic type with parameters
    Generic {
        name: String,
        params: Vec<InferredType>,
    },

    /// Tuple type: (type1, type2, ...)
    Tuple(Vec<InferredType>),

    /// Unknown type (inference in progress)
    Unknown,
}

impl InferredType {
    /// Check if type is concrete (no variables)
    pub fn is_concrete(&self) -> bool {
        match self {
            InferredType::Var(_) | InferredType::Unknown => false,
            InferredType::Function {
                params,
                return_type,
            } => params.iter().all(|p| p.is_concrete()) && return_type.is_concrete(),
            InferredType::Generic { params, .. } => params.iter().all(|p| p.is_concrete()),
            InferredType::Tensor(_) => true,
            _ => true,
        }
    }

    /// Convert AST type annotation to inferred type
    pub fn from_ast(ast_type: &TypeAnnotation) -> Self {
        match ast_type {
            TypeAnnotation::Int32 | TypeAnnotation::Int64 => InferredType::Int,
            TypeAnnotation::Float32 | TypeAnnotation::Float64 => InferredType::Float,
            TypeAnnotation::Bool => InferredType::Bool,
            TypeAnnotation::String => InferredType::String,
            TypeAnnotation::Concept => InferredType::Concept,
            TypeAnnotation::Array(element_type) => {
                InferredType::Array(Box::new(InferredType::from_ast(element_type)))
            }
            TypeAnnotation::ArraySized { element_type, size: _ } => {
                InferredType::Array(Box::new(InferredType::from_ast(element_type)))
            } // Size is for compile-time checking, inferred type is Array
            TypeAnnotation::Tensor { shape, .. } => InferredType::Tensor(Some(shape.clone())),
            TypeAnnotation::Tuple(element_types) => InferredType::Tuple(
                element_types
                    .iter()
                    .map(|t| InferredType::from_ast(t))
                    .collect(),
            ),
        }
    }
}

impl std::fmt::Display for InferredType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferredType::Int => write!(f, "int"),
            InferredType::Float => write!(f, "float"),
            InferredType::Bool => write!(f, "bool"),
            InferredType::String => write!(f, "string"),
            InferredType::Concept => write!(f, "concept"),
            InferredType::Void => write!(f, "void"),
            InferredType::Array(element_type) => write!(f, "[{}]", element_type),
            InferredType::Tensor(Some(shape)) => write!(f, "tensor{:?}", shape),
            InferredType::Tensor(None) => write!(f, "tensor"),
            InferredType::Function {
                params,
                return_type,
            } => {
                write!(
                    f,
                    "({}) -> {}",
                    params
                        .iter()
                        .map(|p| p.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    return_type
                )
            }
            InferredType::Var(TypeVar(name)) => write!(f, "'{}", name),
            InferredType::Generic { name, params } => {
                write!(
                    f,
                    "{}<{}>",
                    name,
                    params
                        .iter()
                        .map(|p| p.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            InferredType::Tuple(element_types) => {
                write!(
                    f,
                    "({})",
                    element_types
                        .iter()
                        .map(|t| t.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            InferredType::Unknown => write!(f, "?"),
        }
    }
}

/// Type substitution for unification
type Substitution = HashMap<TypeVar, InferredType>;

/// Type constraint for inference
#[derive(Debug, Clone)]
pub struct TypeConstraint {
    pub expected: InferredType,
    pub actual: InferredType,
    pub location: String, // For error reporting
}

/// Type inference error
#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub location: String,
    pub expected: Option<InferredType>,
    pub actual: Option<InferredType>,
}

impl TypeError {
    pub fn mismatch(
        location: impl Into<String>,
        expected: InferredType,
        actual: InferredType,
    ) -> Self {
        TypeError {
            message: format!("Type mismatch: expected {}, found {}", expected, actual),
            location: location.into(),
            expected: Some(expected),
            actual: Some(actual),
        }
    }

    pub fn undefined(location: impl Into<String>, name: impl Into<String>) -> Self {
        TypeError {
            message: format!("Undefined variable: {}", name.into()),
            location: location.into(),
            expected: None,
            actual: None,
        }
    }
}

/// Type inference engine
pub struct TypeInference {
    /// Type environment (variable -> type)
    env: HashMap<String, InferredType>,

    /// Type constraints collected during inference
    constraints: Vec<TypeConstraint>,

    /// Type substitutions from unification
    substitution: Substitution,

    /// Counter for generating fresh type variables
    type_var_counter: usize,

    /// Knowledge graph for integration
    knowledge_graph: Option<KnowledgeGraph>,

    /// Collected type errors
    errors: Vec<TypeError>,
}

impl TypeInference {
    /// Create a new type inference engine
    pub fn new() -> Self {
        TypeInference {
            env: HashMap::new(),
            constraints: Vec::new(),
            substitution: HashMap::new(),
            type_var_counter: 0,
            knowledge_graph: None,
            errors: Vec::new(),
        }
    }

    /// Set knowledge graph for integration
    pub fn with_knowledge_graph(mut self, kg: KnowledgeGraph) -> Self {
        self.knowledge_graph = Some(kg);
        self
    }

    /// Generate a fresh type variable
    fn fresh_type_var(&mut self) -> InferredType {
        let var = InferredType::Var(TypeVar(format!("t{}", self.type_var_counter)));
        self.type_var_counter += 1;
        var
    }

    /// Add a type constraint
    fn add_constraint(&mut self, expected: InferredType, actual: InferredType, location: String) {
        self.constraints.push(TypeConstraint {
            expected,
            actual,
            location,
        });
    }

    /// Apply substitution to a type
    fn apply_subst(&self, ty: &InferredType) -> InferredType {
        match ty {
            InferredType::Var(var) => {
                if let Some(substituted) = self.substitution.get(var) {
                    // Recursive substitution
                    self.apply_subst(substituted)
                } else {
                    ty.clone()
                }
            }
            InferredType::Function {
                params,
                return_type,
            } => InferredType::Function {
                params: params.iter().map(|p| self.apply_subst(p)).collect(),
                return_type: Box::new(self.apply_subst(return_type)),
            },
            InferredType::Generic { name, params } => InferredType::Generic {
                name: name.clone(),
                params: params.iter().map(|p| self.apply_subst(p)).collect(),
            },
            _ => ty.clone(),
        }
    }

    /// Unify two types (Hindley-Milner unification)
    fn unify(
        &mut self,
        t1: &InferredType,
        t2: &InferredType,
        location: &str,
    ) -> Result<(), TypeError> {
        let t1 = self.apply_subst(t1);
        let t2 = self.apply_subst(t2);

        match (&t1, &t2) {
            // Same type
            _ if t1 == t2 => Ok(()),

            // Variable unification
            (InferredType::Var(var), ty) | (ty, InferredType::Var(var)) => {
                // Occurs check: prevent infinite types like t = t -> t
                if self.occurs_in(var, ty) {
                    return Err(TypeError {
                        message: format!("Infinite type: {} occurs in {}", var.0, ty),
                        location: location.to_string(),
                        expected: Some(t1.clone()),
                        actual: Some(t2.clone()),
                    });
                }

                self.substitution.insert(var.clone(), ty.clone());
                Ok(())
            }

            // Function unification
            (
                InferredType::Function {
                    params: p1,
                    return_type: r1,
                },
                InferredType::Function {
                    params: p2,
                    return_type: r2,
                },
            ) => {
                if p1.len() != p2.len() {
                    return Err(TypeError::mismatch(location, t1.clone(), t2.clone()));
                }

                for (param1, param2) in p1.iter().zip(p2.iter()) {
                    self.unify(param1, param2, location)?;
                }

                self.unify(r1, r2, location)?;
                Ok(())
            }

            // Generic unification
            (
                InferredType::Generic {
                    name: n1,
                    params: p1,
                },
                InferredType::Generic {
                    name: n2,
                    params: p2,
                },
            ) if n1 == n2 => {
                if p1.len() != p2.len() {
                    return Err(TypeError::mismatch(location, t1.clone(), t2.clone()));
                }

                for (param1, param2) in p1.iter().zip(p2.iter()) {
                    self.unify(param1, param2, location)?;
                }
                Ok(())
            }

            // Unknown can unify with anything
            (InferredType::Unknown, _) | (_, InferredType::Unknown) => Ok(()),

            // Type mismatch
            _ => Err(TypeError::mismatch(location, t1, t2)),
        }
    }

    /// Check if type variable occurs in type (for occurs check)
    fn occurs_in(&self, var: &TypeVar, ty: &InferredType) -> bool {
        let ty = self.apply_subst(ty);

        match ty {
            InferredType::Var(v) => v == *var,
            InferredType::Function {
                params,
                return_type,
            } => params.iter().any(|p| self.occurs_in(var, p)) || self.occurs_in(var, &return_type),
            InferredType::Generic { params, .. } => params.iter().any(|p| self.occurs_in(var, p)),
            _ => false,
        }
    }

    /// Infer type of an expression
    pub fn infer_expression(&mut self, expr: &Expression) -> Result<InferredType, TypeError> {
        match expr {
            Expression::IntegerLiteral(_) => Ok(InferredType::Int),
            Expression::FloatLiteral(_) => Ok(InferredType::Float),
            Expression::BooleanLiteral(_) => Ok(InferredType::Bool),
            Expression::StringLiteral(_) => Ok(InferredType::String),

            Expression::Identifier(name) => self
                .env
                .get(name)
                .cloned()
                .ok_or_else(|| TypeError::undefined("expression", name)),

            Expression::Binary {
                left,
                operator,
                right,
            } => {
                let left_type = self.infer_expression(left)?;
                let right_type = self.infer_expression(right)?;

                use BinaryOperator::*;
                match operator {
                    // Arithmetic operations
                    Add | Subtract | Multiply | Divide | Modulo | MatMul => {
                        self.unify(&left_type, &right_type, "binary operation")?;
                        self.unify(&left_type, &InferredType::Float, "binary operation")?;
                        Ok(InferredType::Float)
                    }
                    // Comparison operations
                    Equal | NotEqual | LessThan | LessEqual | GreaterThan | GreaterEqual => {
                        self.unify(&left_type, &right_type, "comparison")?;
                        Ok(InferredType::Bool)
                    }
                    // Logical operations
                    And | Or => {
                        self.unify(&left_type, &InferredType::Bool, "logical operation")?;
                        self.unify(&right_type, &InferredType::Bool, "logical operation")?;
                        Ok(InferredType::Bool)
                    }
                }
            }

            Expression::Unary { operator, operand } => {
                let operand_type = self.infer_expression(operand)?;

                use UnaryOperator::*;
                match operator {
                    Not => {
                        self.unify(&operand_type, &InferredType::Bool, "unary not")?;
                        Ok(InferredType::Bool)
                    }
                    Negate => {
                        self.unify(&operand_type, &InferredType::Float, "unary negate")?;
                        Ok(InferredType::Float)
                    }
                }
            }

            Expression::Call {
                function,
                arguments,
            } => {
                // Infer function type
                let func_type = self.infer_expression(function)?;

                // Infer argument types
                let arg_types: Result<Vec<_>, _> = arguments
                    .iter()
                    .map(|arg| self.infer_expression(arg))
                    .collect();
                let arg_types = arg_types?;

                // Create expected function type
                let return_type = self.fresh_type_var();
                let expected_func_type = InferredType::Function {
                    params: arg_types,
                    return_type: Box::new(return_type.clone()),
                };

                // Unify with actual function type
                self.unify(&func_type, &expected_func_type, "function call")?;

                Ok(self.apply_subst(&return_type))
            }

            Expression::TensorLiteral(elements) => {
                // Infer element types
                for elem in elements {
                    self.infer_expression(elem)?;
                }
                Ok(InferredType::Tensor(None))
            }

            _ => Ok(InferredType::Unknown),
        }
    }

    /// Infer types for a statement
    pub fn infer_statement(&mut self, stmt: &Statement) -> Result<(), TypeError> {
        match stmt {
            Statement::Let(let_stmt) => {
                // Infer value type
                let value_type = self.infer_expression(&let_stmt.value)?;

                // Check against annotation
                if let Some(ann) = &let_stmt.type_annotation {
                    let expected_type = InferredType::from_ast(ann);
                    self.unify(
                        &expected_type,
                        &value_type,
                        &format!("variable {}", let_stmt.name),
                    )?;
                    self.env.insert(let_stmt.name.clone(), expected_type);
                } else {
                    self.env.insert(let_stmt.name.clone(), value_type);
                }

                Ok(())
            }

            Statement::Assign(assign_stmt) => {
                // Infer value type
                let value_type = self.infer_expression(&assign_stmt.value)?;

                // Infer target type
                match &assign_stmt.target {
                    Expression::Identifier(name) => {
                        // Simple assignment: check if variable exists and types match
                        if let Some(var_type) = self.env.get(name).cloned() {
                            self.unify(
                                &var_type,
                                &value_type,
                                &format!("assignment to {}", name),
                            )?;
                        } else {
                            // Variable doesn't exist - will be caught at runtime
                            // For now, just add to environment
                            self.env.insert(name.clone(), value_type);
                        }
                    }
                    Expression::Index { object, index } => {
                        // Indexed assignment: infer types of object and index
                        self.infer_expression(object)?;
                        self.infer_expression(index)?;
                        // Type checking for indexed assignment is done at runtime
                    }
                    _ => {
                        // Invalid assignment target - just skip for now
                        // Will be caught at runtime
                    }
                }

                Ok(())
            }

            Statement::Function(func) => {
                // Build function type
                let param_types: Vec<InferredType> = func
                    .parameters
                    .iter()
                    .map(|p| InferredType::from_ast(&p.type_annotation))
                    .collect();

                let return_type = if let Some(ret_ty) = &func.return_type {
                    InferredType::from_ast(ret_ty)
                } else {
                    InferredType::Void
                };

                let func_type = InferredType::Function {
                    params: param_types.clone(),
                    return_type: Box::new(return_type.clone()),
                };

                // Add function to environment
                self.env.insert(func.name.clone(), func_type);

                // Type check body in new scope
                let saved_env = self.env.clone();

                // Add parameters to environment
                for (param, param_type) in func.parameters.iter().zip(param_types.iter()) {
                    self.env.insert(param.name.clone(), param_type.clone());
                }

                // Check body statements
                for body_stmt in &func.body {
                    self.infer_statement(body_stmt)?;
                }

                // Restore environment
                self.env = saved_env;

                Ok(())
            }

            Statement::Return(ret_stmt) => {
                self.infer_expression(&ret_stmt.value)?;
                Ok(())
            }

            Statement::Expression(expr_stmt) => {
                self.infer_expression(&expr_stmt.expression)?;
                Ok(())
            }

            Statement::If(if_stmt) => {
                // Infer condition type
                self.infer_expression(&if_stmt.condition)?;

                // Infer consequence statements
                for stmt in &if_stmt.consequence {
                    self.infer_statement(stmt)?;
                }

                // Infer alternative statements if present
                if let Some(alt) = &if_stmt.alternative {
                    for stmt in alt {
                        self.infer_statement(stmt)?;
                    }
                }

                Ok(())
            }

            Statement::While(while_stmt) => {
                // Infer condition type
                self.infer_expression(&while_stmt.condition)?;

                // Infer body statements
                for stmt in &while_stmt.body {
                    self.infer_statement(stmt)?;
                }

                Ok(())
            }

            Statement::For(for_stmt) => {
                // Infer iterable type
                self.infer_expression(&for_stmt.iterable)?;

                // TODO: Add loop variable to environment with proper type

                // Infer body statements
                for stmt in &for_stmt.body {
                    self.infer_statement(stmt)?;
                }

                Ok(())
            }

            Statement::Break => Ok(()),
            Statement::Continue => Ok(()),
        }
    }

    /// Infer types for entire program
    pub fn infer_program(&mut self, program: &Program) -> Result<(), Vec<TypeError>> {
        let mut errors = Vec::new();

        for statement in &program.statements {
            if let Err(err) = self.infer_statement(statement) {
                errors.push(err);
            }
        }

        // Solve constraints
        for constraint in self.constraints.clone() {
            if let Err(err) = self.unify(
                &constraint.expected,
                &constraint.actual,
                &constraint.location,
            ) {
                errors.push(err);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get inferred type for a variable
    pub fn get_type(&self, name: &str) -> Option<InferredType> {
        self.env.get(name).map(|ty| self.apply_subst(ty))
    }

    /// Get all type errors
    pub fn get_errors(&self) -> &[TypeError] {
        &self.errors
    }

    /// Convert type errors to rule violations
    pub fn errors_to_violations(&self) -> Vec<(String, Action)> {
        self.errors
            .iter()
            .map(|err| {
                (
                    err.location.clone(),
                    Action::Violation {
                        severity: Severity::High,
                        message: err.message.clone(),
                    },
                )
            })
            .collect()
    }

    /// Add type information to knowledge graph
    pub fn add_to_knowledge_graph(&self, graph: &mut KnowledgeGraph) {
        for (name, ty) in &self.env {
            // Find entity in graph
            let entity_id = (0..graph.num_entities()).find(|&id| {
                graph
                    .get_entity(id)
                    .map(|e| e.name == *name)
                    .unwrap_or(false)
            });

            if let Some(entity_id) = entity_id {
                // Add type information as entity attribute
                // (In a real implementation, we'd extend Entity to store metadata)

                // For now, we can create a "Type" entity and link it
                let type_entity = graph.add_entity(
                    EntityType::Variable, // Or create EntityType::Type
                    format!("{}::type::{}", name, ty),
                );

                graph.add_triple(crate::knowledge_graph::Triple::new(
                    entity_id,
                    RelationType::HasType,
                    type_entity,
                ));
            }
        }
    }
}

impl Default for TypeInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    #[test]
    fn test_literal_inference() {
        let mut inference = TypeInference::new();

        let int_expr = Expression::IntegerLiteral(42);
        assert_eq!(
            inference.infer_expression(&int_expr).unwrap(),
            InferredType::Int
        );

        let float_expr = Expression::FloatLiteral(3.14);
        assert_eq!(
            inference.infer_expression(&float_expr).unwrap(),
            InferredType::Float
        );

        let bool_expr = Expression::BooleanLiteral(true);
        assert_eq!(
            inference.infer_expression(&bool_expr).unwrap(),
            InferredType::Bool
        );

        let string_expr = Expression::StringLiteral("hello".to_string());
        assert_eq!(
            inference.infer_expression(&string_expr).unwrap(),
            InferredType::String
        );
    }

    #[test]
    fn test_variable_inference() {
        // Manually create AST: let x: int32 = 42;
        let program = Program {
            statements: vec![Statement::Let(LetStatement {
                name: "x".to_string(),
                type_annotation: Some(TypeAnnotation::Int32),
                value: Expression::IntegerLiteral(42),
            })],
        };

        let mut inference = TypeInference::new();
        inference.infer_program(&program).unwrap();

        assert_eq!(inference.get_type("x"), Some(InferredType::Int));
    }

    #[test]
    fn test_binary_op_inference() {
        let mut inference = TypeInference::new();
        inference.env.insert("x".to_string(), InferredType::Float);
        inference.env.insert("y".to_string(), InferredType::Float);

        let expr = Expression::Binary {
            left: Box::new(Expression::Identifier("x".to_string())),
            operator: BinaryOperator::Add,
            right: Box::new(Expression::Identifier("y".to_string())),
        };

        let result = inference.infer_expression(&expr).unwrap();
        assert_eq!(result, InferredType::Float);
    }

    #[test]
    fn test_comparison_inference() {
        let mut inference = TypeInference::new();
        inference.env.insert("x".to_string(), InferredType::Float);
        inference.env.insert("y".to_string(), InferredType::Float);

        let expr = Expression::Binary {
            left: Box::new(Expression::Identifier("x".to_string())),
            operator: BinaryOperator::LessThan,
            right: Box::new(Expression::Identifier("y".to_string())),
        };

        let result = inference.infer_expression(&expr).unwrap();
        assert_eq!(result, InferredType::Bool);
    }

    #[test]
    fn test_unary_op_inference() {
        let mut inference = TypeInference::new();

        let negate_expr = Expression::Unary {
            operator: UnaryOperator::Negate,
            operand: Box::new(Expression::FloatLiteral(3.14)),
        };

        let result = inference.infer_expression(&negate_expr).unwrap();
        assert_eq!(result, InferredType::Float);

        let not_expr = Expression::Unary {
            operator: UnaryOperator::Not,
            operand: Box::new(Expression::BooleanLiteral(true)),
        };

        let result = inference.infer_expression(&not_expr).unwrap();
        assert_eq!(result, InferredType::Bool);
    }

    #[test]
    fn test_type_mismatch() {
        let mut inference = TypeInference::new();

        // Try to unify incompatible types
        let result = inference.unify(&InferredType::Int, &InferredType::Bool, "test");

        assert!(result.is_err());
    }

    #[test]
    fn test_function_inference() {
        // fn add(a: float32, b: float32) -> float32 { return a + b; }
        let program = Program {
            statements: vec![Statement::Function(FunctionStatement {
                name: "add".to_string(),
                parameters: vec![
                    Parameter {
                        name: "a".to_string(),
                        type_annotation: TypeAnnotation::Float32,
                    },
                    Parameter {
                        name: "b".to_string(),
                        type_annotation: TypeAnnotation::Float32,
                    },
                ],
                return_type: Some(TypeAnnotation::Float32),
                body: vec![Statement::Return(ReturnStatement {
                    value: Expression::Binary {
                        left: Box::new(Expression::Identifier("a".to_string())),
                        operator: BinaryOperator::Add,
                        right: Box::new(Expression::Identifier("b".to_string())),
                    },
                })],
            })],
        };

        let mut inference = TypeInference::new();
        inference.infer_program(&program).unwrap();

        let func_type = inference.get_type("add").unwrap();
        match func_type {
            InferredType::Function {
                params,
                return_type,
            } => {
                assert_eq!(params.len(), 2);
                assert_eq!(params[0], InferredType::Float);
                assert_eq!(params[1], InferredType::Float);
                assert_eq!(*return_type, InferredType::Float);
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_function_call_inference() {
        let mut inference = TypeInference::new();

        // Register function: fn double(x: float32) -> float32
        let func_type = InferredType::Function {
            params: vec![InferredType::Float],
            return_type: Box::new(InferredType::Float),
        };
        inference.env.insert("double".to_string(), func_type);

        // Call: double(3.14)
        let call_expr = Expression::Call {
            function: Box::new(Expression::Identifier("double".to_string())),
            arguments: vec![Expression::FloatLiteral(3.14)],
        };

        let result = inference.infer_expression(&call_expr).unwrap();
        assert_eq!(result, InferredType::Float);
    }

    #[test]
    fn test_type_display() {
        assert_eq!(InferredType::Int.to_string(), "int");
        assert_eq!(InferredType::Float.to_string(), "float");
        assert_eq!(InferredType::Bool.to_string(), "bool");
        assert_eq!(InferredType::String.to_string(), "string");

        let func_type = InferredType::Function {
            params: vec![InferredType::Int, InferredType::Float],
            return_type: Box::new(InferredType::Bool),
        };
        assert_eq!(func_type.to_string(), "(int, float) -> bool");

        let var_type = InferredType::Var(TypeVar("t0".to_string()));
        assert_eq!(var_type.to_string(), "'t0");
    }

    #[test]
    fn test_occurs_check() {
        let mut inference = TypeInference::new();
        let var = TypeVar("t0".to_string());

        // Try to create infinite type: t0 = t0 -> t0
        let infinite_type = InferredType::Function {
            params: vec![InferredType::Var(var.clone())],
            return_type: Box::new(InferredType::Var(var.clone())),
        };

        let result = inference.unify(
            &InferredType::Var(var.clone()),
            &infinite_type,
            "occurs check test",
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_inference() {
        let mut inference = TypeInference::new();

        let tensor_expr = Expression::TensorLiteral(vec![
            Expression::FloatLiteral(1.0),
            Expression::FloatLiteral(2.0),
            Expression::FloatLiteral(3.0),
        ]);

        let result = inference.infer_expression(&tensor_expr).unwrap();
        assert_eq!(result, InferredType::Tensor(None));
    }

    #[test]
    fn test_substitution() {
        let mut inference = TypeInference::new();

        let var = TypeVar("t0".to_string());
        inference
            .substitution
            .insert(var.clone(), InferredType::Int);

        let result = inference.apply_subst(&InferredType::Var(var));
        assert_eq!(result, InferredType::Int);
    }
}

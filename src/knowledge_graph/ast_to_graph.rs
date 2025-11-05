// AST to Knowledge Graph Converter
// Converts Charl AST into a knowledge graph representation
//
// This enables:
// - Static code analysis
// - Dependency tracking
// - Call graph construction
// - Variable usage analysis
//
// Usage:
// ```rust
// let program = parser.parse(code)?;
// let graph = AstToGraphConverter::convert(&program);
// ```

use crate::ast::{
    Program, Statement, Expression, FunctionStatement,
    LetStatement, Parameter
};
use crate::knowledge_graph::{KnowledgeGraph, EntityId, EntityType, RelationType, Triple};
use std::collections::HashMap;

/// Converter that builds a knowledge graph from AST
pub struct AstToGraphConverter {
    /// The knowledge graph being built
    graph: KnowledgeGraph,

    /// Symbol table: name → EntityId
    /// Tracks variables and functions in scope
    symbols: HashMap<String, EntityId>,

    /// Current function being analyzed (for scoping)
    current_function: Option<EntityId>,
}

impl AstToGraphConverter {
    /// Create a new converter
    pub fn new() -> Self {
        AstToGraphConverter {
            graph: KnowledgeGraph::new(),
            symbols: HashMap::new(),
            current_function: None,
        }
    }

    /// Convert a program to a knowledge graph
    pub fn convert(program: &Program) -> KnowledgeGraph {
        let mut converter = AstToGraphConverter::new();
        converter.process_program(program);
        converter.graph
    }

    /// Process entire program
    fn process_program(&mut self, program: &Program) {
        for statement in &program.statements {
            self.process_statement(statement);
        }
    }

    /// Process a statement
    fn process_statement(&mut self, statement: &Statement) {
        match statement {
            Statement::Let(let_stmt) => self.process_let(let_stmt),
            Statement::Function(func_stmt) => self.process_function(func_stmt),
            Statement::Return(ret_stmt) => {
                self.process_expression(&ret_stmt.value);
            }
            Statement::Expression(expr_stmt) => {
                self.process_expression(&expr_stmt.expression);
            }
        }
    }

    /// Process let statement (variable declaration)
    fn process_let(&mut self, let_stmt: &LetStatement) {
        // Create variable entity
        let var_id = self.graph.add_entity(
            EntityType::Variable,
            let_stmt.name.clone(),
        );

        // Add to symbol table
        self.symbols.insert(let_stmt.name.clone(), var_id);

        // If inside a function, mark that function contains this variable
        if let Some(func_id) = self.current_function {
            let triple = Triple::new(func_id, RelationType::Contains, var_id);
            self.graph.add_triple(triple);
        }

        // Process the value expression
        self.process_expression(&let_stmt.value);

        // Extract dependencies from the value
        self.extract_expression_dependencies(&let_stmt.value, var_id, RelationType::Uses);
    }

    /// Process function statement
    fn process_function(&mut self, func_stmt: &FunctionStatement) {
        // Create function entity
        let func_id = self.graph.add_entity(
            EntityType::Function,
            func_stmt.name.clone(),
        );

        // Add to symbol table
        self.symbols.insert(func_stmt.name.clone(), func_id);

        // Set as current function for scoping
        let previous_function = self.current_function;
        self.current_function = Some(func_id);

        // Process parameters
        for param in &func_stmt.parameters {
            self.process_parameter(param, func_id);
        }

        // Process function body
        for statement in &func_stmt.body {
            self.process_statement(statement);
        }

        // Restore previous function
        self.current_function = previous_function;
    }

    /// Process function parameter
    fn process_parameter(&mut self, param: &Parameter, func_id: EntityId) {
        // Create parameter as a variable
        let param_id = self.graph.add_entity(
            EntityType::Variable,
            param.name.clone(),
        );

        // Add to symbol table
        self.symbols.insert(param.name.clone(), param_id);

        // Function "Takes" this parameter
        let triple = Triple::new(func_id, RelationType::Takes, param_id);
        self.graph.add_triple(triple);
    }

    /// Process expression (recursively)
    fn process_expression(&mut self, expr: &Expression) {
        match expr {
            Expression::Call { function, arguments } => {
                self.process_call(function, arguments);
            }

            Expression::Binary { left, operator: _, right } => {
                self.process_expression(left);
                self.process_expression(right);
            }

            Expression::Unary { operator: _, operand } => {
                self.process_expression(operand);
            }

            Expression::Index { object, index } => {
                self.process_expression(object);
                self.process_expression(index);
            }

            Expression::Autograd { expression } => {
                self.process_expression(expression);
            }

            Expression::ArrayLiteral(exprs) | Expression::TensorLiteral(exprs) => {
                for expr in exprs {
                    self.process_expression(expr);
                }
            }

            // Literals don't need processing
            Expression::Identifier(_) |
            Expression::IntegerLiteral(_) |
            Expression::FloatLiteral(_) |
            Expression::BooleanLiteral(_) |
            Expression::StringLiteral(_) => {}
        }
    }

    /// Process function call
    fn process_call(&mut self, function: &Expression, arguments: &[Expression]) {
        // If calling a named function, create call relationship
        if let Expression::Identifier(func_name) = function {
            if let Some(&callee_id) = self.symbols.get(func_name) {
                if let Some(caller_id) = self.current_function {
                    // Current function calls callee
                    let triple = Triple::new(caller_id, RelationType::Calls, callee_id);
                    self.graph.add_triple(triple);
                }
            }
        }

        // Process arguments
        for arg in arguments {
            self.process_expression(arg);
        }
    }

    /// Extract dependencies from an expression
    /// Creates "Uses" relationships for identifiers found in the expression
    fn extract_expression_dependencies(
        &mut self,
        expr: &Expression,
        dependent_id: EntityId,
        relation: RelationType,
    ) {
        match expr {
            Expression::Identifier(name) => {
                if let Some(&dependency_id) = self.symbols.get(name) {
                    let triple = Triple::new(dependent_id, relation.clone(), dependency_id);
                    self.graph.add_triple(triple);
                }
            }

            Expression::Binary { left, operator: _, right } => {
                self.extract_expression_dependencies(left, dependent_id, relation.clone());
                self.extract_expression_dependencies(right, dependent_id, relation);
            }

            Expression::Unary { operator: _, operand } => {
                self.extract_expression_dependencies(operand, dependent_id, relation);
            }

            Expression::Call { function, arguments } => {
                self.extract_expression_dependencies(function, dependent_id, relation.clone());
                for arg in arguments {
                    self.extract_expression_dependencies(arg, dependent_id, relation.clone());
                }
            }

            Expression::Index { object, index } => {
                self.extract_expression_dependencies(object, dependent_id, relation.clone());
                self.extract_expression_dependencies(index, dependent_id, relation);
            }

            Expression::Autograd { expression } => {
                self.extract_expression_dependencies(expression, dependent_id, relation);
            }

            Expression::ArrayLiteral(exprs) | Expression::TensorLiteral(exprs) => {
                for expr in exprs {
                    self.extract_expression_dependencies(expr, dependent_id, relation.clone());
                }
            }

            // Literals don't have dependencies
            Expression::IntegerLiteral(_) |
            Expression::FloatLiteral(_) |
            Expression::BooleanLiteral(_) |
            Expression::StringLiteral(_) => {}
        }
    }
}

impl Default for AstToGraphConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{*, BinaryOperator};

    #[test]
    fn test_empty_program() {
        let program = Program { statements: vec![] };
        let graph = AstToGraphConverter::convert(&program);

        assert_eq!(graph.num_entities(), 0);
        assert_eq!(graph.num_triples(), 0);
    }

    #[test]
    fn test_single_variable() {
        let program = Program {
            statements: vec![
                Statement::Let(LetStatement {
                    name: "x".to_string(),
                    type_annotation: Some(TypeAnnotation::Int32),
                    value: Expression::IntegerLiteral(42),
                }),
            ],
        };

        let graph = AstToGraphConverter::convert(&program);

        // Should have 1 variable entity
        assert_eq!(graph.num_entities(), 1);

        let vars = graph.find_entities_by_type(&EntityType::Variable);
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].name, "x");
    }

    #[test]
    fn test_variable_dependency() {
        // let x = 10
        // let y = x
        let program = Program {
            statements: vec![
                Statement::Let(LetStatement {
                    name: "x".to_string(),
                    type_annotation: Some(TypeAnnotation::Int32),
                    value: Expression::IntegerLiteral(10),
                }),
                Statement::Let(LetStatement {
                    name: "y".to_string(),
                    type_annotation: Some(TypeAnnotation::Int32),
                    value: Expression::Identifier("x".to_string()),
                }),
            ],
        };

        let graph = AstToGraphConverter::convert(&program);

        // Should have 2 variables
        assert_eq!(graph.num_entities(), 2);

        // y should "Uses" x
        let y_entity = graph.find_entities_by_name("y");
        assert_eq!(y_entity.len(), 1);
        let y_id = y_entity[0].id;

        let uses = graph.get_related(y_id, &RelationType::Uses);
        assert_eq!(uses.len(), 1);
    }

    #[test]
    fn test_function_declaration() {
        // fn add(a: Int32, b: Int32) { return a + b }
        let program = Program {
            statements: vec![
                Statement::Function(FunctionStatement {
                    name: "add".to_string(),
                    parameters: vec![
                        Parameter {
                            name: "a".to_string(),
                            type_annotation: TypeAnnotation::Int32,
                        },
                        Parameter {
                            name: "b".to_string(),
                            type_annotation: TypeAnnotation::Int32,
                        },
                    ],
                    return_type: Some(TypeAnnotation::Int32),
                    body: vec![
                        Statement::Return(ReturnStatement {
                            value: Expression::Binary {
                                left: Box::new(Expression::Identifier("a".to_string())),
                                operator: BinaryOperator::Add,
                                right: Box::new(Expression::Identifier("b".to_string())),
                            },
                        }),
                    ],
                }),
            ],
        };

        let graph = AstToGraphConverter::convert(&program);

        // Should have: 1 function + 2 parameters
        assert_eq!(graph.num_entities(), 3);

        let funcs = graph.find_entities_by_type(&EntityType::Function);
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0].name, "add");

        // Function should "Takes" both parameters
        let func_id = funcs[0].id;
        let params = graph.get_related(func_id, &RelationType::Takes);
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_function_call() {
        // fn foo() { }
        // fn bar() { foo() }
        let program = Program {
            statements: vec![
                Statement::Function(FunctionStatement {
                    name: "foo".to_string(),
                    parameters: vec![],
                    return_type: None,
                    body: vec![],
                }),
                Statement::Function(FunctionStatement {
                    name: "bar".to_string(),
                    parameters: vec![],
                    return_type: None,
                    body: vec![
                        Statement::Expression(ExpressionStatement {
                            expression: Expression::Call {
                                function: Box::new(Expression::Identifier("foo".to_string())),
                                arguments: vec![],
                            },
                        }),
                    ],
                }),
            ],
        };

        let graph = AstToGraphConverter::convert(&program);

        // Should have 2 functions
        let funcs = graph.find_entities_by_type(&EntityType::Function);
        assert_eq!(funcs.len(), 2);

        // bar should call foo
        let bar = graph.find_entities_by_name("bar");
        assert_eq!(bar.len(), 1);
        let bar_id = bar[0].id;

        let calls = graph.get_related(bar_id, &RelationType::Calls);
        assert_eq!(calls.len(), 1);

        // Verify it's calling foo
        let foo = graph.find_entities_by_name("foo");
        assert_eq!(foo.len(), 1);
        assert_eq!(calls[0], foo[0].id);
    }

    #[test]
    fn test_complex_expression() {
        // let result = (a + b) * c
        let program = Program {
            statements: vec![
                Statement::Let(LetStatement {
                    name: "a".to_string(),
                    type_annotation: Some(TypeAnnotation::Int32),
                    value: Expression::IntegerLiteral(1),
                }),
                Statement::Let(LetStatement {
                    name: "b".to_string(),
                    type_annotation: Some(TypeAnnotation::Int32),
                    value: Expression::IntegerLiteral(2),
                }),
                Statement::Let(LetStatement {
                    name: "c".to_string(),
                    type_annotation: Some(TypeAnnotation::Int32),
                    value: Expression::IntegerLiteral(3),
                }),
                Statement::Let(LetStatement {
                    name: "result".to_string(),
                    type_annotation: Some(TypeAnnotation::Int32),
                    value: Expression::Binary {
                        left: Box::new(Expression::Binary {
                            left: Box::new(Expression::Identifier("a".to_string())),
                            operator: BinaryOperator::Add,
                            right: Box::new(Expression::Identifier("b".to_string())),
                        }),
                        operator: BinaryOperator::Multiply,
                        right: Box::new(Expression::Identifier("c".to_string())),
                    },
                }),
            ],
        };

        let graph = AstToGraphConverter::convert(&program);

        // Should have 4 variables
        assert_eq!(graph.num_entities(), 4);

        // result should use a, b, and c
        let result = graph.find_entities_by_name("result");
        assert_eq!(result.len(), 1);
        let result_id = result[0].id;

        let uses = graph.get_related(result_id, &RelationType::Uses);
        assert_eq!(uses.len(), 3); // a, b, c
    }

    #[test]
    fn test_function_with_local_variables() {
        // fn calculate() {
        //     let x = 10
        //     let y = 20
        //     return x + y
        // }
        let program = Program {
            statements: vec![
                Statement::Function(FunctionStatement {
                    name: "calculate".to_string(),
                    parameters: vec![],
                    return_type: Some(TypeAnnotation::Int32),
                    body: vec![
                        Statement::Let(LetStatement {
                            name: "x".to_string(),
                            type_annotation: Some(TypeAnnotation::Int32),
                            value: Expression::IntegerLiteral(10),
                        }),
                        Statement::Let(LetStatement {
                            name: "y".to_string(),
                            type_annotation: Some(TypeAnnotation::Int32),
                            value: Expression::IntegerLiteral(20),
                        }),
                        Statement::Return(ReturnStatement {
                            value: Expression::Binary {
                                left: Box::new(Expression::Identifier("x".to_string())),
                                operator: BinaryOperator::Add,
                                right: Box::new(Expression::Identifier("y".to_string())),
                            },
                        }),
                    ],
                }),
            ],
        };

        let graph = AstToGraphConverter::convert(&program);

        // Should have: 1 function + 2 variables
        assert_eq!(graph.num_entities(), 3);

        // Function should contain both variables
        let func = graph.find_entities_by_name("calculate");
        assert_eq!(func.len(), 1);
        let func_id = func[0].id;

        let contains = graph.get_related(func_id, &RelationType::Contains);
        assert_eq!(contains.len(), 2); // x and y
    }
}

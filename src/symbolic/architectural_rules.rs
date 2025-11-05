// Architectural Rules
// Predefined rules for common software architecture patterns
//
// This module provides ready-to-use rules for:
// - Clean Architecture
// - SOLID Principles
// - Design Patterns
// - Code Smells
//
// Usage:
// ```rust
// use charl::symbolic::ArchitecturalRules;
//
// let rules = ArchitecturalRules::clean_architecture();
// let violations = rules.execute(&knowledge_graph);
// ```

use super::rule_engine::{Action, Condition, Rule, RuleEngine, Severity};
use crate::knowledge_graph::{EntityType, RelationType};

/// Builder for architectural rule sets
pub struct ArchitecturalRules;

impl ArchitecturalRules {
    /// Rules for Clean Architecture
    ///
    /// Enforces layering: Controller → Service → Repository
    pub fn clean_architecture() -> RuleEngine {
        let mut engine = RuleEngine::new();

        // Rule 1: Controllers should not depend on Repositories
        engine.add_rule(
            Rule::new("no_controller_to_repository")
                .description("Controllers should not directly depend on Repositories")
                .condition(Condition::HasRelation {
                    subject_pattern: "*Controller".to_string(),
                    relation: RelationType::DependsOn,
                    object_pattern: "*Repository".to_string(),
                })
                .action(Action::Violation {
                    severity: Severity::High,
                    message: "Clean Architecture violation: Controller directly depends on Repository. Use a Service layer.".to_string(),
                })
        );

        // Rule 2: Repositories should not depend on Controllers
        engine.add_rule(
            Rule::new("no_repository_to_controller")
                .description("Repositories should not depend on Controllers")
                .condition(Condition::HasRelation {
                    subject_pattern: "*Repository".to_string(),
                    relation: RelationType::DependsOn,
                    object_pattern: "*Controller".to_string(),
                })
                .action(Action::Violation {
                    severity: Severity::Critical,
                    message: "Clean Architecture violation: Repository depends on Controller (wrong direction).".to_string(),
                })
        );

        // Rule 3: Services should not depend on Controllers
        engine.add_rule(
            Rule::new("no_service_to_controller")
                .description("Services should not depend on Controllers")
                .condition(Condition::HasRelation {
                    subject_pattern: "*Service".to_string(),
                    relation: RelationType::DependsOn,
                    object_pattern: "*Controller".to_string(),
                })
                .action(Action::Violation {
                    severity: Severity::Critical,
                    message: "Clean Architecture violation: Service depends on Controller (wrong direction).".to_string(),
                })
        );

        engine
    }

    /// Rules for detecting code smells
    pub fn code_smells() -> RuleEngine {
        let mut engine = RuleEngine::new();

        // Rule: Detect circular dependencies
        engine.add_rule(
            Rule::new("no_circular_dependencies")
                .description("Circular dependencies detected")
                .condition(Condition::CircularDependency {
                    relation: RelationType::DependsOn,
                    max_depth: 10,
                })
                .action(Action::Violation {
                    severity: Severity::High,
                    message: "Circular dependency detected".to_string(),
                }),
        );

        // Rule: God classes (classes with too many dependencies)
        // Note: This is simplified - in real implementation would count dependencies
        engine.add_rule(
            Rule::new("warn_god_class")
                .description("Potential god class pattern")
                .condition(Condition::HasType {
                    entity_pattern: "*Manager".to_string(),
                    entity_type: EntityType::Class,
                })
                .action(Action::Warning {
                    message: "Classes ending in 'Manager' often become god classes".to_string(),
                }),
        );

        engine
    }

    /// Rules for naming conventions
    pub fn naming_conventions() -> RuleEngine {
        let mut engine = RuleEngine::new();

        // Rule: Controllers should end with "Controller"
        engine.add_rule(
            Rule::new("controller_naming")
                .description("Controller classes should end with 'Controller'")
                .condition(Condition::And(
                    Box::new(Condition::HasType {
                        entity_pattern: "*".to_string(),
                        entity_type: EntityType::Class,
                    }),
                    Box::new(Condition::Not(
                        Box::new(Condition::Or(
                            Box::new(Condition::NameMatches {
                                entity_id: None,
                                pattern: "*Controller".to_string(),
                            }),
                            Box::new(Condition::NameMatches {
                                entity_id: None,
                                pattern: "*Service".to_string(),
                            }),
                        ))
                    ))
                ))
                .action(Action::Info {
                    message: "Consider following naming conventions: Controllers end with 'Controller', Services with 'Service'".to_string(),
                })
        );

        engine
    }

    /// Rules for SOLID principles
    pub fn solid_principles() -> RuleEngine {
        let mut engine = RuleEngine::new();

        // Single Responsibility: Warn about classes with many methods
        // (Simplified - actual implementation would count methods)

        // Dependency Inversion: Depend on abstractions (interfaces) not concretions
        engine.add_rule(
            Rule::new("depend_on_interfaces")
                .description("Prefer depending on interfaces/traits over concrete classes")
                .condition(Condition::And(
                    Box::new(Condition::HasRelation {
                        subject_pattern: "*".to_string(),
                        relation: RelationType::DependsOn,
                        object_pattern: "*Impl".to_string(), // Concrete implementations often end with Impl
                    }),
                    Box::new(Condition::Not(
                        Box::new(Condition::NameMatches {
                            entity_id: None,
                            pattern: "*Test*".to_string(),
                        })
                    ))
                ))
                .action(Action::Warning {
                    message: "Consider depending on an interface/trait instead of concrete implementation".to_string(),
                })
        );

        engine
    }

    /// Combine multiple rule sets
    pub fn all_rules() -> RuleEngine {
        let mut engine = RuleEngine::new();

        // Add all rule sets
        engine.add_rules(Self::clean_architecture().rules());
        engine.add_rules(Self::code_smells().rules());
        engine.add_rules(Self::naming_conventions().rules());
        engine.add_rules(Self::solid_principles().rules());

        engine
    }
}

// Note: RuleEngine extension is in rule_engine.rs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_graph::CodeGraphBuilder;

    #[test]
    fn test_clean_architecture_violation() {
        let rules = ArchitecturalRules::clean_architecture();

        // Create a violation: Controller depends on Repository
        let mut builder = CodeGraphBuilder::new();
        let controller = builder.add_class("UserController");
        let repo = builder.add_class("UserRepository");
        builder.add_dependency(controller, repo);

        let graph = builder.build();

        let violations = rules.get_violations(&graph);
        assert!(!violations.is_empty());
        assert_eq!(violations[0].rule_name, "no_controller_to_repository");
    }

    #[test]
    fn test_clean_architecture_valid() {
        let rules = ArchitecturalRules::clean_architecture();

        // Valid architecture: Controller → Service → Repository
        let mut builder = CodeGraphBuilder::new();
        let controller = builder.add_class("UserController");
        let service = builder.add_class("UserService");
        let repo = builder.add_class("UserRepository");

        builder.add_dependency(controller, service);
        builder.add_dependency(service, repo);

        let graph = builder.build();

        let violations = rules.get_violations(&graph);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_code_smells() {
        let rules = ArchitecturalRules::code_smells();

        // Create a god class indicator
        let mut builder = CodeGraphBuilder::new();
        builder.add_class("UserManager");

        let graph = builder.build();

        let warnings = rules.get_warnings(&graph);
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_all_rules() {
        let rules = ArchitecturalRules::all_rules();

        // Should have rules from all categories
        assert!(rules.num_rules() > 5);
    }
}

// Symbolic Reasoning Module
// Provides logic-based reasoning capabilities
//
// This module implements:
// - Rule-based reasoning (if-then rules)
// - Pattern matching
// - Logical inference
// - Constraint verification
//
// Usage:
// ```rust
// use charl::symbolic::{Rule, RuleEngine, Condition, Action, Severity};
//
// // Create a rule
// let rule = Rule::new("clean_architecture")
//     .condition(Condition::HasRelation {
//         subject_pattern: "*Controller".to_string(),
//         relation: RelationType::DependsOn,
//         object_pattern: "*Repository".to_string(),
//     })
//     .action(Action::Violation {
//         severity: Severity::High,
//         message: "Controllers should not depend on Repositories".to_string(),
//     });
//
// // Create engine and add rule
// let mut engine = RuleEngine::new();
// engine.add_rule(rule);
//
// // Execute against knowledge graph
// let violations = engine.get_violations(&graph);
// ```

pub mod rule_engine;
pub mod architectural_rules;
pub mod type_inference;
pub mod fol;
pub mod differentiable_logic;
pub mod concept_learning;

// Re-export main types
pub use rule_engine::{
    Rule, RuleEngine, RuleMatch,
    Condition, Action, Severity,
};
pub use architectural_rules::ArchitecturalRules;
pub use type_inference::{
    TypeInference, InferredType, TypeVar,
    TypeConstraint, TypeError,
};
pub use fol::{
    Term, Formula, Clause, FOLSolver,
    Substitution, UnificationResult, unify,
};
pub use differentiable_logic::{
    FuzzyValue, FuzzyLogic, TNorm, TConorm,
    DifferentiableGate, ProbabilisticTruth, soft_unify,
};
pub use concept_learning::{
    Concept, ConceptGraph, ConceptLearner, ConceptRelation,
};

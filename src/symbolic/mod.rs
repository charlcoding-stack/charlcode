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

pub mod architectural_rules;
pub mod concept_learning;
pub mod differentiable_logic;
pub mod fol;
pub mod rule_engine;
pub mod type_inference;

// Re-export main types
pub use architectural_rules::ArchitecturalRules;
pub use concept_learning::{Concept, ConceptGraph, ConceptLearner, ConceptRelation};
pub use differentiable_logic::{
    soft_unify, DifferentiableGate, FuzzyLogic, FuzzyValue, ProbabilisticTruth, TConorm, TNorm,
};
pub use fol::{unify, Clause, FOLSolver, Formula, Substitution, Term, UnificationResult};
pub use rule_engine::{Action, Condition, Rule, RuleEngine, RuleMatch, Severity};
pub use type_inference::{InferredType, TypeConstraint, TypeError, TypeInference, TypeVar};

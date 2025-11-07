// Symbolic Rule Engine
// Executes logical rules over knowledge graphs
//
// Rules follow the pattern: IF <condition> THEN <action>
//
// Example:
// ```
// Rule: "Clean Architecture Violation"
// IF Controller.dependsOn(Repository)
// THEN Violation("Controllers should not depend on Repositories directly")
// ```
//
// Usage:
// ```rust
// let rule = Rule::new("no_circular_deps")
//     .condition(Condition::CircularDependency)
//     .action(Action::Violation("Circular dependency detected"));
//
// let engine = RuleEngine::new();
// engine.add_rule(rule);
// let violations = engine.execute(&knowledge_graph);
// ```

use crate::knowledge_graph::{EntityId, EntityType, KnowledgeGraph, RelationType};
use std::collections::HashSet;

/// Condition that can be evaluated against a knowledge graph
#[derive(Debug, Clone)]
pub enum Condition {
    /// Check if entity has specific type
    HasType {
        entity_pattern: String, // e.g., "*Controller"
        entity_type: EntityType,
    },

    /// Check if relationship exists
    HasRelation {
        subject_pattern: String,
        relation: RelationType,
        object_pattern: String,
    },

    /// Check for circular dependencies
    CircularDependency {
        relation: RelationType,
        max_depth: usize,
    },

    /// Check if entity name matches pattern
    NameMatches {
        entity_id: Option<EntityId>,
        pattern: String, // Supports * wildcard
    },

    /// Logical AND
    And(Box<Condition>, Box<Condition>),

    /// Logical OR
    Or(Box<Condition>, Box<Condition>),

    /// Logical NOT
    Not(Box<Condition>),

    /// Always true
    Always,
}

/// Action to take when rule fires
#[derive(Debug, Clone)]
pub enum Action {
    /// Report a violation
    Violation { severity: Severity, message: String },

    /// Report a warning
    Warning { message: String },

    /// Report an info message
    Info { message: String },

    /// Add a new fact to the knowledge graph (inference)
    AddFact {
        subject: EntityId,
        relation: RelationType,
        object: EntityId,
    },
}

/// Severity of a violation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
}

/// Result of evaluating a rule
#[derive(Debug, Clone)]
pub struct RuleMatch {
    pub rule_name: String,
    pub matched_entities: Vec<EntityId>,
    pub action: Action,
}

/// A logical rule
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub description: String,
    pub condition: Condition,
    pub action: Action,
    pub enabled: bool,
}

impl Rule {
    /// Create a new rule
    pub fn new(name: impl Into<String>) -> Self {
        Rule {
            name: name.into(),
            description: String::new(),
            condition: Condition::Always,
            action: Action::Info {
                message: "Rule matched".to_string(),
            },
            enabled: true,
        }
    }

    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set condition
    pub fn condition(mut self, condition: Condition) -> Self {
        self.condition = condition;
        self
    }

    /// Set action
    pub fn action(mut self, action: Action) -> Self {
        self.action = action;
        self
    }

    /// Enable/disable rule
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Evaluate this rule against a knowledge graph
    pub fn evaluate(&self, graph: &KnowledgeGraph) -> Vec<RuleMatch> {
        if !self.enabled {
            return Vec::new();
        }

        let matched_entities = self.evaluate_condition(&self.condition, graph);

        if matched_entities.is_empty() {
            Vec::new()
        } else {
            vec![RuleMatch {
                rule_name: self.name.clone(),
                matched_entities,
                action: self.action.clone(),
            }]
        }
    }

    /// Evaluate a condition and return matching entities
    fn evaluate_condition(&self, condition: &Condition, graph: &KnowledgeGraph) -> Vec<EntityId> {
        match condition {
            Condition::Always => {
                // Return all entities
                (0..graph.num_entities()).collect()
            }

            Condition::HasType {
                entity_pattern,
                entity_type,
            } => graph
                .find_entities_by_type(entity_type)
                .into_iter()
                .filter(|e| pattern_matches(&e.name, entity_pattern))
                .map(|e| e.id)
                .collect(),

            Condition::HasRelation {
                subject_pattern,
                relation,
                object_pattern,
            } => {
                let mut matches = Vec::new();

                // Find all triples with this relation
                let triples = graph.query(None, Some(relation), None);

                for triple in triples {
                    if let (Some(subj), Some(obj)) = (
                        graph.get_entity(triple.subject),
                        graph.get_entity(triple.object),
                    ) {
                        if pattern_matches(&subj.name, subject_pattern)
                            && pattern_matches(&obj.name, object_pattern)
                        {
                            matches.push(triple.subject);
                        }
                    }
                }

                matches
            }

            Condition::CircularDependency {
                relation: _,
                max_depth,
            } => {
                let mut circular_entities = Vec::new();

                for entity_id in 0..graph.num_entities() {
                    // Check if there's a path from entity back to itself
                    let paths = graph.find_paths(entity_id, entity_id, *max_depth);
                    if !paths.is_empty() {
                        // Filter paths that use the specified relation
                        // For now, just check if any path exists
                        circular_entities.push(entity_id);
                    }
                }

                circular_entities
            }

            Condition::NameMatches { entity_id, pattern } => {
                if let Some(id) = entity_id {
                    if let Some(entity) = graph.get_entity(*id) {
                        if pattern_matches(&entity.name, pattern) {
                            return vec![*id];
                        }
                    }
                    Vec::new()
                } else {
                    // Check all entities
                    (0..graph.num_entities())
                        .filter_map(|id| graph.get_entity(id))
                        .filter(|e| pattern_matches(&e.name, pattern))
                        .map(|e| e.id)
                        .collect()
                }
            }

            Condition::And(left, right) => {
                let left_matches: HashSet<_> =
                    self.evaluate_condition(left, graph).into_iter().collect();
                let right_matches: HashSet<_> =
                    self.evaluate_condition(right, graph).into_iter().collect();

                left_matches.intersection(&right_matches).copied().collect()
            }

            Condition::Or(left, right) => {
                let mut matches: HashSet<_> =
                    self.evaluate_condition(left, graph).into_iter().collect();
                matches.extend(self.evaluate_condition(right, graph));

                matches.into_iter().collect()
            }

            Condition::Not(inner) => {
                let inner_matches: HashSet<_> =
                    self.evaluate_condition(inner, graph).into_iter().collect();
                let all_entities: HashSet<_> = (0..graph.num_entities()).collect();

                all_entities.difference(&inner_matches).copied().collect()
            }
        }
    }
}

/// Pattern matching with * wildcard
fn pattern_matches(text: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if pattern.starts_with('*') && pattern.ends_with('*') {
        // *substring*
        let substring = &pattern[1..pattern.len() - 1];
        text.contains(substring)
    } else if let Some(suffix) = pattern.strip_prefix('*') {
        // *suffix
        text.ends_with(suffix)
    } else if let Some(prefix) = pattern.strip_suffix('*') {
        // prefix*
        text.starts_with(prefix)
    } else {
        // Exact match
        text == pattern
    }
}

#[derive(Debug, Clone)]
/// Rule Engine - executes multiple rules
pub struct RuleEngine {
    rules: Vec<Rule>,
}

impl RuleEngine {
    /// Create a new rule engine
    pub fn new() -> Self {
        RuleEngine { rules: Vec::new() }
    }

    /// Add a rule
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Add multiple rules
    pub fn add_rules(&mut self, rules: Vec<Rule>) {
        self.rules.extend(rules);
    }

    /// Execute all rules against a knowledge graph
    pub fn execute(&self, graph: &KnowledgeGraph) -> Vec<RuleMatch> {
        let mut all_matches = Vec::new();

        for rule in &self.rules {
            let matches = rule.evaluate(graph);
            all_matches.extend(matches);
        }

        all_matches
    }

    /// Get violations only
    pub fn get_violations(&self, graph: &KnowledgeGraph) -> Vec<RuleMatch> {
        self.execute(graph)
            .into_iter()
            .filter(|m| matches!(m.action, Action::Violation { .. }))
            .collect()
    }

    /// Get warnings only
    pub fn get_warnings(&self, graph: &KnowledgeGraph) -> Vec<RuleMatch> {
        self.execute(graph)
            .into_iter()
            .filter(|m| matches!(m.action, Action::Warning { .. }))
            .collect()
    }

    /// Number of rules
    pub fn num_rules(&self) -> usize {
        self.rules.len()
    }

    /// Clear all rules
    pub fn clear(&mut self) {
        self.rules.clear();
    }

    /// Get a copy of all rules (for combining rule engines)
    pub fn rules(&self) -> Vec<Rule> {
        self.rules.clone()
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_graph::CodeGraphBuilder;

    #[test]
    fn test_pattern_matches() {
        assert!(pattern_matches("UserController", "*Controller"));
        assert!(pattern_matches("UserController", "User*"));
        assert!(pattern_matches("UserController", "*Control*"));
        assert!(pattern_matches("UserController", "UserController"));
        assert!(pattern_matches("anything", "*"));

        assert!(!pattern_matches("UserService", "*Controller"));
        assert!(!pattern_matches("UserController", "Admin*"));
    }

    #[test]
    fn test_rule_creation() {
        let rule = Rule::new("test_rule")
            .description("Test description")
            .condition(Condition::Always)
            .action(Action::Info {
                message: "Test".to_string(),
            });

        assert_eq!(rule.name, "test_rule");
        assert_eq!(rule.description, "Test description");
        assert!(rule.enabled);
    }

    #[test]
    fn test_condition_has_type() {
        let mut builder = CodeGraphBuilder::new();
        builder.add_function("test_func");
        builder.add_class("TestClass");
        let graph = builder.build();

        let rule = Rule::new("find_functions").condition(Condition::HasType {
            entity_pattern: "*".to_string(),
            entity_type: EntityType::Function,
        });

        let matches = rule.evaluate(&graph);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched_entities.len(), 1);
    }

    #[test]
    fn test_condition_has_relation() {
        let mut builder = CodeGraphBuilder::new();
        let a = builder.add_function("funcA");
        let b = builder.add_function("funcB");
        builder.add_call(a, b);
        let graph = builder.build();

        let rule = Rule::new("find_calls").condition(Condition::HasRelation {
            subject_pattern: "*".to_string(),
            relation: RelationType::Calls,
            object_pattern: "*".to_string(),
        });

        let matches = rule.evaluate(&graph);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_condition_name_matches() {
        let mut builder = CodeGraphBuilder::new();
        builder.add_class("UserController");
        builder.add_class("PostController");
        builder.add_class("UserService");
        let graph = builder.build();

        let rule = Rule::new("find_controllers").condition(Condition::NameMatches {
            entity_id: None,
            pattern: "*Controller".to_string(),
        });

        let matches = rule.evaluate(&graph);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched_entities.len(), 2); // UserController and PostController
    }

    #[test]
    fn test_condition_and() {
        let mut builder = CodeGraphBuilder::new();
        builder.add_class("UserController");
        builder.add_function("helper");
        let graph = builder.build();

        let rule = Rule::new("find_class_controllers").condition(Condition::And(
            Box::new(Condition::HasType {
                entity_pattern: "*".to_string(),
                entity_type: EntityType::Class,
            }),
            Box::new(Condition::NameMatches {
                entity_id: None,
                pattern: "*Controller".to_string(),
            }),
        ));

        let matches = rule.evaluate(&graph);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched_entities.len(), 1);
    }

    #[test]
    fn test_condition_or() {
        let mut builder = CodeGraphBuilder::new();
        builder.add_class("User");
        builder.add_function("login");
        let graph = builder.build();

        let rule = Rule::new("find_class_or_function").condition(Condition::Or(
            Box::new(Condition::HasType {
                entity_pattern: "*".to_string(),
                entity_type: EntityType::Class,
            }),
            Box::new(Condition::HasType {
                entity_pattern: "*".to_string(),
                entity_type: EntityType::Function,
            }),
        ));

        let matches = rule.evaluate(&graph);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched_entities.len(), 2); // Both User and login
    }

    #[test]
    fn test_rule_engine() {
        let mut engine = RuleEngine::new();

        let rule1 = Rule::new("rule1")
            .condition(Condition::Always)
            .action(Action::Info {
                message: "Test".to_string(),
            });

        let rule2 = Rule::new("rule2")
            .condition(Condition::Always)
            .action(Action::Warning {
                message: "Warning".to_string(),
            });

        engine.add_rule(rule1);
        engine.add_rule(rule2);

        assert_eq!(engine.num_rules(), 2);

        // Create a graph with some entities
        let mut builder = CodeGraphBuilder::new();
        builder.add_function("test");
        let graph = builder.build();

        let matches = engine.execute(&graph);

        // Both rules should match (Condition::Always)
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_violation_detection() {
        let mut builder = CodeGraphBuilder::new();
        let controller = builder.add_class("UserController");
        let repo = builder.add_class("UserRepository");
        builder.add_dependency(controller, repo);
        let graph = builder.build();

        let mut engine = RuleEngine::new();
        engine.add_rule(
            Rule::new("no_controller_to_repo")
                .description("Controllers should not depend on Repositories directly")
                .condition(Condition::HasRelation {
                    subject_pattern: "*Controller".to_string(),
                    relation: RelationType::DependsOn,
                    object_pattern: "*Repository".to_string(),
                })
                .action(Action::Violation {
                    severity: Severity::High,
                    message: "Clean architecture violation detected".to_string(),
                }),
        );

        let violations = engine.get_violations(&graph);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].rule_name, "no_controller_to_repo");
    }

    #[test]
    fn test_disabled_rule() {
        let rule = Rule::new("disabled")
            .condition(Condition::Always)
            .enabled(false);

        let graph = CodeGraphBuilder::new().build();
        let matches = rule.evaluate(&graph);

        assert!(matches.is_empty());
    }
}

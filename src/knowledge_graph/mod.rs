// Knowledge Graph Module
// Provides knowledge representation and reasoning capabilities
//
// This module implements:
// - Triple store (Subject-Predicate-Object)
// - Knowledge graph data structure
// - Query engine for pattern matching
// - Graph neural networks (future)
//
// Usage:
// ```rust
// use charl::knowledge_graph::{KnowledgeGraph, EntityType, RelationType};
//
// let mut graph = KnowledgeGraph::new();
//
// // Add entities
// let user = graph.add_entity(EntityType::Class, "User".to_string());
// let entity = graph.add_entity(EntityType::Class, "Entity".to_string());
//
// // Add relationship
// graph.add_triple(Triple::new(user, RelationType::Inherits, entity));
//
// // Query
// let results = graph.query(Some(user), None, None);
// ```

pub mod ast_to_graph;
pub mod gnn;
pub mod graph;
pub mod triple;

// Re-export main types for convenience
pub use ast_to_graph::AstToGraphConverter;
pub use gnn::{GraphAttentionLayer, GraphNeuralNetwork, NodeEmbedding};
pub use graph::{GraphStats, KnowledgeGraph};
pub use triple::{Entity, EntityId, EntityType, RelationType, Triple};

/// Knowledge Graph configuration
#[derive(Debug, Clone)]
pub struct KGConfig {
    /// Enable fuzzy logic (confidence scores)
    pub enable_fuzzy_logic: bool,

    /// Maximum path depth for graph traversal
    pub max_path_depth: usize,

    /// Enable transitive closure computation
    pub enable_transitive_closure: bool,
}

impl KGConfig {
    /// Create default configuration
    pub fn default() -> Self {
        KGConfig {
            enable_fuzzy_logic: false,
            max_path_depth: 10,
            enable_transitive_closure: false,
        }
    }

    /// Configuration for code analysis
    pub fn for_code_analysis() -> Self {
        KGConfig {
            enable_fuzzy_logic: false,       // Code relationships are binary
            max_path_depth: 20,              // Allow deeper traversal for dependencies
            enable_transitive_closure: true, // Important for transitive deps
        }
    }

    /// Configuration for concept learning
    pub fn for_concept_learning() -> Self {
        KGConfig {
            enable_fuzzy_logic: true, // Concepts have confidence scores
            max_path_depth: 15,
            enable_transitive_closure: true,
        }
    }
}

/// Helper to build a simple code knowledge graph
pub struct CodeGraphBuilder {
    graph: KnowledgeGraph,
}

impl CodeGraphBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        CodeGraphBuilder {
            graph: KnowledgeGraph::new(),
        }
    }

    /// Add a class
    pub fn add_class(&mut self, name: &str) -> EntityId {
        self.graph.add_entity(EntityType::Class, name.to_string())
    }

    /// Add a function
    pub fn add_function(&mut self, name: &str) -> EntityId {
        self.graph
            .add_entity(EntityType::Function, name.to_string())
    }

    /// Add a module
    pub fn add_module(&mut self, name: &str) -> EntityId {
        self.graph.add_entity(EntityType::Module, name.to_string())
    }

    /// Add inheritance relationship
    pub fn add_inheritance(&mut self, child: EntityId, parent: EntityId) {
        let triple = Triple::new(child, RelationType::Inherits, parent);
        self.graph.add_triple(triple);
    }

    /// Add call relationship
    pub fn add_call(&mut self, caller: EntityId, callee: EntityId) {
        let triple = Triple::new(caller, RelationType::Calls, callee);
        self.graph.add_triple(triple);
    }

    /// Add dependency relationship
    pub fn add_dependency(&mut self, from: EntityId, to: EntityId) {
        let triple = Triple::new(from, RelationType::DependsOn, to);
        self.graph.add_triple(triple);
    }

    /// Build and return the graph
    pub fn build(self) -> KnowledgeGraph {
        self.graph
    }
}

impl Default for CodeGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kg_config_default() {
        let config = KGConfig::default();
        assert_eq!(config.enable_fuzzy_logic, false);
        assert_eq!(config.max_path_depth, 10);
    }

    #[test]
    fn test_kg_config_for_code() {
        let config = KGConfig::for_code_analysis();
        assert_eq!(config.enable_fuzzy_logic, false);
        assert_eq!(config.max_path_depth, 20);
        assert_eq!(config.enable_transitive_closure, true);
    }

    #[test]
    fn test_code_graph_builder() {
        let mut builder = CodeGraphBuilder::new();

        let user = builder.add_class("User");
        let entity = builder.add_class("Entity");
        let login = builder.add_function("login");

        builder.add_inheritance(user, entity);
        builder.add_call(login, user);

        let graph = builder.build();

        assert_eq!(graph.num_entities(), 3);
        assert_eq!(graph.num_triples(), 2);

        // Verify inheritance
        let parents = graph.get_related(user, &RelationType::Inherits);
        assert_eq!(parents.len(), 1);
        assert_eq!(parents[0], entity);

        // Verify calls
        let calls = graph.get_related(login, &RelationType::Calls);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0], user);
    }

    #[test]
    fn test_builder_pattern() {
        let graph = CodeGraphBuilder::new().build();

        assert_eq!(graph.num_entities(), 0);
        assert_eq!(graph.num_triples(), 0);
    }
}

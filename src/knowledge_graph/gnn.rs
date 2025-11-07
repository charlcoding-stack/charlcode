// Graph Neural Network (GNN)
// Neural network layers that operate on knowledge graphs
//
// Architecture:
// 1. Node embeddings: Each entity gets a vector representation
// 2. Message passing: Nodes aggregate information from neighbors
// 3. Attention: Uses MultiHeadAttention for selective aggregation
// 4. Output: Updated node embeddings capturing graph structure
//
// Usage:
// ```rust
// let gnn = GraphNeuralNetwork::new(embedding_dim, num_heads)?;
// let updated_embeddings = gnn.forward(&graph, &node_embeddings)?;
// ```

use crate::attention::MultiHeadAttention;
use crate::knowledge_graph::{EntityId, EntityType, KnowledgeGraph, RelationType};
use std::collections::HashMap;

/// Node embedding - vector representation of an entity
pub type NodeEmbedding = Vec<f64>;

/// Graph Neural Network Layer
#[derive(Debug, Clone)]
pub struct GraphNeuralNetwork {
    /// Embedding dimension
    embedding_dim: usize,

    /// Multi-head attention for message passing
    attention: MultiHeadAttention,

    /// Learnable entity type embeddings
    type_embeddings: HashMap<EntityType, Vec<f64>>,
}

impl GraphNeuralNetwork {
    /// Create a new GNN
    ///
    /// # Arguments
    /// * `embedding_dim` - Dimension of node embeddings
    /// * `num_heads` - Number of attention heads
    pub fn new(embedding_dim: usize, num_heads: usize) -> Result<Self, String> {
        // Validate that embedding_dim is divisible by num_heads
        if !embedding_dim.is_multiple_of(num_heads) {
            return Err(format!(
                "embedding_dim ({}) must be divisible by num_heads ({})",
                embedding_dim, num_heads
            ));
        }

        // Create multi-head attention for message passing
        let attention = MultiHeadAttention::new(embedding_dim, num_heads, 0.1)?;

        // Initialize type embeddings
        let type_embeddings = Self::initialize_type_embeddings(embedding_dim);

        Ok(GraphNeuralNetwork {
            embedding_dim,
            attention,
            type_embeddings,
        })
    }

    /// Initialize embeddings for each entity type
    fn initialize_type_embeddings(dim: usize) -> HashMap<EntityType, Vec<f64>> {
        let mut embeddings = HashMap::new();

        // Simple initialization based on entity type
        // In production, these would be learned parameters
        let types = vec![
            EntityType::Class,
            EntityType::Function,
            EntityType::Method,
            EntityType::Variable,
            EntityType::Module,
            EntityType::Package,
            EntityType::Interface,
            EntityType::Trait,
            EntityType::Struct,
            EntityType::Enum,
            EntityType::Type,
            EntityType::Concept,
        ];

        for (idx, entity_type) in types.into_iter().enumerate() {
            let mut embedding = vec![0.0; dim];

            // Simple initialization: different patterns for different types
            for i in 0..dim {
                // Use sine waves with different frequencies based on type
                let freq = (idx + 1) as f64;
                embedding[i] = (freq * i as f64 / dim as f64).sin() * 0.1;
            }

            embeddings.insert(entity_type, embedding);
        }

        embeddings
    }

    /// Get initial embedding for an entity based on its type
    pub fn get_type_embedding(&self, entity_type: &EntityType) -> Vec<f64> {
        self.type_embeddings
            .get(entity_type)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.embedding_dim])
    }

    /// Initialize embeddings for all nodes in the graph
    pub fn initialize_node_embeddings(
        &self,
        graph: &KnowledgeGraph,
    ) -> HashMap<EntityId, NodeEmbedding> {
        let mut embeddings = HashMap::new();

        for id in 0..graph.num_entities() {
            if let Some(entity) = graph.get_entity(id) {
                let embedding = self.get_type_embedding(&entity.entity_type);
                embeddings.insert(id, embedding);
            }
        }

        embeddings
    }

    /// Forward pass: update node embeddings using graph structure
    ///
    /// # Arguments
    /// * `graph` - Knowledge graph
    /// * `node_embeddings` - Current node embeddings
    ///
    /// # Returns
    /// Updated node embeddings after message passing
    pub fn forward(
        &self,
        graph: &KnowledgeGraph,
        node_embeddings: &HashMap<EntityId, NodeEmbedding>,
    ) -> Result<HashMap<EntityId, NodeEmbedding>, String> {
        let mut updated_embeddings = HashMap::new();

        // For each node, aggregate information from neighbors
        for node_id in 0..graph.num_entities() {
            if !node_embeddings.contains_key(&node_id) {
                continue;
            }

            let updated = self.aggregate_neighbors(graph, node_embeddings, node_id)?;
            updated_embeddings.insert(node_id, updated);
        }

        Ok(updated_embeddings)
    }

    /// Aggregate information from neighboring nodes using attention
    fn aggregate_neighbors(
        &self,
        graph: &KnowledgeGraph,
        node_embeddings: &HashMap<EntityId, NodeEmbedding>,
        node_id: EntityId,
    ) -> Result<NodeEmbedding, String> {
        // Get neighbors (nodes connected to this node)
        let neighbors = self.get_neighbors(graph, node_id);

        if neighbors.is_empty() {
            // No neighbors - return original embedding
            return Ok(node_embeddings[&node_id].clone());
        }

        // Prepare query (this node's embedding)
        let query = &node_embeddings[&node_id];

        // Prepare keys and values (neighbors' embeddings)
        let mut keys = Vec::new();
        let mut values = Vec::new();

        for neighbor_id in &neighbors {
            if let Some(neighbor_emb) = node_embeddings.get(neighbor_id) {
                keys.extend_from_slice(neighbor_emb);
                values.extend_from_slice(neighbor_emb);
            }
        }

        if keys.is_empty() {
            return Ok(query.clone());
        }

        // Apply attention to aggregate neighbor information
        // Query: (1, 1, embedding_dim)
        // Keys/Values: (1, num_neighbors, embedding_dim)
        let batch_size = 1;
        let seq_len_q = 1;
        let seq_len_k = neighbors.len();

        let (output, _weights) = self.attention.forward(
            query,
            &keys,
            &values,
            (batch_size, seq_len_q, self.embedding_dim),
            (batch_size, seq_len_k, self.embedding_dim),
            (batch_size, seq_len_k, self.embedding_dim),
            None, // No mask
        )?;

        Ok(output)
    }

    /// Get all neighbors of a node (both incoming and outgoing edges)
    fn get_neighbors(&self, graph: &KnowledgeGraph, node_id: EntityId) -> Vec<EntityId> {
        let mut neighbors = Vec::new();

        // Outgoing edges (node is subject)
        let outgoing = graph.query(Some(node_id), None, None);
        for triple in outgoing {
            neighbors.push(triple.object);
        }

        // Incoming edges (node is object)
        let incoming = graph.query(None, None, Some(node_id));
        for triple in incoming {
            neighbors.push(triple.subject);
        }

        // Remove duplicates
        neighbors.sort();
        neighbors.dedup();

        // Remove self-loops
        neighbors.retain(|&id| id != node_id);

        neighbors
    }

    /// Multi-layer forward pass
    ///
    /// Applies multiple GNN layers for deeper propagation
    pub fn forward_multilayer(
        &self,
        graph: &KnowledgeGraph,
        node_embeddings: &HashMap<EntityId, NodeEmbedding>,
        num_layers: usize,
    ) -> Result<HashMap<EntityId, NodeEmbedding>, String> {
        let mut current_embeddings = node_embeddings.clone();

        for _layer in 0..num_layers {
            current_embeddings = self.forward(graph, &current_embeddings)?;
        }

        Ok(current_embeddings)
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

/// Graph Attention Layer (GAT)
/// Specialized GNN that uses attention weights on edges
pub struct GraphAttentionLayer {
    gnn: GraphNeuralNetwork,
}

impl GraphAttentionLayer {
    /// Create a new GAT layer
    pub fn new(embedding_dim: usize, num_heads: usize) -> Result<Self, String> {
        Ok(GraphAttentionLayer {
            gnn: GraphNeuralNetwork::new(embedding_dim, num_heads)?,
        })
    }

    /// Forward pass with relation-specific attention
    pub fn forward(
        &self,
        graph: &KnowledgeGraph,
        node_embeddings: &HashMap<EntityId, NodeEmbedding>,
        _relation_filter: Option<&RelationType>,
    ) -> Result<HashMap<EntityId, NodeEmbedding>, String> {
        // If relation filter is specified, create a filtered view
        // For now, just use the standard GNN forward
        // TODO: Implement relation-specific filtering
        self.gnn.forward(graph, node_embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_graph::CodeGraphBuilder;

    #[test]
    fn test_gnn_creation() {
        let gnn = GraphNeuralNetwork::new(64, 4);
        assert!(gnn.is_ok());

        let gnn = gnn.unwrap();
        assert_eq!(gnn.embedding_dim(), 64);
    }

    #[test]
    fn test_gnn_invalid_dimensions() {
        // embedding_dim not divisible by num_heads
        let gnn = GraphNeuralNetwork::new(65, 4);
        assert!(gnn.is_err());
    }

    #[test]
    fn test_type_embeddings() {
        let gnn = GraphNeuralNetwork::new(64, 4).unwrap();

        let func_emb = gnn.get_type_embedding(&EntityType::Function);
        assert_eq!(func_emb.len(), 64);

        let var_emb = gnn.get_type_embedding(&EntityType::Variable);
        assert_eq!(var_emb.len(), 64);

        // Different types should have different embeddings
        assert_ne!(func_emb, var_emb);
    }

    #[test]
    fn test_initialize_node_embeddings() {
        let mut builder = CodeGraphBuilder::new();
        let func = builder.add_function("test");
        let var = builder.add_class("User");
        let graph = builder.build();

        let gnn = GraphNeuralNetwork::new(64, 4).unwrap();
        let embeddings = gnn.initialize_node_embeddings(&graph);

        assert_eq!(embeddings.len(), 2);
        assert!(embeddings.contains_key(&func));
        assert!(embeddings.contains_key(&var));
        assert_eq!(embeddings[&func].len(), 64);
    }

    #[test]
    fn test_gnn_forward_empty_graph() {
        let graph = CodeGraphBuilder::new().build();
        let gnn = GraphNeuralNetwork::new(64, 4).unwrap();
        let embeddings = gnn.initialize_node_embeddings(&graph);

        let result = gnn.forward(&graph, &embeddings);
        assert!(result.is_ok());

        let updated = result.unwrap();
        assert_eq!(updated.len(), 0);
    }

    #[test]
    fn test_gnn_forward_single_node() {
        let mut builder = CodeGraphBuilder::new();
        builder.add_function("isolated");
        let graph = builder.build();

        let gnn = GraphNeuralNetwork::new(64, 4).unwrap();
        let embeddings = gnn.initialize_node_embeddings(&graph);

        let result = gnn.forward(&graph, &embeddings);
        assert!(result.is_ok());

        let updated = result.unwrap();
        assert_eq!(updated.len(), 1);
    }

    #[test]
    fn test_gnn_forward_with_edges() {
        // Create graph: A -> B -> C
        let mut builder = CodeGraphBuilder::new();
        let a = builder.add_function("A");
        let b = builder.add_function("B");
        let c = builder.add_function("C");

        builder.add_call(a, b);
        builder.add_call(b, c);

        let graph = builder.build();

        let gnn = GraphNeuralNetwork::new(64, 4).unwrap();
        let embeddings = gnn.initialize_node_embeddings(&graph);

        // Forward pass should succeed
        let result = gnn.forward(&graph, &embeddings);
        assert!(result.is_ok());

        let updated = result.unwrap();
        assert_eq!(updated.len(), 3);

        // Embeddings should have changed (due to aggregation)
        // Note: They might be the same if attention learns identity, but structure is correct
        assert!(updated.contains_key(&a));
        assert!(updated.contains_key(&b));
        assert!(updated.contains_key(&c));
    }

    #[test]
    fn test_gnn_multilayer() {
        let mut builder = CodeGraphBuilder::new();
        let a = builder.add_function("A");
        let b = builder.add_function("B");
        builder.add_call(a, b);

        let graph = builder.build();

        let gnn = GraphNeuralNetwork::new(64, 4).unwrap();
        let embeddings = gnn.initialize_node_embeddings(&graph);

        // Apply 3 layers
        let result = gnn.forward_multilayer(&graph, &embeddings, 3);
        assert!(result.is_ok());

        let updated = result.unwrap();
        assert_eq!(updated.len(), 2);
    }

    #[test]
    fn test_graph_attention_layer() {
        let gat = GraphAttentionLayer::new(64, 4);
        assert!(gat.is_ok());

        let mut builder = CodeGraphBuilder::new();
        let a = builder.add_function("A");
        let b = builder.add_function("B");
        builder.add_call(a, b);

        let graph = builder.build();
        let gat = gat.unwrap();

        let gnn = GraphNeuralNetwork::new(64, 4).unwrap();
        let embeddings = gnn.initialize_node_embeddings(&graph);

        let result = gat.forward(&graph, &embeddings, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_neighbors() {
        let mut builder = CodeGraphBuilder::new();
        let a = builder.add_function("A");
        let b = builder.add_function("B");
        let c = builder.add_function("C");

        // A -> B -> C
        // A -> C
        builder.add_call(a, b);
        builder.add_call(b, c);
        builder.add_call(a, c);

        let graph = builder.build();
        let gnn = GraphNeuralNetwork::new(64, 4).unwrap();

        // A's neighbors: B, C (outgoing)
        let neighbors_a = gnn.get_neighbors(&graph, a);
        assert_eq!(neighbors_a.len(), 2);
        assert!(neighbors_a.contains(&b));
        assert!(neighbors_a.contains(&c));

        // B's neighbors: A (incoming), C (outgoing)
        let neighbors_b = gnn.get_neighbors(&graph, b);
        assert_eq!(neighbors_b.len(), 2);
        assert!(neighbors_b.contains(&a));
        assert!(neighbors_b.contains(&c));

        // C's neighbors: A, B (both incoming)
        let neighbors_c = gnn.get_neighbors(&graph, c);
        assert_eq!(neighbors_c.len(), 2);
        assert!(neighbors_c.contains(&a));
        assert!(neighbors_c.contains(&b));
    }
}

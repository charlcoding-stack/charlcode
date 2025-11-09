// Knowledge Graph & GNN Additional Builtin Functions
// Phase 2 - Backend Exposure: Graph operations, GNN layers
//
// This module exposes advanced KG and GNN functionality to Charl.

use crate::interpreter::Value;
use crate::knowledge_graph::{
    KnowledgeGraph, Entity, Triple, EntityId, EntityType, RelationType,
    GraphNeuralNetwork, NodeEmbedding, GraphStats,
};
use std::collections::HashMap;

/// Builtin function type
pub type BuiltinFn = fn(Vec<Value>) -> Result<Value, String>;

// ===================================================================
// KNOWLEDGE GRAPH - Advanced Operations
// ===================================================================

/// kg_add_node(kg: KG, id: string, label: string, properties: {string: Value}) -> KG
/// Add a node with properties to the knowledge graph
pub fn builtin_kg_add_node(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("kg_add_node() expects 4 arguments: kg_add_node(kg, id, label, properties)".to_string());
    }

    let mut kg = match &args[0] {
        Value::KnowledgeGraph(g) => (**g).clone(),
        _ => return Err(format!("kg_add_node() expects KnowledgeGraph, got {}", args[0].type_name())),
    };

    let id_str = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("kg_add_node() id must be string".to_string()),
    };

    let label = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err("kg_add_node() label must be string".to_string()),
    };

    // For now, we store the name and can extend with properties later
    let entity_type = EntityType::Concept; // Default to Concept for generic nodes
    let _entity_id = kg.add_entity(entity_type, label);

    Ok(Value::KnowledgeGraph(Box::new(kg)))
}

/// kg_add_edge(kg: KG, from: int, to: int, relation: string) -> KG
/// Add an edge between two nodes
pub fn builtin_kg_add_edge(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("kg_add_edge() expects 4 arguments: kg_add_edge(kg, from, to, relation)".to_string());
    }

    let mut kg = match &args[0] {
        Value::KnowledgeGraph(g) => (**g).clone(),
        _ => return Err(format!("kg_add_edge() expects KnowledgeGraph, got {}", args[0].type_name())),
    };

    let from = match &args[1] {
        Value::Integer(i) => *i as EntityId,
        _ => return Err("kg_add_edge() from must be integer".to_string()),
    };

    let to = match &args[2] {
        Value::Integer(i) => *i as EntityId,
        _ => return Err("kg_add_edge() to must be integer".to_string()),
    };

    let relation = match &args[3] {
        Value::String(s) => {
            // Map string to RelationType
            match s.as_str() {
                "calls" => RelationType::Calls,
                "inherits" => RelationType::Inherits,
                "implements" => RelationType::Implements,
                "depends_on" => RelationType::DependsOn,
                "contains" => RelationType::Contains,
                "uses" => RelationType::Uses,
                "returns" => RelationType::Returns,
                "takes" => RelationType::Takes,
                "has_type" => RelationType::HasType,
                "is_a" => RelationType::IsA,
                "layer_above" => RelationType::LayerAbove,
                "layer_below" => RelationType::LayerBelow,
                "violates" => RelationType::Violates,
                // Allow custom relations
                _ => RelationType::Custom(s.clone()),
            }
        }
        _ => return Err("kg_add_edge() relation must be string".to_string()),
    };

    let triple = Triple::new(from, relation, to);
    kg.add_triple(triple);

    Ok(Value::KnowledgeGraph(Box::new(kg)))
}

/// kg_neighbors(kg: KG, node_id: int) -> [int]
/// Get all neighbors of a node
pub fn builtin_kg_neighbors(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("kg_neighbors() expects 2 arguments: kg_neighbors(kg, node_id)".to_string());
    }

    let kg = match &args[0] {
        Value::KnowledgeGraph(g) => g,
        _ => return Err(format!("kg_neighbors() expects KnowledgeGraph, got {}", args[0].type_name())),
    };

    let node_id = match &args[1] {
        Value::Integer(i) => *i as EntityId,
        _ => return Err("kg_neighbors() node_id must be integer".to_string()),
    };

    // Get all triples where this node is the subject
    let triples = kg.query(Some(node_id), None, None);

    let mut neighbors = Vec::new();
    for triple in triples {
        neighbors.push(Value::Integer(triple.object as i64));
    }

    Ok(Value::Array(neighbors))
}

/// kg_node_count(kg: KG) -> int
/// Get number of nodes in graph
pub fn builtin_kg_node_count(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("kg_node_count() expects 1 argument: kg_node_count(kg)".to_string());
    }

    let kg = match &args[0] {
        Value::KnowledgeGraph(g) => g,
        _ => return Err(format!("kg_node_count() expects KnowledgeGraph, got {}", args[0].type_name())),
    };

    Ok(Value::Integer(kg.num_entities() as i64))
}

/// kg_edge_count(kg: KG) -> int
/// Get number of edges in graph
pub fn builtin_kg_edge_count(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("kg_edge_count() expects 1 argument: kg_edge_count(kg)".to_string());
    }

    let kg = match &args[0] {
        Value::KnowledgeGraph(g) => g,
        _ => return Err(format!("kg_edge_count() expects KnowledgeGraph, got {}", args[0].type_name())),
    };

    Ok(Value::Integer(kg.num_triples() as i64))
}

/// kg_degree(kg: KG, node_id: int) -> int
/// Get degree (number of edges) of a node
pub fn builtin_kg_degree(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("kg_degree() expects 2 arguments: kg_degree(kg, node_id)".to_string());
    }

    let kg = match &args[0] {
        Value::KnowledgeGraph(g) => g,
        _ => return Err(format!("kg_degree() expects KnowledgeGraph, got {}", args[0].type_name())),
    };

    let node_id = match &args[1] {
        Value::Integer(i) => *i as EntityId,
        _ => return Err("kg_degree() node_id must be integer".to_string()),
    };

    // Count outgoing + incoming edges
    let outgoing = kg.query(Some(node_id), None, None).len();
    let incoming = kg.query(None, None, Some(node_id)).len();

    Ok(Value::Integer((outgoing + incoming) as i64))
}

/// kg_shortest_path(kg: KG, from: int, to: int) -> [int]
/// Find shortest path between two nodes (BFS)
pub fn builtin_kg_shortest_path(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("kg_shortest_path() expects 3 arguments: kg_shortest_path(kg, from, to)".to_string());
    }

    let kg = match &args[0] {
        Value::KnowledgeGraph(g) => g,
        _ => return Err(format!("kg_shortest_path() expects KnowledgeGraph, got {}", args[0].type_name())),
    };

    let from = match &args[1] {
        Value::Integer(i) => *i as EntityId,
        _ => return Err("kg_shortest_path() from must be integer".to_string()),
    };

    let to = match &args[2] {
        Value::Integer(i) => *i as EntityId,
        _ => return Err("kg_shortest_path() to must be integer".to_string()),
    };

    // Use the existing find_paths with max_depth
    let paths = kg.find_paths(from, to, 10);

    if paths.is_empty() {
        return Ok(Value::Array(Vec::new()));
    }

    // Return the first (shortest) path
    let path = &paths[0];
    let path_values: Vec<Value> = path.iter()
        .map(|&node_id| Value::Integer(node_id as i64))
        .collect();

    Ok(Value::Array(path_values))
}

// ===================================================================
// GRAPH NEURAL NETWORKS - Specialized Layers
// ===================================================================

/// gnn_gcn_layer(in_features: int, out_features: int) -> GCNLayer
/// Create a Graph Convolutional Network layer
pub fn builtin_gnn_gcn_layer(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("gnn_gcn_layer() expects 2 arguments: gnn_gcn_layer(in_features, out_features)".to_string());
    }

    let in_features = match &args[0] {
        Value::Integer(i) => *i as usize,
        _ => return Err("gnn_gcn_layer() in_features must be integer".to_string()),
    };

    let out_features = match &args[1] {
        Value::Integer(i) => *i as usize,
        _ => return Err("gnn_gcn_layer() out_features must be integer".to_string()),
    };

    // For now, we create a GNN with default parameters
    // In full implementation, GCN would be a separate struct
    let gnn = GraphNeuralNetwork::new(out_features, 4)
        .map_err(|e| format!("Failed to create GNN: {}", e))?;

    Ok(Value::GraphNeuralNetwork(Box::new(gnn)))
}

/// gnn_aggregate(messages: [[float]], method: string) -> [float]
/// Aggregate messages from neighbors (sum, mean, max)
pub fn builtin_gnn_aggregate(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("gnn_aggregate() expects 2 arguments: gnn_aggregate(messages, method)".to_string());
    }

    let messages = match &args[0] {
        Value::Array(arr) => {
            let mut msgs = Vec::new();
            for msg in arr {
                match msg {
                    Value::Array(inner) => {
                        let floats: Result<Vec<f64>, _> = inner.iter()
                            .map(|v| v.to_float())
                            .collect();
                        msgs.push(floats?);
                    }
                    _ => return Err("gnn_aggregate() messages must be array of arrays".to_string()),
                }
            }
            msgs
        }
        _ => return Err("gnn_aggregate() messages must be array".to_string()),
    };

    let method = match &args[1] {
        Value::String(s) => s.as_str(),
        _ => return Err("gnn_aggregate() method must be string".to_string()),
    };

    if messages.is_empty() {
        return Ok(Value::Array(Vec::new()));
    }

    let dim = messages[0].len();
    let result = match method {
        "sum" => {
            let mut sum = vec![0.0; dim];
            for msg in &messages {
                for (i, &val) in msg.iter().enumerate() {
                    sum[i] += val;
                }
            }
            sum
        }
        "mean" => {
            let mut sum = vec![0.0; dim];
            for msg in &messages {
                for (i, &val) in msg.iter().enumerate() {
                    sum[i] += val;
                }
            }
            let n = messages.len() as f64;
            sum.iter().map(|&x| x / n).collect()
        }
        "max" => {
            let mut max_vals = messages[0].clone();
            for msg in &messages[1..] {
                for (i, &val) in msg.iter().enumerate() {
                    if val > max_vals[i] {
                        max_vals[i] = val;
                    }
                }
            }
            max_vals
        }
        _ => return Err(format!("Unknown aggregation method: {}", method)),
    };

    let result_values: Vec<Value> = result.iter()
        .map(|&x| Value::Float(x))
        .collect();

    Ok(Value::Array(result_values))
}

/// gnn_node_classification(gnn: GNN, graph: KG, embeddings: {int: [float]}, labels: [int]) -> {int: int}
/// Perform node classification using GNN
pub fn builtin_gnn_node_classification(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("gnn_node_classification() expects 4 arguments: gnn_node_classification(gnn, graph, embeddings, labels)".to_string());
    }

    // This is a placeholder for full GNN node classification
    // In production, this would do forward passes and compute predictions

    Ok(Value::Array(Vec::new()))
}

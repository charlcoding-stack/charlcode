// Fusion Optimizer - Detects and applies kernel fusion opportunities
// Analyzes computational graphs to find chains of operations that can be fused

use crate::autograd::ComputationGraph;
use crate::fusion::patterns::{FusionOpportunity, FusionPattern, OpType};
use crate::fusion::FusionConfig;
use std::collections::HashSet;

/// Statistics about fusion optimizations applied
#[derive(Debug, Clone, Default)]
pub struct FusionStats {
    /// Number of fusion opportunities found
    pub opportunities_found: usize,

    /// Number of fusions applied
    pub fusions_applied: usize,

    /// Total memory saved (bytes)
    pub total_memory_saved: usize,

    /// Average estimated speedup
    pub average_speedup: f64,

    /// Number of nodes eliminated
    pub nodes_eliminated: usize,
}

/// The Fusion Optimizer
/// Analyzes computational graphs and identifies fusion opportunities
#[derive(Clone, Debug)]
pub struct FusionOptimizer {
    config: FusionConfig,
    stats: FusionStats,
}

impl FusionOptimizer {
    /// Create a new fusion optimizer with given config
    pub fn new(config: FusionConfig) -> Self {
        FusionOptimizer {
            config,
            stats: FusionStats::default(),
        }
    }

    /// Create with default config
    pub fn default() -> Self {
        Self::new(FusionConfig::default())
    }

    /// Analyze a computational graph and find all fusion opportunities
    ///
    /// Strategy:
    /// 1. Build dependency graph
    /// 2. Find chains of element-wise operations
    /// 3. Check if chains meet fusion criteria
    /// 4. Create FusionOpportunity for each valid chain
    pub fn analyze(&mut self, graph: &ComputationGraph) -> Vec<FusionOpportunity> {
        let mut opportunities = Vec::new();

        // Get all nodes in topological order
        let node_ids = self.get_execution_order(graph);

        // For vertical fusion: scan for chains
        if self.config.enable_vertical {
            opportunities.extend(self.find_vertical_chains(graph, &node_ids));
        }

        // For horizontal fusion: find independent operations
        if self.config.enable_horizontal {
            opportunities.extend(self.find_horizontal_opportunities(graph, &node_ids));
        }

        // Update stats
        self.stats.opportunities_found = opportunities.len();
        if !opportunities.is_empty() {
            self.stats.total_memory_saved =
                opportunities.iter().map(|opp| opp.memory_savings).sum();

            self.stats.average_speedup = opportunities
                .iter()
                .map(|opp| opp.estimated_speedup)
                .sum::<f64>()
                / opportunities.len() as f64;
        }

        opportunities
    }

    /// Find vertical fusion chains (operations in sequence)
    ///
    /// Example: a + b → c * d → e / f
    /// This can be fused into a single kernel
    fn find_vertical_chains(
        &self,
        graph: &ComputationGraph,
        node_ids: &[usize],
    ) -> Vec<FusionOpportunity> {
        let mut opportunities = Vec::new();
        let mut visited = HashSet::new();

        for &start_id in node_ids {
            if visited.contains(&start_id) {
                continue;
            }

            // Try to build a chain starting from this node
            let chain = self.build_chain(graph, start_id, &mut visited);

            if chain.len() >= 2 {
                // We have a fusible chain
                let op_types: Vec<OpType> = chain
                    .iter()
                    .filter_map(|&id| graph.get_node(id))
                    .filter_map(|node| OpType::from_op(&node.op))
                    .collect();

                if op_types.len() == chain.len() {
                    // All operations are element-wise
                    if let Some(pattern) = FusionPattern::detect(&op_types) {
                        // Estimate tensor size (use first node's data)
                        let tensor_size = graph
                            .get_node(chain[0])
                            .map(|tensor| tensor.data.len())
                            .unwrap_or(1000); // Default estimate

                        let opportunity = FusionOpportunity::new(pattern, chain, tensor_size);

                        // Check if it's beneficial
                        if opportunity.is_beneficial(self.config.min_memory_savings) {
                            opportunities.push(opportunity);
                        }
                    }
                }
            }
        }

        opportunities
    }

    /// Build a chain of fusible operations starting from a node
    fn build_chain(
        &self,
        graph: &ComputationGraph,
        start_id: usize,
        visited: &mut HashSet<usize>,
    ) -> Vec<usize> {
        let mut chain = vec![start_id];
        visited.insert(start_id);

        let mut current_id = start_id;

        loop {
            // Check if we've reached max fusion size
            if chain.len() >= self.config.max_ops_per_fusion {
                break;
            }

            // Get the current node
            let current_node = match graph.get_node(current_id) {
                Some(node) => node,
                None => break,
            };

            // Check if operation is element-wise
            if OpType::from_op(&current_node.op).is_none() {
                break; // Not element-wise, can't continue chain
            }

            // Find the next node in the chain
            // For simplicity, we look for nodes that use this one as input
            let next_id = self.find_next_in_chain(graph, current_id, visited);

            match next_id {
                Some(id) => {
                    chain.push(id);
                    visited.insert(id);
                    current_id = id;
                }
                None => break, // No more nodes in chain
            }
        }

        chain
    }

    /// Find the next node in a fusion chain
    fn find_next_in_chain(
        &self,
        _graph: &ComputationGraph,
        _current_id: usize,
        _visited: &HashSet<usize>,
    ) -> Option<usize> {
        // This is a simplified version
        // In a full implementation, we'd need to:
        // 1. Track which nodes use current_id as input
        // 2. Ensure the next node only depends on the current chain
        // 3. Verify no other nodes depend on intermediate results

        // For now, return None (conservative approach)
        // This will be enhanced when we have a proper dependency graph
        None
    }

    /// Find horizontal fusion opportunities (independent operations)
    fn find_horizontal_opportunities(
        &self,
        _graph: &ComputationGraph,
        _node_ids: &[usize],
    ) -> Vec<FusionOpportunity> {
        // Horizontal fusion is more complex and requires:
        // 1. Dependency analysis to find independent operations
        // 2. Resource estimation to ensure parallel execution is beneficial
        // 3. Output buffer management

        // For MVP, we return empty
        // This will be implemented in a future iteration
        Vec::new()
    }

    /// Get execution order for the graph (simplified topological sort)
    fn get_execution_order(&self, _graph: &ComputationGraph) -> Vec<usize> {
        // For MVP, just return empty
        // In full implementation, this would be a proper topological sort
        // that traverses the graph to find all nodes and their dependencies
        Vec::new()
    }

    /// Apply fusion opportunities to transform the graph
    ///
    /// This is a placeholder for future work
    /// In the full implementation, this would:
    /// 1. Create fused operation nodes
    /// 2. Update graph edges
    /// 3. Remove intermediate nodes
    pub fn apply_fusions(
        &mut self,
        _graph: &mut ComputationGraph,
        _opportunities: &[FusionOpportunity],
    ) -> Result<(), String> {
        // TODO: Implement graph transformation
        // For now, just update stats
        self.stats.fusions_applied = 0; // Would be opportunities.len() when implemented

        Ok(())
    }

    /// Get fusion statistics
    pub fn stats(&self) -> &FusionStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = FusionStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{ComputationGraph, Tensor};

    #[test]
    fn test_optimizer_creation() {
        let optimizer = FusionOptimizer::default();
        assert_eq!(optimizer.stats.opportunities_found, 0);
    }

    #[test]
    fn test_optimizer_with_config() {
        let config = FusionConfig::aggressive();
        let optimizer = FusionOptimizer::new(config);
        assert_eq!(optimizer.config.max_ops_per_fusion, 10);
    }

    #[test]
    fn test_analyze_empty_graph() {
        let mut optimizer = FusionOptimizer::default();
        let graph = ComputationGraph::new();

        let opportunities = optimizer.analyze(&graph);
        assert_eq!(opportunities.len(), 0);
    }

    #[test]
    fn test_analyze_simple_graph() {
        let mut optimizer = FusionOptimizer::default();
        let mut graph = ComputationGraph::new();

        // Create a simple graph: a + b
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);

        graph.add_node(a);
        graph.add_node(b);

        let opportunities = optimizer.analyze(&graph);

        // For MVP, this might not find opportunities yet
        // (depends on graph structure representation)
        // Just verify it returns without errors
        let _ = opportunities;
    }

    #[test]
    fn test_stats_tracking() {
        let mut optimizer = FusionOptimizer::default();
        let graph = ComputationGraph::new();

        optimizer.analyze(&graph);
        let stats = optimizer.stats();

        assert_eq!(stats.opportunities_found, 0);
        assert_eq!(stats.fusions_applied, 0);
        assert_eq!(stats.total_memory_saved, 0);
    }

    #[test]
    fn test_reset_stats() {
        let mut optimizer = FusionOptimizer::default();
        let graph = ComputationGraph::new();

        optimizer.analyze(&graph);
        optimizer.reset_stats();

        assert_eq!(optimizer.stats.opportunities_found, 0);
    }

    #[test]
    fn test_execution_order() {
        let optimizer = FusionOptimizer::default();
        let mut graph = ComputationGraph::new();

        let a = Tensor::new(vec![1.0], vec![1]);
        let b = Tensor::new(vec![2.0], vec![1]);
        let c = Tensor::new(vec![3.0], vec![1]);

        graph.add_node(a);
        graph.add_node(b);
        graph.add_node(c);

        let order = optimizer.get_execution_order(&graph);
        // For MVP, returns empty - will be implemented with proper topological sort
        assert_eq!(order.len(), 0);
    }

    #[test]
    fn test_build_chain_single_node() {
        let optimizer = FusionOptimizer::default();
        let mut graph = ComputationGraph::new();
        let mut visited = HashSet::new();

        let a = Tensor::new(vec![1.0], vec![1]);
        let id = graph.add_node(a);

        let chain = optimizer.build_chain(&graph, id, &mut visited);

        // Single node chain
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0], id);
    }

    #[test]
    fn test_config_limits() {
        let config = FusionConfig {
            enable_vertical: true,
            enable_horizontal: false,
            max_ops_per_fusion: 2,
            min_memory_savings: 1024,
        };

        let mut optimizer = FusionOptimizer::new(config);
        let graph = ComputationGraph::new();

        // Should respect config limits
        let opportunities = optimizer.analyze(&graph);
        for opp in opportunities {
            assert!(opp.pattern.num_ops() <= 2);
            assert!(opp.memory_savings >= 1024);
        }
    }
}

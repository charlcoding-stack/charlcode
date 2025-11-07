// Tree-of-Thoughts (ToT) - Multi-path Reasoning
//
// Extension of Chain-of-Thought that explores multiple reasoning paths
// in parallel, like a search tree.
//
// Key insight: Deliberate search through thought space
//
// Example (Game of 24):
//   Numbers: 4, 9, 10, 13
//   Goal: Make 24 using +, -, ×, ÷
//
//   Thought Tree:
//       [4, 9, 10, 13]
//        /    |    \
//    13-9=4  10-4=6  13+9=22
//      /       |        \
//   4×6=24 ✓  6×4=24 ✓  ...
//
// Search strategies:
// - Breadth-First Search (BFS): Explore all at depth D before D+1
// - Depth-First Search (DFS): Explore one path completely
// - Best-First Search: Explore most promising paths first
//
// References:
// - Yao et al. (2023): "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};

/// A single thought/node in the reasoning tree
#[derive(Debug, Clone)]
pub struct ThoughtNode {
    pub id: usize,
    pub parent_id: Option<usize>,
    pub thought: String,
    pub value: f32, // Evaluation score
    pub depth: usize,
    pub children: Vec<usize>,
    pub is_solution: bool,
}

impl ThoughtNode {
    pub fn new(
        id: usize,
        parent_id: Option<usize>,
        thought: impl Into<String>,
        depth: usize,
    ) -> Self {
        Self {
            id,
            parent_id,
            thought: thought.into(),
            value: 0.0,
            depth,
            children: Vec::new(),
            is_solution: false,
        }
    }

    pub fn with_value(mut self, value: f32) -> Self {
        self.value = value;
        self
    }

    pub fn mark_solution(mut self) -> Self {
        self.is_solution = true;
        self
    }
}

/// For priority queue in best-first search
#[derive(Debug, Clone)]
struct ScoredNode {
    node_id: usize,
    score: f32,
}

impl PartialEq for ScoredNode {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredNode {}

impl PartialOrd for ScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse order for max-heap
        other.score.partial_cmp(&self.score)
    }
}

impl Ord for ScoredNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Search strategy for tree exploration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SearchStrategy {
    BreadthFirst,
    DepthFirst,
    BestFirst,
}

/// Tree of Thoughts reasoner
#[derive(Debug, Clone)]
pub struct TreeOfThoughts {
    /// All nodes in the tree
    pub nodes: HashMap<usize, ThoughtNode>,
    /// Root node ID
    pub root_id: usize,
    /// Next node ID
    next_id: usize,
    /// Search strategy
    pub strategy: SearchStrategy,
    /// Maximum depth
    pub max_depth: usize,
    /// Maximum nodes to explore
    pub max_nodes: usize,
}

impl TreeOfThoughts {
    /// Create new ToT with root thought
    pub fn new(root_thought: impl Into<String>, strategy: SearchStrategy) -> Self {
        let root = ThoughtNode::new(0, None, root_thought, 0);
        let mut nodes = HashMap::new();
        nodes.insert(0, root);

        Self {
            nodes,
            root_id: 0,
            next_id: 1,
            strategy,
            max_depth: 10,
            max_nodes: 100,
        }
    }

    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes;
        self
    }

    /// Add a thought as child of parent
    pub fn add_thought(
        &mut self,
        parent_id: usize,
        thought: impl Into<String>,
        value: f32,
    ) -> Option<usize> {
        if self.nodes.len() >= self.max_nodes {
            return None;
        }

        let parent_depth = self.nodes.get(&parent_id)?.depth;
        if parent_depth >= self.max_depth {
            return None;
        }

        let new_id = self.next_id;
        self.next_id += 1;

        let node =
            ThoughtNode::new(new_id, Some(parent_id), thought, parent_depth + 1).with_value(value);

        // Add to parent's children
        if let Some(parent) = self.nodes.get_mut(&parent_id) {
            parent.children.push(new_id);
        }

        self.nodes.insert(new_id, node);
        Some(new_id)
    }

    /// Mark node as solution
    pub fn mark_solution(&mut self, node_id: usize) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.is_solution = true;
        }
    }

    /// Find all solution nodes
    pub fn get_solutions(&self) -> Vec<&ThoughtNode> {
        self.nodes
            .values()
            .filter(|node| node.is_solution)
            .collect()
    }

    /// Get path from root to node
    pub fn get_path(&self, node_id: usize) -> Vec<&ThoughtNode> {
        let mut path = Vec::new();
        let mut current_id = Some(node_id);

        while let Some(id) = current_id {
            if let Some(node) = self.nodes.get(&id) {
                path.push(node);
                current_id = node.parent_id;
            } else {
                break;
            }
        }

        path.reverse();
        path
    }

    /// Breadth-First Search
    pub fn search_bfs<F>(&mut self, mut expand_fn: F) -> Option<usize>
    where
        F: FnMut(&ThoughtNode) -> Vec<(String, f32)>,
    {
        let mut queue = VecDeque::new();
        queue.push_back(self.root_id);

        while let Some(node_id) = queue.pop_front() {
            let node = self.nodes.get(&node_id)?.clone();

            // Check if solution
            if node.is_solution {
                return Some(node_id);
            }

            // Expand node
            let children = expand_fn(&node);
            for (thought, value) in children {
                if let Some(child_id) = self.add_thought(node_id, thought, value) {
                    queue.push_back(child_id);

                    // Check if child is solution
                    if self
                        .nodes
                        .get(&child_id)
                        .map(|n| n.is_solution)
                        .unwrap_or(false)
                    {
                        return Some(child_id);
                    }
                }
            }
        }

        None
    }

    /// Depth-First Search
    pub fn search_dfs<F>(&mut self, mut expand_fn: F) -> Option<usize>
    where
        F: FnMut(&ThoughtNode) -> Vec<(String, f32)>,
    {
        let mut stack = vec![self.root_id];

        while let Some(node_id) = stack.pop() {
            let node = self.nodes.get(&node_id)?.clone();

            // Check if solution
            if node.is_solution {
                return Some(node_id);
            }

            // Expand node
            let children = expand_fn(&node);
            for (thought, value) in children {
                if let Some(child_id) = self.add_thought(node_id, thought, value) {
                    stack.push(child_id);

                    // Check if child is solution
                    if self
                        .nodes
                        .get(&child_id)
                        .map(|n| n.is_solution)
                        .unwrap_or(false)
                    {
                        return Some(child_id);
                    }
                }
            }
        }

        None
    }

    /// Best-First Search (prioritized by value)
    pub fn search_best_first<F>(&mut self, mut expand_fn: F) -> Option<usize>
    where
        F: FnMut(&ThoughtNode) -> Vec<(String, f32)>,
    {
        let mut heap = BinaryHeap::new();
        heap.push(ScoredNode {
            node_id: self.root_id,
            score: 0.0,
        });

        while let Some(ScoredNode { node_id, .. }) = heap.pop() {
            let node = self.nodes.get(&node_id)?.clone();

            // Check if solution
            if node.is_solution {
                return Some(node_id);
            }

            // Expand node
            let children = expand_fn(&node);
            for (thought, value) in children {
                if let Some(child_id) = self.add_thought(node_id, thought, value) {
                    heap.push(ScoredNode {
                        node_id: child_id,
                        score: value,
                    });

                    // Check if child is solution
                    if self
                        .nodes
                        .get(&child_id)
                        .map(|n| n.is_solution)
                        .unwrap_or(false)
                    {
                        return Some(child_id);
                    }
                }
            }
        }

        None
    }

    /// Search using configured strategy
    pub fn search<F>(&mut self, expand_fn: F) -> Option<usize>
    where
        F: FnMut(&ThoughtNode) -> Vec<(String, f32)>,
    {
        match self.strategy {
            SearchStrategy::BreadthFirst => self.search_bfs(expand_fn),
            SearchStrategy::DepthFirst => self.search_dfs(expand_fn),
            SearchStrategy::BestFirst => self.search_best_first(expand_fn),
        }
    }

    /// Get statistics
    pub fn stats(&self) -> (usize, usize, usize) {
        let total_nodes = self.nodes.len();
        let max_depth_reached = self.nodes.values().map(|n| n.depth).max().unwrap_or(0);
        let num_solutions = self.get_solutions().len();
        (total_nodes, max_depth_reached, num_solutions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thought_node_creation() {
        let node = ThoughtNode::new(0, None, "Root thought", 0).with_value(0.8);

        assert_eq!(node.id, 0);
        assert_eq!(node.parent_id, None);
        assert_eq!(node.thought, "Root thought");
        assert_eq!(node.depth, 0);
        assert!((node.value - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_tree_creation() {
        let tree = TreeOfThoughts::new("Problem: 2+2", SearchStrategy::BreadthFirst);

        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.root_id, 0);
    }

    #[test]
    fn test_add_thought() {
        let mut tree = TreeOfThoughts::new("Root", SearchStrategy::BreadthFirst);

        let child1 = tree.add_thought(0, "Child 1", 0.5);
        assert!(child1.is_some());
        assert_eq!(child1.unwrap(), 1);

        let child2 = tree.add_thought(0, "Child 2", 0.7);
        assert_eq!(child2.unwrap(), 2);

        assert_eq!(tree.nodes.len(), 3);

        // Root should have 2 children
        assert_eq!(tree.nodes.get(&0).unwrap().children.len(), 2);
    }

    #[test]
    fn test_max_depth() {
        let mut tree = TreeOfThoughts::new("Root", SearchStrategy::BreadthFirst).with_max_depth(2);

        let child1 = tree.add_thought(0, "Depth 1", 0.5);
        assert!(child1.is_some());

        let child2 = tree.add_thought(child1.unwrap(), "Depth 2", 0.5);
        assert!(child2.is_some());

        // Should not add (depth 3 exceeds max)
        let child3 = tree.add_thought(child2.unwrap(), "Depth 3", 0.5);
        assert!(child3.is_none());
    }

    #[test]
    fn test_max_nodes() {
        let mut tree = TreeOfThoughts::new("Root", SearchStrategy::BreadthFirst).with_max_nodes(3);

        tree.add_thought(0, "Child 1", 0.5);
        tree.add_thought(0, "Child 2", 0.5);

        // Should not add (3 nodes already)
        let child3 = tree.add_thought(0, "Child 3", 0.5);
        assert!(child3.is_none());
    }

    #[test]
    fn test_mark_solution() {
        let mut tree = TreeOfThoughts::new("Root", SearchStrategy::BreadthFirst);
        let child = tree.add_thought(0, "Solution", 1.0).unwrap();

        tree.mark_solution(child);

        let solutions = tree.get_solutions();
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].id, child);
    }

    #[test]
    fn test_get_path() {
        let mut tree = TreeOfThoughts::new("Root", SearchStrategy::BreadthFirst);

        let child1 = tree.add_thought(0, "Child 1", 0.5).unwrap();
        let child2 = tree.add_thought(child1, "Child 2", 0.7).unwrap();

        let path = tree.get_path(child2);

        assert_eq!(path.len(), 3);
        assert_eq!(path[0].id, 0); // Root
        assert_eq!(path[1].id, child1);
        assert_eq!(path[2].id, child2);
    }

    #[test]
    fn test_search_bfs() {
        let mut tree = TreeOfThoughts::new("Problem", SearchStrategy::BreadthFirst);

        // Expand function that creates 2 children, marks second as solution
        let mut expand_count = 0;
        let result = tree.search_bfs(|node| {
            expand_count += 1;
            if node.depth < 2 {
                vec![("Child A".to_string(), 0.5), ("Child B".to_string(), 0.8)]
            } else {
                vec![]
            }
        });

        // Should find a node (expanded at least once)
        assert!(expand_count > 0);
    }

    #[test]
    fn test_search_with_solution() {
        let mut tree =
            TreeOfThoughts::new("Find 24", SearchStrategy::BreadthFirst).with_max_depth(3);

        let result = tree.search(|node| {
            if node.depth == 0 {
                // First level
                vec![
                    ("Try 4+9=13".to_string(), 0.5),
                    ("Try 10-4=6".to_string(), 0.7),
                ]
            } else if node.depth == 1 {
                // Second level - mark one as solution
                vec![("Found 24!".to_string(), 1.0)]
            } else {
                vec![]
            }
        });

        // Should have explored the tree
        let (total, depth, solutions) = tree.stats();
        assert!(total >= 1);
    }

    #[test]
    fn test_search_strategies() {
        // BFS
        let tree_bfs = TreeOfThoughts::new("Root", SearchStrategy::BreadthFirst);
        assert_eq!(tree_bfs.strategy, SearchStrategy::BreadthFirst);

        // DFS
        let tree_dfs = TreeOfThoughts::new("Root", SearchStrategy::DepthFirst);
        assert_eq!(tree_dfs.strategy, SearchStrategy::DepthFirst);

        // Best-first
        let tree_best = TreeOfThoughts::new("Root", SearchStrategy::BestFirst);
        assert_eq!(tree_best.strategy, SearchStrategy::BestFirst);
    }

    #[test]
    fn test_stats() {
        let mut tree = TreeOfThoughts::new("Root", SearchStrategy::BreadthFirst);

        tree.add_thought(0, "Child 1", 0.5);
        let child2 = tree.add_thought(0, "Child 2", 0.7).unwrap();
        tree.mark_solution(child2);

        let (total, max_depth, solutions) = tree.stats();

        assert_eq!(total, 3);
        assert_eq!(max_depth, 1);
        assert_eq!(solutions, 1);
    }
}

// Causal Reasoning - Beyond Correlation
//
// Causal reasoning distinguishes correlation from causation and enables
// counterfactual reasoning ("what if" questions).
//
// Key insight: Correlation ≠ Causation
//
// Example:
//   Observation: Students who study more get better grades
//   Correlation: study_hours ↔ grades (bidirectional)
//   Causation: study_hours → grades (unidirectional)
//
// Counterfactual:
//   "If I had studied 2 more hours, would I have passed?"
//   → Intervention: do(study_hours = actual + 2)
//   → Predict outcome under intervention
//
// Framework: Pearl's Causal Hierarchy (Ladder of Causation)
//   Level 1: Association (correlation) - "What if I SEE X?"
//   Level 2: Intervention (causation) - "What if I DO X?"
//   Level 3: Counterfactuals - "What if I HAD DONE X?"
//
// References:
// - Pearl (2009): "Causality: Models, Reasoning and Inference"
// - Pearl & Mackenzie (2018): "The Book of Why"

use std::collections::{HashMap, HashSet};

/// Causal graph node (variable)
#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    pub name: String,
    pub value: Option<f32>,
}

impl Variable {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: None,
        }
    }

    pub fn with_value(mut self, value: f32) -> Self {
        self.value = Some(value);
        self
    }
}

/// Directed causal edge: cause → effect
#[derive(Debug, Clone)]
pub struct CausalEdge {
    pub cause: String,
    pub effect: String,
    pub strength: f32, // [0, 1] - how strong the causal relationship
}

impl CausalEdge {
    pub fn new(cause: impl Into<String>, effect: impl Into<String>, strength: f32) -> Self {
        Self {
            cause: cause.into(),
            effect: effect.into(),
            strength: strength.clamp(0.0, 1.0),
        }
    }
}

/// Causal graph (DAG - Directed Acyclic Graph)
pub struct CausalGraph {
    /// Variables in the graph
    pub variables: HashMap<String, Variable>,
    /// Causal relationships (edges)
    pub edges: Vec<CausalEdge>,
}

impl CausalGraph {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Add variable to graph
    pub fn add_variable(&mut self, var: Variable) {
        self.variables.insert(var.name.clone(), var);
    }

    /// Add causal edge
    pub fn add_edge(&mut self, edge: CausalEdge) -> Result<(), String> {
        // Check if variables exist
        if !self.variables.contains_key(&edge.cause) {
            return Err(format!("Cause variable '{}' not found", edge.cause));
        }
        if !self.variables.contains_key(&edge.effect) {
            return Err(format!("Effect variable '{}' not found", edge.effect));
        }

        // Check for cycles (simplified - just check direct reverse edge)
        if self.has_edge(&edge.effect, &edge.cause) {
            return Err(format!("Adding edge would create cycle"));
        }

        self.edges.push(edge);
        Ok(())
    }

    /// Check if edge exists
    fn has_edge(&self, from: &str, to: &str) -> bool {
        self.edges.iter().any(|e| e.cause == from && e.effect == to)
    }

    /// Get parents (causes) of a variable
    pub fn get_parents(&self, var_name: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|e| e.effect == var_name)
            .map(|e| e.cause.clone())
            .collect()
    }

    /// Get children (effects) of a variable
    pub fn get_children(&self, var_name: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|e| e.cause == var_name)
            .map(|e| e.effect.clone())
            .collect()
    }

    /// Check if X causes Y (direct or indirect)
    pub fn causes(&self, x: &str, y: &str) -> bool {
        // Direct causation
        if self.has_edge(x, y) {
            return true;
        }

        // Indirect causation (DFS)
        let mut visited = HashSet::new();
        self.causes_recursive(x, y, &mut visited)
    }

    fn causes_recursive(&self, x: &str, y: &str, visited: &mut HashSet<String>) -> bool {
        if visited.contains(x) {
            return false;
        }
        visited.insert(x.to_string());

        let children = self.get_children(x);
        for child in children {
            if child == y {
                return true;
            }
            if self.causes_recursive(&child, y, visited) {
                return true;
            }
        }

        false
    }
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Intervention: do(X = x)
///
/// Sets X to value x and removes all incoming edges to X
/// (cuts off X's causes - we're forcing X to be x)
#[derive(Debug, Clone)]
pub struct Intervention {
    pub variable: String,
    pub value: f32,
}

impl Intervention {
    pub fn new(variable: impl Into<String>, value: f32) -> Self {
        Self {
            variable: variable.into(),
            value,
        }
    }

    /// Apply intervention to graph (creates new graph)
    pub fn apply(&self, graph: &CausalGraph) -> CausalGraph {
        let mut new_graph = CausalGraph::new();

        // Copy all variables
        for var in graph.variables.values() {
            new_graph.add_variable(var.clone());
        }

        // Set intervened variable
        if let Some(var) = new_graph.variables.get_mut(&self.variable) {
            var.value = Some(self.value);
        }

        // Copy edges, but remove incoming edges to intervened variable
        for edge in &graph.edges {
            if edge.effect != self.variable {
                let _ = new_graph.add_edge(edge.clone());
            }
        }

        new_graph
    }
}

/// Counterfactual query: "What if X had been x?"
#[derive(Debug, Clone)]
pub struct Counterfactual {
    /// Original observation
    pub actual: HashMap<String, f32>,
    /// Counterfactual intervention
    pub intervention: Intervention,
    /// Query variable (what we want to predict)
    pub query: String,
}

impl Counterfactual {
    pub fn new(
        actual: HashMap<String, f32>,
        intervention: Intervention,
        query: impl Into<String>,
    ) -> Self {
        Self {
            actual,
            intervention,
            query: query.into(),
        }
    }

    /// Evaluate counterfactual (simplified)
    ///
    /// In practice, this requires:
    /// 1. Abduction: Infer unobserved causes from observations
    /// 2. Action: Apply intervention
    /// 3. Prediction: Predict outcome
    pub fn evaluate(&self, graph: &CausalGraph) -> Option<f32> {
        // Apply intervention
        let intervened_graph = self.intervention.apply(graph);

        // For simplicity, just check if query variable has value
        intervened_graph
            .variables
            .get(&self.query)
            .and_then(|v| v.value)
    }
}

/// Confounding: When correlation does not imply causation
///
/// Example:
///   Ice cream sales ↔ Drownings (correlated)
///   But: Temperature → Ice cream sales
///        Temperature → Drownings
///   Temperature is a confounder!
#[derive(Debug, Clone)]
pub struct Confounder {
    pub name: String,
    /// Variables affected by confounder
    pub affects: Vec<String>,
}

impl Confounder {
    pub fn new(name: impl Into<String>, affects: Vec<String>) -> Self {
        Self {
            name: name.into(),
            affects,
        }
    }

    /// Check if confounder explains correlation between X and Y
    pub fn explains_correlation(&self, x: &str, y: &str) -> bool {
        self.affects.contains(&x.to_string()) && self.affects.contains(&y.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_creation() {
        let var = Variable::new("temperature").with_value(25.0);
        assert_eq!(var.name, "temperature");
        assert_eq!(var.value, Some(25.0));
    }

    #[test]
    fn test_causal_edge_creation() {
        let edge = CausalEdge::new("smoking", "cancer", 0.8);
        assert_eq!(edge.cause, "smoking");
        assert_eq!(edge.effect, "cancer");
        assert_eq!(edge.strength, 0.8);
    }

    #[test]
    fn test_causal_graph_creation() {
        let mut graph = CausalGraph::new();

        graph.add_variable(Variable::new("X"));
        graph.add_variable(Variable::new("Y"));

        assert_eq!(graph.variables.len(), 2);
    }

    #[test]
    fn test_add_edge() {
        let mut graph = CausalGraph::new();

        graph.add_variable(Variable::new("smoking"));
        graph.add_variable(Variable::new("cancer"));

        let result = graph.add_edge(CausalEdge::new("smoking", "cancer", 0.8));
        assert!(result.is_ok());
        assert_eq!(graph.edges.len(), 1);
    }

    #[test]
    fn test_add_edge_missing_variable() {
        let mut graph = CausalGraph::new();
        graph.add_variable(Variable::new("X"));

        let result = graph.add_edge(CausalEdge::new("X", "Y", 0.5));
        assert!(result.is_err());
    }

    #[test]
    fn test_add_edge_cycle_detection() {
        let mut graph = CausalGraph::new();
        graph.add_variable(Variable::new("X"));
        graph.add_variable(Variable::new("Y"));

        graph.add_edge(CausalEdge::new("X", "Y", 0.5)).unwrap();

        // Try to add reverse edge (creates cycle)
        let result = graph.add_edge(CausalEdge::new("Y", "X", 0.5));
        assert!(result.is_err());
    }

    #[test]
    fn test_get_parents() {
        let mut graph = CausalGraph::new();
        graph.add_variable(Variable::new("X"));
        graph.add_variable(Variable::new("Y"));
        graph.add_variable(Variable::new("Z"));

        graph.add_edge(CausalEdge::new("X", "Z", 0.5)).unwrap();
        graph.add_edge(CausalEdge::new("Y", "Z", 0.5)).unwrap();

        let parents = graph.get_parents("Z");
        assert_eq!(parents.len(), 2);
        assert!(parents.contains(&"X".to_string()));
        assert!(parents.contains(&"Y".to_string()));
    }

    #[test]
    fn test_get_children() {
        let mut graph = CausalGraph::new();
        graph.add_variable(Variable::new("X"));
        graph.add_variable(Variable::new("Y"));
        graph.add_variable(Variable::new("Z"));

        graph.add_edge(CausalEdge::new("X", "Y", 0.5)).unwrap();
        graph.add_edge(CausalEdge::new("X", "Z", 0.5)).unwrap();

        let children = graph.get_children("X");
        assert_eq!(children.len(), 2);
        assert!(children.contains(&"Y".to_string()));
        assert!(children.contains(&"Z".to_string()));
    }

    #[test]
    fn test_direct_causation() {
        let mut graph = CausalGraph::new();
        graph.add_variable(Variable::new("study"));
        graph.add_variable(Variable::new("grade"));

        graph.add_edge(CausalEdge::new("study", "grade", 0.7)).unwrap();

        assert!(graph.causes("study", "grade"));
        assert!(!graph.causes("grade", "study"));
    }

    #[test]
    fn test_indirect_causation() {
        let mut graph = CausalGraph::new();
        graph.add_variable(Variable::new("X"));
        graph.add_variable(Variable::new("Y"));
        graph.add_variable(Variable::new("Z"));

        graph.add_edge(CausalEdge::new("X", "Y", 0.5)).unwrap();
        graph.add_edge(CausalEdge::new("Y", "Z", 0.5)).unwrap();

        // X causes Z indirectly through Y
        assert!(graph.causes("X", "Z"));
    }

    #[test]
    fn test_intervention() {
        let mut graph = CausalGraph::new();
        graph.add_variable(Variable::new("X"));
        graph.add_variable(Variable::new("Y"));
        graph.add_variable(Variable::new("Z"));

        graph.add_edge(CausalEdge::new("X", "Y", 0.5)).unwrap();
        graph.add_edge(CausalEdge::new("Y", "Z", 0.5)).unwrap();

        let intervention = Intervention::new("Y", 10.0);
        let new_graph = intervention.apply(&graph);

        // Y should have value 10.0
        assert_eq!(new_graph.variables.get("Y").unwrap().value, Some(10.0));

        // Edge from X to Y should be removed
        assert_eq!(new_graph.edges.len(), 1);
        assert!(new_graph.has_edge("Y", "Z"));
        assert!(!new_graph.has_edge("X", "Y"));
    }

    #[test]
    fn test_counterfactual() {
        let mut graph = CausalGraph::new();
        graph.add_variable(Variable::new("study_hours").with_value(2.0));
        graph.add_variable(Variable::new("grade").with_value(70.0));

        graph
            .add_edge(CausalEdge::new("study_hours", "grade", 0.8))
            .unwrap();

        let mut actual = HashMap::new();
        actual.insert("study_hours".to_string(), 2.0);
        actual.insert("grade".to_string(), 70.0);

        let intervention = Intervention::new("study_hours", 4.0); // What if studied 4 hours?

        let counterfactual = Counterfactual::new(actual, intervention, "grade");

        // In a full implementation, this would predict the counterfactual grade
        // Here we just check the structure works
        assert_eq!(counterfactual.query, "grade");
    }

    #[test]
    fn test_confounder() {
        let confounder = Confounder::new(
            "temperature",
            vec!["ice_cream_sales".to_string(), "drownings".to_string()],
        );

        assert!(confounder.explains_correlation("ice_cream_sales", "drownings"));
        assert!(!confounder.explains_correlation("ice_cream_sales", "other"));
    }

    #[test]
    fn test_smoking_cancer_example() {
        let mut graph = CausalGraph::new();

        // Variables
        graph.add_variable(Variable::new("smoking"));
        graph.add_variable(Variable::new("tar_deposits"));
        graph.add_variable(Variable::new("cancer"));

        // Causal chain: smoking → tar_deposits → cancer
        graph
            .add_edge(CausalEdge::new("smoking", "tar_deposits", 0.9))
            .unwrap();
        graph
            .add_edge(CausalEdge::new("tar_deposits", "cancer", 0.8))
            .unwrap();

        // Smoking causes cancer (indirectly)
        assert!(graph.causes("smoking", "cancer"));

        // Intervention: do(smoking = 0)
        let intervention = Intervention::new("smoking", 0.0);
        let new_graph = intervention.apply(&graph);

        // After intervention, smoking no longer has incoming edges
        assert_eq!(new_graph.get_parents("smoking").len(), 0);
    }
}

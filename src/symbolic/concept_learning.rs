// Advanced Concept Learning
// Extract abstract concepts and enable compositional generalization
//
// This module implements:
// - Abstract concept extraction from examples
// - Hierarchical concept graphs
// - Compositional generalization (combine concepts)
// - Zero-shot concept transfer
// - Concept embeddings and similarity
//
// Usage:
// ```rust
// use charl::symbolic::concept_learning::{Concept, ConceptGraph, ConceptLearner};
//
// // Create a concept
// let controller = Concept::new("Controller")
//     .with_property("handles_requests", 1.0)
//     .with_property("depends_on_service", 0.9);
//
// // Build concept hierarchy
// let mut graph = ConceptGraph::new();
// graph.add_concept(controller);
// graph.add_subconcept_relation("Controller", "Class");
//
// // Learn concepts from examples
// let mut learner = ConceptLearner::new();
// let new_concept = learner.learn_from_examples(&examples)?;
// ```

use crate::knowledge_graph::KnowledgeGraph;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// A concept: abstract representation of a category or pattern
#[derive(Debug, Clone)]
pub struct Concept {
    /// Name of the concept
    pub name: String,

    /// Properties and their strengths (0-1)
    pub properties: HashMap<String, f64>,

    /// Examples that instantiate this concept
    pub examples: Vec<String>,

    /// Embedding vector for similarity computation
    pub embedding: Option<Vec<f64>>,

    /// Confidence in this concept definition
    pub confidence: f64,
}

impl Concept {
    /// Create a new concept
    pub fn new(name: impl Into<String>) -> Self {
        Concept {
            name: name.into(),
            properties: HashMap::new(),
            examples: Vec::new(),
            embedding: None,
            confidence: 1.0,
        }
    }

    /// Add a property with strength
    pub fn with_property(mut self, property: impl Into<String>, strength: f64) -> Self {
        self.properties
            .insert(property.into(), strength.clamp(0.0, 1.0));
        self
    }

    /// Add an example
    pub fn with_example(mut self, example: impl Into<String>) -> Self {
        self.examples.push(example.into());
        self
    }

    /// Set embedding
    pub fn with_embedding(mut self, embedding: Vec<f64>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Get property strength (0 if not present)
    pub fn get_property(&self, property: &str) -> f64 {
        self.properties.get(property).copied().unwrap_or(0.0)
    }

    /// Check if has property (strength >= 0.5)
    pub fn has_property(&self, property: &str) -> bool {
        self.get_property(property) >= 0.5
    }

    /// Number of examples
    pub fn num_examples(&self) -> usize {
        self.examples.len()
    }

    /// Compute similarity with another concept (Jaccard similarity on properties)
    pub fn similarity(&self, other: &Concept) -> f64 {
        if self.properties.is_empty() && other.properties.is_empty() {
            return 1.0;
        }

        // Get all properties
        let all_props: HashSet<_> = self
            .properties
            .keys()
            .chain(other.properties.keys())
            .collect();

        if all_props.is_empty() {
            return 1.0;
        }

        // Compute weighted Jaccard similarity
        let mut intersection = 0.0;
        let mut union = 0.0;

        for prop in all_props {
            let s1 = self.properties.get(prop).copied().unwrap_or(0.0);
            let s2 = other.properties.get(prop).copied().unwrap_or(0.0);

            intersection += s1.min(s2);
            union += s1.max(s2);
        }

        if union == 0.0 {
            1.0
        } else {
            intersection / union
        }
    }

    /// Embedding-based similarity (cosine similarity)
    pub fn embedding_similarity(&self, other: &Concept) -> Option<f64> {
        match (&self.embedding, &other.embedding) {
            (Some(e1), Some(e2)) => Some(cosine_similarity(e1, e2)),
            _ => None,
        }
    }

    /// Compose with another concept (intersection of properties)
    pub fn compose(&self, other: &Concept) -> Concept {
        let name = format!("{}_{}", self.name, other.name);
        let mut concept = Concept::new(name);

        // Intersection of properties (minimum strength)
        for (prop, &strength1) in &self.properties {
            if let Some(&strength2) = other.properties.get(prop) {
                concept
                    .properties
                    .insert(prop.clone(), strength1.min(strength2));
            }
        }

        // Combined examples
        concept.examples.extend(self.examples.clone());
        concept.examples.extend(other.examples.clone());

        // Average confidence
        concept.confidence = (self.confidence + other.confidence) / 2.0;

        concept
    }

    /// Generalize from this concept (weaken constraints)
    pub fn generalize(&self, relaxation: f64) -> Concept {
        let mut generalized = self.clone();
        generalized.name = format!("{}_generalized", self.name);

        // Reduce property strengths
        for strength in generalized.properties.values_mut() {
            *strength = (*strength * (1.0 - relaxation)).max(0.1);
        }

        // Reduce confidence
        generalized.confidence *= 1.0 - relaxation;

        generalized
    }

    /// Specialize this concept (strengthen constraints)
    pub fn specialize(&self, new_property: String, strength: f64) -> Concept {
        let mut specialized = self.clone();
        specialized.name = format!("{}_{}", self.name, new_property);
        specialized.properties.insert(new_property, strength);

        // Increase confidence for more specific concepts
        specialized.confidence = (specialized.confidence * 1.1).min(1.0);

        specialized
    }
}

impl fmt::Display for Concept {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Concept '{}' (conf: {:.2})", self.name, self.confidence)?;
        if !self.properties.is_empty() {
            write!(f, " [")?;
            for (i, (prop, strength)) in self.properties.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}: {:.2}", prop, strength)?;
            }
            write!(f, "]")?;
        }
        Ok(())
    }
}

/// Relation between concepts
#[derive(Debug, Clone, PartialEq)]
pub enum ConceptRelation {
    /// Subconcept (is-a)
    IsA,

    /// Part-of (composition)
    PartOf,

    /// Similar to
    SimilarTo(f64),

    /// Opposite of
    OppositeOf,

    /// Custom relation
    Custom(String),
}

#[derive(Debug, Clone)]
/// Hierarchical concept graph
pub struct ConceptGraph {
    /// All concepts
    concepts: HashMap<String, Concept>,

    /// Relations between concepts
    relations: Vec<(String, ConceptRelation, String)>,
}

impl ConceptGraph {
    /// Create a new concept graph
    pub fn new() -> Self {
        ConceptGraph {
            concepts: HashMap::new(),
            relations: Vec::new(),
        }
    }

    /// Add a concept
    pub fn add_concept(&mut self, concept: Concept) {
        self.concepts.insert(concept.name.clone(), concept);
    }

    /// Get a concept
    pub fn get_concept(&self, name: &str) -> Option<&Concept> {
        self.concepts.get(name)
    }

    /// Get mutable concept
    pub fn get_concept_mut(&mut self, name: &str) -> Option<&mut Concept> {
        self.concepts.get_mut(name)
    }

    /// Add a relation
    pub fn add_relation(
        &mut self,
        from: impl Into<String>,
        relation: ConceptRelation,
        to: impl Into<String>,
    ) {
        self.relations.push((from.into(), relation, to.into()));
    }

    /// Add subconcept relation
    pub fn add_subconcept_relation(
        &mut self,
        subconcept: impl Into<String>,
        superconcept: impl Into<String>,
    ) {
        self.add_relation(subconcept, ConceptRelation::IsA, superconcept);
    }

    /// Get all subconcepts
    pub fn get_subconcepts(&self, concept_name: &str) -> Vec<&Concept> {
        let mut subconcepts = Vec::new();

        for (from, relation, to) in &self.relations {
            if to == concept_name && *relation == ConceptRelation::IsA {
                if let Some(concept) = self.concepts.get(from) {
                    subconcepts.push(concept);
                }
            }
        }

        subconcepts
    }

    /// Get all superconcepts (ancestors)
    pub fn get_superconcepts(&self, concept_name: &str) -> Vec<&Concept> {
        let mut superconcepts = Vec::new();
        let mut to_visit = vec![concept_name.to_string()];
        let mut visited = HashSet::new();

        while let Some(current) = to_visit.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            for (from, relation, to) in &self.relations {
                if from == &current && *relation == ConceptRelation::IsA {
                    if let Some(concept) = self.concepts.get(to) {
                        superconcepts.push(concept);
                        to_visit.push(to.clone());
                    }
                }
            }
        }

        superconcepts
    }

    /// Find similar concepts
    pub fn find_similar(&self, concept_name: &str, threshold: f64) -> Vec<(&Concept, f64)> {
        let concept = match self.concepts.get(concept_name) {
            Some(c) => c,
            None => return Vec::new(),
        };

        let mut similar = Vec::new();

        for other in self.concepts.values() {
            if other.name == concept_name {
                continue;
            }

            let similarity = concept.similarity(other);
            if similarity >= threshold {
                similar.push((other, similarity));
            }
        }

        // Sort by similarity (descending)
        similar.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        similar
    }

    /// Compose two concepts
    pub fn compose_concepts(&self, name1: &str, name2: &str) -> Option<Concept> {
        let c1 = self.concepts.get(name1)?;
        let c2 = self.concepts.get(name2)?;
        Some(c1.compose(c2))
    }

    /// Number of concepts
    pub fn num_concepts(&self) -> usize {
        self.concepts.len()
    }

    /// Number of relations
    pub fn num_relations(&self) -> usize {
        self.relations.len()
    }
}

impl Default for ConceptGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Concept learner: extract concepts from examples
pub struct ConceptLearner {
    /// Minimum examples to form a concept
    min_examples: usize,

    /// Minimum property frequency
    min_property_frequency: f64,

    /// Learned concepts
    learned_concepts: Vec<Concept>,
}

impl ConceptLearner {
    /// Create a new concept learner
    pub fn new() -> Self {
        ConceptLearner {
            min_examples: 3,
            min_property_frequency: 0.5,
            learned_concepts: Vec::new(),
        }
    }

    /// Set minimum examples
    pub fn with_min_examples(mut self, min: usize) -> Self {
        self.min_examples = min;
        self
    }

    /// Set minimum property frequency
    pub fn with_min_frequency(mut self, freq: f64) -> Self {
        self.min_property_frequency = freq;
        self
    }

    /// Learn a concept from examples (property vectors)
    pub fn learn_from_examples(
        &mut self,
        name: impl Into<String>,
        examples: &[(String, HashMap<String, f64>)],
    ) -> Result<Concept, String> {
        if examples.len() < self.min_examples {
            return Err(format!(
                "Not enough examples: {} < {}",
                examples.len(),
                self.min_examples
            ));
        }

        let name = name.into();
        let mut concept = Concept::new(&name);

        // Count property occurrences
        let mut property_sums: HashMap<String, f64> = HashMap::new();
        let mut property_counts: HashMap<String, usize> = HashMap::new();

        for (_example_name, properties) in examples {
            for (prop, &value) in properties {
                *property_sums.entry(prop.clone()).or_insert(0.0) += value;
                *property_counts.entry(prop.clone()).or_insert(0) += 1;
            }
        }

        // Compute average strengths
        let n = examples.len() as f64;
        for (prop, sum) in property_sums {
            let count = property_counts[&prop];
            let frequency = count as f64 / n;

            if frequency >= self.min_property_frequency {
                let avg_strength = sum / count as f64;
                concept = concept.with_property(prop, avg_strength);
            }
        }

        // Add examples
        for (example_name, _) in examples {
            concept = concept.with_example(example_name);
        }

        // Confidence based on number of examples and property agreement
        let confidence =
            (examples.len() as f64 / (self.min_examples as f64 + 10.0)).min(1.0) * 0.8 + 0.2; // Base confidence 0.2
        concept = concept.with_confidence(confidence);

        self.learned_concepts.push(concept.clone());

        Ok(concept)
    }

    /// Learn from knowledge graph
    pub fn learn_from_knowledge_graph(
        &mut self,
        graph: &KnowledgeGraph,
        entity_type: crate::knowledge_graph::EntityType,
    ) -> Vec<Concept> {
        let mut concepts = Vec::new();

        // Group entities by type
        let entities = graph.find_entities_by_type(&entity_type);

        if entities.len() < self.min_examples {
            return concepts;
        }

        // Extract properties from graph
        let mut examples = Vec::new();
        for entity in entities {
            let mut properties = HashMap::new();

            // Count relations
            let outgoing = graph.query(Some(entity.id), None, None);
            let incoming = graph.query(None, None, Some(entity.id));

            properties.insert("num_outgoing".to_string(), outgoing.len() as f64);
            properties.insert("num_incoming".to_string(), incoming.len() as f64);

            // Specific relation types
            for triple in outgoing {
                let rel_name = format!("has_{:?}", triple.predicate);
                *properties.entry(rel_name).or_insert(0.0) += 1.0;
            }

            examples.push((entity.name.clone(), properties));
        }

        // Learn concept
        let concept_name = format!("{:?}_concept", entity_type);
        if let Ok(concept) = self.learn_from_examples(concept_name, &examples) {
            concepts.push(concept);
        }

        concepts
    }

    /// Get all learned concepts
    pub fn learned_concepts(&self) -> &[Concept] {
        &self.learned_concepts
    }
}

impl Default for ConceptLearner {
    fn default() -> Self {
        Self::new()
    }
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concept_creation() {
        let concept = Concept::new("Controller")
            .with_property("handles_http", 0.9)
            .with_property("depends_on_service", 0.8)
            .with_example("UserController");

        assert_eq!(concept.name, "Controller");
        assert_eq!(concept.get_property("handles_http"), 0.9);
        assert!(concept.has_property("handles_http"));
        assert_eq!(concept.num_examples(), 1);
    }

    #[test]
    fn test_concept_similarity() {
        let c1 = Concept::new("Controller")
            .with_property("handles_http", 0.9)
            .with_property("depends_on_service", 0.8);

        let c2 = Concept::new("APIController")
            .with_property("handles_http", 0.95)
            .with_property("depends_on_service", 0.7);

        let similarity = c1.similarity(&c2);
        assert!(similarity > 0.7);
    }

    #[test]
    fn test_concept_composition() {
        let c1 = Concept::new("WebComponent")
            .with_property("is_web", 1.0)
            .with_property("handles_requests", 0.8);

        let c2 = Concept::new("DataComponent")
            .with_property("accesses_db", 1.0)
            .with_property("handles_requests", 0.9);

        let composed = c1.compose(&c2);

        // Should have intersection of properties
        assert!(composed.get_property("handles_requests") > 0.0);
        assert_eq!(composed.get_property("is_web"), 0.0); // Not in c2
    }

    #[test]
    fn test_concept_generalization() {
        let concept = Concept::new("UserController")
            .with_property("handles_users", 0.9)
            .with_property("validates", 0.8);

        let generalized = concept.generalize(0.3);

        // Properties should be weaker
        assert!(generalized.get_property("handles_users") < concept.get_property("handles_users"));
        assert!(generalized.confidence < concept.confidence);
    }

    #[test]
    fn test_concept_specialization() {
        let concept = Concept::new("Controller").with_property("handles_requests", 0.8);

        let specialized = concept.specialize("handles_users".to_string(), 0.9);

        // Should have new property
        assert_eq!(specialized.get_property("handles_users"), 0.9);
        assert!(specialized.name.contains("handles_users"));
    }

    #[test]
    fn test_concept_graph() {
        let mut graph = ConceptGraph::new();

        let class_concept = Concept::new("Class");
        let controller_concept = Concept::new("Controller");
        let service_concept = Concept::new("Service");

        graph.add_concept(class_concept);
        graph.add_concept(controller_concept);
        graph.add_concept(service_concept);

        // Build hierarchy
        graph.add_subconcept_relation("Controller", "Class");
        graph.add_subconcept_relation("Service", "Class");

        assert_eq!(graph.num_concepts(), 3);
        assert_eq!(graph.num_relations(), 2);

        // Check hierarchy
        let subconcepts = graph.get_subconcepts("Class");
        assert_eq!(subconcepts.len(), 2);
    }

    #[test]
    fn test_find_similar_concepts() {
        let mut graph = ConceptGraph::new();

        let c1 = Concept::new("UserController")
            .with_property("handles_http", 0.9)
            .with_property("validates_users", 0.8);

        let c2 = Concept::new("PostController")
            .with_property("handles_http", 0.95)
            .with_property("validates_posts", 0.7);

        let c3 = Concept::new("Repository").with_property("accesses_db", 0.9);

        graph.add_concept(c1);
        graph.add_concept(c2);
        graph.add_concept(c3);

        let similar = graph.find_similar("UserController", 0.3);

        // UserController should be similar to PostController, not Repository
        assert!(!similar.is_empty());
        assert_eq!(similar[0].0.name, "PostController");
    }

    #[test]
    fn test_concept_learner() {
        let mut learner = ConceptLearner::new()
            .with_min_examples(2)
            .with_min_frequency(0.5);

        let mut examples = Vec::new();

        // Example 1
        let mut props1 = HashMap::new();
        props1.insert("handles_http".to_string(), 0.9);
        props1.insert("depends_on_service".to_string(), 0.8);
        examples.push(("UserController".to_string(), props1));

        // Example 2
        let mut props2 = HashMap::new();
        props2.insert("handles_http".to_string(), 0.95);
        props2.insert("depends_on_service".to_string(), 0.7);
        examples.push(("PostController".to_string(), props2));

        // Example 3
        let mut props3 = HashMap::new();
        props3.insert("handles_http".to_string(), 0.85);
        props3.insert("validates_input".to_string(), 0.9);
        examples.push(("CommentController".to_string(), props3));

        let concept = learner
            .learn_from_examples("Controller", &examples)
            .unwrap();

        // Should have handles_http (present in all examples)
        assert!(concept.has_property("handles_http"));

        // depends_on_service only in 2/3 examples (66%)
        assert!(concept.get_property("depends_on_service") > 0.0);

        assert_eq!(concept.num_examples(), 3);
    }

    #[test]
    fn test_embedding_similarity() {
        let c1 = Concept::new("Concept1").with_embedding(vec![1.0, 0.0, 0.0]);

        let c2 = Concept::new("Concept2").with_embedding(vec![0.8, 0.6, 0.0]);

        let similarity = c1.embedding_similarity(&c2).unwrap();
        assert!(similarity > 0.6); // Should be similar
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);

        let c = vec![0.0, 1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &c), 0.0);

        let d = vec![1.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &d);
        assert!(sim > 0.7 && sim < 0.72);
    }
}

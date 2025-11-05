// Knowledge Graph
// Stores entities and triples, supports efficient querying
//
// Structure:
// - Entities: HashMap of EntityId → Entity
// - Triples: Vec of all facts
// - Indexes: For efficient querying by subject/predicate/object
//
// This is optimized for:
// - Fast triple insertion O(1)
// - Fast queries by subject/predicate/object O(log n)
// - Reasonable memory usage

use std::collections::{HashMap, HashSet};
use super::triple::{Entity, EntityId, EntityType, RelationType, Triple};

/// Knowledge Graph - stores entities and their relationships
pub struct KnowledgeGraph {
    /// All entities in the graph
    entities: HashMap<EntityId, Entity>,

    /// All triples (facts) in the graph
    triples: Vec<Triple>,

    /// Index: subject → list of triple indices
    subject_index: HashMap<EntityId, Vec<usize>>,

    /// Index: predicate → list of triple indices
    predicate_index: HashMap<RelationType, Vec<usize>>,

    /// Index: object → list of triple indices
    object_index: HashMap<EntityId, Vec<usize>>,

    /// Next available entity ID
    next_entity_id: EntityId,
}

impl KnowledgeGraph {
    /// Create a new empty knowledge graph
    pub fn new() -> Self {
        KnowledgeGraph {
            entities: HashMap::new(),
            triples: Vec::new(),
            subject_index: HashMap::new(),
            predicate_index: HashMap::new(),
            object_index: HashMap::new(),
            next_entity_id: 0,
        }
    }

    /// Add a new entity to the graph
    ///
    /// # Returns
    /// EntityId of the newly created entity
    pub fn add_entity(&mut self, entity_type: EntityType, name: String) -> EntityId {
        let id = self.next_entity_id;
        self.next_entity_id += 1;

        let entity = Entity::new(id, entity_type, name);
        self.entities.insert(id, entity);

        id
    }

    /// Get entity by ID
    pub fn get_entity(&self, id: EntityId) -> Option<&Entity> {
        self.entities.get(&id)
    }

    /// Get mutable entity by ID
    pub fn get_entity_mut(&mut self, id: EntityId) -> Option<&mut Entity> {
        self.entities.get_mut(&id)
    }

    /// Find entities by name
    pub fn find_entities_by_name(&self, name: &str) -> Vec<&Entity> {
        self.entities
            .values()
            .filter(|e| e.name == name)
            .collect()
    }

    /// Find entities by type
    pub fn find_entities_by_type(&self, entity_type: &EntityType) -> Vec<&Entity> {
        self.entities
            .values()
            .filter(|e| &e.entity_type == entity_type)
            .collect()
    }

    /// Add a triple to the graph
    pub fn add_triple(&mut self, triple: Triple) {
        // Validate that entities exist
        if !self.entities.contains_key(&triple.subject) {
            panic!("Subject entity {} does not exist", triple.subject);
        }
        if !self.entities.contains_key(&triple.object) {
            panic!("Object entity {} does not exist", triple.object);
        }

        let triple_idx = self.triples.len();

        // Update indexes
        self.subject_index
            .entry(triple.subject)
            .or_insert_with(Vec::new)
            .push(triple_idx);

        self.predicate_index
            .entry(triple.predicate.clone())
            .or_insert_with(Vec::new)
            .push(triple_idx);

        self.object_index
            .entry(triple.object)
            .or_insert_with(Vec::new)
            .push(triple_idx);

        // Store triple
        self.triples.push(triple);
    }

    /// Query triples by pattern (None = wildcard)
    ///
    /// # Example
    /// ```
    /// // Find all triples where entity 1 is the subject
    /// graph.query(Some(1), None, None);
    ///
    /// // Find all "Inherits" relationships
    /// graph.query(None, Some(&RelationType::Inherits), None);
    ///
    /// // Find what entity 5 inherits from
    /// graph.query(Some(5), Some(&RelationType::Inherits), None);
    /// ```
    pub fn query(
        &self,
        subject: Option<EntityId>,
        predicate: Option<&RelationType>,
        object: Option<EntityId>,
    ) -> Vec<&Triple> {
        // Use indexes to narrow down search space
        let candidate_indices = if let Some(s) = subject {
            self.subject_index.get(&s).map(|v| v.as_slice())
        } else if let Some(p) = predicate {
            self.predicate_index.get(p).map(|v| v.as_slice())
        } else if let Some(o) = object {
            self.object_index.get(&o).map(|v| v.as_slice())
        } else {
            // No filters - return all
            None
        };

        // If we have candidates from index, filter them
        if let Some(indices) = candidate_indices {
            indices
                .iter()
                .map(|&idx| &self.triples[idx])
                .filter(|t| t.matches(subject, predicate, object))
                .collect()
        } else {
            // No index available, scan all triples
            self.triples
                .iter()
                .filter(|t| t.matches(subject, predicate, object))
                .collect()
        }
    }

    /// Get all entities that entity `id` has relation `predicate` with
    ///
    /// # Example
    /// ```
    /// // Get all classes that User inherits from
    /// graph.get_related(user_id, &RelationType::Inherits);
    /// ```
    pub fn get_related(&self, id: EntityId, predicate: &RelationType) -> Vec<EntityId> {
        self.query(Some(id), Some(predicate), None)
            .into_iter()
            .map(|t| t.object)
            .collect()
    }

    /// Get all entities that have relation `predicate` with entity `id`
    ///
    /// # Example
    /// ```
    /// // Get all classes that inherit from Entity
    /// graph.get_inverse_related(entity_id, &RelationType::Inherits);
    /// ```
    pub fn get_inverse_related(&self, id: EntityId, predicate: &RelationType) -> Vec<EntityId> {
        self.query(None, Some(predicate), Some(id))
            .into_iter()
            .map(|t| t.subject)
            .collect()
    }

    /// Find all paths between two entities (BFS)
    ///
    /// Returns list of paths, where each path is a list of entity IDs
    pub fn find_paths(&self, from: EntityId, to: EntityId, max_depth: usize) -> Vec<Vec<EntityId>> {
        let mut paths = Vec::new();
        let mut queue = vec![(from, vec![from])];
        let mut visited = HashSet::new();

        while let Some((current, path)) = queue.pop() {
            if current == to {
                paths.push(path);
                continue;
            }

            if path.len() > max_depth {
                continue;
            }

            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            // Get all outgoing edges from current
            if let Some(indices) = self.subject_index.get(&current) {
                for &idx in indices {
                    let triple = &self.triples[idx];
                    let next = triple.object;

                    if !path.contains(&next) {
                        let mut new_path = path.clone();
                        new_path.push(next);
                        queue.push((next, new_path));
                    }
                }
            }
        }

        paths
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        GraphStats {
            num_entities: self.entities.len(),
            num_triples: self.triples.len(),
            num_relations: self.predicate_index.len(),
        }
    }

    /// Get number of entities
    pub fn num_entities(&self) -> usize {
        self.entities.len()
    }

    /// Get number of triples
    pub fn num_triples(&self) -> usize {
        self.triples.len()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.entities.clear();
        self.triples.clear();
        self.subject_index.clear();
        self.predicate_index.clear();
        self.object_index.clear();
        self.next_entity_id = 0;
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub num_entities: usize,
    pub num_triples: usize,
    pub num_relations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = KnowledgeGraph::new();
        assert_eq!(graph.num_entities(), 0);
        assert_eq!(graph.num_triples(), 0);
    }

    #[test]
    fn test_add_entity() {
        let mut graph = KnowledgeGraph::new();

        let id1 = graph.add_entity(EntityType::Class, "User".to_string());
        let id2 = graph.add_entity(EntityType::Class, "Entity".to_string());

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(graph.num_entities(), 2);

        let entity1 = graph.get_entity(id1).unwrap();
        assert_eq!(entity1.name, "User");
        assert_eq!(entity1.entity_type, EntityType::Class);
    }

    #[test]
    fn test_find_entities_by_name() {
        let mut graph = KnowledgeGraph::new();

        graph.add_entity(EntityType::Class, "User".to_string());
        graph.add_entity(EntityType::Function, "login".to_string());
        graph.add_entity(EntityType::Class, "User".to_string()); // Duplicate name

        let users = graph.find_entities_by_name("User");
        assert_eq!(users.len(), 2);

        let logins = graph.find_entities_by_name("login");
        assert_eq!(logins.len(), 1);
    }

    #[test]
    fn test_find_entities_by_type() {
        let mut graph = KnowledgeGraph::new();

        graph.add_entity(EntityType::Class, "User".to_string());
        graph.add_entity(EntityType::Class, "Post".to_string());
        graph.add_entity(EntityType::Function, "login".to_string());

        let classes = graph.find_entities_by_type(&EntityType::Class);
        assert_eq!(classes.len(), 2);

        let functions = graph.find_entities_by_type(&EntityType::Function);
        assert_eq!(functions.len(), 1);
    }

    #[test]
    fn test_add_triple() {
        let mut graph = KnowledgeGraph::new();

        let user = graph.add_entity(EntityType::Class, "User".to_string());
        let entity = graph.add_entity(EntityType::Class, "Entity".to_string());

        let triple = Triple::new(user, RelationType::Inherits, entity);
        graph.add_triple(triple);

        assert_eq!(graph.num_triples(), 1);
    }

    #[test]
    #[should_panic(expected = "Subject entity")]
    fn test_add_triple_invalid_subject() {
        let mut graph = KnowledgeGraph::new();

        let entity = graph.add_entity(EntityType::Class, "Entity".to_string());
        let triple = Triple::new(999, RelationType::Inherits, entity);

        graph.add_triple(triple); // Should panic
    }

    #[test]
    fn test_query_by_subject() {
        let mut graph = KnowledgeGraph::new();

        let user = graph.add_entity(EntityType::Class, "User".to_string());
        let entity = graph.add_entity(EntityType::Class, "Entity".to_string());
        let model = graph.add_entity(EntityType::Class, "Model".to_string());

        graph.add_triple(Triple::new(user, RelationType::Inherits, entity));
        graph.add_triple(Triple::new(user, RelationType::Implements, model));

        let results = graph.query(Some(user), None, None);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_predicate() {
        let mut graph = KnowledgeGraph::new();

        let user = graph.add_entity(EntityType::Class, "User".to_string());
        let post = graph.add_entity(EntityType::Class, "Post".to_string());
        let entity = graph.add_entity(EntityType::Class, "Entity".to_string());

        graph.add_triple(Triple::new(user, RelationType::Inherits, entity));
        graph.add_triple(Triple::new(post, RelationType::Inherits, entity));

        let results = graph.query(None, Some(&RelationType::Inherits), None);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_object() {
        let mut graph = KnowledgeGraph::new();

        let user = graph.add_entity(EntityType::Class, "User".to_string());
        let post = graph.add_entity(EntityType::Class, "Post".to_string());
        let entity = graph.add_entity(EntityType::Class, "Entity".to_string());

        graph.add_triple(Triple::new(user, RelationType::Inherits, entity));
        graph.add_triple(Triple::new(post, RelationType::Inherits, entity));

        let results = graph.query(None, None, Some(entity));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_get_related() {
        let mut graph = KnowledgeGraph::new();

        let user = graph.add_entity(EntityType::Class, "User".to_string());
        let entity = graph.add_entity(EntityType::Class, "Entity".to_string());
        let model = graph.add_entity(EntityType::Class, "Model".to_string());

        graph.add_triple(Triple::new(user, RelationType::Inherits, entity));
        graph.add_triple(Triple::new(user, RelationType::Implements, model));

        let inherits = graph.get_related(user, &RelationType::Inherits);
        assert_eq!(inherits.len(), 1);
        assert_eq!(inherits[0], entity);

        let implements = graph.get_related(user, &RelationType::Implements);
        assert_eq!(implements.len(), 1);
        assert_eq!(implements[0], model);
    }

    #[test]
    fn test_get_inverse_related() {
        let mut graph = KnowledgeGraph::new();

        let user = graph.add_entity(EntityType::Class, "User".to_string());
        let post = graph.add_entity(EntityType::Class, "Post".to_string());
        let entity = graph.add_entity(EntityType::Class, "Entity".to_string());

        graph.add_triple(Triple::new(user, RelationType::Inherits, entity));
        graph.add_triple(Triple::new(post, RelationType::Inherits, entity));

        let children = graph.get_inverse_related(entity, &RelationType::Inherits);
        assert_eq!(children.len(), 2);
        assert!(children.contains(&user));
        assert!(children.contains(&post));
    }

    #[test]
    fn test_find_paths() {
        let mut graph = KnowledgeGraph::new();

        // A → B → C → D
        let a = graph.add_entity(EntityType::Class, "A".to_string());
        let b = graph.add_entity(EntityType::Class, "B".to_string());
        let c = graph.add_entity(EntityType::Class, "C".to_string());
        let d = graph.add_entity(EntityType::Class, "D".to_string());

        graph.add_triple(Triple::new(a, RelationType::Calls, b));
        graph.add_triple(Triple::new(b, RelationType::Calls, c));
        graph.add_triple(Triple::new(c, RelationType::Calls, d));

        let paths = graph.find_paths(a, d, 10);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec![a, b, c, d]);
    }

    #[test]
    fn test_graph_stats() {
        let mut graph = KnowledgeGraph::new();

        let user = graph.add_entity(EntityType::Class, "User".to_string());
        let entity = graph.add_entity(EntityType::Class, "Entity".to_string());

        graph.add_triple(Triple::new(user, RelationType::Inherits, entity));

        let stats = graph.stats();
        assert_eq!(stats.num_entities, 2);
        assert_eq!(stats.num_triples, 1);
        assert_eq!(stats.num_relations, 1);
    }

    #[test]
    fn test_clear() {
        let mut graph = KnowledgeGraph::new();

        let user = graph.add_entity(EntityType::Class, "User".to_string());
        let entity = graph.add_entity(EntityType::Class, "Entity".to_string());
        graph.add_triple(Triple::new(user, RelationType::Inherits, entity));

        graph.clear();

        assert_eq!(graph.num_entities(), 0);
        assert_eq!(graph.num_triples(), 0);
    }
}

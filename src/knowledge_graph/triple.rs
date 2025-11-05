// Knowledge Graph Triple
// Represents a single fact: (Subject, Predicate, Object)
//
// Examples:
// - (Class::User, Inherits, Class::Entity)
// - (Function::login, Calls, Function::authenticate)
// - (Module::auth, DependsOn, Module::database)
//
// This is the fundamental unit of knowledge representation in knowledge graphs.

use std::fmt;

/// Entity ID - unique identifier for nodes in the graph
pub type EntityId = usize;

/// Relation type - represents the type of relationship between entities
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RelationType {
    // Code structure relations
    Inherits,
    Implements,
    Contains,
    Uses,
    Calls,
    DependsOn,
    Returns,
    Takes,

    // Type relations
    HasType,
    IsA,

    // Architectural relations
    LayerAbove,
    LayerBelow,
    Violates,

    // Custom relation (for extensibility)
    Custom(String),
}

impl fmt::Display for RelationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RelationType::Inherits => write!(f, "inherits"),
            RelationType::Implements => write!(f, "implements"),
            RelationType::Contains => write!(f, "contains"),
            RelationType::Uses => write!(f, "uses"),
            RelationType::Calls => write!(f, "calls"),
            RelationType::DependsOn => write!(f, "dependsOn"),
            RelationType::Returns => write!(f, "returns"),
            RelationType::Takes => write!(f, "takes"),
            RelationType::HasType => write!(f, "hasType"),
            RelationType::IsA => write!(f, "isA"),
            RelationType::LayerAbove => write!(f, "layerAbove"),
            RelationType::LayerBelow => write!(f, "layerBelow"),
            RelationType::Violates => write!(f, "violates"),
            RelationType::Custom(s) => write!(f, "{}", s),
        }
    }
}

impl RelationType {
    /// Parse relation type from string
    pub fn from_str(s: &str) -> Self {
        match s {
            "inherits" => RelationType::Inherits,
            "implements" => RelationType::Implements,
            "contains" => RelationType::Contains,
            "uses" => RelationType::Uses,
            "calls" => RelationType::Calls,
            "dependsOn" => RelationType::DependsOn,
            "returns" => RelationType::Returns,
            "takes" => RelationType::Takes,
            "hasType" => RelationType::HasType,
            "isA" => RelationType::IsA,
            "layerAbove" => RelationType::LayerAbove,
            "layerBelow" => RelationType::LayerBelow,
            "violates" => RelationType::Violates,
            _ => RelationType::Custom(s.to_string()),
        }
    }
}

/// Entity - represents a node in the knowledge graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Entity {
    pub id: EntityId,
    pub entity_type: EntityType,
    pub name: String,
    pub metadata: Vec<(String, String)>,
}

/// Entity type - what kind of code entity this is
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntityType {
    Class,
    Function,
    Method,
    Variable,
    Module,
    Package,
    Interface,
    Trait,
    Struct,
    Enum,
    Type,
    Concept,  // Abstract concept (for meta-learning)
}

impl fmt::Display for EntityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntityType::Class => write!(f, "Class"),
            EntityType::Function => write!(f, "Function"),
            EntityType::Method => write!(f, "Method"),
            EntityType::Variable => write!(f, "Variable"),
            EntityType::Module => write!(f, "Module"),
            EntityType::Package => write!(f, "Package"),
            EntityType::Interface => write!(f, "Interface"),
            EntityType::Trait => write!(f, "Trait"),
            EntityType::Struct => write!(f, "Struct"),
            EntityType::Enum => write!(f, "Enum"),
            EntityType::Type => write!(f, "Type"),
            EntityType::Concept => write!(f, "Concept"),
        }
    }
}

impl Entity {
    /// Create a new entity
    pub fn new(id: EntityId, entity_type: EntityType, name: String) -> Self {
        Entity {
            id,
            entity_type,
            name,
            metadata: Vec::new(),
        }
    }

    /// Add metadata to entity
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.push((key, value));
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// Triple - fundamental knowledge representation (Subject-Predicate-Object)
#[derive(Debug, Clone, PartialEq)]
pub struct Triple {
    pub subject: EntityId,
    pub predicate: RelationType,
    pub object: EntityId,
    pub confidence: Option<f64>,  // Optional confidence score (for fuzzy logic)
}

impl Triple {
    /// Create a new triple
    pub fn new(subject: EntityId, predicate: RelationType, object: EntityId) -> Self {
        Triple {
            subject,
            predicate,
            object,
            confidence: None,
        }
    }

    /// Create triple with confidence score
    pub fn with_confidence(
        subject: EntityId,
        predicate: RelationType,
        object: EntityId,
        confidence: f64,
    ) -> Self {
        Triple {
            subject,
            predicate,
            object,
            confidence: Some(confidence),
        }
    }

    /// Check if triple matches pattern (None = wildcard)
    pub fn matches(
        &self,
        subject: Option<EntityId>,
        predicate: Option<&RelationType>,
        object: Option<EntityId>,
    ) -> bool {
        let subject_match = subject.map_or(true, |s| s == self.subject);
        let predicate_match = predicate.map_or(true, |p| p == &self.predicate);
        let object_match = object.map_or(true, |o| o == self.object);

        subject_match && predicate_match && object_match
    }
}

impl fmt::Display for Triple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(conf) = self.confidence {
            write!(
                f,
                "({}, {}, {}) [confidence: {:.2}]",
                self.subject, self.predicate, self.object, conf
            )
        } else {
            write!(f, "({}, {}, {})", self.subject, self.predicate, self.object)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_creation() {
        let entity = Entity::new(1, EntityType::Class, "User".to_string());

        assert_eq!(entity.id, 1);
        assert_eq!(entity.entity_type, EntityType::Class);
        assert_eq!(entity.name, "User");
        assert_eq!(entity.metadata.len(), 0);
    }

    #[test]
    fn test_entity_metadata() {
        let mut entity = Entity::new(1, EntityType::Function, "login".to_string());

        entity.add_metadata("file".to_string(), "auth.rs".to_string());
        entity.add_metadata("line".to_string(), "42".to_string());

        assert_eq!(entity.get_metadata("file"), Some("auth.rs"));
        assert_eq!(entity.get_metadata("line"), Some("42"));
        assert_eq!(entity.get_metadata("nonexistent"), None);
    }

    #[test]
    fn test_triple_creation() {
        let triple = Triple::new(1, RelationType::Inherits, 2);

        assert_eq!(triple.subject, 1);
        assert_eq!(triple.predicate, RelationType::Inherits);
        assert_eq!(triple.object, 2);
        assert_eq!(triple.confidence, None);
    }

    #[test]
    fn test_triple_with_confidence() {
        let triple = Triple::with_confidence(1, RelationType::Calls, 2, 0.95);

        assert_eq!(triple.confidence, Some(0.95));
    }

    #[test]
    fn test_triple_matches() {
        let triple = Triple::new(1, RelationType::Inherits, 2);

        // Exact match
        assert!(triple.matches(Some(1), Some(&RelationType::Inherits), Some(2)));

        // Wildcard subject
        assert!(triple.matches(None, Some(&RelationType::Inherits), Some(2)));

        // Wildcard predicate
        assert!(triple.matches(Some(1), None, Some(2)));

        // Wildcard object
        assert!(triple.matches(Some(1), Some(&RelationType::Inherits), None));

        // All wildcards
        assert!(triple.matches(None, None, None));

        // No match
        assert!(!triple.matches(Some(3), Some(&RelationType::Inherits), Some(2)));
        assert!(!triple.matches(Some(1), Some(&RelationType::Calls), Some(2)));
        assert!(!triple.matches(Some(1), Some(&RelationType::Inherits), Some(3)));
    }

    #[test]
    fn test_relation_type_display() {
        assert_eq!(format!("{}", RelationType::Inherits), "inherits");
        assert_eq!(format!("{}", RelationType::Calls), "calls");
        assert_eq!(format!("{}", RelationType::Custom("customRel".to_string())), "customRel");
    }

    #[test]
    fn test_relation_type_from_str() {
        assert_eq!(RelationType::from_str("inherits"), RelationType::Inherits);
        assert_eq!(RelationType::from_str("calls"), RelationType::Calls);
        assert_eq!(RelationType::from_str("customRel"), RelationType::Custom("customRel".to_string()));
    }

    #[test]
    fn test_entity_type_display() {
        assert_eq!(format!("{}", EntityType::Class), "Class");
        assert_eq!(format!("{}", EntityType::Function), "Function");
        assert_eq!(format!("{}", EntityType::Module), "Module");
    }

    #[test]
    fn test_triple_display() {
        let triple1 = Triple::new(1, RelationType::Inherits, 2);
        assert_eq!(format!("{}", triple1), "(1, inherits, 2)");

        let triple2 = Triple::with_confidence(1, RelationType::Calls, 2, 0.87);
        assert_eq!(format!("{}", triple2), "(1, calls, 2) [confidence: 0.87]");
    }
}

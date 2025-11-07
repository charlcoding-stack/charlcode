// Symbolic Scene Understanding
//
// Converts visual scenes into symbolic representations with objects,
// spatial relationships, and temporal reasoning.
//
// Key capabilities:
// 1. Object detection → symbolic entities
// 2. Spatial relationship extraction (on, above, left_of, etc.)
// 3. Scene graph generation
// 4. Temporal reasoning (before, after, during)
//
// Example:
//   Scene: [Cat on mat, ball next to cat]
//   → SceneGraph:
//       Objects: {cat, mat, ball}
//       Relations: {on(cat, mat), next_to(ball, cat)}
//
// References:
// - Krishna et al. (2017): "Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations"
// - Johnson et al. (2015): "Image retrieval using scene graphs"

use std::collections::HashMap;

/// Detected object in a scene
#[derive(Debug, Clone, PartialEq)]
pub struct SceneObject {
    /// Unique identifier
    pub id: String,
    /// Object class/type
    pub class: String,
    /// Bounding box (x, y, width, height) - normalized [0, 1]
    pub bbox: (f32, f32, f32, f32),
    /// Detection confidence
    pub confidence: f32,
    /// Object attributes
    pub attributes: Vec<String>,
}

impl SceneObject {
    pub fn new(
        id: impl Into<String>,
        class: impl Into<String>,
        bbox: (f32, f32, f32, f32),
    ) -> Self {
        Self {
            id: id.into(),
            class: class.into(),
            bbox,
            confidence: 1.0,
            attributes: Vec::new(),
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_attribute(mut self, attribute: impl Into<String>) -> Self {
        self.attributes.push(attribute.into());
        self
    }

    /// Get object center position
    pub fn center(&self) -> (f32, f32) {
        let (x, y, w, h) = self.bbox;
        (x + w / 2.0, y + h / 2.0)
    }

    /// Get object area
    pub fn area(&self) -> f32 {
        self.bbox.2 * self.bbox.3
    }
}

/// Spatial relationship types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpatialRelation {
    /// Object A is on top of object B
    On,
    /// Object A is above object B (but not touching)
    Above,
    /// Object A is below object B
    Below,
    /// Object A is to the left of object B
    LeftOf,
    /// Object A is to the right of object B
    RightOf,
    /// Object A is near/next to object B
    Near,
    /// Object A contains object B
    Contains,
    /// Object A is inside object B
    Inside,
}

impl SpatialRelation {
    pub fn as_str(&self) -> &'static str {
        match self {
            SpatialRelation::On => "on",
            SpatialRelation::Above => "above",
            SpatialRelation::Below => "below",
            SpatialRelation::LeftOf => "left_of",
            SpatialRelation::RightOf => "right_of",
            SpatialRelation::Near => "near",
            SpatialRelation::Contains => "contains",
            SpatialRelation::Inside => "inside",
        }
    }

    /// Inverse relation
    pub fn inverse(&self) -> SpatialRelation {
        match self {
            SpatialRelation::On => SpatialRelation::Below,
            SpatialRelation::Above => SpatialRelation::Below,
            SpatialRelation::Below => SpatialRelation::Above,
            SpatialRelation::LeftOf => SpatialRelation::RightOf,
            SpatialRelation::RightOf => SpatialRelation::LeftOf,
            SpatialRelation::Near => SpatialRelation::Near,
            SpatialRelation::Contains => SpatialRelation::Inside,
            SpatialRelation::Inside => SpatialRelation::Contains,
        }
    }
}

/// Relationship between two objects
#[derive(Debug, Clone)]
pub struct ObjectRelation {
    /// Subject object ID
    pub subject: String,
    /// Relation type
    pub relation: SpatialRelation,
    /// Object ID
    pub object: String,
    /// Confidence score
    pub confidence: f32,
}

impl ObjectRelation {
    pub fn new(
        subject: impl Into<String>,
        relation: SpatialRelation,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            relation,
            object: object.into(),
            confidence: 1.0,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        format!(
            "{}({}, {})",
            self.relation.as_str(),
            self.subject,
            self.object
        )
    }
}

/// Scene graph: symbolic representation of a visual scene
#[derive(Debug, Clone)]
pub struct SceneGraph {
    /// All objects in the scene
    pub objects: HashMap<String, SceneObject>,
    /// All relationships between objects
    pub relations: Vec<ObjectRelation>,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
            relations: Vec::new(),
        }
    }

    /// Add object to scene
    pub fn add_object(&mut self, object: SceneObject) {
        self.objects.insert(object.id.clone(), object);
    }

    /// Add relation between objects
    pub fn add_relation(&mut self, relation: ObjectRelation) -> Result<(), String> {
        // Verify both objects exist
        if !self.objects.contains_key(&relation.subject) {
            return Err(format!("Subject object '{}' not found", relation.subject));
        }
        if !self.objects.contains_key(&relation.object) {
            return Err(format!("Object '{}' not found", relation.object));
        }

        self.relations.push(relation);
        Ok(())
    }

    /// Get all relations for an object (as subject)
    pub fn get_relations_for(&self, object_id: &str) -> Vec<&ObjectRelation> {
        self.relations
            .iter()
            .filter(|r| r.subject == object_id)
            .collect()
    }

    /// Get all objects with a specific relation to an object
    pub fn get_objects_with_relation(
        &self,
        object_id: &str,
        relation: SpatialRelation,
    ) -> Vec<&SceneObject> {
        self.relations
            .iter()
            .filter(|r| r.object == object_id && r.relation == relation)
            .filter_map(|r| self.objects.get(&r.subject))
            .collect()
    }

    /// Query: What is on top of X?
    pub fn what_is_on(&self, object_id: &str) -> Vec<&SceneObject> {
        self.get_objects_with_relation(object_id, SpatialRelation::On)
    }

    /// Convert to natural language description
    pub fn to_description(&self) -> String {
        let mut parts = Vec::new();

        // List objects
        for obj in self.objects.values() {
            parts.push(format!("There is a {}", obj.class));
        }

        // List relations
        for rel in &self.relations {
            if let (Some(subj), Some(obj)) = (
                self.objects.get(&rel.subject),
                self.objects.get(&rel.object),
            ) {
                parts.push(format!(
                    "The {} is {} the {}",
                    subj.class,
                    rel.relation.as_str(),
                    obj.class
                ));
            }
        }

        parts.join(". ")
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Scene graph generator
#[derive(Debug, Clone)]
pub struct SceneGraphGenerator {
    /// Minimum confidence for object detection
    pub min_object_confidence: f32,
    /// Minimum confidence for relations
    pub min_relation_confidence: f32,
    /// Distance threshold for "near" relation (normalized units)
    pub near_threshold: f32,
}

impl SceneGraphGenerator {
    pub fn new() -> Self {
        Self {
            min_object_confidence: 0.5,
            min_relation_confidence: 0.7,
            near_threshold: 0.2,
        }
    }

    /// Generate scene graph from detected objects
    pub fn generate(&self, objects: Vec<SceneObject>) -> SceneGraph {
        let mut graph = SceneGraph::new();

        // Add objects (filter by confidence)
        for obj in objects {
            if obj.confidence >= self.min_object_confidence {
                graph.add_object(obj);
            }
        }

        // Infer spatial relations
        let object_ids: Vec<String> = graph.objects.keys().cloned().collect();
        for i in 0..object_ids.len() {
            for j in 0..object_ids.len() {
                if i == j {
                    continue;
                }

                let id_a = &object_ids[i];
                let id_b = &object_ids[j];

                if let Some(relation) = self.infer_relation(
                    graph.objects.get(id_a).unwrap(),
                    graph.objects.get(id_b).unwrap(),
                ) {
                    if relation.confidence >= self.min_relation_confidence {
                        let _ = graph.add_relation(relation);
                    }
                }
            }
        }

        graph
    }

    /// Infer spatial relation between two objects
    fn infer_relation(&self, obj_a: &SceneObject, obj_b: &SceneObject) -> Option<ObjectRelation> {
        let (ax, ay) = obj_a.center();
        let (bx, by) = obj_b.center();

        let dx = (ax - bx).abs();
        let dy = (ay - by).abs();

        // Near: objects are close
        if dx < self.near_threshold && dy < self.near_threshold {
            return Some(
                ObjectRelation::new(obj_a.id.clone(), SpatialRelation::Near, obj_b.id.clone())
                    .with_confidence(0.9),
            );
        }

        // Vertical relations
        if dy > dx {
            if ay < by {
                // A is above B
                return Some(
                    ObjectRelation::new(obj_a.id.clone(), SpatialRelation::Above, obj_b.id.clone())
                        .with_confidence(0.8),
                );
            } else {
                // A is below B
                return Some(
                    ObjectRelation::new(obj_a.id.clone(), SpatialRelation::Below, obj_b.id.clone())
                        .with_confidence(0.8),
                );
            }
        }

        // Horizontal relations
        if dx > dy {
            if ax < bx {
                // A is left of B
                return Some(
                    ObjectRelation::new(
                        obj_a.id.clone(),
                        SpatialRelation::LeftOf,
                        obj_b.id.clone(),
                    )
                    .with_confidence(0.8),
                );
            } else {
                // A is right of B
                return Some(
                    ObjectRelation::new(
                        obj_a.id.clone(),
                        SpatialRelation::RightOf,
                        obj_b.id.clone(),
                    )
                    .with_confidence(0.8),
                );
            }
        }

        None
    }
}

impl Default for SceneGraphGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Temporal relationship types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporalRelation {
    Before,
    After,
    During,
    Overlaps,
}

/// Temporal event in a scene
#[derive(Debug, Clone)]
pub struct TemporalEvent {
    /// Event ID
    pub id: String,
    /// Event description
    pub description: String,
    /// Start time (frame number or timestamp)
    pub start_time: f32,
    /// End time
    pub end_time: f32,
    /// Involved objects
    pub objects: Vec<String>,
}

impl TemporalEvent {
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        start: f32,
        end: f32,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            start_time: start,
            end_time: end,
            objects: Vec::new(),
        }
    }

    pub fn with_object(mut self, object_id: impl Into<String>) -> Self {
        self.objects.push(object_id.into());
        self
    }

    /// Check temporal relation with another event
    pub fn relation_to(&self, other: &TemporalEvent) -> TemporalRelation {
        if self.end_time <= other.start_time {
            TemporalRelation::Before
        } else if self.start_time >= other.end_time {
            TemporalRelation::After
        } else if self.start_time >= other.start_time && self.end_time <= other.end_time {
            TemporalRelation::During
        } else {
            TemporalRelation::Overlaps
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_object_creation() {
        let obj = SceneObject::new("obj1", "cat", (0.1, 0.2, 0.3, 0.4));
        assert_eq!(obj.id, "obj1");
        assert_eq!(obj.class, "cat");
        assert_eq!(obj.bbox, (0.1, 0.2, 0.3, 0.4));
    }

    #[test]
    fn test_scene_object_center() {
        let obj = SceneObject::new("obj1", "cat", (0.0, 0.0, 0.4, 0.6));
        let (cx, cy) = obj.center();
        assert!((cx - 0.2).abs() < 1e-5);
        assert!((cy - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_scene_object_area() {
        let obj = SceneObject::new("obj1", "cat", (0.0, 0.0, 0.5, 0.4));
        let area = obj.area();
        assert!((area - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_scene_object_attributes() {
        let obj = SceneObject::new("obj1", "cat", (0.0, 0.0, 0.1, 0.1))
            .with_attribute("fluffy")
            .with_attribute("orange");
        assert_eq!(obj.attributes.len(), 2);
        assert!(obj.attributes.contains(&"fluffy".to_string()));
    }

    #[test]
    fn test_spatial_relation_inverse() {
        assert_eq!(SpatialRelation::Above.inverse(), SpatialRelation::Below);
        assert_eq!(SpatialRelation::LeftOf.inverse(), SpatialRelation::RightOf);
        assert_eq!(SpatialRelation::Contains.inverse(), SpatialRelation::Inside);
    }

    #[test]
    fn test_object_relation_creation() {
        let rel = ObjectRelation::new("cat", SpatialRelation::On, "mat");
        assert_eq!(rel.subject, "cat");
        assert_eq!(rel.object, "mat");
        assert_eq!(rel.relation, SpatialRelation::On);
    }

    #[test]
    fn test_object_relation_to_string() {
        let rel = ObjectRelation::new("cat", SpatialRelation::On, "mat");
        assert_eq!(rel.to_string(), "on(cat, mat)");
    }

    #[test]
    fn test_scene_graph_creation() {
        let graph = SceneGraph::new();
        assert_eq!(graph.objects.len(), 0);
        assert_eq!(graph.relations.len(), 0);
    }

    #[test]
    fn test_scene_graph_add_object() {
        let mut graph = SceneGraph::new();
        let obj = SceneObject::new("obj1", "cat", (0.1, 0.1, 0.2, 0.2));
        graph.add_object(obj);
        assert_eq!(graph.objects.len(), 1);
    }

    #[test]
    fn test_scene_graph_add_relation() {
        let mut graph = SceneGraph::new();
        graph.add_object(SceneObject::new("cat", "cat", (0.1, 0.1, 0.2, 0.2)));
        graph.add_object(SceneObject::new("mat", "mat", (0.0, 0.0, 0.5, 0.1)));

        let rel = ObjectRelation::new("cat", SpatialRelation::On, "mat");
        let result = graph.add_relation(rel);
        assert!(result.is_ok());
        assert_eq!(graph.relations.len(), 1);
    }

    #[test]
    fn test_scene_graph_add_relation_missing_object() {
        let mut graph = SceneGraph::new();
        graph.add_object(SceneObject::new("cat", "cat", (0.1, 0.1, 0.2, 0.2)));

        let rel = ObjectRelation::new("cat", SpatialRelation::On, "mat");
        let result = graph.add_relation(rel);
        assert!(result.is_err());
    }

    #[test]
    fn test_scene_graph_get_relations_for() {
        let mut graph = SceneGraph::new();
        graph.add_object(SceneObject::new("cat", "cat", (0.1, 0.1, 0.2, 0.2)));
        graph.add_object(SceneObject::new("mat", "mat", (0.0, 0.0, 0.5, 0.1)));

        let _ = graph.add_relation(ObjectRelation::new("cat", SpatialRelation::On, "mat"));

        let relations = graph.get_relations_for("cat");
        assert_eq!(relations.len(), 1);
    }

    #[test]
    fn test_scene_graph_what_is_on() {
        let mut graph = SceneGraph::new();
        graph.add_object(SceneObject::new("cat", "cat", (0.1, 0.1, 0.2, 0.2)));
        graph.add_object(SceneObject::new("mat", "mat", (0.0, 0.0, 0.5, 0.1)));

        let _ = graph.add_relation(ObjectRelation::new("cat", SpatialRelation::On, "mat"));

        let on_mat = graph.what_is_on("mat");
        assert_eq!(on_mat.len(), 1);
        assert_eq!(on_mat[0].id, "cat");
    }

    #[test]
    fn test_scene_graph_to_description() {
        let mut graph = SceneGraph::new();
        graph.add_object(SceneObject::new("cat", "cat", (0.1, 0.1, 0.2, 0.2)));
        graph.add_object(SceneObject::new("mat", "mat", (0.0, 0.0, 0.5, 0.1)));
        let _ = graph.add_relation(ObjectRelation::new("cat", SpatialRelation::On, "mat"));

        let desc = graph.to_description();
        assert!(desc.contains("cat"));
        assert!(desc.contains("mat"));
        assert!(desc.contains("on"));
    }

    #[test]
    fn test_scene_graph_generator() {
        let generator = SceneGraphGenerator::new();
        assert_eq!(generator.min_object_confidence, 0.5);
    }

    #[test]
    fn test_generate_scene_graph() {
        let generator = SceneGraphGenerator::new();
        let objects = vec![
            SceneObject::new("cat", "cat", (0.3, 0.1, 0.2, 0.2)).with_confidence(0.9),
            SceneObject::new("mat", "mat", (0.0, 0.5, 0.8, 0.2)).with_confidence(0.8),
        ];

        let graph = generator.generate(objects);
        assert_eq!(graph.objects.len(), 2);
    }

    #[test]
    fn test_infer_spatial_relation_above() {
        let generator = SceneGraphGenerator::new();
        let obj_a = SceneObject::new("obj1", "cat", (0.5, 0.2, 0.1, 0.1)); // center: (0.55, 0.25)
        let obj_b = SceneObject::new("obj2", "mat", (0.5, 0.6, 0.1, 0.1)); // center: (0.55, 0.65)

        let relation = generator.infer_relation(&obj_a, &obj_b);
        assert!(relation.is_some());
        assert_eq!(relation.unwrap().relation, SpatialRelation::Above);
    }

    #[test]
    fn test_infer_spatial_relation_left_of() {
        let generator = SceneGraphGenerator::new();
        let obj_a = SceneObject::new("obj1", "cat", (0.1, 0.5, 0.1, 0.1)); // center: (0.15, 0.55)
        let obj_b = SceneObject::new("obj2", "dog", (0.7, 0.5, 0.1, 0.1)); // center: (0.75, 0.55)

        let relation = generator.infer_relation(&obj_a, &obj_b);
        assert!(relation.is_some());
        assert_eq!(relation.unwrap().relation, SpatialRelation::LeftOf);
    }

    #[test]
    fn test_temporal_event_creation() {
        let event = TemporalEvent::new("event1", "Cat jumps", 0.0, 1.0);
        assert_eq!(event.id, "event1");
        assert_eq!(event.start_time, 0.0);
        assert_eq!(event.end_time, 1.0);
    }

    #[test]
    fn test_temporal_event_with_object() {
        let event = TemporalEvent::new("event1", "Cat jumps", 0.0, 1.0).with_object("cat");
        assert_eq!(event.objects.len(), 1);
        assert_eq!(event.objects[0], "cat");
    }

    #[test]
    fn test_temporal_relation_before() {
        let event1 = TemporalEvent::new("e1", "Event 1", 0.0, 1.0);
        let event2 = TemporalEvent::new("e2", "Event 2", 2.0, 3.0);

        assert_eq!(event1.relation_to(&event2), TemporalRelation::Before);
    }

    #[test]
    fn test_temporal_relation_during() {
        let event1 = TemporalEvent::new("e1", "Event 1", 1.0, 2.0);
        let event2 = TemporalEvent::new("e2", "Event 2", 0.0, 3.0);

        assert_eq!(event1.relation_to(&event2), TemporalRelation::During);
    }

    #[test]
    fn test_temporal_relation_overlaps() {
        let event1 = TemporalEvent::new("e1", "Event 1", 0.0, 2.0);
        let event2 = TemporalEvent::new("e2", "Event 2", 1.0, 3.0);

        assert_eq!(event1.relation_to(&event2), TemporalRelation::Overlaps);
    }
}

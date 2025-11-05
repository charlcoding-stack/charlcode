// Cross-Modal Reasoning
//
// Enables reasoning across vision and language modalities, combining
// visual understanding with symbolic reasoning.
//
// Key capabilities:
// 1. Multimodal Chain-of-Thought reasoning
// 2. Visual + textual reasoning integration
// 3. Grounding language in visual scenes
// 4. Embodied AI foundations
//
// Example:
//   Image: [Red ball, blue box]
//   Question: "What color is the object on the left?"
//   → Multimodal CoT:
//     1. Detect objects: red ball (left), blue box (right)
//     2. Identify left object: red ball
//     3. Extract color: red
//     Answer: "red"
//
// References:
// - Zhang et al. (2023): "Multimodal Chain-of-Thought Reasoning in Language Models"
// - Zellers et al. (2021): "From Recognition to Cognition: Visual Commonsense Reasoning"

use crate::reasoning::chain_of_thought::{ChainOfThought, ReasoningStep};
use crate::multimodal::scene_understanding::{SceneGraph, SceneObject, SpatialRelation};
use crate::multimodal::vision_language::{Image, CLIPEncoder, MultimodalEmbedding};

/// Modality of a reasoning step
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReasoningModality {
    Visual,
    Textual,
    CrossModal,
}

/// Multimodal reasoning step
#[derive(Debug, Clone)]
pub struct MultimodalReasoningStep {
    /// Step number
    pub step_number: usize,
    /// Reasoning text
    pub thought: String,
    /// Modality used
    pub modality: ReasoningModality,
    /// Referenced objects (if visual)
    pub objects: Vec<String>,
    /// Confidence
    pub confidence: f32,
}

impl MultimodalReasoningStep {
    pub fn new(step_number: usize, thought: impl Into<String>, modality: ReasoningModality) -> Self {
        Self {
            step_number,
            thought: thought.into(),
            modality,
            objects: Vec::new(),
            confidence: 1.0,
        }
    }

    pub fn with_object(mut self, object_id: impl Into<String>) -> Self {
        self.objects.push(object_id.into());
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Multimodal Chain-of-Thought
#[derive(Debug, Clone)]
pub struct MultimodalCoT {
    /// Question or task
    pub question: String,
    /// Visual context (scene graph)
    pub scene: Option<SceneGraph>,
    /// Image (optional)
    pub image: Option<String>, // Image ID
    /// Reasoning steps
    pub steps: Vec<MultimodalReasoningStep>,
    /// Final answer
    pub answer: String,
    /// Overall confidence
    pub confidence: f32,
}

impl MultimodalCoT {
    pub fn new(question: impl Into<String>) -> Self {
        Self {
            question: question.into(),
            scene: None,
            image: None,
            steps: Vec::new(),
            answer: String::new(),
            confidence: 0.0,
        }
    }

    pub fn with_scene(mut self, scene: SceneGraph) -> Self {
        self.scene = Some(scene);
        self
    }

    pub fn with_image(mut self, image_id: impl Into<String>) -> Self {
        self.image = Some(image_id.into());
        self
    }

    pub fn add_step(mut self, step: MultimodalReasoningStep) -> Self {
        self.steps.push(step);
        self
    }

    pub fn with_answer(mut self, answer: impl Into<String>) -> Self {
        self.answer = answer.into();
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Compute confidence from steps
    pub fn compute_confidence(&mut self) {
        if self.steps.is_empty() {
            self.confidence = 0.0;
            return;
        }

        let sum: f32 = self.steps.iter().map(|s| s.confidence).sum();
        self.confidence = sum / self.steps.len() as f32;
    }
}

/// Visual grounding: link text to visual elements
pub struct VisualGrounding {
    /// CLIP encoder for similarity
    pub encoder: CLIPEncoder,
    /// Similarity threshold
    pub threshold: f32,
}

impl VisualGrounding {
    pub fn new(encoder: CLIPEncoder) -> Self {
        Self {
            encoder,
            threshold: 0.5,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Ground a text phrase to objects in a scene
    pub fn ground_phrase(&self, phrase: &str, scene: &SceneGraph) -> Vec<(String, f32)> {
        let phrase_emb = self.encoder.encode_text(phrase);

        let mut matches = Vec::new();

        for (obj_id, obj) in &scene.objects {
            // Create pseudo-embedding for object (in practice: visual features)
            let obj_text = format!("{} {}", obj.class, obj.attributes.join(" "));
            let obj_emb = self.encoder.encode_text(&obj_text);

            let similarity = phrase_emb.cosine_similarity(&obj_emb);

            if similarity >= self.threshold {
                matches.push((obj_id.clone(), similarity));
            }
        }

        // Sort by similarity
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        matches
    }

    /// Check if phrase refers to an object
    pub fn refers_to(&self, phrase: &str, object_class: &str) -> bool {
        let phrase_lower = phrase.to_lowercase();
        let class_lower = object_class.to_lowercase();

        phrase_lower.contains(&class_lower)
    }
}

/// Multimodal reasoner
pub struct MultimodalReasoner {
    /// Visual grounding
    pub grounding: VisualGrounding,
}

impl MultimodalReasoner {
    pub fn new(grounding: VisualGrounding) -> Self {
        Self { grounding }
    }

    /// Reason about a visual question
    pub fn reason(&self, question: &str, scene: &SceneGraph) -> MultimodalCoT {
        let mut cot = MultimodalCoT::new(question).with_scene(scene.clone());

        // Parse question type
        let q_lower = question.to_lowercase();

        if q_lower.contains("what") {
            cot = self.reason_what(question, scene, cot);
        } else if q_lower.contains("where") {
            cot = self.reason_where(question, scene, cot);
        } else if q_lower.contains("how many") {
            cot = self.reason_count(question, scene, cot);
        } else {
            // Generic reasoning
            cot = cot.add_step(MultimodalReasoningStep::new(
                1,
                "Analyzing scene",
                ReasoningModality::Visual
            ));
        }

        cot.compute_confidence();
        cot
    }

    /// Reason about "what" questions
    fn reason_what(&self, question: &str, scene: &SceneGraph, mut cot: MultimodalCoT) -> MultimodalCoT {
        // Step 1: Identify objects
        let obj_count = scene.objects.len();
        cot = cot.add_step(MultimodalReasoningStep::new(
            1,
            format!("Detected {} objects in scene", obj_count),
            ReasoningModality::Visual
        ).with_confidence(0.9));

        // Step 2: Match question to objects
        let matches = self.grounding.ground_phrase(question, scene);

        if let Some((obj_id, sim)) = matches.first() {
            if let Some(obj) = scene.objects.get(obj_id) {
                cot = cot.add_step(MultimodalReasoningStep::new(
                    2,
                    format!("Found relevant object: {}", obj.class),
                    ReasoningModality::CrossModal
                ).with_object(obj_id.clone()).with_confidence(*sim));

                cot = cot.with_answer(obj.class.clone());
            }
        }

        cot
    }

    /// Reason about "where" questions
    fn reason_where(&self, question: &str, scene: &SceneGraph, mut cot: MultimodalCoT) -> MultimodalCoT {
        // Find referenced object
        let matches = self.grounding.ground_phrase(question, scene);

        if let Some((obj_id, _)) = matches.first() {
            // Step 1: Identify object
            if let Some(obj) = scene.objects.get(obj_id) {
                cot = cot.add_step(MultimodalReasoningStep::new(
                    1,
                    format!("Located object: {}", obj.class),
                    ReasoningModality::Visual
                ).with_object(obj_id.clone()));

                // Step 2: Find spatial relations
                let relations = scene.get_relations_for(obj_id);

                if let Some(rel) = relations.first() {
                    if let Some(ref_obj) = scene.objects.get(&rel.object) {
                        cot = cot.add_step(MultimodalReasoningStep::new(
                            2,
                            format!("The {} is {} the {}", obj.class, rel.relation.as_str(), ref_obj.class),
                            ReasoningModality::CrossModal
                        ));

                        cot = cot.with_answer(format!("{} the {}", rel.relation.as_str(), ref_obj.class));
                    }
                }
            }
        }

        cot
    }

    /// Reason about counting questions
    fn reason_count(&self, question: &str, scene: &SceneGraph, mut cot: MultimodalCoT) -> MultimodalCoT {
        // Step 1: Count objects
        let count = scene.objects.len();
        cot = cot.add_step(MultimodalReasoningStep::new(
            1,
            format!("Counting objects in scene"),
            ReasoningModality::Visual
        ));

        cot = cot.add_step(MultimodalReasoningStep::new(
            2,
            format!("Found {} objects", count),
            ReasoningModality::CrossModal
        ));

        cot = cot.with_answer(count.to_string());

        cot
    }

    /// Convert to standard Chain-of-Thought
    pub fn to_standard_cot(&self, multimodal_cot: &MultimodalCoT) -> ChainOfThought {
        let mut cot = ChainOfThought::new(&multimodal_cot.question);

        for step in &multimodal_cot.steps {
            cot = cot.add_step(
                ReasoningStep::new(step.step_number, &step.thought)
                    .with_confidence(step.confidence)
            );
        }

        cot.with_final_answer(&multimodal_cot.answer)
            .with_confidence(multimodal_cot.confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimodal::scene_understanding::ObjectRelation;

    #[test]
    fn test_multimodal_reasoning_step() {
        let step = MultimodalReasoningStep::new(1, "Analyzing scene", ReasoningModality::Visual);
        assert_eq!(step.step_number, 1);
        assert_eq!(step.modality, ReasoningModality::Visual);
    }

    #[test]
    fn test_multimodal_reasoning_step_with_object() {
        let step = MultimodalReasoningStep::new(1, "Found cat", ReasoningModality::Visual)
            .with_object("cat1");
        assert_eq!(step.objects.len(), 1);
        assert_eq!(step.objects[0], "cat1");
    }

    #[test]
    fn test_multimodal_cot_creation() {
        let cot = MultimodalCoT::new("What is this?");
        assert_eq!(cot.question, "What is this?");
        assert_eq!(cot.steps.len(), 0);
    }

    #[test]
    fn test_multimodal_cot_with_scene() {
        let scene = SceneGraph::new();
        let cot = MultimodalCoT::new("Question").with_scene(scene);
        assert!(cot.scene.is_some());
    }

    #[test]
    fn test_multimodal_cot_add_step() {
        let cot = MultimodalCoT::new("Question")
            .add_step(MultimodalReasoningStep::new(1, "Step 1", ReasoningModality::Visual));
        assert_eq!(cot.steps.len(), 1);
    }

    #[test]
    fn test_multimodal_cot_compute_confidence() {
        let mut cot = MultimodalCoT::new("Question")
            .add_step(MultimodalReasoningStep::new(1, "Step 1", ReasoningModality::Visual).with_confidence(0.8))
            .add_step(MultimodalReasoningStep::new(2, "Step 2", ReasoningModality::Textual).with_confidence(0.9));

        cot.compute_confidence();
        assert!((cot.confidence - 0.85).abs() < 1e-5); // Average of 0.8 and 0.9
    }

    #[test]
    fn test_visual_grounding_creation() {
        let encoder = CLIPEncoder::new(128);
        let grounding = VisualGrounding::new(encoder);
        assert_eq!(grounding.threshold, 0.5);
    }

    #[test]
    fn test_visual_grounding_refers_to() {
        let encoder = CLIPEncoder::new(128);
        let grounding = VisualGrounding::new(encoder);

        assert!(grounding.refers_to("the cat", "cat"));
        assert!(grounding.refers_to("big dog", "dog"));
        assert!(!grounding.refers_to("the cat", "dog"));
    }

    #[test]
    fn test_visual_grounding_ground_phrase() {
        let encoder = CLIPEncoder::new(128);
        let grounding = VisualGrounding::new(encoder);

        let mut scene = SceneGraph::new();
        scene.add_object(SceneObject::new("cat1", "cat", (0.1, 0.1, 0.2, 0.2)));
        scene.add_object(SceneObject::new("dog1", "dog", (0.5, 0.5, 0.2, 0.2)));

        let matches = grounding.ground_phrase("cat", &scene);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_multimodal_reasoner() {
        let encoder = CLIPEncoder::new(128);
        let grounding = VisualGrounding::new(encoder);
        let reasoner = MultimodalReasoner::new(grounding);

        let scene = SceneGraph::new();
        let cot = reasoner.reason("What is this?", &scene);

        assert_eq!(cot.question, "What is this?");
    }

    #[test]
    fn test_reason_what_question() {
        let encoder = CLIPEncoder::new(128);
        let grounding = VisualGrounding::new(encoder).with_threshold(0.0); // Lower threshold for test
        let reasoner = MultimodalReasoner::new(grounding);

        let mut scene = SceneGraph::new();
        scene.add_object(SceneObject::new("cat1", "cat", (0.1, 0.1, 0.2, 0.2)));

        let cot = reasoner.reason("What animal is this?", &scene);

        assert!(!cot.steps.is_empty());
        assert!(!cot.answer.is_empty());
    }

    #[test]
    fn test_reason_where_question() {
        let encoder = CLIPEncoder::new(128);
        let grounding = VisualGrounding::new(encoder).with_threshold(0.0); // Lower threshold for test
        let reasoner = MultimodalReasoner::new(grounding);

        let mut scene = SceneGraph::new();
        scene.add_object(SceneObject::new("cat1", "cat", (0.3, 0.1, 0.2, 0.2)));
        scene.add_object(SceneObject::new("mat1", "mat", (0.0, 0.5, 0.6, 0.2)));
        let _ = scene.add_relation(ObjectRelation::new("cat1", SpatialRelation::Above, "mat1"));

        let cot = reasoner.reason("Where is the cat?", &scene);

        assert!(!cot.steps.is_empty());
    }

    #[test]
    fn test_reason_count_question() {
        let encoder = CLIPEncoder::new(128);
        let grounding = VisualGrounding::new(encoder);
        let reasoner = MultimodalReasoner::new(grounding);

        let mut scene = SceneGraph::new();
        scene.add_object(SceneObject::new("cat1", "cat", (0.1, 0.1, 0.2, 0.2)));
        scene.add_object(SceneObject::new("dog1", "dog", (0.5, 0.5, 0.2, 0.2)));

        let cot = reasoner.reason("How many objects?", &scene);

        assert!(!cot.steps.is_empty());
        assert_eq!(cot.answer, "2");
    }

    #[test]
    fn test_to_standard_cot() {
        let encoder = CLIPEncoder::new(128);
        let grounding = VisualGrounding::new(encoder);
        let reasoner = MultimodalReasoner::new(grounding);

        let multimodal_cot = MultimodalCoT::new("Test question")
            .add_step(MultimodalReasoningStep::new(1, "Step 1", ReasoningModality::Visual))
            .with_answer("Test answer")
            .with_confidence(0.9);

        let standard_cot = reasoner.to_standard_cot(&multimodal_cot);

        assert_eq!(standard_cot.problem, "Test question");
        assert_eq!(standard_cot.steps.len(), 1);
        assert_eq!(standard_cot.final_answer, "Test answer");
    }

    #[test]
    fn test_multimodal_reasoning_modality_types() {
        let visual = ReasoningModality::Visual;
        let textual = ReasoningModality::Textual;
        let cross_modal = ReasoningModality::CrossModal;

        assert_ne!(visual, textual);
        assert_ne!(visual, cross_modal);
        assert_ne!(textual, cross_modal);
    }

    #[test]
    fn test_complex_multimodal_reasoning() {
        let encoder = CLIPEncoder::new(128);
        let grounding = VisualGrounding::new(encoder);
        let reasoner = MultimodalReasoner::new(grounding);

        // Create complex scene
        let mut scene = SceneGraph::new();
        scene.add_object(SceneObject::new("cat1", "cat", (0.2, 0.1, 0.2, 0.2))
            .with_attribute("orange"));
        scene.add_object(SceneObject::new("mat1", "mat", (0.0, 0.6, 0.8, 0.2))
            .with_attribute("blue"));
        let _ = scene.add_relation(ObjectRelation::new("cat1", SpatialRelation::Above, "mat1"));

        let cot = reasoner.reason("What is above the mat?", &scene);

        assert!(!cot.steps.is_empty());
        assert!(cot.confidence > 0.0);
    }
}

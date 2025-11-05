// Multimodal Neuro-Symbolic Module
//
// This module integrates vision, language, and symbolic reasoning for
// multimodal AI systems.
//
// Components:
// 1. Vision-Language Integration: CLIP-like embeddings, VQA
// 2. Symbolic Scene Understanding: Scene graphs, spatial/temporal reasoning
// 3. Cross-Modal Reasoning: Multimodal Chain-of-Thought
//
// Usage:
// ```rust
// use charl::multimodal::{
//     CLIPEncoder, VQASystem, Image,
//     SceneGraph, SceneObject, ObjectRelation, SpatialRelation,
//     MultimodalReasoner, VisualGrounding, MultimodalCoT
// };
//
// // Vision-Language
// let encoder = CLIPEncoder::new(512);
// let image = Image::new("img1", 224, 224, pixels).with_caption("A cat");
// let image_emb = encoder.encode_image(&image);
// let text_emb = encoder.encode_text("A cat sitting");
// let similarity = image_emb.cosine_similarity(&text_emb);
//
// // VQA
// let mut vqa = VQASystem::new(encoder);
// vqa.add_qa("img1", "What animal?", "A cat");
// let answer = vqa.answer_question(&image, "What animal?");
//
// // Scene Understanding
// let mut scene = SceneGraph::new();
// scene.add_object(SceneObject::new("cat", "cat", (0.1, 0.1, 0.2, 0.2)));
// scene.add_object(SceneObject::new("mat", "mat", (0.0, 0.5, 0.8, 0.2)));
// scene.add_relation(ObjectRelation::new("cat", SpatialRelation::Above, "mat"));
//
// // Cross-Modal Reasoning
// let grounding = VisualGrounding::new(CLIPEncoder::new(512));
// let reasoner = MultimodalReasoner::new(grounding);
// let cot = reasoner.reason("Where is the cat?", &scene);
// ```

pub mod cross_modal_reasoning;
pub mod scene_understanding;
pub mod vision_language;

// Re-export main types
pub use vision_language::{
    CLIPEncoder, CrossModalRetrieval, Image, Modality, MultimodalEmbedding, VQAAnswer, VQASystem,
};

pub use scene_understanding::{
    ObjectRelation, SceneGraph, SceneGraphGenerator, SceneObject, SpatialRelation, TemporalEvent,
    TemporalRelation,
};

pub use cross_modal_reasoning::{
    MultimodalCoT, MultimodalReasoner, MultimodalReasoningStep, ReasoningModality, VisualGrounding,
};

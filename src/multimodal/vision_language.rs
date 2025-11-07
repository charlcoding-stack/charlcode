// Vision-Language Integration
//
// Implements CLIP-like shared embeddings for vision and language, enabling
// cross-modal understanding and visual question answering.
//
// Key capabilities:
// 1. Shared embedding space for images and text
// 2. Visual reasoning with natural language
// 3. Visual Question Answering (VQA)
// 4. Image-text similarity and retrieval
//
// Example:
//   Image: [cat sitting on mat]
//   Text: "What animal is in the image?"
//   → Embedding similarity → "A cat"
//
// References:
// - Radford et al. (2021): "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
// - Alayrac et al. (2022): "Flamingo: a Visual Language Model for Few-Shot Learning"

use std::collections::HashMap;

/// Modality type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Vision,
    Language,
}

/// Embedding in shared vision-language space
#[derive(Debug, Clone)]
pub struct MultimodalEmbedding {
    /// Embedding vector
    pub vector: Vec<f32>,
    /// Source modality
    pub modality: Modality,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl MultimodalEmbedding {
    pub fn new(vector: Vec<f32>, modality: Modality) -> Self {
        Self {
            vector,
            modality,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Dimensionality of embedding
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Compute cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &MultimodalEmbedding) -> f32 {
        if self.vector.len() != other.vector.len() {
            return 0.0;
        }

        let dot: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_self: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_other: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_self == 0.0 || norm_other == 0.0 {
            return 0.0;
        }

        dot / (norm_self * norm_other)
    }

    /// L2 distance to another embedding
    pub fn l2_distance(&self, other: &MultimodalEmbedding) -> f32 {
        if self.vector.len() != other.vector.len() {
            return f32::INFINITY;
        }

        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Normalize embedding to unit length
    pub fn normalize(&mut self) {
        let norm: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut self.vector {
                *x /= norm;
            }
        }
    }
}

/// Image representation (simplified - in practice would be actual image data)
#[derive(Debug, Clone)]
pub struct Image {
    /// Image identifier
    pub id: String,
    /// Image dimensions (width, height)
    pub dimensions: (usize, usize),
    /// Raw pixel data (simplified as Vec<f32> for grayscale or flattened RGB)
    pub pixels: Vec<f32>,
    /// Optional caption/description
    pub caption: Option<String>,
}

impl Image {
    pub fn new(id: impl Into<String>, width: usize, height: usize, pixels: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            dimensions: (width, height),
            pixels,
            caption: None,
        }
    }

    pub fn with_caption(mut self, caption: impl Into<String>) -> Self {
        self.caption = Some(caption.into());
        self
    }

    pub fn pixel_count(&self) -> usize {
        self.dimensions.0 * self.dimensions.1
    }
}

/// CLIP-like encoder for vision and language
#[derive(Debug, Clone)]
pub struct CLIPEncoder {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Temperature parameter for softmax
    pub temperature: f32,
}

impl CLIPEncoder {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            temperature: 0.07, // Standard CLIP temperature
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Encode image to embedding (simplified - no actual CNN)
    pub fn encode_image(&self, image: &Image) -> MultimodalEmbedding {
        // Simplified: average pooling over patches + random projection
        // In practice: CNN/ViT → linear projection
        let patch_size = 16;
        let _num_patches = (image.dimensions.0 / patch_size) * (image.dimensions.1 / patch_size);

        let mut embedding_vec = vec![0.0; self.embedding_dim];

        // Simple feature extraction (in practice: learned CNN/ViT)
        for i in 0..self.embedding_dim.min(image.pixels.len()) {
            embedding_vec[i] = image.pixels[i % image.pixels.len()] / 255.0;
        }

        // Simulate learned transformation
        for i in 0..embedding_vec.len() {
            embedding_vec[i] = (embedding_vec[i] * 2.0 - 1.0).tanh();
        }

        let mut emb = MultimodalEmbedding::new(embedding_vec, Modality::Vision);
        emb.normalize();
        emb.with_metadata("source", "image")
            .with_metadata("id", image.id.clone())
    }

    /// Encode text to embedding (simplified - no actual transformer)
    pub fn encode_text(&self, text: &str) -> MultimodalEmbedding {
        // Simplified: character-level features
        // In practice: Transformer → linear projection
        let mut embedding_vec = vec![0.0; self.embedding_dim];

        let chars: Vec<char> = text.chars().collect();
        for (i, &ch) in chars.iter().enumerate() {
            let idx = i % self.embedding_dim;
            embedding_vec[idx] += (ch as u32 as f32) / 1000.0;
        }

        // Simulate learned transformation
        for i in 0..embedding_vec.len() {
            embedding_vec[i] = (embedding_vec[i]).tanh();
        }

        let mut emb = MultimodalEmbedding::new(embedding_vec, Modality::Language);
        emb.normalize();
        emb.with_metadata("source", "text")
            .with_metadata("content", text.to_string())
    }

    /// Compute contrastive loss between image and text embeddings
    pub fn contrastive_loss(
        &self,
        image_emb: &MultimodalEmbedding,
        text_emb: &MultimodalEmbedding,
    ) -> f32 {
        let similarity = image_emb.cosine_similarity(text_emb);
        let logits = similarity / self.temperature;

        // Simplified cross-entropy loss
        -logits.ln_1p()
    }
}

impl Default for CLIPEncoder {
    fn default() -> Self {
        Self::new(512) // Standard CLIP embedding size
    }
}

/// Visual Question Answering system
#[derive(Debug, Clone)]
pub struct VQASystem {
    /// CLIP encoder for embeddings
    pub encoder: CLIPEncoder,
    /// Question-answer pairs database
    pub qa_database: HashMap<String, Vec<(String, String)>>, // image_id -> [(question, answer)]
}

impl VQASystem {
    pub fn new(encoder: CLIPEncoder) -> Self {
        Self {
            encoder,
            qa_database: HashMap::new(),
        }
    }

    /// Add question-answer pair for an image
    pub fn add_qa(
        &mut self,
        image_id: impl Into<String>,
        question: impl Into<String>,
        answer: impl Into<String>,
    ) {
        let image_id = image_id.into();
        self.qa_database
            .entry(image_id)
            .or_default()
            .push((question.into(), answer.into()));
    }

    /// Answer a question about an image
    pub fn answer_question(&self, image: &Image, question: &str) -> Option<VQAAnswer> {
        // Encode image and question
        let image_emb = self.encoder.encode_image(image);
        let question_emb = self.encoder.encode_text(question);

        // Compute similarity
        let similarity = image_emb.cosine_similarity(&question_emb);

        // Retrieve from database (simplified)
        if let Some(qa_pairs) = self.qa_database.get(&image.id) {
            for (q, a) in qa_pairs {
                if q.to_lowercase().contains(&question.to_lowercase()) {
                    return Some(VQAAnswer {
                        answer: a.clone(),
                        confidence: similarity,
                        reasoning: format!("Matched question: {}", q),
                    });
                }
            }
        }

        // Generate answer based on image caption (if available)
        if let Some(caption) = &image.caption {
            return Some(VQAAnswer {
                answer: caption.clone(),
                confidence: similarity * 0.8,
                reasoning: "Generated from image caption".to_string(),
            });
        }

        None
    }

    /// Find most relevant image for a text query
    pub fn retrieve_image<'a>(&self, query: &str, images: &'a [Image]) -> Option<(&'a Image, f32)> {
        let query_emb = self.encoder.encode_text(query);

        let mut best_match: Option<(&'a Image, f32)> = None;

        for image in images {
            let image_emb = self.encoder.encode_image(image);
            let similarity = query_emb.cosine_similarity(&image_emb);

            if let Some((_, best_sim)) = best_match {
                if similarity > best_sim {
                    best_match = Some((image, similarity));
                }
            } else {
                best_match = Some((image, similarity));
            }
        }

        best_match
    }
}

/// VQA answer with confidence and reasoning
#[derive(Debug, Clone)]
pub struct VQAAnswer {
    pub answer: String,
    pub confidence: f32,
    pub reasoning: String,
}

impl VQAAnswer {
    pub fn is_confident(&self) -> bool {
        self.confidence > 0.7
    }
}

/// Cross-modal retrieval
#[derive(Debug, Clone)]
pub struct CrossModalRetrieval {
    /// CLIP encoder
    pub encoder: CLIPEncoder,
    /// Top-k results to return
    pub top_k: usize,
}

impl CrossModalRetrieval {
    pub fn new(encoder: CLIPEncoder) -> Self {
        Self { encoder, top_k: 5 }
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Text-to-image retrieval
    pub fn text_to_image<'a>(&self, query: &str, images: &'a [Image]) -> Vec<(&'a Image, f32)> {
        let query_emb = self.encoder.encode_text(query);

        let mut scores: Vec<(&'a Image, f32)> = images
            .iter()
            .map(|img| {
                let img_emb = self.encoder.encode_image(img);
                let sim = query_emb.cosine_similarity(&img_emb);
                (img, sim)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(self.top_k);
        scores
    }

    /// Image-to-text retrieval
    pub fn image_to_text<'a>(&self, image: &Image, texts: &'a [String]) -> Vec<(&'a String, f32)> {
        let image_emb = self.encoder.encode_image(image);

        let mut scores: Vec<(&'a String, f32)> = texts
            .iter()
            .map(|text| {
                let text_emb = self.encoder.encode_text(text);
                let sim = image_emb.cosine_similarity(&text_emb);
                (text, sim)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(self.top_k);
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_embedding_creation() {
        let emb = MultimodalEmbedding::new(vec![0.1, 0.2, 0.3], Modality::Vision);
        assert_eq!(emb.dim(), 3);
        assert_eq!(emb.modality, Modality::Vision);
    }

    #[test]
    fn test_embedding_cosine_similarity() {
        let emb1 = MultimodalEmbedding::new(vec![1.0, 0.0, 0.0], Modality::Vision);
        let emb2 = MultimodalEmbedding::new(vec![1.0, 0.0, 0.0], Modality::Language);

        let sim = emb1.cosine_similarity(&emb2);
        assert!((sim - 1.0).abs() < 1e-5); // Should be 1.0 (identical)
    }

    #[test]
    fn test_embedding_cosine_similarity_orthogonal() {
        let emb1 = MultimodalEmbedding::new(vec![1.0, 0.0], Modality::Vision);
        let emb2 = MultimodalEmbedding::new(vec![0.0, 1.0], Modality::Language);

        let sim = emb1.cosine_similarity(&emb2);
        assert!(sim.abs() < 1e-5); // Should be 0.0 (orthogonal)
    }

    #[test]
    fn test_embedding_l2_distance() {
        let emb1 = MultimodalEmbedding::new(vec![0.0, 0.0], Modality::Vision);
        let emb2 = MultimodalEmbedding::new(vec![3.0, 4.0], Modality::Vision);

        let dist = emb1.l2_distance(&emb2);
        assert!((dist - 5.0).abs() < 1e-5); // Should be 5.0 (3-4-5 triangle)
    }

    #[test]
    fn test_embedding_normalize() {
        let mut emb = MultimodalEmbedding::new(vec![3.0, 4.0], Modality::Vision);
        emb.normalize();

        let norm: f32 = emb.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5); // Should be unit length
    }

    #[test]
    fn test_image_creation() {
        let pixels = vec![128.0; 256]; // 16x16 grayscale
        let image = Image::new("img1", 16, 16, pixels);

        assert_eq!(image.id, "img1");
        assert_eq!(image.dimensions, (16, 16));
        assert_eq!(image.pixel_count(), 256);
    }

    #[test]
    fn test_image_with_caption() {
        let pixels = vec![0.0; 100];
        let image = Image::new("img1", 10, 10, pixels).with_caption("A cat");

        assert_eq!(image.caption, Some("A cat".to_string()));
    }

    #[test]
    fn test_clip_encoder_creation() {
        let encoder = CLIPEncoder::new(512);
        assert_eq!(encoder.embedding_dim, 512);
        assert!((encoder.temperature - 0.07).abs() < 1e-5);
    }

    #[test]
    fn test_encode_image() {
        let encoder = CLIPEncoder::new(128);
        let pixels = vec![128.0; 256];
        let image = Image::new("img1", 16, 16, pixels);

        let embedding = encoder.encode_image(&image);
        assert_eq!(embedding.dim(), 128);
        assert_eq!(embedding.modality, Modality::Vision);
    }

    #[test]
    fn test_encode_text() {
        let encoder = CLIPEncoder::new(128);
        let embedding = encoder.encode_text("Hello world");

        assert_eq!(embedding.dim(), 128);
        assert_eq!(embedding.modality, Modality::Language);
    }

    #[test]
    fn test_vqa_system_creation() {
        let encoder = CLIPEncoder::new(256);
        let vqa = VQASystem::new(encoder);
        assert_eq!(vqa.encoder.embedding_dim, 256);
    }

    #[test]
    fn test_vqa_add_qa() {
        let encoder = CLIPEncoder::new(256);
        let mut vqa = VQASystem::new(encoder);

        vqa.add_qa("img1", "What is this?", "A cat");
        assert_eq!(vqa.qa_database.get("img1").unwrap().len(), 1);
    }

    #[test]
    fn test_vqa_answer_with_caption() {
        let encoder = CLIPEncoder::new(256);
        let vqa = VQASystem::new(encoder);

        let pixels = vec![100.0; 256];
        let image = Image::new("img1", 16, 16, pixels).with_caption("A cat sitting");

        let answer = vqa.answer_question(&image, "What is in the image?");
        assert!(answer.is_some());
        assert_eq!(answer.unwrap().answer, "A cat sitting");
    }

    #[test]
    fn test_vqa_answer_from_database() {
        let encoder = CLIPEncoder::new(256);
        let mut vqa = VQASystem::new(encoder);

        vqa.add_qa("img1", "What animal is this?", "A dog");

        let pixels = vec![100.0; 256];
        let image = Image::new("img1", 16, 16, pixels);

        let answer = vqa.answer_question(&image, "What animal");
        assert!(answer.is_some());
        assert_eq!(answer.unwrap().answer, "A dog");
    }

    #[test]
    fn test_vqa_answer_confidence() {
        let encoder = CLIPEncoder::new(256);
        let vqa = VQASystem::new(encoder);

        let pixels = vec![200.0; 256];
        let image = Image::new("img1", 16, 16, pixels).with_caption("Test");

        let answer = vqa.answer_question(&image, "Test?").unwrap();
        assert!(answer.confidence >= 0.0 && answer.confidence <= 1.0);
    }

    #[test]
    fn test_retrieve_image() {
        let encoder = CLIPEncoder::new(256);
        let vqa = VQASystem::new(encoder);

        let images = vec![
            Image::new("img1", 10, 10, vec![50.0; 100]),
            Image::new("img2", 10, 10, vec![150.0; 100]),
        ];

        let result = vqa.retrieve_image("test query", &images);
        assert!(result.is_some());
    }

    #[test]
    fn test_cross_modal_retrieval() {
        let encoder = CLIPEncoder::new(256);
        let retrieval = CrossModalRetrieval::new(encoder);
        assert_eq!(retrieval.top_k, 5);
    }

    #[test]
    fn test_text_to_image_retrieval() {
        let encoder = CLIPEncoder::new(256);
        let retrieval = CrossModalRetrieval::new(encoder).with_top_k(2);

        let images = vec![
            Image::new("img1", 10, 10, vec![50.0; 100]),
            Image::new("img2", 10, 10, vec![150.0; 100]),
            Image::new("img3", 10, 10, vec![200.0; 100]),
        ];

        let results = retrieval.text_to_image("cat", &images);
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_image_to_text_retrieval() {
        let encoder = CLIPEncoder::new(256);
        let retrieval = CrossModalRetrieval::new(encoder).with_top_k(2);

        let image = Image::new("img1", 10, 10, vec![100.0; 100]);
        let texts = vec![
            "A cat sitting".to_string(),
            "A dog running".to_string(),
            "A bird flying".to_string(),
        ];

        let results = retrieval.image_to_text(&image, &texts);
        assert!(results.len() <= 2);
    }
}

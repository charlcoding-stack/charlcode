// Prototypical Networks for Few-Shot Learning
//
// Prototypical Networks learn a metric space where classification
// is performed by computing distances to prototype representations
// of each class.
//
// Algorithm:
// 1. Encode support examples into embedding space
// 2. Compute class prototypes (mean of embeddings per class)
// 3. Classify query examples by distance to nearest prototype
//
// N-way K-shot classification:
// - N classes
// - K examples per class in support set
// - Classify query examples into one of N classes
//
// References:
// - Snell et al. (2017): "Prototypical Networks for Few-shot Learning"

/// Embedding function (maps input to embedding space)
pub type EmbeddingFn = Box<dyn Fn(&[f32]) -> Vec<f32>>;

/// Distance metric
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance: ||a - b||₂
    Euclidean,
    /// Cosine distance: 1 - (a·b / ||a|| ||b||)
    Cosine,
    /// Manhattan distance: Σ|aᵢ - bᵢ|
    Manhattan,
}

impl DistanceMetric {
    /// Compute distance between two embeddings
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Embeddings must have same dimension");

        match self {
            DistanceMetric::Euclidean => {
                let mut sum = 0.0;
                for (ai, bi) in a.iter().zip(b.iter()) {
                    let diff = ai - bi;
                    sum += diff * diff;
                }
                sum.sqrt()
            }
            DistanceMetric::Cosine => {
                let mut dot = 0.0;
                let mut norm_a = 0.0;
                let mut norm_b = 0.0;

                for (ai, bi) in a.iter().zip(b.iter()) {
                    dot += ai * bi;
                    norm_a += ai * ai;
                    norm_b += bi * bi;
                }

                let norm_a = norm_a.sqrt();
                let norm_b = norm_b.sqrt();

                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0
                } else {
                    1.0 - (dot / (norm_a * norm_b))
                }
            }
            DistanceMetric::Manhattan => {
                let mut sum = 0.0;
                for (ai, bi) in a.iter().zip(b.iter()) {
                    sum += (ai - bi).abs();
                }
                sum
            }
        }
    }
}

/// N-way K-shot episode
#[derive(Debug, Clone)]
pub struct Episode {
    /// Support set: K examples per class, N classes
    pub support_set: Vec<(Vec<f32>, usize)>, // (input, class_id)
    /// Query set: examples to classify
    pub query_set: Vec<(Vec<f32>, usize)>,
    /// Number of classes (N-way)
    pub n_way: usize,
    /// Number of examples per class (K-shot)
    pub k_shot: usize,
}

impl Episode {
    /// Create new episode
    pub fn new(n_way: usize, k_shot: usize) -> Self {
        Self {
            support_set: Vec::new(),
            query_set: Vec::new(),
            n_way,
            k_shot,
        }
    }

    /// Add support example
    pub fn add_support(mut self, input: Vec<f32>, class_id: usize) -> Self {
        assert!(class_id < self.n_way, "Class ID must be < N-way");
        self.support_set.push((input, class_id));
        self
    }

    /// Add query example
    pub fn add_query(mut self, input: Vec<f32>, class_id: usize) -> Self {
        assert!(class_id < self.n_way, "Class ID must be < N-way");
        self.query_set.push((input, class_id));
        self
    }

    /// Validate episode structure
    pub fn validate(&self) -> Result<(), String> {
        // Check support set has K examples per class
        let mut class_counts = vec![0; self.n_way];
        for (_, class_id) in &self.support_set {
            class_counts[*class_id] += 1;
        }

        for (class_id, count) in class_counts.iter().enumerate() {
            if *count != self.k_shot {
                return Err(format!(
                    "Class {} has {} examples, expected {} (K-shot)",
                    class_id, count, self.k_shot
                ));
            }
        }

        Ok(())
    }
}

/// Prototypical Network classifier
#[derive(Debug, Clone)]
pub struct PrototypicalNetwork {
    /// Distance metric to use
    pub metric: DistanceMetric,
    /// Embedding dimension
    pub embedding_dim: usize,
}

impl PrototypicalNetwork {
    /// Create new prototypical network
    pub fn new(embedding_dim: usize, metric: DistanceMetric) -> Self {
        Self {
            metric,
            embedding_dim,
        }
    }

    /// Compute class prototypes from support set
    ///
    /// Prototype for class c: mean of all embeddings for examples in class c
    pub fn compute_prototypes(
        &self,
        support_set: &[(Vec<f32>, usize)],
        n_way: usize,
        embed_fn: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> Vec<Vec<f32>> {
        let mut prototypes = vec![vec![0.0; self.embedding_dim]; n_way];
        let mut class_counts = vec![0; n_way];

        // Sum embeddings for each class
        for (input, class_id) in support_set {
            let embedding = embed_fn(input);
            assert_eq!(
                embedding.len(),
                self.embedding_dim,
                "Embedding dimension mismatch"
            );

            for (i, val) in embedding.iter().enumerate() {
                prototypes[*class_id][i] += val;
            }
            class_counts[*class_id] += 1;
        }

        // Average to get prototypes
        for class_id in 0..n_way {
            let count = class_counts[class_id] as f32;
            if count > 0.0 {
                for val in prototypes[class_id].iter_mut() {
                    *val /= count;
                }
            }
        }

        prototypes
    }

    /// Classify query example using prototypes
    ///
    /// Returns (predicted_class, distances_to_prototypes)
    pub fn classify_query(
        &self,
        query: &[f32],
        prototypes: &[Vec<f32>],
        embed_fn: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> (usize, Vec<f32>) {
        let query_embedding = embed_fn(query);

        // Compute distances to all prototypes
        let distances: Vec<f32> = prototypes
            .iter()
            .map(|prototype| self.metric.distance(&query_embedding, prototype))
            .collect();

        // Find nearest prototype
        let (min_class, _) = distances
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        (min_class, distances)
    }

    /// Evaluate episode accuracy
    pub fn evaluate_episode(
        &self,
        episode: &Episode,
        embed_fn: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> f32 {
        // Compute prototypes from support set
        let prototypes = self.compute_prototypes(&episode.support_set, episode.n_way, embed_fn);

        // Classify each query
        let mut correct = 0;
        for (query_input, true_class) in &episode.query_set {
            let (predicted_class, _) = self.classify_query(query_input, &prototypes, embed_fn);
            if predicted_class == *true_class {
                correct += 1;
            }
        }

        correct as f32 / episode.query_set.len() as f32
    }

    /// Compute prototypical loss (negative log probability)
    ///
    /// Uses softmax over negative distances to convert to probabilities
    pub fn prototypical_loss(
        &self,
        episode: &Episode,
        embed_fn: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> f32 {
        let prototypes = self.compute_prototypes(&episode.support_set, episode.n_way, embed_fn);

        let mut total_loss = 0.0;

        for (query_input, true_class) in &episode.query_set {
            let query_embedding = embed_fn(query_input);

            // Compute negative distances (logits)
            let logits: Vec<f32> = prototypes
                .iter()
                .map(|prototype| -self.metric.distance(&query_embedding, prototype))
                .collect();

            // Softmax
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_logits: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
            let sum_exp: f32 = exp_logits.iter().sum();

            // Negative log probability of true class
            let log_prob = (exp_logits[*true_class] / sum_exp).ln();
            total_loss -= log_prob;
        }

        total_loss / episode.query_set.len() as f32
    }
}

/// Matching Networks (related to Prototypical Networks)
///
/// Uses attention mechanism over support set instead of class prototypes
pub struct MatchingNetwork {
    /// Distance metric
    pub metric: DistanceMetric,
    /// Embedding dimension
    pub embedding_dim: usize,
}

impl MatchingNetwork {
    /// Create new matching network
    pub fn new(embedding_dim: usize, metric: DistanceMetric) -> Self {
        Self {
            metric,
            embedding_dim,
        }
    }

    /// Classify using attention over support set
    ///
    /// Weighted k-NN: aggregate labels weighted by similarity
    pub fn classify_query(
        &self,
        query: &[f32],
        support_set: &[(Vec<f32>, usize)],
        n_way: usize,
        embed_fn: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> (usize, Vec<f32>) {
        let query_embedding = embed_fn(query);

        // Compute attention weights (softmax over similarities)
        let similarities: Vec<f32> = support_set
            .iter()
            .map(|(input, _)| {
                let support_embedding = embed_fn(input);
                -self.metric.distance(&query_embedding, &support_embedding)
            })
            .collect();

        // Softmax
        let max_sim = similarities
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sims: Vec<f32> = similarities.iter().map(|s| (s - max_sim).exp()).collect();
        let sum_exp: f32 = exp_sims.iter().sum();
        let attention: Vec<f32> = exp_sims.iter().map(|e| e / sum_exp).collect();

        // Weighted vote for each class
        let mut class_scores = vec![0.0; n_way];
        for (i, (_, class_id)) in support_set.iter().enumerate() {
            class_scores[*class_id] += attention[i];
        }

        // Predict class with highest score
        let (max_class, _) = class_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        (max_class, class_scores)
    }

    /// Evaluate episode accuracy
    pub fn evaluate_episode(
        &self,
        episode: &Episode,
        embed_fn: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> f32 {
        let mut correct = 0;
        for (query_input, true_class) in &episode.query_set {
            let (predicted_class, _) =
                self.classify_query(query_input, &episode.support_set, episode.n_way, embed_fn);
            if predicted_class == *true_class {
                correct += 1;
            }
        }

        correct as f32 / episode.query_set.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple embedding function for testing (identity or simple transformation)
    fn simple_embed(input: &[f32]) -> Vec<f32> {
        // Just normalize the input
        let norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            input.iter().map(|x| x / norm).collect()
        } else {
            input.to_vec()
        }
    }

    #[test]
    fn test_euclidean_distance() {
        let metric = DistanceMetric::Euclidean;
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        let dist = metric.distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-5); // 3-4-5 triangle
    }

    #[test]
    fn test_cosine_distance() {
        let metric = DistanceMetric::Cosine;
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];

        let dist = metric.distance(&a, &b);
        assert!(dist.abs() < 1e-5); // Same direction = 0 distance

        let c = vec![0.0, 1.0];
        let dist2 = metric.distance(&a, &c);
        assert!((dist2 - 1.0).abs() < 1e-5); // Orthogonal = distance 1
    }

    #[test]
    fn test_manhattan_distance() {
        let metric = DistanceMetric::Manhattan;
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        let dist = metric.distance(&a, &b);
        assert!((dist - 7.0).abs() < 1e-5); // |3| + |4| = 7
    }

    #[test]
    fn test_episode_creation() {
        let episode = Episode::new(3, 2) // 3-way 2-shot
            .add_support(vec![1.0, 0.0], 0)
            .add_support(vec![1.1, 0.1], 0)
            .add_support(vec![0.0, 1.0], 1)
            .add_support(vec![0.1, 1.1], 1)
            .add_support(vec![0.5, 0.5], 2)
            .add_support(vec![0.6, 0.4], 2)
            .add_query(vec![1.05, 0.05], 0);

        assert_eq!(episode.n_way, 3);
        assert_eq!(episode.k_shot, 2);
        assert_eq!(episode.support_set.len(), 6);
        assert_eq!(episode.query_set.len(), 1);
    }

    #[test]
    fn test_episode_validation() {
        let valid_episode = Episode::new(2, 2)
            .add_support(vec![1.0], 0)
            .add_support(vec![1.1], 0)
            .add_support(vec![2.0], 1)
            .add_support(vec![2.1], 1);

        assert!(valid_episode.validate().is_ok());

        let invalid_episode = Episode::new(2, 2)
            .add_support(vec![1.0], 0)
            .add_support(vec![2.0], 1)
            .add_support(vec![2.1], 1);

        assert!(invalid_episode.validate().is_err());
    }

    #[test]
    fn test_prototypical_network_creation() {
        let net = PrototypicalNetwork::new(128, DistanceMetric::Euclidean);
        assert_eq!(net.embedding_dim, 128);
        assert_eq!(net.metric, DistanceMetric::Euclidean);
    }

    #[test]
    fn test_compute_prototypes() {
        let net = PrototypicalNetwork::new(2, DistanceMetric::Euclidean);

        let support_set = vec![
            (vec![1.0, 0.0], 0),
            (vec![1.2, 0.0], 0),
            (vec![0.0, 1.0], 1),
            (vec![0.0, 1.2], 1),
        ];

        let prototypes = net.compute_prototypes(&support_set, 2, &simple_embed);

        assert_eq!(prototypes.len(), 2);
        // Class 0 prototype should be close to [1, 0] (normalized)
        // Class 1 prototype should be close to [0, 1] (normalized)
        assert!(prototypes[0][0] > 0.9); // First component of class 0
        assert!(prototypes[1][1] > 0.9); // Second component of class 1
    }

    #[test]
    fn test_classify_query() {
        let net = PrototypicalNetwork::new(2, DistanceMetric::Euclidean);

        let support_set = vec![
            (vec![1.0, 0.0], 0),
            (vec![1.2, 0.0], 0),
            (vec![0.0, 1.0], 1),
            (vec![0.0, 1.2], 1),
        ];

        let prototypes = net.compute_prototypes(&support_set, 2, &simple_embed);

        // Query close to class 0
        let query1 = vec![1.1, 0.1];
        let (pred1, _) = net.classify_query(&query1, &prototypes, &simple_embed);
        assert_eq!(pred1, 0);

        // Query close to class 1
        let query2 = vec![0.1, 1.1];
        let (pred2, _) = net.classify_query(&query2, &prototypes, &simple_embed);
        assert_eq!(pred2, 1);
    }

    #[test]
    fn test_evaluate_episode() {
        let net = PrototypicalNetwork::new(2, DistanceMetric::Euclidean);

        let episode = Episode::new(2, 2)
            .add_support(vec![1.0, 0.0], 0)
            .add_support(vec![1.2, 0.0], 0)
            .add_support(vec![0.0, 1.0], 1)
            .add_support(vec![0.0, 1.2], 1)
            .add_query(vec![1.1, 0.0], 0)
            .add_query(vec![0.0, 1.1], 1)
            .add_query(vec![1.0, 0.1], 0);

        let accuracy = net.evaluate_episode(&episode, &simple_embed);

        // Should classify all queries correctly
        assert!((accuracy - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_prototypical_loss() {
        let net = PrototypicalNetwork::new(2, DistanceMetric::Euclidean);

        let episode = Episode::new(2, 2)
            .add_support(vec![1.0, 0.0], 0)
            .add_support(vec![1.2, 0.0], 0)
            .add_support(vec![0.0, 1.0], 1)
            .add_support(vec![0.0, 1.2], 1)
            .add_query(vec![1.1, 0.0], 0);

        let loss = net.prototypical_loss(&episode, &simple_embed);

        // Loss should be positive and relatively small for easy classification
        assert!(loss > 0.0);
        assert!(loss < 2.0);
    }

    #[test]
    fn test_matching_network_creation() {
        let net = MatchingNetwork::new(64, DistanceMetric::Cosine);
        assert_eq!(net.embedding_dim, 64);
        assert_eq!(net.metric, DistanceMetric::Cosine);
    }

    #[test]
    fn test_matching_network_classify() {
        let net = MatchingNetwork::new(2, DistanceMetric::Euclidean);

        let support_set = vec![
            (vec![1.0, 0.0], 0),
            (vec![1.2, 0.0], 0),
            (vec![0.0, 1.0], 1),
            (vec![0.0, 1.2], 1),
        ];

        // Query close to class 0
        let query = vec![1.1, 0.1];
        let (pred, _) = net.classify_query(&query, &support_set, 2, &simple_embed);
        assert_eq!(pred, 0);
    }

    #[test]
    fn test_matching_network_evaluate() {
        let net = MatchingNetwork::new(2, DistanceMetric::Euclidean);

        let episode = Episode::new(2, 2)
            .add_support(vec![1.0, 0.0], 0)
            .add_support(vec![1.2, 0.0], 0)
            .add_support(vec![0.0, 1.0], 1)
            .add_support(vec![0.0, 1.2], 1)
            .add_query(vec![1.1, 0.0], 0)
            .add_query(vec![0.0, 1.1], 1);

        let accuracy = net.evaluate_episode(&episode, &simple_embed);

        // Should classify queries correctly
        assert!((accuracy - 1.0).abs() < 1e-5);
    }
}

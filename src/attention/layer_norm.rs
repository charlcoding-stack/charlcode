// Layer Normalization
// Normalizes inputs across the feature dimension to stabilize training
//
// Formula: LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
//
// Where:
// - μ (mu) = mean(x) over feature dimension
// - σ (sigma) = std(x) over feature dimension
// - γ (gamma) = learnable scale parameter (initialized to 1)
// - β (beta) = learnable shift parameter (initialized to 0)
// - ε (epsilon) = small constant for numerical stability
//
// Unlike Batch Normalization which normalizes across the batch dimension,
// Layer Normalization normalizes across features, making it better for
// variable-length sequences and recurrent architectures.

/// Layer Normalization Layer
///
/// Stabilizes training by normalizing activations across features.
/// Critical for deep networks and hybrid neural-symbolic architectures.
pub struct LayerNorm {
    /// Number of features to normalize
    normalized_shape: usize,

    /// Learnable scale parameter (gamma)
    /// Initialized to ones
    gamma: Vec<f64>,

    /// Learnable shift parameter (beta)
    /// Initialized to zeros
    beta: Vec<f64>,

    /// Small constant for numerical stability
    epsilon: f64,
}

impl LayerNorm {
    /// Create a new layer normalization layer
    ///
    /// # Arguments
    /// * `normalized_shape` - Size of the feature dimension to normalize
    /// * `epsilon` - Small constant for numerical stability (default: 1e-5)
    pub fn new(normalized_shape: usize, epsilon: f64) -> Self {
        // Initialize gamma to ones (no scaling initially)
        let gamma = vec![1.0; normalized_shape];

        // Initialize beta to zeros (no shift initially)
        let beta = vec![0.0; normalized_shape];

        LayerNorm {
            normalized_shape,
            gamma,
            beta,
            epsilon,
        }
    }

    /// Create with default epsilon
    pub fn default(normalized_shape: usize) -> Self {
        Self::new(normalized_shape, 1e-5)
    }

    /// Forward pass of layer normalization
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size, seq_len, features)
    /// * `shape` - Input shape tuple (batch, seq_len, features)
    ///
    /// # Returns
    /// * Normalized output with same shape as input
    pub fn forward(
        &self,
        input: &[f64],
        shape: (usize, usize, usize), // (batch, seq_len, features)
    ) -> Result<Vec<f64>, String> {
        let (batch_size, seq_len, features) = shape;

        // Validate feature dimension
        if features != self.normalized_shape {
            return Err(format!(
                "Input features ({}) don't match normalized_shape ({})",
                features, self.normalized_shape
            ));
        }

        if input.len() != batch_size * seq_len * features {
            return Err("Input length doesn't match shape".to_string());
        }

        let mut output = vec![0.0; input.len()];

        // Normalize each (batch, position) independently
        for b in 0..batch_size {
            for s in 0..seq_len {
                let start_idx = b * (seq_len * features) + s * features;
                let end_idx = start_idx + features;

                // Step 1: Compute mean (μ)
                let mean = Self::compute_mean(&input[start_idx..end_idx]);

                // Step 2: Compute standard deviation (σ)
                let std = Self::compute_std(&input[start_idx..end_idx], mean, self.epsilon);

                // Step 3: Normalize and apply learnable parameters
                for (i, f) in (0..features).enumerate() {
                    let idx = start_idx + f;

                    // Normalize: (x - μ) / σ
                    let normalized = (input[idx] - mean) / std;

                    // Scale and shift: γ * normalized + β
                    output[idx] = self.gamma[i] * normalized + self.beta[i];
                }
            }
        }

        Ok(output)
    }

    /// Compute mean of a slice
    fn compute_mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let sum: f64 = values.iter().sum();
        sum / values.len() as f64
    }

    /// Compute standard deviation
    fn compute_std(values: &[f64], mean: f64, epsilon: f64) -> f64 {
        if values.is_empty() {
            return epsilon;
        }

        // Variance = E[(x - μ)²]
        let variance: f64 = values
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;

        // Standard deviation = sqrt(variance + ε)
        (variance + epsilon).sqrt()
    }

    /// Get references to learnable parameters (for training/optimization)
    pub fn parameters(&self) -> (&[f64], &[f64]) {
        (&self.gamma, &self.beta)
    }

    /// Get mutable references to parameters (for updates during training)
    pub fn parameters_mut(&mut self) -> (&mut [f64], &mut [f64]) {
        (&mut self.gamma, &mut self.beta)
    }

    /// Get number of learnable parameters
    pub fn num_parameters(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }

    /// Reset parameters to default initialization
    pub fn reset_parameters(&mut self) {
        // Gamma back to ones
        for g in &mut self.gamma {
            *g = 1.0;
        }

        // Beta back to zeros
        for b in &mut self.beta {
            *b = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_creation() {
        let ln = LayerNorm::new(128, 1e-5);

        assert_eq!(ln.normalized_shape, 128);
        assert_eq!(ln.epsilon, 1e-5);
        assert_eq!(ln.gamma.len(), 128);
        assert_eq!(ln.beta.len(), 128);

        // Check initialization
        assert!(ln.gamma.iter().all(|&g| g == 1.0));
        assert!(ln.beta.iter().all(|&b| b == 0.0));
    }

    #[test]
    fn test_layer_norm_default() {
        let ln = LayerNorm::default(64);
        assert_eq!(ln.normalized_shape, 64);
        assert_eq!(ln.epsilon, 1e-5);
    }

    #[test]
    fn test_compute_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = LayerNorm::compute_mean(&values);
        assert!((mean - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_std() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let std = LayerNorm::compute_std(&values, mean, 0.0);

        // Expected: sqrt(((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²) / 5)
        //         = sqrt((4 + 1 + 0 + 1 + 4) / 5) = sqrt(2)
        assert!((std - 2.0_f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_layer_norm_forward_simple() {
        let ln = LayerNorm::new(3, 1e-5);

        // Input: (1, 1, 3) - single sample, single position, 3 features
        // Values: [1.0, 2.0, 3.0]
        // Mean: 2.0
        // Std: sqrt(2/3) ≈ 0.816
        let input = vec![1.0, 2.0, 3.0];

        let output = ln.forward(&input, (1, 1, 3)).unwrap();

        // After normalization, mean should be ~0, std should be ~1
        let out_mean = LayerNorm::compute_mean(&output);
        assert!(
            out_mean.abs() < 1e-5,
            "Mean should be close to 0, got {}",
            out_mean
        );

        let out_std = LayerNorm::compute_std(&output, out_mean, 0.0);
        assert!(
            (out_std - 1.0).abs() < 1e-4,
            "Std should be close to 1, got {}",
            out_std
        );
    }

    #[test]
    fn test_layer_norm_forward_batch() {
        let ln = LayerNorm::new(4, 1e-5);

        // Input: (2, 3, 4) - 2 samples, 3 positions, 4 features each
        let batch_size = 2;
        let seq_len = 3;
        let features = 4;

        let input = vec![1.0; batch_size * seq_len * features];

        let output = ln.forward(&input, (batch_size, seq_len, features)).unwrap();

        // Output should have same shape
        assert_eq!(output.len(), batch_size * seq_len * features);

        // Each position should be normalized independently
        // Since all inputs are 1.0, after normalization they should all be the same
        // (mean=1, std=0+eps, so all become (1-1)/eps = 0, then gamma*0 + beta = 0)
        for &val in &output {
            assert!(val.abs() < 1e-4, "Expected ~0, got {}", val);
        }
    }

    #[test]
    fn test_layer_norm_with_different_values() {
        let ln = LayerNorm::new(4, 1e-5);

        // Input with varying values
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // position 0
        ];

        let output = ln.forward(&input, (1, 1, 4)).unwrap();

        // Check that normalization happened
        let out_mean = LayerNorm::compute_mean(&output);
        let out_std = LayerNorm::compute_std(&output, out_mean, 0.0);

        assert!(out_mean.abs() < 1e-5);
        assert!((out_std - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_layer_norm_learnable_params() {
        let mut ln = LayerNorm::new(3, 1e-5);

        // Modify gamma and beta
        ln.gamma[0] = 2.0;
        ln.beta[1] = 1.0;

        let input = vec![1.0, 2.0, 3.0];
        let output = ln.forward(&input, (1, 1, 3)).unwrap();

        // First feature should be scaled by 2
        // Second feature should be shifted by 1
        // Check that parameters affect output
        assert_ne!(output[0], output[1]);
        assert_ne!(output[1], output[2]);
    }

    #[test]
    fn test_layer_norm_dimension_mismatch() {
        let ln = LayerNorm::new(3, 1e-5);

        // Wrong feature dimension
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = ln.forward(&input, (1, 1, 4));

        assert!(result.is_err());
    }

    #[test]
    fn test_reset_parameters() {
        let mut ln = LayerNorm::new(3, 1e-5);

        // Modify parameters
        ln.gamma[0] = 5.0;
        ln.beta[1] = 3.0;

        // Reset
        ln.reset_parameters();

        // Check they're back to defaults
        assert!(ln.gamma.iter().all(|&g| g == 1.0));
        assert!(ln.beta.iter().all(|&b| b == 0.0));
    }

    #[test]
    fn test_num_parameters() {
        let ln = LayerNorm::new(128, 1e-5);

        // gamma (128) + beta (128) = 256 parameters
        assert_eq!(ln.num_parameters(), 256);
    }

    #[test]
    fn test_layer_norm_multiple_positions() {
        let ln = LayerNorm::new(2, 1e-5);

        // Input: (1, 3, 2) - 1 sample, 3 positions, 2 features
        let input = vec![
            1.0, 5.0, // pos 0: mean=3, will normalize
            2.0, 4.0, // pos 1: mean=3, will normalize
            0.0, 6.0, // pos 2: mean=3, will normalize
        ];

        let output = ln.forward(&input, (1, 3, 2)).unwrap();

        // Each position normalized independently
        // Check first position
        let pos0 = &output[0..2];
        let mean0 = LayerNorm::compute_mean(pos0);
        assert!(mean0.abs() < 1e-5);

        // Check second position
        let pos1 = &output[2..4];
        let mean1 = LayerNorm::compute_mean(pos1);
        assert!(mean1.abs() < 1e-5);
    }
}

// Self-Attention
// A special case of attention where Query, Key, and Value come from the same source
//
// In self-attention, each position in the sequence can attend to all positions
// in the same sequence, allowing the model to capture dependencies within the input.
//
// Used in Transformer encoders and the first attention layer in decoders.

use crate::attention::scaled_attention::ScaledDotProductAttention;

/// Self-Attention Layer
///
/// Projects the input into Query, Key, and Value spaces using learned weights,
/// then applies scaled dot-product attention.
pub struct SelfAttention {
    /// Dimension of input/output
    d_model: usize,

    /// Dimension of keys and queries
    d_k: usize,

    /// Dimension of values
    d_v: usize,

    /// Query projection weights (d_model x d_k)
    w_q: Vec<f64>,

    /// Key projection weights (d_model x d_k)
    w_k: Vec<f64>,

    /// Value projection weights (d_model x d_v)
    w_v: Vec<f64>,

    /// Output projection weights (d_v x d_model)
    w_o: Vec<f64>,

    /// Scaled dot-product attention mechanism
    attention: ScaledDotProductAttention,

    /// Dropout probability
    dropout: f64,
}

impl SelfAttention {
    /// Create a new self-attention layer
    ///
    /// # Arguments
    /// * `d_model` - Dimension of the model (input/output)
    /// * `d_k` - Dimension of keys and queries
    /// * `d_v` - Dimension of values
    /// * `dropout` - Dropout probability
    pub fn new(d_model: usize, d_k: usize, d_v: usize, dropout: f64) -> Self {
        // Initialize weights with Xavier/Glorot initialization
        let w_q = Self::init_weights(d_model, d_k);
        let w_k = Self::init_weights(d_model, d_k);
        let w_v = Self::init_weights(d_model, d_v);
        let w_o = Self::init_weights(d_v, d_model);

        let attention = ScaledDotProductAttention::new(d_k, dropout);

        SelfAttention {
            d_model,
            d_k,
            d_v,
            w_q,
            w_k,
            w_v,
            w_o,
            attention,
            dropout,
        }
    }

    /// Initialize weights using simple random initialization
    /// TODO: Replace with proper Xavier/He initialization when RNG is available
    fn init_weights(in_dim: usize, out_dim: usize) -> Vec<f64> {
        // For now, use simple identity-like initialization for testing
        let total_size = in_dim * out_dim;
        let mut weights = vec![0.0; total_size];

        // Simple diagonal initialization for square matrices
        if in_dim == out_dim {
            for i in 0..in_dim.min(out_dim) {
                weights[i * out_dim + i] = 1.0;
            }
        } else {
            // For non-square, use small random-like values
            // In production, this should be proper random initialization
            for i in 0..total_size {
                weights[i] = 0.1 * ((i as f64 * 0.7).sin());
            }
        }

        weights
    }

    /// Forward pass of self-attention
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size, seq_len, d_model)
    /// * `mask` - Optional attention mask (batch_size, seq_len, seq_len)
    ///
    /// # Returns
    /// * Output tensor (batch_size, seq_len, d_model)
    /// * Attention weights (batch_size, seq_len, seq_len)
    pub fn forward(
        &self,
        input: &[f64],
        input_shape: (usize, usize, usize), // (batch, seq_len, d_model)
        mask: Option<&[f64]>,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let (batch_size, seq_len, d_model) = input_shape;

        // Validate input dimension
        if d_model != self.d_model {
            return Err(format!(
                "Input d_model ({}) doesn't match configured d_model ({})",
                d_model, self.d_model
            ));
        }

        // Step 1: Project input to Q, K, V
        let query = self.linear_projection(input, &self.w_q, input_shape, self.d_k)?;
        let key = self.linear_projection(input, &self.w_k, input_shape, self.d_k)?;
        let value = self.linear_projection(input, &self.w_v, input_shape, self.d_v)?;

        // Step 2: Apply scaled dot-product attention
        let (attention_output, attention_weights) = self.attention.forward(
            &query,
            &key,
            &value,
            (batch_size, seq_len, self.d_k),
            (batch_size, seq_len, self.d_k),
            (batch_size, seq_len, self.d_v),
            mask,
        )?;

        // Step 3: Project back to d_model
        let output = self.linear_projection(
            &attention_output,
            &self.w_o,
            (batch_size, seq_len, self.d_v),
            self.d_model,
        )?;

        Ok((output, attention_weights))
    }

    /// Linear projection: input @ weights
    fn linear_projection(
        &self,
        input: &[f64],
        weights: &[f64],
        input_shape: (usize, usize, usize), // (batch, seq_len, in_dim)
        out_dim: usize,
    ) -> Result<Vec<f64>, String> {
        let (batch_size, seq_len, in_dim) = input_shape;

        if weights.len() != in_dim * out_dim {
            return Err(format!(
                "Weights size ({}) doesn't match expected ({})",
                weights.len(),
                in_dim * out_dim
            ));
        }

        let mut output = vec![0.0; batch_size * seq_len * out_dim];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for o in 0..out_dim {
                    let mut sum = 0.0;
                    for i in 0..in_dim {
                        let input_idx = b * (seq_len * in_dim) + s * in_dim + i;
                        let weight_idx = i * out_dim + o;
                        sum += input[input_idx] * weights[weight_idx];
                    }
                    let output_idx = b * (seq_len * out_dim) + s * out_dim + o;
                    output[output_idx] = sum;
                }
            }
        }

        Ok(output)
    }

    /// Get reference to attention weights (for visualization/debugging)
    pub fn parameters(&self) -> (&[f64], &[f64], &[f64], &[f64]) {
        (&self.w_q, &self.w_k, &self.w_v, &self.w_o)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_attention_creation() {
        let self_attn = SelfAttention::new(128, 64, 64, 0.1);

        assert_eq!(self_attn.d_model, 128);
        assert_eq!(self_attn.d_k, 64);
        assert_eq!(self_attn.d_v, 64);

        // Check parameter counts
        let expected_params = 128 * 64 + 128 * 64 + 128 * 64 + 64 * 128;
        assert_eq!(self_attn.num_parameters(), expected_params);
    }

    #[test]
    fn test_linear_projection() {
        let self_attn = SelfAttention::new(4, 2, 2, 0.0);

        // Input: (1, 1, 4) - one sample, one position, 4 features
        let input = vec![1.0, 2.0, 3.0, 4.0];

        // Weights: (4, 2) - simple identity-like for first 2
        let weights = vec![
            1.0, 0.0, // row 0
            0.0, 1.0, // row 1
            0.0, 0.0, // row 2
            0.0, 0.0, // row 3
        ];

        let output = self_attn
            .linear_projection(&input, &weights, (1, 1, 4), 2)
            .unwrap();

        // Expected: [1.0, 2.0]
        assert_eq!(output.len(), 2);
        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_self_attention_forward_shape() {
        let self_attn = SelfAttention::new(64, 32, 32, 0.0);

        // Input: (2, 3, 64) - 2 batches, 3 tokens, 64 features
        let batch_size = 2;
        let seq_len = 3;
        let d_model = 64;

        let input = vec![0.1; batch_size * seq_len * d_model];

        let (output, attention_weights) = self_attn
            .forward(&input, (batch_size, seq_len, d_model), None)
            .unwrap();

        // Output shape should be (2, 3, 64)
        assert_eq!(output.len(), batch_size * seq_len * d_model);

        // Attention weights shape should be (2, 3, 3)
        assert_eq!(attention_weights.len(), batch_size * seq_len * seq_len);
    }

    #[test]
    fn test_self_attention_with_mask() {
        let self_attn = SelfAttention::new(32, 16, 16, 0.0);

        let batch_size = 1;
        let seq_len = 4;
        let d_model = 32;

        let input = vec![1.0; batch_size * seq_len * d_model];

        // Create a causal mask (lower triangular)
        // Each token can only attend to previous tokens and itself
        let mut mask = vec![0.0; batch_size * seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask[i * seq_len + j] = 1.0;
            }
        }

        let (output, attention_weights) = self_attn
            .forward(&input, (batch_size, seq_len, d_model), Some(&mask))
            .unwrap();

        assert_eq!(output.len(), batch_size * seq_len * d_model);
        assert_eq!(attention_weights.len(), batch_size * seq_len * seq_len);

        // Verify causal mask was applied
        // attention_weights[0, i, j] should be 0 if j > i (future positions)
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                let idx = i * seq_len + j;
                assert!(
                    attention_weights[idx].abs() < 1e-6,
                    "Future position {} should not attend to {}",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_self_attention_dimension_mismatch() {
        let self_attn = SelfAttention::new(64, 32, 32, 0.0);

        // Wrong d_model
        let input = vec![0.1; 2 * 3 * 128]; // d_model = 128, not 64

        let result = self_attn.forward(&input, (2, 3, 128), None);
        assert!(result.is_err());
    }
}

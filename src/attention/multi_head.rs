// Multi-Head Attention
// Allows the model to jointly attend to information from different representation subspaces
//
// Formula: MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
// where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
//
// Key insight: Instead of performing a single attention with d_model dimensions,
// we split into h heads with d_k dimensions each (d_model = h * d_k).
// This allows the model to focus on different aspects of the input.

use crate::attention::scaled_attention::ScaledDotProductAttention;

/// Multi-Head Attention Layer
///
/// Performs attention in parallel across multiple representation subspaces.
/// This is the core building block of Transformer models.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Number of attention heads
    num_heads: usize,

    /// Dimension of the model (d_model)
    d_model: usize,

    /// Dimension of each head (d_k = d_model / num_heads)
    d_k: usize,

    /// Dimension of values per head (typically same as d_k)
    d_v: usize,

    /// Query projection weights (d_model x d_model)
    /// Will be split into num_heads of size (d_model x d_k)
    w_q: Vec<f64>,

    /// Key projection weights (d_model x d_model)
    w_k: Vec<f64>,

    /// Value projection weights (d_model x d_model)
    w_v: Vec<f64>,

    /// Output projection weights (d_model x d_model)
    w_o: Vec<f64>,

    /// Attention mechanism (shared across all heads)
    attention: ScaledDotProductAttention,

    /// Dropout probability
    dropout: f64,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (must be divisible by num_heads)
    /// * `num_heads` - Number of attention heads
    /// * `dropout` - Dropout probability
    pub fn new(d_model: usize, num_heads: usize, dropout: f64) -> Result<Self, String> {
        // Validate that d_model is divisible by num_heads
        if !d_model.is_multiple_of(num_heads) {
            return Err(format!(
                "d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            ));
        }

        let d_k = d_model / num_heads;
        let d_v = d_k; // Typically same as d_k

        // Initialize weights (simple initialization for now)
        let w_q = Self::init_weights(d_model, d_model);
        let w_k = Self::init_weights(d_model, d_model);
        let w_v = Self::init_weights(d_model, d_model);
        let w_o = Self::init_weights(d_model, d_model);

        let attention = ScaledDotProductAttention::new(d_k, dropout);

        Ok(MultiHeadAttention {
            num_heads,
            d_model,
            d_k,
            d_v,
            w_q,
            w_k,
            w_v,
            w_o,
            attention,
            dropout,
        })
    }

    /// Initialize weights with simple pattern
    /// TODO: Replace with proper Xavier/He initialization
    fn init_weights(in_dim: usize, out_dim: usize) -> Vec<f64> {
        let total_size = in_dim * out_dim;
        let mut weights = vec![0.0; total_size];

        // Diagonal initialization for square matrices
        if in_dim == out_dim {
            for i in 0..in_dim.min(out_dim) {
                weights[i * out_dim + i] = 1.0;
            }
        } else {
            for i in 0..total_size {
                weights[i] = 0.1 * ((i as f64 * 0.7).sin());
            }
        }

        weights
    }

    /// Forward pass of multi-head attention
    ///
    /// # Arguments
    /// * `query` - Query tensor (batch, seq_len_q, d_model)
    /// * `key` - Key tensor (batch, seq_len_k, d_model)
    /// * `value` - Value tensor (batch, seq_len_v, d_model)
    /// * `mask` - Optional attention mask (batch, seq_len_q, seq_len_k)
    ///
    /// # Returns
    /// * Output tensor (batch, seq_len_q, d_model)
    /// * Attention weights (batch, num_heads, seq_len_q, seq_len_k)
    pub fn forward(
        &self,
        query: &[f64],
        key: &[f64],
        value: &[f64],
        query_shape: (usize, usize, usize), // (batch, seq_len_q, d_model)
        key_shape: (usize, usize, usize),   // (batch, seq_len_k, d_model)
        value_shape: (usize, usize, usize), // (batch, seq_len_v, d_model)
        mask: Option<&[f64]>,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let (batch_size, seq_len_q, d_model_q) = query_shape;
        let (_, seq_len_k, d_model_k) = key_shape;
        let (_, seq_len_v, d_model_v) = value_shape;

        // Validate dimensions
        if d_model_q != self.d_model || d_model_k != self.d_model || d_model_v != self.d_model {
            return Err("Input dimensions don't match d_model".to_string());
        }

        if seq_len_k != seq_len_v {
            return Err("Key and Value sequence lengths must match".to_string());
        }

        // Step 1: Linear projections
        let q_projected = self.linear_projection(query, &self.w_q, query_shape)?;
        let k_projected = self.linear_projection(key, &self.w_k, key_shape)?;
        let v_projected = self.linear_projection(value, &self.w_v, value_shape)?;

        // Step 2: Split into heads
        // (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        let q_heads = self.split_heads(&q_projected, (batch_size, seq_len_q, self.d_model))?;
        let k_heads = self.split_heads(&k_projected, (batch_size, seq_len_k, self.d_model))?;
        let v_heads = self.split_heads(&v_projected, (batch_size, seq_len_v, self.d_model))?;

        // Step 3: Apply scaled dot-product attention for each head
        let mut all_head_outputs = Vec::new();
        let mut all_attention_weights = Vec::new();

        for h in 0..self.num_heads {
            // Extract data for this head
            let q_head = self.extract_head(&q_heads, h, batch_size, seq_len_q, self.d_k);
            let k_head = self.extract_head(&k_heads, h, batch_size, seq_len_k, self.d_k);
            let v_head = self.extract_head(&v_heads, h, batch_size, seq_len_v, self.d_v);

            // Apply attention for this head
            let (head_output, head_weights) = self.attention.forward(
                &q_head,
                &k_head,
                &v_head,
                (batch_size, seq_len_q, self.d_k),
                (batch_size, seq_len_k, self.d_k),
                (batch_size, seq_len_v, self.d_v),
                mask,
            )?;

            all_head_outputs.push(head_output);
            all_attention_weights.push(head_weights);
        }

        // Step 4: Concatenate all heads
        // (batch, num_heads, seq_len_q, d_v) -> (batch, seq_len_q, d_model)
        let concat = self.concatenate_heads(&all_head_outputs, batch_size, seq_len_q)?;

        // Step 5: Final linear projection
        let output =
            self.linear_projection(&concat, &self.w_o, (batch_size, seq_len_q, self.d_model))?;

        // Combine attention weights from all heads
        // (num_heads, batch, seq_len_q, seq_len_k) -> (batch, num_heads, seq_len_q, seq_len_k)
        let combined_weights = self.combine_attention_weights(
            &all_attention_weights,
            batch_size,
            seq_len_q,
            seq_len_k,
        );

        Ok((output, combined_weights))
    }

    /// Linear projection: input @ weights
    fn linear_projection(
        &self,
        input: &[f64],
        weights: &[f64],
        input_shape: (usize, usize, usize), // (batch, seq_len, in_dim)
    ) -> Result<Vec<f64>, String> {
        let (batch_size, seq_len, in_dim) = input_shape;
        let out_dim = self.d_model;

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

    /// Split tensor into multiple heads
    /// (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
    fn split_heads(&self, input: &[f64], shape: (usize, usize, usize)) -> Result<Vec<f64>, String> {
        let (batch_size, seq_len, d_model) = shape;

        if d_model != self.d_model {
            return Err("Input d_model doesn't match".to_string());
        }

        // Output: (batch, num_heads, seq_len, d_k)
        let mut output = vec![0.0; batch_size * self.num_heads * seq_len * self.d_k];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.num_heads {
                    for k in 0..self.d_k {
                        let input_idx = b * (seq_len * d_model) + s * d_model + h * self.d_k + k;
                        let output_idx = b * (self.num_heads * seq_len * self.d_k)
                            + h * (seq_len * self.d_k)
                            + s * self.d_k
                            + k;
                        output[output_idx] = input[input_idx];
                    }
                }
            }
        }

        Ok(output)
    }

    /// Extract a single head from the split heads tensor
    fn extract_head(
        &self,
        heads: &[f64],
        head_idx: usize,
        batch_size: usize,
        seq_len: usize,
        d_k: usize,
    ) -> Vec<f64> {
        let mut head_data = vec![0.0; batch_size * seq_len * d_k];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for k in 0..d_k {
                    let src_idx = b * (self.num_heads * seq_len * d_k)
                        + head_idx * (seq_len * d_k)
                        + s * d_k
                        + k;
                    let dst_idx = b * (seq_len * d_k) + s * d_k + k;
                    head_data[dst_idx] = heads[src_idx];
                }
            }
        }

        head_data
    }

    /// Concatenate outputs from all heads
    /// Vec of (batch, seq_len, d_v) -> (batch, seq_len, d_model)
    fn concatenate_heads(
        &self,
        heads: &[Vec<f64>],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Vec<f64>, String> {
        if heads.len() != self.num_heads {
            return Err(format!(
                "Expected {} heads, got {}",
                self.num_heads,
                heads.len()
            ));
        }

        let mut output = vec![0.0; batch_size * seq_len * self.d_model];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.num_heads {
                    for v in 0..self.d_v {
                        let head_idx = b * (seq_len * self.d_v) + s * self.d_v + v;
                        let output_idx =
                            b * (seq_len * self.d_model) + s * self.d_model + h * self.d_v + v;
                        output[output_idx] = heads[h][head_idx];
                    }
                }
            }
        }

        Ok(output)
    }

    /// Combine attention weights from all heads
    fn combine_attention_weights(
        &self,
        weights: &[Vec<f64>],
        batch_size: usize,
        seq_len_q: usize,
        seq_len_k: usize,
    ) -> Vec<f64> {
        let mut combined = vec![0.0; batch_size * self.num_heads * seq_len_q * seq_len_k];

        for h in 0..self.num_heads {
            for b in 0..batch_size {
                for i in 0..seq_len_q {
                    for j in 0..seq_len_k {
                        let src_idx = b * (seq_len_q * seq_len_k) + i * seq_len_k + j;
                        let dst_idx = b * (self.num_heads * seq_len_q * seq_len_k)
                            + h * (seq_len_q * seq_len_k)
                            + i * seq_len_k
                            + j;
                        combined[dst_idx] = weights[h][src_idx];
                    }
                }
            }
        }

        combined
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
    fn test_multi_head_creation() {
        let mha = MultiHeadAttention::new(512, 8, 0.1).unwrap();

        assert_eq!(mha.d_model, 512);
        assert_eq!(mha.num_heads, 8);
        assert_eq!(mha.d_k, 64); // 512 / 8

        // Check parameter count: 4 * (d_model * d_model)
        assert_eq!(mha.num_parameters(), 4 * 512 * 512);
    }

    #[test]
    fn test_multi_head_invalid_division() {
        let result = MultiHeadAttention::new(100, 7, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_head_forward_shape() {
        let mha = MultiHeadAttention::new(64, 4, 0.0).unwrap();

        let batch = 2;
        let seq_len = 3;
        let d_model = 64;

        let query = vec![0.1; batch * seq_len * d_model];
        let key = vec![0.1; batch * seq_len * d_model];
        let value = vec![0.1; batch * seq_len * d_model];

        let (output, attention_weights) = mha
            .forward(
                &query,
                &key,
                &value,
                (batch, seq_len, d_model),
                (batch, seq_len, d_model),
                (batch, seq_len, d_model),
                None,
            )
            .unwrap();

        // Output shape: (batch, seq_len, d_model)
        assert_eq!(output.len(), batch * seq_len * d_model);

        // Attention weights shape: (batch, num_heads, seq_len, seq_len)
        assert_eq!(attention_weights.len(), batch * 4 * seq_len * seq_len);
    }

    #[test]
    fn test_multi_head_with_mask() {
        let mha = MultiHeadAttention::new(32, 2, 0.0).unwrap();

        let batch = 1;
        let seq_len = 4;
        let d_model = 32;

        let query = vec![1.0; batch * seq_len * d_model];
        let key = vec![1.0; batch * seq_len * d_model];
        let value = vec![1.0; batch * seq_len * d_model];

        // Causal mask (lower triangular)
        let mut mask = vec![0.0; batch * seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask[i * seq_len + j] = 1.0;
            }
        }

        let (output, _weights) = mha
            .forward(
                &query,
                &key,
                &value,
                (batch, seq_len, d_model),
                (batch, seq_len, d_model),
                (batch, seq_len, d_model),
                Some(&mask),
            )
            .unwrap();

        assert_eq!(output.len(), batch * seq_len * d_model);
    }

    #[test]
    fn test_split_heads() {
        let mha = MultiHeadAttention::new(64, 4, 0.0).unwrap();

        // Input: (1, 2, 64)
        let input = (0..128).map(|i| i as f64).collect::<Vec<_>>();

        let split = mha.split_heads(&input, (1, 2, 64)).unwrap();

        // Output: (1, 4, 2, 16) = 128 elements
        assert_eq!(split.len(), 128);
    }

    #[test]
    fn test_concatenate_heads() {
        let mha = MultiHeadAttention::new(64, 4, 0.0).unwrap();

        // 4 heads, each with (1, 2, 16)
        let heads = vec![
            vec![1.0; 1 * 2 * 16],
            vec![2.0; 1 * 2 * 16],
            vec![3.0; 1 * 2 * 16],
            vec![4.0; 1 * 2 * 16],
        ];

        let concat = mha.concatenate_heads(&heads, 1, 2).unwrap();

        // Output: (1, 2, 64) = 128 elements
        assert_eq!(concat.len(), 128);
    }

    #[test]
    fn test_different_seq_lengths() {
        let mha = MultiHeadAttention::new(64, 4, 0.0).unwrap();

        let batch = 1;
        let seq_len_q = 3;
        let seq_len_k = 5;
        let d_model = 64;

        let query = vec![0.1; batch * seq_len_q * d_model];
        let key = vec![0.1; batch * seq_len_k * d_model];
        let value = vec![0.1; batch * seq_len_k * d_model];

        let (output, weights) = mha
            .forward(
                &query,
                &key,
                &value,
                (batch, seq_len_q, d_model),
                (batch, seq_len_k, d_model),
                (batch, seq_len_k, d_model),
                None,
            )
            .unwrap();

        // Output shape: (batch, seq_len_q, d_model)
        assert_eq!(output.len(), batch * seq_len_q * d_model);

        // Weights shape: (batch, num_heads, seq_len_q, seq_len_k)
        assert_eq!(weights.len(), batch * 4 * seq_len_q * seq_len_k);
    }
}

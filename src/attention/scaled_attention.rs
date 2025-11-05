// Scaled Dot-Product Attention
// The fundamental attention mechanism used in Transformers
//
// Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
//
// Where:
// - Q: Query matrix (seq_len x d_k)
// - K: Key matrix (seq_len x d_k)
// - V: Value matrix (seq_len x d_v)
// - d_k: Dimension of keys (used for scaling)
//
// The scaling by sqrt(d_k) prevents dot products from growing too large

/// Scaled Dot-Product Attention
///
/// This is the core attention mechanism that computes attention scores
/// between queries and keys, then uses those scores to weight the values.
pub struct ScaledDotProductAttention {
    /// Dimension of keys (for scaling)
    d_k: usize,

    /// Dropout probability
    dropout: f64,
}

impl ScaledDotProductAttention {
    /// Create a new scaled dot-product attention
    pub fn new(d_k: usize, dropout: f64) -> Self {
        ScaledDotProductAttention { d_k, dropout }
    }

    /// Compute attention scores
    ///
    /// # Arguments
    /// * `query` - Query matrix (batch_size, seq_len, d_k)
    /// * `key` - Key matrix (batch_size, seq_len, d_k)
    /// * `value` - Value matrix (batch_size, seq_len, d_v)
    /// * `mask` - Optional attention mask (batch_size, seq_len, seq_len)
    ///
    /// # Returns
    /// * Attention output (batch_size, seq_len, d_v)
    /// * Attention weights (batch_size, seq_len, seq_len)
    pub fn forward(
        &self,
        query: &[f64],
        key: &[f64],
        value: &[f64],
        query_shape: (usize, usize, usize), // (batch, seq_len, d_k)
        key_shape: (usize, usize, usize),
        value_shape: (usize, usize, usize),
        mask: Option<&[f64]>,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let (batch_size, seq_len_q, d_k) = query_shape;
        let (_, seq_len_k, _) = key_shape;
        let (_, _, d_v) = value_shape;

        // Validate dimensions
        if d_k != self.d_k {
            return Err(format!(
                "Query d_k ({}) doesn't match configured d_k ({})",
                d_k, self.d_k
            ));
        }

        // Note: seq_len_q and seq_len_k can be different (e.g., in encoder-decoder attention)
        // But seq_len_k MUST equal seq_len_v (K and V come from same source)
        let (_, seq_len_v_check, _) = value_shape;
        if seq_len_k != seq_len_v_check {
            return Err(format!(
                "Key seq_len ({}) doesn't match Value seq_len ({})",
                seq_len_k, seq_len_v_check
            ));
        }

        // Step 1: Compute QK^T
        // Shape: (batch, seq_len, seq_len)
        let scores = self.matmul_transpose(
            query,
            key,
            (batch_size, seq_len_q, d_k),
            (batch_size, seq_len_k, d_k),
        )?;

        // Step 2: Scale by sqrt(d_k)
        let scale = (self.d_k as f64).sqrt();
        let scaled_scores: Vec<f64> = scores.iter().map(|&s| s / scale).collect();

        // Step 3: Apply mask (if provided)
        let masked_scores = if let Some(mask_data) = mask {
            self.apply_mask(&scaled_scores, mask_data, (batch_size, seq_len_q, seq_len_k))?
        } else {
            scaled_scores
        };

        // Step 4: Apply softmax
        let attention_weights =
            self.softmax(&masked_scores, (batch_size, seq_len_q, seq_len_k))?;

        // Step 5: Apply dropout (in training mode)
        // For now, we skip dropout in this MVP
        // TODO: Add dropout when training flag is available

        // Step 6: Multiply attention weights by values
        // attention_weights: (batch, seq_len_q, seq_len_k)
        // value: (batch, seq_len_k, d_v)
        // output: (batch, seq_len_q, d_v)
        let output = self.matmul(
            &attention_weights,
            value,
            (batch_size, seq_len_q, seq_len_k),
            (batch_size, seq_len_k, d_v),
        )?;

        Ok((output, attention_weights))
    }

    /// Matrix multiplication: A @ B
    fn matmul(
        &self,
        a: &[f64],
        b: &[f64],
        a_shape: (usize, usize, usize), // (batch, m, k)
        b_shape: (usize, usize, usize), // (batch, k, n)
    ) -> Result<Vec<f64>, String> {
        let (batch, m, k) = a_shape;
        let (batch_b, k_b, n) = b_shape;

        if batch != batch_b {
            return Err("Batch sizes don't match".to_string());
        }

        if k != k_b {
            return Err(format!("Inner dimensions don't match: {} vs {}", k, k_b));
        }

        let mut result = vec![0.0; batch * m * n];

        for batch_idx in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k_idx in 0..k {
                        let a_idx = batch_idx * (m * k) + i * k + k_idx;
                        let b_idx = batch_idx * (k * n) + k_idx * n + j;
                        sum += a[a_idx] * b[b_idx];
                    }
                    let result_idx = batch_idx * (m * n) + i * n + j;
                    result[result_idx] = sum;
                }
            }
        }

        Ok(result)
    }

    /// Matrix multiplication with transpose: A @ B^T
    fn matmul_transpose(
        &self,
        a: &[f64],
        b: &[f64],
        a_shape: (usize, usize, usize), // (batch, m, k)
        b_shape: (usize, usize, usize), // (batch, n, k) - will transpose to (batch, k, n)
    ) -> Result<Vec<f64>, String> {
        let (batch, m, k) = a_shape;
        let (batch_b, n, k_b) = b_shape;

        if batch != batch_b {
            return Err("Batch sizes don't match".to_string());
        }

        if k != k_b {
            return Err(format!("K dimensions don't match: {} vs {}", k, k_b));
        }

        let mut result = vec![0.0; batch * m * n];

        for batch_idx in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k_idx in 0..k {
                        let a_idx = batch_idx * (m * k) + i * k + k_idx;
                        let b_idx = batch_idx * (n * k) + j * k + k_idx; // Note: j and k swapped for transpose
                        sum += a[a_idx] * b[b_idx];
                    }
                    let result_idx = batch_idx * (m * n) + i * n + j;
                    result[result_idx] = sum;
                }
            }
        }

        Ok(result)
    }

    /// Apply attention mask
    fn apply_mask(
        &self,
        scores: &[f64],
        mask: &[f64],
        _shape: (usize, usize, usize),
    ) -> Result<Vec<f64>, String> {
        if scores.len() != mask.len() {
            return Err(format!(
                "Scores length ({}) doesn't match mask length ({})",
                scores.len(),
                mask.len()
            ));
        }

        // Mask: 1.0 = keep, 0.0 = mask out (set to -inf)
        let masked: Vec<f64> = scores
            .iter()
            .zip(mask.iter())
            .map(|(&s, &m)| if m > 0.5 { s } else { f64::NEG_INFINITY })
            .collect();

        Ok(masked)
    }

    /// Apply softmax along the last dimension
    fn softmax(
        &self,
        scores: &[f64],
        shape: (usize, usize, usize),
    ) -> Result<Vec<f64>, String> {
        let (batch, seq_len_q, seq_len_k) = shape;

        if scores.len() != batch * seq_len_q * seq_len_k {
            return Err("Scores length doesn't match shape".to_string());
        }

        let mut result = vec![0.0; scores.len()];

        // Apply softmax for each (batch, query position)
        for b in 0..batch {
            for i in 0..seq_len_q {
                let start_idx = b * (seq_len_q * seq_len_k) + i * seq_len_k;
                let end_idx = start_idx + seq_len_k;

                // Find max for numerical stability
                let max_score = scores[start_idx..end_idx]
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                // Compute exp and sum
                let mut sum = 0.0;
                let mut exp_scores = vec![0.0; seq_len_k];

                for (k, exp_score) in exp_scores.iter_mut().enumerate() {
                    *exp_score = (scores[start_idx + k] - max_score).exp();
                    sum += *exp_score;
                }

                // Normalize
                for (k, exp_score) in exp_scores.iter().enumerate() {
                    result[start_idx + k] = exp_score / sum;
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_attention_creation() {
        let attention = ScaledDotProductAttention::new(64, 0.1);
        assert_eq!(attention.d_k, 64);
        assert_eq!(attention.dropout, 0.1);
    }

    #[test]
    fn test_matmul_simple() {
        let attention = ScaledDotProductAttention::new(2, 0.0);

        // A: 1x2x2
        let a = vec![1.0, 2.0, 3.0, 4.0];
        // B: 1x2x2
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = attention
            .matmul(&a, &b, (1, 2, 2), (1, 2, 2))
            .unwrap();

        // Expected: [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
        //         = [[19, 22], [43, 50]]
        assert_eq!(result.len(), 4);
        assert!((result[0] - 19.0).abs() < 1e-6);
        assert!((result[1] - 22.0).abs() < 1e-6);
        assert!((result[2] - 43.0).abs() < 1e-6);
        assert!((result[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let attention = ScaledDotProductAttention::new(64, 0.0);

        // Simple case: 1 batch, 1 query, 3 keys
        let scores = vec![1.0, 2.0, 3.0];
        let result = attention.softmax(&scores, (1, 1, 3)).unwrap();

        // Should sum to 1
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Larger values should have larger probabilities
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_apply_mask() {
        let attention = ScaledDotProductAttention::new(64, 0.0);

        let scores = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![1.0, 1.0, 0.0, 0.0]; // Mask out last 2

        let result = attention.apply_mask(&scores, &mask, (1, 1, 4)).unwrap();

        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], f64::NEG_INFINITY);
        assert_eq!(result[3], f64::NEG_INFINITY);
    }

    #[test]
    fn test_attention_forward_simple() {
        let attention = ScaledDotProductAttention::new(2, 0.0);

        // Batch=1, seq_len=2, d_k=2, d_v=2
        let query = vec![1.0, 0.0, 0.0, 1.0]; // 2x2
        let key = vec![1.0, 0.0, 0.0, 1.0]; // 2x2
        let value = vec![1.0, 2.0, 3.0, 4.0]; // 2x2

        let (output, weights) = attention
            .forward(
                &query,
                &key,
                &value,
                (1, 2, 2), // query shape
                (1, 2, 2), // key shape
                (1, 2, 2), // value shape
                None,      // no mask
            )
            .unwrap();

        // Output should have shape (1, 2, 2) = 4 elements
        assert_eq!(output.len(), 4);

        // Weights should have shape (1, 2, 2) = 4 elements
        assert_eq!(weights.len(), 4);

        // Weights should sum to 1 for each query
        let w1_sum = weights[0] + weights[1];
        let w2_sum = weights[2] + weights[3];
        assert!((w1_sum - 1.0).abs() < 1e-6);
        assert!((w2_sum - 1.0).abs() < 1e-6);
    }
}

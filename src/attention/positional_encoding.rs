// Positional Encoding
// Adds position information to sequence embeddings using sine and cosine functions
//
// Formula:
//   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
//   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
//
// Where:
//   pos: position in the sequence
//   i: dimension index
//   d_model: model dimension
//
// This allows the model to learn to attend by relative positions,
// since for any fixed offset k, PE(pos+k) can be represented as a
// linear function of PE(pos).

use std::f64::consts::PI;

/// Positional Encoding for Transformer models
///
/// Generates sinusoidal position encodings that can be added to input embeddings
/// to inject information about the position of tokens in a sequence.
pub struct PositionalEncoding {
    /// Model dimension
    d_model: usize,

    /// Maximum sequence length to cache
    max_len: usize,

    /// Cached positional encodings (max_len x d_model)
    /// Computed once and reused for efficiency
    encodings: Vec<f64>,
}

impl Default for PositionalEncoding {
    fn default() -> Self {
        Self::new(512, 5000)
    }
}

impl PositionalEncoding {
    /// Create a new positional encoding
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (must be even)
    /// * `max_len` - Maximum sequence length to support
    pub fn new(d_model: usize, max_len: usize) -> Self {
        if d_model % 2 != 0 {
            panic!("d_model must be even for positional encoding, got {}", d_model);
        }

        let encodings = Self::compute_encodings(d_model, max_len);

        PositionalEncoding {
            d_model,
            max_len,
            encodings,
        }
    }

    /// Compute positional encodings for all positions
    fn compute_encodings(d_model: usize, max_len: usize) -> Vec<f64> {
        let mut encodings = vec![0.0; max_len * d_model];

        // Compute encoding for each position
        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                // Compute the divisor: 10000^(2i/d_model)
                let div_term = 10000_f64.powf(2.0 * i as f64 / d_model as f64);
                let angle = pos as f64 / div_term;

                // Apply sin to even indices
                encodings[pos * d_model + 2 * i] = angle.sin();

                // Apply cos to odd indices
                encodings[pos * d_model + 2 * i + 1] = angle.cos();
            }
        }

        encodings
    }

    /// Get positional encoding for a specific sequence length
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// * Positional encodings (seq_len x d_model)
    pub fn forward(&self, seq_len: usize) -> Result<Vec<f64>, String> {
        if seq_len > self.max_len {
            return Err(format!(
                "Sequence length ({}) exceeds maximum length ({})",
                seq_len, self.max_len
            ));
        }

        // Return the first seq_len positions
        let encoding = self.encodings[..seq_len * self.d_model].to_vec();
        Ok(encoding)
    }

    /// Get positional encoding with batch dimension
    ///
    /// # Arguments
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// * Positional encodings (batch x seq_len x d_model)
    pub fn forward_batch(&self, batch_size: usize, seq_len: usize) -> Result<Vec<f64>, String> {
        if seq_len > self.max_len {
            return Err(format!(
                "Sequence length ({}) exceeds maximum length ({})",
                seq_len, self.max_len
            ));
        }

        let mut batch_encoding = vec![0.0; batch_size * seq_len * self.d_model];

        // Repeat the same positional encoding for each batch
        let base_encoding = &self.encodings[..seq_len * self.d_model];
        for b in 0..batch_size {
            let start = b * seq_len * self.d_model;
            let end = start + seq_len * self.d_model;
            batch_encoding[start..end].copy_from_slice(base_encoding);
        }

        Ok(batch_encoding)
    }

    /// Get encoding for a single position
    pub fn get_position(&self, pos: usize) -> Result<&[f64], String> {
        if pos >= self.max_len {
            return Err(format!(
                "Position ({}) exceeds maximum length ({})",
                pos, self.max_len
            ));
        }

        let start = pos * self.d_model;
        let end = start + self.d_model;
        Ok(&self.encodings[start..end])
    }

    /// Get the model dimension
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Get the maximum sequence length
    pub fn max_len(&self) -> usize {
        self.max_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positional_encoding_creation() {
        let pe = PositionalEncoding::new(512, 100);
        assert_eq!(pe.d_model(), 512);
        assert_eq!(pe.max_len(), 100);
        assert_eq!(pe.encodings.len(), 100 * 512);
    }

    #[test]
    #[should_panic]
    fn test_odd_d_model() {
        // Should panic because d_model must be even
        PositionalEncoding::new(513, 100);
    }

    #[test]
    fn test_forward() {
        let pe = PositionalEncoding::new(128, 100);
        let encoding = pe.forward(50).unwrap();

        // Should return 50 positions × 128 dimensions
        assert_eq!(encoding.len(), 50 * 128);
    }

    #[test]
    fn test_forward_exceeds_max_len() {
        let pe = PositionalEncoding::new(128, 100);
        let result = pe.forward(150);

        assert!(result.is_err());
    }

    #[test]
    fn test_forward_batch() {
        let pe = PositionalEncoding::new(64, 100);
        let batch_encoding = pe.forward_batch(4, 20).unwrap();

        // Should return 4 batches × 20 positions × 64 dimensions
        assert_eq!(batch_encoding.len(), 4 * 20 * 64);

        // Each batch should have the same encoding
        let batch_0 = &batch_encoding[0..20 * 64];
        let batch_1 = &batch_encoding[20 * 64..2 * 20 * 64];

        for i in 0..batch_0.len() {
            assert!((batch_0[i] - batch_1[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_get_position() {
        let pe = PositionalEncoding::new(128, 100);

        // Get encoding for position 0
        let pos_0 = pe.get_position(0).unwrap();
        assert_eq!(pos_0.len(), 128);

        // Position 0, dimension 0 should be sin(0) = 0
        assert!((pos_0[0] - 0.0).abs() < 1e-10);

        // Position 0, dimension 1 should be cos(0) = 1
        assert!((pos_0[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_encoding_properties() {
        let pe = PositionalEncoding::new(128, 10);

        // For each position, check that encodings follow sin/cos pattern
        for pos in 0..5 {
            let encoding = pe.get_position(pos).unwrap();

            // Even indices should be sin, odd should be cos
            // Both should be in range [-1, 1]
            for i in 0..encoding.len() {
                assert!(encoding[i] >= -1.0 && encoding[i] <= 1.0,
                    "Encoding value out of range: {}", encoding[i]);
            }
        }
    }

    #[test]
    fn test_different_positions_different_encodings() {
        let pe = PositionalEncoding::new(128, 100);

        let pos_0 = pe.get_position(0).unwrap();
        let pos_1 = pe.get_position(1).unwrap();
        let pos_5 = pe.get_position(5).unwrap();

        // Different positions should have different encodings
        let mut differs_0_1 = false;
        let mut differs_0_5 = false;

        for i in 0..128 {
            if (pos_0[i] - pos_1[i]).abs() > 1e-6 {
                differs_0_1 = true;
            }
            if (pos_0[i] - pos_5[i]).abs() > 1e-6 {
                differs_0_5 = true;
            }
        }

        assert!(differs_0_1, "Positions 0 and 1 should have different encodings");
        assert!(differs_0_5, "Positions 0 and 5 should have different encodings");
    }
}

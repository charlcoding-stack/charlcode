// Attention Mechanisms - Mini-Fase 11
// Foundation for Transformers and Neuro-Symbolic AI
//
// Implements:
// 1. Scaled Dot-Product Attention
// 2. Self-Attention
// 3. Multi-Head Attention
// 4. Layer Normalization
// 5. Positional Encoding
//
// Key concepts:
// - Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
// - Multi-head allows model to attend to different representation subspaces
// - Layer norm stabilizes training
// - Positional encoding adds sequence order information

pub mod scaled_attention;
pub mod self_attention;
pub mod multi_head;
pub mod layer_norm;
pub mod positional_encoding;

pub use scaled_attention::ScaledDotProductAttention;
pub use self_attention::SelfAttention;
pub use multi_head::MultiHeadAttention;
pub use layer_norm::LayerNorm;
pub use positional_encoding::PositionalEncoding;

/// Attention configuration
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Dimensionality of the model (d_model)
    pub d_model: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Dimensionality of each head (d_k = d_model / num_heads)
    pub d_k: usize,

    /// Dropout probability
    pub dropout: f64,

    /// Use bias in linear projections
    pub use_bias: bool,
}

impl AttentionConfig {
    /// Create a standard transformer config
    pub fn transformer_base() -> Self {
        AttentionConfig {
            d_model: 512,
            num_heads: 8,
            d_k: 64, // 512 / 8
            dropout: 0.1,
            use_bias: true,
        }
    }

    /// Create a small transformer config (for testing)
    pub fn small() -> Self {
        AttentionConfig {
            d_model: 128,
            num_heads: 4,
            d_k: 32, // 128 / 4
            dropout: 0.1,
            use_bias: true,
        }
    }

    /// Create a large transformer config
    pub fn transformer_large() -> Self {
        AttentionConfig {
            d_model: 1024,
            num_heads: 16,
            d_k: 64, // 1024 / 16
            dropout: 0.1,
            use_bias: true,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model % self.num_heads != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by num_heads ({})",
                self.d_model, self.num_heads
            ));
        }

        let expected_d_k = self.d_model / self.num_heads;
        if self.d_k != expected_d_k {
            return Err(format!(
                "d_k ({}) should be d_model / num_heads = {}",
                self.d_k, expected_d_k
            ));
        }

        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err(format!("dropout ({}) must be in [0, 1]", self.dropout));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_transformer_base() {
        let config = AttentionConfig::transformer_base();
        assert_eq!(config.d_model, 512);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.d_k, 64);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_small() {
        let config = AttentionConfig::small();
        assert_eq!(config.d_model, 128);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.d_k, 32);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_division() {
        let config = AttentionConfig {
            d_model: 100,
            num_heads: 7, // 100 not divisible by 7
            d_k: 14,
            dropout: 0.1,
            use_bias: true,
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_d_k() {
        let config = AttentionConfig {
            d_model: 128,
            num_heads: 4,
            d_k: 30, // Should be 32
            dropout: 0.1,
            use_bias: true,
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_dropout() {
        let config = AttentionConfig {
            d_model: 128,
            num_heads: 4,
            d_k: 32,
            dropout: 1.5, // Invalid
            use_bias: true,
        };

        assert!(config.validate().is_err());
    }
}

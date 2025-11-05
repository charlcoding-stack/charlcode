// Quantization Module - Phase 9
// Reduces model size 4-8x and speeds up inference 2-4x
//
// Key concepts:
// - INT8: 8-bit integers (-128 to 127) → 4x memory reduction from FP32
// - INT4: 4-bit integers (-8 to 7) → 8x memory reduction from FP32
// - FP16: 16-bit floating point → 2x memory reduction from FP32
//
// Quantization formula: q = round(x / scale) + zero_point
// Dequantization formula: x = (q - zero_point) * scale

pub mod calibration;
pub mod ops;
pub mod types;

pub use calibration::{CalibrationMethod, Calibrator};
pub use ops::{
    dequantize, dequantize_tensor, post_training_quantization, quantize, quantize_tensor,
    quantize_tensor_auto, quantize_tensor_percentile, QuantizationMetrics,
};
pub use types::{QuantParams, QuantType, QuantizedTensor};

/// Quantization scheme
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantScheme {
    /// Symmetric quantization: zero_point = 0
    /// Range: [-127, 127] for INT8, [-7, 7] for INT4
    Symmetric,

    /// Asymmetric quantization: zero_point != 0
    /// Range: [-128, 127] for INT8, [-8, 7] for INT4
    /// Better for distributions not centered at 0
    Asymmetric,
}

/// Quantization granularity
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantGranularity {
    /// Per-tensor quantization: single scale/zero_point for entire tensor
    /// Memory efficient, faster, but less accurate
    PerTensor,

    /// Per-channel quantization: different scale/zero_point per output channel
    /// More accurate for weights, common in production
    PerChannel,

    /// Per-group quantization: different scale/zero_point per group of values
    /// Balance between accuracy and efficiency
    PerGroup { group_size: usize },
}

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub quant_type: QuantType,
    pub scheme: QuantScheme,
    pub granularity: QuantGranularity,
}

impl QuantConfig {
    /// Create INT8 symmetric per-tensor config (most common)
    pub fn int8_symmetric() -> Self {
        QuantConfig {
            quant_type: QuantType::INT8,
            scheme: QuantScheme::Symmetric,
            granularity: QuantGranularity::PerTensor,
        }
    }

    /// Create INT4 symmetric per-group config (good for LLMs)
    pub fn int4_per_group(group_size: usize) -> Self {
        QuantConfig {
            quant_type: QuantType::INT4,
            scheme: QuantScheme::Symmetric,
            granularity: QuantGranularity::PerGroup { group_size },
        }
    }

    /// Create FP16 config (mixed precision training)
    pub fn fp16() -> Self {
        QuantConfig {
            quant_type: QuantType::FP16,
            scheme: QuantScheme::Symmetric, // Not really used for FP16
            granularity: QuantGranularity::PerTensor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_config_int8() {
        let config = QuantConfig::int8_symmetric();
        assert_eq!(config.quant_type, QuantType::INT8);
        assert_eq!(config.scheme, QuantScheme::Symmetric);
    }

    #[test]
    fn test_quant_config_int4() {
        let config = QuantConfig::int4_per_group(128);
        assert_eq!(config.quant_type, QuantType::INT4);
        if let QuantGranularity::PerGroup { group_size } = config.granularity {
            assert_eq!(group_size, 128);
        } else {
            panic!("Expected PerGroup granularity");
        }
    }
}

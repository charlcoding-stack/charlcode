// Quantization Operations
// Core functions for quantizing and dequantizing data

use super::types::{QuantParams, QuantType, QuantizedTensor};
use super::calibration::{Calibrator, CalibrationMethod};

/// Quantize a single f32 value
#[inline]
pub fn quantize(value: f32, params: &QuantParams) -> i32 {
    params.quantize(value)
}

/// Dequantize a single quantized value
#[inline]
pub fn dequantize(quantized: i32, params: &QuantParams) -> f32 {
    params.dequantize(quantized)
}

/// Quantize a tensor from f32 to quantized representation
///
/// # Arguments
/// * `data` - F32 tensor data
/// * `params` - Quantization parameters
///
/// # Returns
/// Vector of quantized INT8 values
pub fn quantize_tensor(data: &[f32], params: &QuantParams) -> Vec<i8> {
    data.iter()
        .map(|&value| params.quantize(value) as i8)
        .collect()
}

/// Dequantize a tensor from quantized to f32
///
/// # Arguments
/// * `data` - Quantized INT8 data
/// * `params` - Quantization parameters
///
/// # Returns
/// Vector of dequantized f32 values
pub fn dequantize_tensor(data: &[i8], params: &QuantParams) -> Vec<f32> {
    data.iter()
        .map(|&q| params.dequantize(q as i32))
        .collect()
}

/// Post-Training Quantization (PTQ)
///
/// Quantize a pre-trained model's weights using calibration data
///
/// # Arguments
/// * `weights` - Model weights (f32)
/// * `calibration_data` - Representative data for calibration
/// * `quant_type` - Target quantization type
/// * `method` - Calibration method
///
/// # Returns
/// Quantized tensor
pub fn post_training_quantization(
    weights: &[f32],
    calibration_data: &[Vec<f32>],
    quant_type: QuantType,
    method: CalibrationMethod,
) -> Result<QuantizedTensor, String> {
    // Create calibrator
    let mut calibrator = Calibrator::new(method, quant_type, true);

    // Observe calibration data
    for batch in calibration_data {
        calibrator.observe(batch);
    }

    // Compute quantization parameters
    let params = calibrator.compute_params()?;

    // Quantize weights
    let quantized_data = quantize_tensor(weights, &params);

    Ok(QuantizedTensor::new(
        quantized_data,
        vec![weights.len()],
        params,
    ))
}

/// Quantize tensor with automatic calibration
///
/// Simpler API that uses the data itself for calibration
///
/// # Arguments
/// * `data` - Tensor data to quantize
/// * `shape` - Tensor shape
/// * `quant_type` - Target quantization type
///
/// # Returns
/// Quantized tensor
pub fn quantize_tensor_auto(
    data: &[f32],
    shape: Vec<usize>,
    quant_type: QuantType,
) -> Result<QuantizedTensor, String> {
    // Auto-calibration using min-max
    let mut calibrator = Calibrator::new(
        CalibrationMethod::MinMax,
        quant_type,
        true, // Symmetric
    );

    calibrator.observe(data);
    let params = calibrator.compute_params()?;

    let quantized_data = quantize_tensor(data, &params);

    Ok(QuantizedTensor::new(quantized_data, shape, params))
}

/// Quantize with percentile calibration (robust to outliers)
pub fn quantize_tensor_percentile(
    data: &[f32],
    shape: Vec<usize>,
    quant_type: QuantType,
    percentile: f32,
) -> Result<QuantizedTensor, String> {
    let mut calibrator = Calibrator::new(
        CalibrationMethod::Percentile { percentile },
        quant_type,
        true,
    );

    calibrator.observe(data);
    let params = calibrator.compute_params()?;

    let quantized_data = quantize_tensor(data, &params);

    Ok(QuantizedTensor::new(quantized_data, shape, params))
}

/// Compute quantization error metrics
pub struct QuantizationMetrics {
    /// Mean Squared Error
    pub mse: f32,
    /// Mean Absolute Error
    pub mae: f32,
    /// Signal-to-Quantization-Noise Ratio (SQNR) in dB
    pub sqnr_db: f32,
}

impl QuantizationMetrics {
    /// Compute metrics by comparing original and quantized data
    pub fn compute(original: &[f32], quantized_tensor: &QuantizedTensor) -> Self {
        let dequantized = quantized_tensor.dequantize();

        let mut mse = 0.0;
        let mut mae = 0.0;
        let mut signal_power = 0.0;
        let mut noise_power = 0.0;

        for (_i, (&orig, &deq)) in original.iter().zip(dequantized.iter()).enumerate() {
            let error = orig - deq;
            mse += error * error;
            mae += error.abs();
            signal_power += orig * orig;
            noise_power += error * error;
        }

        let n = original.len() as f32;
        mse /= n;
        mae /= n;
        signal_power /= n;
        noise_power /= n;

        let sqnr_db = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f32::INFINITY
        };

        QuantizationMetrics { mse, mae, sqnr_db }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_single() {
        let params = QuantParams::int8_symmetric(100.0);

        let value = 50.0;
        let quantized = quantize(value, &params);
        let dequantized = dequantize(quantized, &params);

        assert!((value - dequantized).abs() < 1.0);
    }

    #[test]
    fn test_quantize_tensor() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = QuantParams::int8_symmetric(5.0);

        let quantized = quantize_tensor(&data, &params);
        assert_eq!(quantized.len(), data.len());

        let dequantized = dequantize_tensor(&quantized, &params);
        for (i, (&orig, &deq)) in data.iter().zip(dequantized.iter()).enumerate() {
            assert!((orig - deq).abs() < 0.1, "Mismatch at {}: {} vs {}", i, orig, deq);
        }
    }

    #[test]
    fn test_quantize_tensor_auto() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0];
        let shape = vec![8];

        let quantized_tensor = quantize_tensor_auto(&data, shape, QuantType::INT8).unwrap();

        assert_eq!(quantized_tensor.numel(), data.len());
        assert_eq!(quantized_tensor.params.quant_type, QuantType::INT8);

        // Verify dequantization accuracy
        let dequantized = quantized_tensor.dequantize();
        for (i, (&orig, &deq)) in data.iter().zip(dequantized.iter()).enumerate() {
            assert!((orig - deq).abs() < 0.5, "Mismatch at {}: {} vs {}", i, orig, deq);
        }
    }

    #[test]
    fn test_quantize_tensor_percentile() {
        // Data with outliers
        let mut data: Vec<f32> = (1..=100).map(|x| x as f32).collect();
        data.push(1000.0); // Outlier

        let quantized_tensor = quantize_tensor_percentile(
            &data,
            vec![data.len()],
            QuantType::INT8,
            0.99,
        ).unwrap();

        assert_eq!(quantized_tensor.numel(), data.len());
    }

    #[test]
    fn test_post_training_quantization() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let calibration_data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];

        let quantized = post_training_quantization(
            &weights,
            &calibration_data,
            QuantType::INT8,
            CalibrationMethod::MinMax,
        ).unwrap();

        assert_eq!(quantized.numel(), weights.len());
        assert_eq!(quantized.params.quant_type, QuantType::INT8);
    }

    #[test]
    fn test_quantization_metrics() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let quantized_tensor = quantize_tensor_auto(&original, vec![5], QuantType::INT8).unwrap();

        let metrics = QuantizationMetrics::compute(&original, &quantized_tensor);

        // MSE and MAE should be small for simple data
        assert!(metrics.mse < 1.0);
        assert!(metrics.mae < 1.0);

        // SQNR should be high (good quality)
        assert!(metrics.sqnr_db > 20.0);
    }

    #[test]
    fn test_int4_quantization() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let quantized_tensor = quantize_tensor_auto(&data, vec![7], QuantType::INT4).unwrap();

        assert_eq!(quantized_tensor.params.quant_type, QuantType::INT4);

        // Verify accuracy (INT4 has less precision)
        let dequantized = quantized_tensor.dequantize();
        for (i, (&orig, &deq)) in data.iter().zip(dequantized.iter()).enumerate() {
            assert!((orig - deq).abs() < 1.0, "Mismatch at {}: {} vs {}", i, orig, deq);
        }
    }

    #[test]
    fn test_large_tensor_quantization() {
        // Test with larger tensor (1000 elements)
        let data: Vec<f32> = (0..1000).map(|x| (x as f32) * 0.1).collect();
        let quantized_tensor = quantize_tensor_auto(&data, vec![1000], QuantType::INT8).unwrap();

        assert_eq!(quantized_tensor.numel(), 1000);

        // Measure memory reduction
        assert_eq!(quantized_tensor.memory_reduction(), 4.0);

        // Check accuracy
        let metrics = QuantizationMetrics::compute(&data, &quantized_tensor);
        println!("Large tensor metrics: MSE={}, MAE={}, SQNR={} dB",
                 metrics.mse, metrics.mae, metrics.sqnr_db);

        assert!(metrics.sqnr_db > 30.0); // Good quality
    }
}

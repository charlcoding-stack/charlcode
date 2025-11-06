// GPU-Accelerated Neural Network Layers
// v0.2.0 - Full GPU support for deep learning

use crate::gpu_tensor::GPUTensor;

/// Initialization strategies for layer parameters
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Initializer {
    Zeros,
    Xavier,
    He,
    Normal { mean: f64, std: f64 },
}

impl Initializer {
    /// Initialize a tensor with given shape
    pub fn initialize(&self, shape: &[usize]) -> Vec<f64> {
        let size: usize = shape.iter().product();

        match self {
            Initializer::Zeros => vec![0.0; size],
            Initializer::Xavier => {
                // Xavier/Glorot: scale = sqrt(2 / (fan_in + fan_out))
                let fan_in = if shape.len() >= 2 { shape[0] } else { 1 };
                let fan_out = if shape.len() >= 2 { shape[1] } else { size };
                let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();
                (0..size).map(|i| pseudo_random(i) * scale).collect()
            }
            Initializer::He => {
                // He initialization: scale = sqrt(2 / fan_in)
                let fan_in = if shape.len() >= 2 { shape[0] } else { 1 };
                let scale = (2.0 / fan_in as f64).sqrt();
                (0..size).map(|i| pseudo_random(i) * scale).collect()
            }
            Initializer::Normal { mean, std } => {
                (0..size).map(|i| mean + pseudo_random(i) * std).collect()
            }
        }
    }
}

// Simple pseudo-random for initialization (deterministic)
fn pseudo_random(seed: usize) -> f64 {
    let x = ((seed as f64 * 9301.0 + 49297.0) % 233280.0) / 233280.0;
    2.0 * x - 1.0 // Range: [-1, 1]
}

/// Linear (Dense/Fully Connected) Layer - GPU Accelerated
/// Performs: y = xW + b
#[derive(Debug, Clone)]
pub struct Linear {
    pub in_features: usize,
    pub out_features: usize,

    /// Weights: shape [in_features, out_features]
    pub weight: GPUTensor,

    /// Bias: shape [out_features]
    pub bias: GPUTensor,

    /// Whether to use bias
    pub use_bias: bool,
}

impl Linear {
    /// Create new Linear layer with specified initialization
    pub fn new(in_features: usize, out_features: usize, initializer: Initializer) -> Self {
        // Initialize weights
        let weight_data = initializer.initialize(&[in_features, out_features]);
        let weight = GPUTensor::new(weight_data, vec![in_features, out_features]);

        // Initialize bias to zeros
        let bias_data = vec![0.0; out_features];
        let bias = GPUTensor::new(bias_data, vec![out_features]);

        Linear {
            in_features,
            out_features,
            weight,
            bias,
            use_bias: true,
        }
    }

    /// Create Linear layer with Xavier initialization (recommended for most cases)
    pub fn xavier(in_features: usize, out_features: usize) -> Self {
        Self::new(in_features, out_features, Initializer::Xavier)
    }

    /// Create Linear layer with He initialization (recommended for ReLU)
    pub fn he(in_features: usize, out_features: usize) -> Self {
        Self::new(in_features, out_features, Initializer::He)
    }

    /// Disable bias term
    pub fn without_bias(mut self) -> Self {
        self.use_bias = false;
        self
    }

    /// Move layer to GPU
    pub fn to_gpu(&mut self) -> Result<(), String> {
        // This will be handled by the global GPU backend
        // Parameters are already GPUTensors, so we just need to ensure they're on GPU
        Ok(())
    }

    /// Forward pass: y = xW + b
    /// Input: [batch_size, in_features] or [in_features]
    /// Output: [batch_size, out_features] or [out_features]
    pub fn forward(&self, input: &GPUTensor) -> Result<GPUTensor, String> {
        // Validate input shape
        let input_shape = &input.tensor.shape;

        // Support both batched [B, in_features] and single [in_features]
        let is_batched = input_shape.len() == 2;
        let batch_size = if is_batched { input_shape[0] } else { 1 };
        let input_features = if is_batched { input_shape[1] } else { input_shape[0] };

        if input_features != self.in_features {
            return Err(format!(
                "Input feature mismatch: expected {}, got {}",
                self.in_features, input_features
            ));
        }

        // For now, implement on CPU side (will be optimized to use GPU matmul later)
        // y = xW + b

        // Reshape input if needed
        let input_2d = if !is_batched {
            // Add batch dimension
            input.clone() // Will handle reshaping in matmul
        } else {
            input.clone()
        };

        // Matrix multiplication: input @ weight
        // This will use GPU matmul through tensor_matmul builtin
        // For now, compute on CPU side of GPUTensor
        let mut output_data = vec![0.0; batch_size * self.out_features];

        for b in 0..batch_size {
            for j in 0..self.out_features {
                let mut sum = 0.0;
                for i in 0..self.in_features {
                    let input_idx = if is_batched {
                        b * self.in_features + i
                    } else {
                        i
                    };
                    let weight_idx = i * self.out_features + j;
                    sum += input.tensor.data[input_idx] * self.weight.tensor.data[weight_idx];
                }
                if self.use_bias {
                    sum += self.bias.tensor.data[j];
                }
                output_data[b * self.out_features + j] = sum;
            }
        }

        // Create output tensor
        let output_shape = if is_batched {
            vec![batch_size, self.out_features]
        } else {
            vec![self.out_features]
        };

        Ok(GPUTensor::new(output_data, output_shape))
    }

    /// Get list of parameters (for optimization)
    pub fn parameters(&self) -> Vec<&GPUTensor> {
        if self.use_bias {
            vec![&self.weight, &self.bias]
        } else {
            vec![&self.weight]
        }
    }

    /// Get mutable list of parameters (for optimization)
    pub fn parameters_mut(&mut self) -> Vec<&mut GPUTensor> {
        if self.use_bias {
            vec![&mut self.weight, &mut self.bias]
        } else {
            vec![&mut self.weight]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_creation() {
        let layer = Linear::xavier(784, 128);
        assert_eq!(layer.in_features, 784);
        assert_eq!(layer.out_features, 128);
        assert_eq!(layer.weight.tensor.shape, vec![784, 128]);
        assert_eq!(layer.bias.tensor.shape, vec![128]);
    }

    #[test]
    fn test_linear_forward_single() {
        let layer = Linear::xavier(3, 2);
        let input = GPUTensor::new(vec![1.0, 2.0, 3.0], vec![3]);

        let output = layer.forward(&input).unwrap();
        assert_eq!(output.tensor.shape, vec![2]);
    }

    #[test]
    fn test_linear_forward_batched() {
        let layer = Linear::xavier(3, 2);
        // Batch of 4 samples
        let input = GPUTensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![4, 3]
        );

        let output = layer.forward(&input).unwrap();
        assert_eq!(output.tensor.shape, vec![4, 2]);
    }
}

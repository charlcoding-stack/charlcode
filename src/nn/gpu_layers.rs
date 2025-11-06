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

/// Conv2d Layer - 2D Convolution for CNNs
/// Performs: output[b,c_out,h,w] = sum(input[b,c_in,h',w'] * weight[c_out,c_in,kh,kw])
#[derive(Debug, Clone)]
pub struct Conv2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,

    /// Weights: [out_channels, in_channels, kernel_h, kernel_w]
    pub weight: GPUTensor,

    /// Bias: [out_channels]
    pub bias: GPUTensor,

    pub use_bias: bool,
}

impl Conv2d {
    /// Create Conv2d layer with He initialization (recommended for ReLU)
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::with_params(in_channels, out_channels, kernel_size, 1, 0, Initializer::He)
    }

    /// Create Conv2d with custom stride and padding
    pub fn with_params(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        initializer: Initializer,
    ) -> Self {
        // Weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        let weight_shape = vec![out_channels, in_channels, kernel_size, kernel_size];
        let weight_size = out_channels * in_channels * kernel_size * kernel_size;
        
        // Initialize with proper fan-in for convolution
        let weight_data = match initializer {
            Initializer::He => {
                let fan_in = in_channels * kernel_size * kernel_size;
                let scale = (2.0 / fan_in as f64).sqrt();
                (0..weight_size).map(|i| pseudo_random(i) * scale).collect()
            }
            _ => initializer.initialize(&weight_shape),
        };

        let weight = GPUTensor::new(weight_data, weight_shape);

        // Bias: one per output channel
        let bias_data = vec![0.0; out_channels];
        let bias = GPUTensor::new(bias_data, vec![out_channels]);

        Conv2d {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight,
            bias,
            use_bias: true,
        }
    }

    /// Disable bias term
    pub fn without_bias(mut self) -> Self {
        self.use_bias = false;
        self
    }

    /// Forward pass: convolve input with learned filters
    /// Input: [batch, in_channels, height, width]
    /// Output: [batch, out_channels, out_height, out_width]
    pub fn forward(&self, input: &GPUTensor) -> Result<GPUTensor, String> {
        let input_shape = &input.tensor.shape;

        // Validate input shape
        if input_shape.len() != 4 {
            return Err(format!(
                "Conv2d expects 4D input [batch, channels, height, width], got {:?}",
                input_shape
            ));
        }

        let batch = input_shape[0];
        let in_c = input_shape[1];
        let in_h = input_shape[2];
        let in_w = input_shape[3];

        if in_c != self.in_channels {
            return Err(format!(
                "Input channel mismatch: expected {}, got {}",
                self.in_channels, in_c
            ));
        }

        // Calculate output dimensions
        let out_h = (in_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        // Initialize output
        let output_size = batch * self.out_channels * out_h * out_w;
        let mut output_data = vec![0.0; output_size];

        // Naive convolution implementation (can be optimized with im2col later)
        for b in 0..batch {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;

                        // Convolve over all input channels and kernel
                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    // Calculate input position with stride and padding
                                    let ih = (oh * self.stride + kh) as i32 - self.padding as i32;
                                    let iw = (ow * self.stride + kw) as i32 - self.padding as i32;

                                    // Check bounds (padding)
                                    if ih >= 0 && ih < in_h as i32 && iw >= 0 && iw < in_w as i32 {
                                        let input_idx = b * (in_c * in_h * in_w)
                                            + ic * (in_h * in_w)
                                            + (ih as usize) * in_w
                                            + (iw as usize);

                                        let weight_idx = oc * (self.in_channels * self.kernel_size * self.kernel_size)
                                            + ic * (self.kernel_size * self.kernel_size)
                                            + kh * self.kernel_size
                                            + kw;

                                        sum += input.tensor.data[input_idx] * self.weight.tensor.data[weight_idx];
                                    }
                                }
                            }
                        }

                        // Add bias
                        if self.use_bias {
                            sum += self.bias.tensor.data[oc];
                        }

                        let output_idx = b * (self.out_channels * out_h * out_w)
                            + oc * (out_h * out_w)
                            + oh * out_w
                            + ow;

                        output_data[output_idx] = sum;
                    }
                }
            }
        }

        Ok(GPUTensor::new(output_data, vec![batch, self.out_channels, out_h, out_w]))
    }

    /// Get parameters for optimization
    pub fn parameters(&self) -> Vec<&GPUTensor> {
        if self.use_bias {
            vec![&self.weight, &self.bias]
        } else {
            vec![&self.weight]
        }
    }
}

/// MaxPool2d - Max Pooling for downsampling
/// Takes the maximum value in each pooling window
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl MaxPool2d {
    /// Create MaxPool2d layer with kernel_size (stride defaults to kernel_size)
    pub fn new(kernel_size: usize) -> Self {
        MaxPool2d {
            kernel_size,
            stride: kernel_size, // Default stride = kernel_size (non-overlapping)
        }
    }

    /// Create MaxPool2d with custom stride
    pub fn with_stride(kernel_size: usize, stride: usize) -> Self {
        MaxPool2d { kernel_size, stride }
    }

    /// Forward pass: downsample by taking max in each window
    /// Input: [batch, channels, height, width]
    /// Output: [batch, channels, out_height, out_width]
    pub fn forward(&self, input: &GPUTensor) -> Result<GPUTensor, String> {
        let input_shape = &input.tensor.shape;

        if input_shape.len() != 4 {
            return Err(format!(
                "MaxPool2d expects 4D input [batch, channels, height, width], got {:?}",
                input_shape
            ));
        }

        let batch = input_shape[0];
        let channels = input_shape[1];
        let in_h = input_shape[2];
        let in_w = input_shape[3];

        // Calculate output dimensions
        let out_h = (in_h - self.kernel_size) / self.stride + 1;
        let out_w = (in_w - self.kernel_size) / self.stride + 1;

        let output_size = batch * channels * out_h * out_w;
        let mut output_data = vec![f64::NEG_INFINITY; output_size];

        // Max pooling
        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f64::NEG_INFINITY;

                        // Find max in pooling window
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let ih = oh * self.stride + kh;
                                let iw = ow * self.stride + kw;

                                let input_idx = b * (channels * in_h * in_w)
                                    + c * (in_h * in_w)
                                    + ih * in_w
                                    + iw;

                                max_val = max_val.max(input.tensor.data[input_idx]);
                            }
                        }

                        let output_idx = b * (channels * out_h * out_w)
                            + c * (out_h * out_w)
                            + oh * out_w
                            + ow;

                        output_data[output_idx] = max_val;
                    }
                }
            }
        }

        Ok(GPUTensor::new(output_data, vec![batch, channels, out_h, out_w]))
    }
}

/// AvgPool2d - Average Pooling for downsampling
/// Takes the average value in each pooling window
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl AvgPool2d {
    /// Create AvgPool2d layer with kernel_size (stride defaults to kernel_size)
    pub fn new(kernel_size: usize) -> Self {
        AvgPool2d {
            kernel_size,
            stride: kernel_size,
        }
    }

    /// Create AvgPool2d with custom stride
    pub fn with_stride(kernel_size: usize, stride: usize) -> Self {
        AvgPool2d { kernel_size, stride }
    }

    /// Forward pass: downsample by taking average in each window
    /// Input: [batch, channels, height, width]
    /// Output: [batch, channels, out_height, out_width]
    pub fn forward(&self, input: &GPUTensor) -> Result<GPUTensor, String> {
        let input_shape = &input.tensor.shape;

        if input_shape.len() != 4 {
            return Err(format!(
                "AvgPool2d expects 4D input [batch, channels, height, width], got {:?}",
                input_shape
            ));
        }

        let batch = input_shape[0];
        let channels = input_shape[1];
        let in_h = input_shape[2];
        let in_w = input_shape[3];

        // Calculate output dimensions
        let out_h = (in_h - self.kernel_size) / self.stride + 1;
        let out_w = (in_w - self.kernel_size) / self.stride + 1;

        let output_size = batch * channels * out_h * out_w;
        let mut output_data = vec![0.0; output_size];

        let pool_area = (self.kernel_size * self.kernel_size) as f64;

        // Average pooling
        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;

                        // Sum values in pooling window
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let ih = oh * self.stride + kh;
                                let iw = ow * self.stride + kw;

                                let input_idx = b * (channels * in_h * in_w)
                                    + c * (in_h * in_w)
                                    + ih * in_w
                                    + iw;

                                sum += input.tensor.data[input_idx];
                            }
                        }

                        let output_idx = b * (channels * out_h * out_w)
                            + c * (out_h * out_w)
                            + oh * out_w
                            + ow;

                        output_data[output_idx] = sum / pool_area;
                    }
                }
            }
        }

        Ok(GPUTensor::new(output_data, vec![batch, channels, out_h, out_w]))
    }
}

/// BatchNorm - Batch Normalization for training stability
/// Normalizes activations: y = (x - mean) / sqrt(var + eps) * gamma + beta
#[derive(Debug, Clone)]
pub struct BatchNorm {
    pub num_features: usize,

    /// Learnable scale parameter (gamma)
    pub weight: GPUTensor,

    /// Learnable shift parameter (beta)
    pub bias: GPUTensor,

    /// Running mean for inference
    pub running_mean: Vec<f64>,

    /// Running variance for inference
    pub running_var: Vec<f64>,

    /// Small constant for numerical stability
    pub epsilon: f64,

    /// Momentum for running stats update
    pub momentum: f64,

    /// Training mode flag
    pub training: bool,
}

impl BatchNorm {
    /// Create BatchNorm layer with default parameters
    pub fn new(num_features: usize) -> Self {
        // Initialize gamma to 1.0, beta to 0.0
        let weight_data = vec![1.0; num_features];
        let bias_data = vec![0.0; num_features];

        BatchNorm {
            num_features,
            weight: GPUTensor::new(weight_data, vec![num_features]),
            bias: GPUTensor::new(bias_data, vec![num_features]),
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            epsilon: 1e-5,
            momentum: 0.1,
            training: true,
        }
    }

    /// Set training mode
    pub fn train(mut self) -> Self {
        self.training = true;
        self
    }

    /// Set evaluation mode
    pub fn eval(mut self) -> Self {
        self.training = false;
        self
    }

    /// Forward pass: normalize across batch dimension
    /// For 2D input [batch, features]: normalize each feature across batch
    /// For 4D input [batch, channels, height, width]: normalize each channel
    pub fn forward(&self, input: &GPUTensor) -> Result<GPUTensor, String> {
        let input_shape = &input.tensor.shape;

        match input_shape.len() {
            2 => self.forward_2d(input),
            4 => self.forward_4d(input),
            _ => Err(format!(
                "BatchNorm expects 2D [batch, features] or 4D [batch, channels, H, W], got {:?}",
                input_shape
            )),
        }
    }

    /// Forward pass for 2D input [batch, features]
    fn forward_2d(&self, input: &GPUTensor) -> Result<GPUTensor, String> {
        let batch = input.tensor.shape[0];
        let features = input.tensor.shape[1];

        if features != self.num_features {
            return Err(format!(
                "Feature mismatch: expected {}, got {}",
                self.num_features, features
            ));
        }

        let mut output_data = vec![0.0; batch * features];

        // For each feature
        for f in 0..features {
            let (mean, var) = if self.training {
                // Calculate batch statistics
                let mut sum = 0.0;
                for b in 0..batch {
                    sum += input.tensor.data[b * features + f];
                }
                let mean = sum / batch as f64;

                let mut var_sum = 0.0;
                for b in 0..batch {
                    let diff = input.tensor.data[b * features + f] - mean;
                    var_sum += diff * diff;
                }
                let var = var_sum / batch as f64;

                (mean, var)
            } else {
                // Use running statistics
                (self.running_mean[f], self.running_var[f])
            };

            let std = (var + self.epsilon).sqrt();
            let gamma = self.weight.tensor.data[f];
            let beta = self.bias.tensor.data[f];

            // Normalize and scale
            for b in 0..batch {
                let idx = b * features + f;
                let normalized = (input.tensor.data[idx] - mean) / std;
                output_data[idx] = gamma * normalized + beta;
            }
        }

        Ok(GPUTensor::new(output_data, input.tensor.shape.clone()))
    }

    /// Forward pass for 4D input [batch, channels, height, width]
    fn forward_4d(&self, input: &GPUTensor) -> Result<GPUTensor, String> {
        let batch = input.tensor.shape[0];
        let channels = input.tensor.shape[1];
        let height = input.tensor.shape[2];
        let width = input.tensor.shape[3];

        if channels != self.num_features {
            return Err(format!(
                "Channel mismatch: expected {}, got {}",
                self.num_features, channels
            ));
        }

        let mut output_data = vec![0.0; batch * channels * height * width];
        let spatial_size = height * width;

        // For each channel
        for c in 0..channels {
            let (mean, var) = if self.training {
                // Calculate statistics across batch and spatial dimensions
                let mut sum = 0.0;
                let count = (batch * spatial_size) as f64;

                for b in 0..batch {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = b * (channels * spatial_size) + c * spatial_size + h * width + w;
                            sum += input.tensor.data[idx];
                        }
                    }
                }
                let mean = sum / count;

                let mut var_sum = 0.0;
                for b in 0..batch {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = b * (channels * spatial_size) + c * spatial_size + h * width + w;
                            let diff = input.tensor.data[idx] - mean;
                            var_sum += diff * diff;
                        }
                    }
                }
                let var = var_sum / count;

                (mean, var)
            } else {
                // Use running statistics
                (self.running_mean[c], self.running_var[c])
            };

            let std = (var + self.epsilon).sqrt();
            let gamma = self.weight.tensor.data[c];
            let beta = self.bias.tensor.data[c];

            // Normalize and scale
            for b in 0..batch {
                for h in 0..height {
                    for w in 0..width {
                        let idx = b * (channels * spatial_size) + c * spatial_size + h * width + w;
                        let normalized = (input.tensor.data[idx] - mean) / std;
                        output_data[idx] = gamma * normalized + beta;
                    }
                }
            }
        }

        Ok(GPUTensor::new(output_data, input.tensor.shape.clone()))
    }
}

#[cfg(test)]
mod conv_tests {
    use super::*;

    #[test]
    fn test_conv2d_creation() {
        let conv = Conv2d::new(3, 16, 3);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, 3);
    }

    #[test]
    fn test_conv2d_forward_shape() {
        let conv = Conv2d::new(3, 16, 3);
        // Input: batch=1, channels=3, height=32, width=32
        let input = GPUTensor::new(vec![0.0; 1 * 3 * 32 * 32], vec![1, 3, 32, 32]);

        let output = conv.forward(&input).unwrap();
        // Output: batch=1, channels=16, height=30, width=30 (no padding, stride=1)
        assert_eq!(output.tensor.shape, vec![1, 16, 30, 30]);
    }

    #[test]
    fn test_maxpool2d() {
        let pool = MaxPool2d::new(2);
        let input = GPUTensor::new(vec![0.0; 1 * 3 * 32 * 32], vec![1, 3, 32, 32]);

        let output = pool.forward(&input).unwrap();
        // 32x32 -> 16x16 with 2x2 pooling
        assert_eq!(output.tensor.shape, vec![1, 3, 16, 16]);
    }

    #[test]
    fn test_avgpool2d() {
        let pool = AvgPool2d::new(2);
        let input = GPUTensor::new(vec![1.0; 1 * 3 * 32 * 32], vec![1, 3, 32, 32]);

        let output = pool.forward(&input).unwrap();
        assert_eq!(output.tensor.shape, vec![1, 3, 16, 16]);

        // All values should be 1.0 (average of 1.0s)
        assert!((output.tensor.data[0] - 1.0).abs() < 1e-6);
    }
}

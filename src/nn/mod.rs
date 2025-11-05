// Neural Network Module
// Phase 5: DSL for defining and training neural networks

use crate::autograd::{ComputationGraph, Tensor};

// Activation functions
#[derive(Debug, Clone, PartialEq)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
}

impl Activation {
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        match self {
            Activation::ReLU => x.iter().map(|&v| v.max(0.0)).collect(),
            Activation::Sigmoid => x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(),
            Activation::Tanh => x.iter().map(|&v| v.tanh()).collect(),
            Activation::Linear => x.to_vec(),
            Activation::Softmax => {
                let max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let exp_values: Vec<f64> = x.iter().map(|&v| (v - max).exp()).collect();
                let sum: f64 = exp_values.iter().sum();
                exp_values.iter().map(|&v| v / sum).collect()
            }
        }
    }

    pub fn backward(&self, x: &[f64], grad: &[f64]) -> Vec<f64> {
        match self {
            Activation::ReLU => x
                .iter()
                .zip(grad.iter())
                .map(|(&v, &g)| if v > 0.0 { g } else { 0.0 })
                .collect(),
            Activation::Sigmoid => {
                let sigmoid: Vec<f64> = x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
                sigmoid
                    .iter()
                    .zip(grad.iter())
                    .map(|(&s, &g)| g * s * (1.0 - s))
                    .collect()
            }
            Activation::Tanh => {
                let tanh: Vec<f64> = x.iter().map(|&v| v.tanh()).collect();
                tanh.iter()
                    .zip(grad.iter())
                    .map(|(&t, &g)| g * (1.0 - t * t))
                    .collect()
            }
            Activation::Linear => grad.to_vec(),
            Activation::Softmax => {
                // Simplified softmax gradient (assumes cross-entropy loss)
                grad.to_vec()
            }
        }
    }
}

// Parameter initialization strategies
#[derive(Debug, Clone, PartialEq)]
pub enum Initializer {
    Zeros,
    Ones,
    Xavier,
    He,
    Uniform { low: f64, high: f64 },
    Normal { mean: f64, std: f64 },
}

impl Initializer {
    pub fn initialize(&self, shape: &[usize]) -> Vec<f64> {
        let size: usize = shape.iter().product();

        match self {
            Initializer::Zeros => vec![0.0; size],
            Initializer::Ones => vec![1.0; size],
            Initializer::Xavier => {
                // Xavier/Glorot initialization: scale = sqrt(2 / (fan_in + fan_out))
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
            Initializer::Uniform { low, high } => {
                let range = high - low;
                (0..size)
                    .map(|i| low + pseudo_random(i).abs() * range)
                    .collect()
            }
            Initializer::Normal { mean, std } => {
                (0..size).map(|i| mean + pseudo_random(i) * std).collect()
            }
        }
    }
}

// Simple pseudo-random number generator for initialization
fn pseudo_random(seed: usize) -> f64 {
    let x = ((seed as f64 * 9301.0 + 49297.0) % 233280.0) / 233280.0;
    2.0 * x - 1.0 // Range: [-1, 1]
}

// Layer trait - all layers must implement this
pub trait Layer: std::fmt::Debug {
    fn forward(&mut self, input: &Tensor, graph: &mut ComputationGraph) -> Result<Tensor, String>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    fn layer_type(&self) -> &str;
}

// Dense (Fully Connected) Layer
#[derive(Debug, Clone)]
pub struct Dense {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Tensor,
    pub bias: Tensor,
    pub activation: Activation,
    pub use_bias: bool,
}

impl Dense {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Activation,
        initializer: Initializer,
    ) -> Self {
        let weight_data = initializer.initialize(&[input_size, output_size]);
        let weights = Tensor::with_grad(weight_data, vec![input_size, output_size]);

        let bias_data = vec![0.0; output_size];
        let bias = Tensor::with_grad(bias_data, vec![output_size]);

        Dense {
            input_size,
            output_size,
            weights,
            bias,
            activation,
            use_bias: true,
        }
    }

    pub fn without_bias(mut self) -> Self {
        self.use_bias = false;
        self
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Tensor, graph: &mut ComputationGraph) -> Result<Tensor, String> {
        // Simple matrix-vector multiplication for now
        // TODO: Implement proper matrix multiplication in autograd

        if input.shape[0] != self.input_size {
            return Err(format!(
                "Input size mismatch: expected {}, got {}",
                self.input_size, input.shape[0]
            ));
        }

        // Compute: output = input @ weights + bias
        let mut output_data = vec![0.0; self.output_size];

        // Matrix multiplication: input (1 x input_size) @ weights (input_size x output_size)
        for j in 0..self.output_size {
            let mut sum = 0.0;
            for i in 0..self.input_size {
                let w_idx = i * self.output_size + j;
                sum += input.data[i] * self.weights.data[w_idx];
            }
            if self.use_bias {
                sum += self.bias.data[j];
            }
            output_data[j] = sum;
        }

        // Apply activation
        let activated = self.activation.forward(&output_data);
        let output = Tensor::with_grad(activated, vec![self.output_size]);

        graph.add_node(output.clone());
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        if self.use_bias {
            vec![&self.weights, &self.bias]
        } else {
            vec![&self.weights]
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        if self.use_bias {
            vec![&mut self.weights, &mut self.bias]
        } else {
            vec![&mut self.weights]
        }
    }

    fn layer_type(&self) -> &str {
        "Dense"
    }
}

// Dropout Layer
#[derive(Debug, Clone)]
pub struct Dropout {
    pub rate: f64,
    pub training: bool,
}

impl Dropout {
    pub fn new(rate: f64) -> Self {
        Dropout {
            rate,
            training: true,
        }
    }

    pub fn eval_mode(&mut self) {
        self.training = false;
    }

    pub fn train_mode(&mut self) {
        self.training = true;
    }
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Tensor, graph: &mut ComputationGraph) -> Result<Tensor, String> {
        if !self.training || self.rate == 0.0 {
            // No dropout during evaluation
            return Ok(input.clone());
        }

        // Apply dropout: randomly set elements to zero
        let scale = 1.0 / (1.0 - self.rate);
        let mut output_data = Vec::with_capacity(input.data.len());

        for (i, &val) in input.data.iter().enumerate() {
            let keep = pseudo_random(i).abs() > self.rate;
            output_data.push(if keep { val * scale } else { 0.0 });
        }

        let output = Tensor::with_grad(output_data, input.shape.clone());
        graph.add_node(output.clone());
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![] // Dropout has no parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn layer_type(&self) -> &str {
        "Dropout"
    }
}

// Sequential Model - chains multiple layers
#[derive(Debug)]
pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
    pub name: String,
}

impl Sequential {
    pub fn new(name: String) -> Self {
        Sequential {
            layers: Vec::new(),
            name,
        }
    }

    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn forward(
        &mut self,
        input: &Tensor,
        graph: &mut ComputationGraph,
    ) -> Result<Tensor, String> {
        let mut current = input.clone();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            current = layer
                .forward(&current, graph)
                .map_err(|e| format!("Error in layer {} ({}): {}", i, layer.layer_type(), e))?;
        }

        Ok(current)
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.layers
            .iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect()
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters().iter().map(|p| p.data.len()).sum()
    }

    pub fn summary(&self) {
        println!("Model: {}", self.name);
        println!("{}", "=".repeat(70));
        println!(
            "{:<20} {:<20} {:<20}",
            "Layer (type)", "Output Shape", "Param #"
        );
        println!("{}", "-".repeat(70));

        let mut total_params = 0;

        for (i, layer) in self.layers.iter().enumerate() {
            let params: usize = layer.parameters().iter().map(|p| p.data.len()).sum();
            total_params += params;

            let layer_name = format!("{} ({})", layer.layer_type(), i);
            println!("{:<20} {:<20} {:<20}", layer_name, "N/A", params);
        }

        println!("{}", "=".repeat(70));
        println!("Total params: {}", total_params);
        println!("Trainable params: {}", total_params);
        println!("Non-trainable params: 0");
        println!("{}", "=".repeat(70));
    }
}

// Loss functions
#[derive(Debug, Clone, PartialEq)]
pub enum Loss {
    MSE,          // Mean Squared Error
    CrossEntropy, // Cross Entropy
    BinaryCrossEntropy,
}

impl Loss {
    pub fn compute(&self, predicted: &[f64], target: &[f64]) -> Result<f64, String> {
        if predicted.len() != target.len() {
            return Err(format!(
                "Shape mismatch: predicted {} vs target {}",
                predicted.len(),
                target.len()
            ));
        }

        match self {
            Loss::MSE => {
                let mse: f64 = predicted
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / predicted.len() as f64;
                Ok(mse)
            }
            Loss::CrossEntropy => {
                let ce: f64 = -predicted
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| {
                        let p_clipped = p.max(1e-15).min(1.0 - 1e-15);
                        t * p_clipped.ln()
                    })
                    .sum::<f64>();
                Ok(ce)
            }
            Loss::BinaryCrossEntropy => {
                let bce: f64 = -predicted
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| {
                        let p_clipped = p.max(1e-15).min(1.0 - 1e-15);
                        t * p_clipped.ln() + (1.0 - t) * (1.0 - p_clipped).ln()
                    })
                    .sum::<f64>()
                    / predicted.len() as f64;
                Ok(bce)
            }
        }
    }

    pub fn gradient(&self, predicted: &[f64], target: &[f64]) -> Result<Vec<f64>, String> {
        if predicted.len() != target.len() {
            return Err(format!(
                "Shape mismatch: predicted {} vs target {}",
                predicted.len(),
                target.len()
            ));
        }

        match self {
            Loss::MSE => {
                let grad: Vec<f64> = predicted
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| 2.0 * (p - t) / predicted.len() as f64)
                    .collect();
                Ok(grad)
            }
            Loss::CrossEntropy => {
                let grad: Vec<f64> = predicted
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| {
                        let p_clipped = p.max(1e-15).min(1.0 - 1e-15);
                        -t / p_clipped
                    })
                    .collect();
                Ok(grad)
            }
            Loss::BinaryCrossEntropy => {
                let grad: Vec<f64> = predicted
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| {
                        let p_clipped = p.max(1e-15).min(1.0 - 1e-15);
                        (-t / p_clipped + (1.0 - t) / (1.0 - p_clipped)) / predicted.len() as f64
                    })
                    .collect();
                Ok(grad)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_relu() {
        let activation = Activation::ReLU;
        let x = vec![-1.0, 0.0, 1.0, 2.0];
        let result = activation.forward(&x);
        assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_activation_sigmoid() {
        let activation = Activation::Sigmoid;
        let x = vec![0.0];
        let result = activation.forward(&x);
        assert!((result[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_activation_tanh() {
        let activation = Activation::Tanh;
        let x = vec![0.0];
        let result = activation.forward(&x);
        assert!((result[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_activation_softmax() {
        let activation = Activation::Softmax;
        let x = vec![1.0, 2.0, 3.0];
        let result = activation.forward(&x);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_initializer_zeros() {
        let init = Initializer::Zeros;
        let weights = init.initialize(&[2, 3]);
        assert_eq!(weights, vec![0.0; 6]);
    }

    #[test]
    fn test_initializer_ones() {
        let init = Initializer::Ones;
        let weights = init.initialize(&[2, 3]);
        assert_eq!(weights, vec![1.0; 6]);
    }

    #[test]
    fn test_initializer_xavier() {
        let init = Initializer::Xavier;
        let weights = init.initialize(&[2, 3]);
        assert_eq!(weights.len(), 6);
        // Check that weights are not all the same
        let all_same = weights.windows(2).all(|w| w[0] == w[1]);
        assert!(!all_same);
    }

    #[test]
    fn test_dense_layer_creation() {
        let layer = Dense::new(10, 5, Activation::ReLU, Initializer::Xavier);
        assert_eq!(layer.input_size, 10);
        assert_eq!(layer.output_size, 5);
        assert_eq!(layer.weights.shape, vec![10, 5]);
        assert_eq!(layer.bias.shape, vec![5]);
    }

    #[test]
    fn test_dense_layer_forward() {
        let mut graph = ComputationGraph::new();
        let mut layer = Dense::new(3, 2, Activation::Linear, Initializer::Zeros);

        // Set specific weights for testing
        layer.weights.data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        layer.bias.data = vec![0.0, 0.0];

        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        graph.add_node(input.clone());

        let output = layer.forward(&input, &mut graph).unwrap();
        assert_eq!(output.shape, vec![2]);
        // Expected: [1*1 + 2*0 + 3*1, 1*0 + 2*1 + 3*1] = [4, 5]
        assert_eq!(output.data[0], 4.0);
        assert_eq!(output.data[1], 5.0);
    }

    #[test]
    fn test_dropout_layer() {
        let mut graph = ComputationGraph::new();
        let mut layer = Dropout::new(0.5);

        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        graph.add_node(input.clone());

        let output = layer.forward(&input, &mut graph).unwrap();
        assert_eq!(output.shape, vec![4]);
    }

    #[test]
    fn test_dropout_eval_mode() {
        let mut graph = ComputationGraph::new();
        let mut layer = Dropout::new(0.5);
        layer.eval_mode();

        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        graph.add_node(input.clone());

        let output = layer.forward(&input, &mut graph).unwrap();
        assert_eq!(output.data, input.data);
    }

    #[test]
    fn test_sequential_model() {
        let mut model = Sequential::new("test_model".to_string());
        model.add(Box::new(Dense::new(
            10,
            5,
            Activation::ReLU,
            Initializer::Xavier,
        )));
        model.add(Box::new(Dense::new(
            5,
            2,
            Activation::Softmax,
            Initializer::Xavier,
        )));

        assert_eq!(model.layers.len(), 2);
        let param_count = model.parameter_count();
        assert_eq!(param_count, 10 * 5 + 5 + 5 * 2 + 2); // weights + biases
    }

    #[test]
    fn test_sequential_forward() {
        let mut graph = ComputationGraph::new();
        let mut model = Sequential::new("test".to_string());

        // Simple 2-layer network: 3 -> 2 -> 1
        model.add(Box::new(Dense::new(
            3,
            2,
            Activation::Linear,
            Initializer::Ones,
        )));
        model.add(Box::new(Dense::new(
            2,
            1,
            Activation::Linear,
            Initializer::Ones,
        )));

        let input = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]);
        graph.add_node(input.clone());

        let output = model.forward(&input, &mut graph).unwrap();
        assert_eq!(output.shape, vec![1]);
    }

    #[test]
    fn test_loss_mse() {
        let loss = Loss::MSE;
        let predicted = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];
        let result = loss.compute(&predicted, &target).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_loss_mse_nonzero() {
        let loss = Loss::MSE;
        let predicted = vec![2.0, 3.0];
        let target = vec![1.0, 2.0];
        let result = loss.compute(&predicted, &target).unwrap();
        assert_eq!(result, 1.0); // ((2-1)^2 + (3-2)^2) / 2 = 1.0
    }

    #[test]
    fn test_loss_gradient_mse() {
        let loss = Loss::MSE;
        let predicted = vec![2.0, 3.0];
        let target = vec![1.0, 2.0];
        let grad = loss.gradient(&predicted, &target).unwrap();
        assert_eq!(grad.len(), 2);
        assert!((grad[0] - 1.0).abs() < 1e-6);
        assert!((grad[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_parameter_access() {
        let layer = Dense::new(3, 2, Activation::ReLU, Initializer::Xavier);
        let params = layer.parameters();
        assert_eq!(params.len(), 2); // weights and bias
    }
}

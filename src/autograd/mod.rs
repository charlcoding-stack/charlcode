// Automatic Differentiation System
// Phase 4: Computational graph and gradient computation

use std::collections::HashMap;
use std::cell::RefCell;

// Unique ID generator for graph nodes
static mut NEXT_ID: usize = 0;

fn next_id() -> usize {
    unsafe {
        let id = NEXT_ID;
        NEXT_ID += 1;
        id
    }
}

// Global computation graph (thread-local for safety)
thread_local! {
    static GLOBAL_GRAPH: RefCell<ComputationGraph> = RefCell::new(ComputationGraph::new());
}

// Access the global graph with a closure
pub fn with_global_graph<F, R>(f: F) -> R
where
    F: FnOnce(&ComputationGraph) -> R,
{
    GLOBAL_GRAPH.with(|graph| f(&graph.borrow()))
}

// Access the global graph mutably with a closure
pub fn with_global_graph_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut ComputationGraph) -> R,
{
    GLOBAL_GRAPH.with(|graph| f(&mut graph.borrow_mut()))
}

// Clear/reset the global graph
pub fn reset_global_graph() {
    GLOBAL_GRAPH.with(|graph| {
        *graph.borrow_mut() = ComputationGraph::new();
    });
}

// Add a tensor to the global graph
pub fn add_to_global_graph(tensor: Tensor) -> usize {
    with_global_graph_mut(|graph| graph.add_node(tensor))
}

// Get a tensor from the global graph
pub fn get_from_global_graph(id: usize) -> Option<Tensor> {
    with_global_graph(|graph| graph.get_node(id).cloned())
}

// Perform backward pass on the global graph
pub fn backward_global(output_id: usize) -> Result<(), String> {
    with_global_graph_mut(|graph| graph.backward(output_id))
}

// Operation types in the computational graph
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Leaf,                 // Leaf node (input/constant)
    Add(usize, usize),    // Addition
    Sub(usize, usize),    // Subtraction
    Mul(usize, usize),    // Multiplication
    Div(usize, usize),    // Division
    Neg(usize),           // Negation
    Pow(usize, f64),      // Power (x^n)
    Sum(usize),           // Sum all elements
    MatMul(usize, usize), // Matrix multiplication
    // NN Operations
    Linear(usize, usize, usize), // Linear layer: input, weight, bias
    ReLU(usize),          // ReLU activation
    Sigmoid(usize),       // Sigmoid activation
    Tanh(usize),          // Tanh activation
    Softmax(usize),       // Softmax activation
    Embedding { input_id: usize, indices: Vec<i64> }, // Embedding lookup
    Concat { input_ids: Vec<usize>, dim: usize }, // Concatenate tensors along dimension
    // Loss functions
    MSELoss(usize, usize), // Mean Squared Error: pred, target
    CrossEntropyLoss(usize, usize), // Cross Entropy Loss: pred, target
    CrossEntropyLogitsLoss(usize, usize), // Cross Entropy Loss with logits: logits, target
}

// Tensor data structure for autograd
#[derive(Debug, Clone)]
pub struct Tensor {
    pub id: usize,
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub grad: Option<Vec<f64>>,
    pub requires_grad: bool,
    pub op: Op,
}

impl Tensor {
    // Create a new tensor from data
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Tensor {
            id: next_id(),
            data,
            shape,
            grad: None,
            requires_grad: false,
            op: Op::Leaf,
        }
    }

    // Create a tensor that requires gradients
    pub fn with_grad(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let mut tensor = Tensor::new(data, shape);
        tensor.requires_grad = true;
        tensor.grad = Some(vec![0.0; tensor.data.len()]);
        tensor
    }

    // Create a scalar tensor
    pub fn scalar(value: f64) -> Self {
        Tensor::new(vec![value], vec![1])
    }

    // Create a scalar tensor with gradients
    pub fn scalar_with_grad(value: f64) -> Self {
        Tensor::with_grad(vec![value], vec![1])
    }

    // Get total number of elements
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    // Check if shapes are compatible for element-wise operations
    pub fn same_shape(&self, other: &Tensor) -> bool {
        self.shape == other.shape
    }

    // Zero out gradients
    pub fn zero_grad(&mut self) {
        if let Some(ref mut grad) = self.grad {
            for g in grad.iter_mut() {
                *g = 0.0;
            }
        }
    }

    // Get scalar value (for 1-element tensors)
    pub fn item(&self) -> Result<f64, String> {
        if self.data.len() == 1 {
            Ok(self.data[0])
        } else {
            Err(format!(
                "item() only works on 1-element tensors, got {} elements",
                self.data.len()
            ))
        }
    }
}

// Computational graph for tracking operations
pub struct ComputationGraph {
    nodes: HashMap<usize, Tensor>,
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputationGraph {
    pub fn new() -> Self {
        ComputationGraph {
            nodes: HashMap::new(),
        }
    }

    // Add a tensor to the graph
    pub fn add_node(&mut self, tensor: Tensor) -> usize {
        let id = tensor.id;
        self.nodes.insert(id, tensor);
        id
    }

    // Get a tensor from the graph
    pub fn get_node(&self, id: usize) -> Option<&Tensor> {
        self.nodes.get(&id)
    }

    // Get mutable tensor from the graph
    pub fn get_node_mut(&mut self, id: usize) -> Option<&mut Tensor> {
        self.nodes.get_mut(&id)
    }

    // Perform backward pass from a given node
    pub fn backward(&mut self, output_id: usize) -> Result<(), String> {
        let output = self
            .get_node(output_id)
            .ok_or_else(|| "Output node not found in graph".to_string())?;

        // Initialize gradient of output to 1.0 (for scalar) or ones (for tensor)
        if !output.requires_grad {
            return Err("Cannot call backward on tensor that doesn't require grad".to_string());
        }

        let grad_size = output.data.len();
        if let Some(output_mut) = self.get_node_mut(output_id) {
            output_mut.grad = Some(vec![1.0; grad_size]);
        }

        // Topological sort to get computation order
        let sorted = self.topological_sort(output_id)?;

        // Backward pass in reverse topological order
        for node_id in sorted.iter().rev() {
            let node = self.get_node(*node_id).unwrap().clone();

            if !node.requires_grad {
                continue;
            }

            let node_grad = node
                .grad
                .clone()
                .unwrap_or_else(|| vec![0.0; node.data.len()]);

            match &node.op {
                Op::Leaf => {
                    // Leaf nodes don't propagate gradients
                }
                Op::Add(left_id, right_id) => {
                    self.backward_add(*left_id, *right_id, &node_grad)?;
                }
                Op::Sub(left_id, right_id) => {
                    self.backward_sub(*left_id, *right_id, &node_grad)?;
                }
                Op::Mul(left_id, right_id) => {
                    self.backward_mul(*left_id, *right_id, &node_grad)?;
                }
                Op::Div(left_id, right_id) => {
                    self.backward_div(*left_id, *right_id, &node_grad)?;
                }
                Op::Neg(input_id) => {
                    self.backward_neg(*input_id, &node_grad)?;
                }
                Op::Pow(input_id, exponent) => {
                    self.backward_pow(*input_id, *exponent, &node_grad)?;
                }
                Op::Sum(input_id) => {
                    self.backward_sum(*input_id, &node_grad)?;
                }
                Op::MatMul(left_id, right_id) => {
                    self.backward_matmul(*left_id, *right_id, &node_grad)?;
                }
                Op::Linear(input_id, weight_id, bias_id) => {
                    self.backward_linear(*input_id, *weight_id, *bias_id, &node_grad)?;
                }
                Op::ReLU(input_id) => {
                    self.backward_relu(*input_id, &node_grad)?;
                }
                Op::Sigmoid(input_id) => {
                    self.backward_sigmoid(*node_id, *input_id, &node_grad)?;
                }
                Op::Tanh(input_id) => {
                    self.backward_tanh(*node_id, *input_id, &node_grad)?;
                }
                Op::Softmax(input_id) => {
                    self.backward_softmax(*node_id, *input_id, &node_grad)?;
                }
                Op::Embedding { input_id, indices } => {
                    self.backward_embedding(*input_id, indices, &node_grad)?;
                }
                Op::Concat { input_ids, dim } => {
                    self.backward_concat(input_ids, *dim, &node_grad)?;
                }
                Op::MSELoss(pred_id, target_id) => {
                    self.backward_mse_loss(*pred_id, *target_id, &node_grad)?;
                }
                Op::CrossEntropyLoss(pred_id, target_id) => {
                    self.backward_cross_entropy_loss(*pred_id, *target_id, &node_grad)?;
                }
                Op::CrossEntropyLogitsLoss(logits_id, target_id) => {
                    self.backward_cross_entropy_logits_loss(*logits_id, *target_id, &node_grad)?;
                }
            }
        }

        Ok(())
    }

    // Topological sort for backward pass
    fn topological_sort(&self, start_id: usize) -> Result<Vec<usize>, String> {
        let mut sorted = Vec::new();
        let mut visited = HashMap::new();
        let mut temp_mark = HashMap::new();

        self.visit(start_id, &mut visited, &mut temp_mark, &mut sorted)?;

        Ok(sorted)
    }

    fn visit(
        &self,
        node_id: usize,
        visited: &mut HashMap<usize, bool>,
        temp_mark: &mut HashMap<usize, bool>,
        sorted: &mut Vec<usize>,
    ) -> Result<(), String> {
        if visited.get(&node_id) == Some(&true) {
            return Ok(());
        }

        if temp_mark.get(&node_id) == Some(&true) {
            return Err("Cycle detected in computation graph".to_string());
        }

        temp_mark.insert(node_id, true);

        let node = self
            .get_node(node_id)
            .ok_or_else(|| format!("Node {} not found", node_id))?;

        // Visit dependencies based on operation
        match &node.op {
            Op::Leaf => {}
            Op::Add(l, r) | Op::Sub(l, r) | Op::Mul(l, r) | Op::Div(l, r) | Op::MatMul(l, r) | Op::MSELoss(l, r) | Op::CrossEntropyLoss(l, r) | Op::CrossEntropyLogitsLoss(l, r) => {
                self.visit(*l, visited, temp_mark, sorted)?;
                self.visit(*r, visited, temp_mark, sorted)?;
            }
            Op::Neg(i) | Op::Pow(i, _) | Op::Sum(i) | Op::ReLU(i) | Op::Sigmoid(i) | Op::Tanh(i) | Op::Softmax(i) => {
                self.visit(*i, visited, temp_mark, sorted)?;
            }
            Op::Linear(input, weight, bias) => {
                self.visit(*input, visited, temp_mark, sorted)?;
                self.visit(*weight, visited, temp_mark, sorted)?;
                self.visit(*bias, visited, temp_mark, sorted)?;
            }
            Op::Embedding { input_id, .. } => {
                self.visit(*input_id, visited, temp_mark, sorted)?;
            }
            Op::Concat { input_ids, .. } => {
                for id in input_ids {
                    self.visit(*id, visited, temp_mark, sorted)?;
                }
            }
        }

        temp_mark.insert(node_id, false);
        visited.insert(node_id, true);
        sorted.push(node_id);

        Ok(())
    }

    // Backward pass implementations for each operation

    fn backward_add(
        &mut self,
        left_id: usize,
        right_id: usize,
        grad: &[f64],
    ) -> Result<(), String> {
        // d(a+b)/da = 1, d(a+b)/db = 1
        if let Some(left) = self.get_node_mut(left_id) {
            if left.requires_grad {
                if let Some(ref mut left_grad) = left.grad {
                    for (i, g) in grad.iter().enumerate() {
                        left_grad[i] += g;
                    }
                }
            }
        }

        if let Some(right) = self.get_node_mut(right_id) {
            if right.requires_grad {
                if let Some(ref mut right_grad) = right.grad {
                    for (i, g) in grad.iter().enumerate() {
                        right_grad[i] += g;
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_sub(
        &mut self,
        left_id: usize,
        right_id: usize,
        grad: &[f64],
    ) -> Result<(), String> {
        // d(a-b)/da = 1, d(a-b)/db = -1
        if let Some(left) = self.get_node_mut(left_id) {
            if left.requires_grad {
                if let Some(ref mut left_grad) = left.grad {
                    for (i, g) in grad.iter().enumerate() {
                        left_grad[i] += g;
                    }
                }
            }
        }

        if let Some(right) = self.get_node_mut(right_id) {
            if right.requires_grad {
                if let Some(ref mut right_grad) = right.grad {
                    for (i, g) in grad.iter().enumerate() {
                        right_grad[i] -= g; // Note: subtract
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_mul(
        &mut self,
        left_id: usize,
        right_id: usize,
        grad: &[f64],
    ) -> Result<(), String> {
        // d(a*b)/da = b, d(a*b)/db = a
        let left_data = self.get_node(left_id).unwrap().data.clone();
        let right_data = self.get_node(right_id).unwrap().data.clone();

        if let Some(left) = self.get_node_mut(left_id) {
            if left.requires_grad {
                if let Some(ref mut left_grad) = left.grad {
                    for (i, g) in grad.iter().enumerate() {
                        left_grad[i] += g * right_data[i];
                    }
                }
            }
        }

        if let Some(right) = self.get_node_mut(right_id) {
            if right.requires_grad {
                if let Some(ref mut right_grad) = right.grad {
                    for (i, g) in grad.iter().enumerate() {
                        right_grad[i] += g * left_data[i];
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_div(
        &mut self,
        left_id: usize,
        right_id: usize,
        grad: &[f64],
    ) -> Result<(), String> {
        // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        let left_data = self.get_node(left_id).unwrap().data.clone();
        let right_data = self.get_node(right_id).unwrap().data.clone();

        if let Some(left) = self.get_node_mut(left_id) {
            if left.requires_grad {
                if let Some(ref mut left_grad) = left.grad {
                    for (i, g) in grad.iter().enumerate() {
                        left_grad[i] += g / right_data[i];
                    }
                }
            }
        }

        if let Some(right) = self.get_node_mut(right_id) {
            if right.requires_grad {
                if let Some(ref mut right_grad) = right.grad {
                    for (i, g) in grad.iter().enumerate() {
                        right_grad[i] -= g * left_data[i] / (right_data[i] * right_data[i]);
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_neg(&mut self, input_id: usize, grad: &[f64]) -> Result<(), String> {
        // d(-a)/da = -1
        if let Some(input) = self.get_node_mut(input_id) {
            if input.requires_grad {
                if let Some(ref mut input_grad) = input.grad {
                    for (i, g) in grad.iter().enumerate() {
                        input_grad[i] -= g;
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_pow(&mut self, input_id: usize, exponent: f64, grad: &[f64]) -> Result<(), String> {
        // d(x^n)/dx = n * x^(n-1)
        let input_data = self.get_node(input_id).unwrap().data.clone();

        if let Some(input) = self.get_node_mut(input_id) {
            if input.requires_grad {
                if let Some(ref mut input_grad) = input.grad {
                    for (i, g) in grad.iter().enumerate() {
                        let derivative = exponent * input_data[i].powf(exponent - 1.0);
                        input_grad[i] += g * derivative;
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_sum(&mut self, input_id: usize, grad: &[f64]) -> Result<(), String> {
        // d(sum(x))/dx_i = 1 for all i
        // grad has shape [1] (scalar), broadcast to all elements
        let grad_value = grad[0];

        if let Some(input) = self.get_node_mut(input_id) {
            if input.requires_grad {
                if let Some(ref mut input_grad) = input.grad {
                    for g in input_grad.iter_mut() {
                        *g += grad_value;
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_matmul(
        &mut self,
        left_id: usize,
        right_id: usize,
        grad_output: &[f64],
    ) -> Result<(), String> {
        // Forward: C = A @ B  (shape: [m,k] @ [k,n] = [m,n])
        // Backward:
        //   dL/dA = dL/dC @ B^T  (shape: [m,n] @ [n,k] = [m,k])
        //   dL/dB = A^T @ dL/dC  (shape: [k,m] @ [m,n] = [k,n])

        let left_node = self.get_node(left_id).unwrap();
        let right_node = self.get_node(right_id).unwrap();

        let left_shape = left_node.shape.clone();
        let right_shape = right_node.shape.clone();
        let left_data = left_node.data.clone();
        let right_data = right_node.data.clone();

        // Assuming 2D matrices for simplicity
        if left_shape.len() != 2 || right_shape.len() != 2 {
            return Err("MatMul gradient only supports 2D matrices".to_string());
        }

        let (m, k) = (left_shape[0], left_shape[1]);
        let (_k2, n) = (right_shape[0], right_shape[1]);

        // Compute dL/dA = dL/dC @ B^T
        if let Some(left) = self.get_node_mut(left_id) {
            if left.requires_grad {
                if let Some(ref mut left_grad) = left.grad {
                    for i in 0..m {
                        for j in 0..k {
                            let mut sum = 0.0;
                            for l in 0..n {
                                sum += grad_output[i * n + l] * right_data[j * n + l];
                            }
                            left_grad[i * k + j] += sum;
                        }
                    }
                }
            }
        }

        // Compute dL/dB = A^T @ dL/dC
        if let Some(right) = self.get_node_mut(right_id) {
            if right.requires_grad {
                if let Some(ref mut right_grad) = right.grad {
                    for i in 0..k {
                        for j in 0..n {
                            let mut sum = 0.0;
                            for l in 0..m {
                                sum += left_data[l * k + i] * grad_output[l * n + j];
                            }
                            right_grad[i * n + j] += sum;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_linear(
        &mut self,
        input_id: usize,
        weight_id: usize,
        bias_id: usize,
        grad_output: &[f64],
    ) -> Result<(), String> {
        // Forward: y = x @ W + b
        // Backward:
        //   dL/dx = dL/dy @ W^T
        //   dL/dW = x^T @ dL/dy
        //   dL/db = sum(dL/dy, axis=0)

        let input_node = self.get_node(input_id).unwrap();
        let weight_node = self.get_node(weight_id).unwrap();

        let input_data = input_node.data.clone();
        let weight_data = weight_node.data.clone();
        let input_shape = input_node.shape.clone();
        let weight_shape = weight_node.shape.clone();

        let (batch, in_features) = if input_shape.len() == 1 {
            (1, input_shape[0])
        } else {
            (input_shape[0], input_shape[1])
        };
        let (_in_features2, out_features) = (weight_shape[0], weight_shape[1]);

        // dL/dx = dL/dy @ W^T
        if let Some(input) = self.get_node_mut(input_id) {
            if input.requires_grad {
                if let Some(ref mut input_grad) = input.grad {
                    for i in 0..batch {
                        for j in 0..in_features {
                            let mut sum = 0.0;
                            for k in 0..out_features {
                                sum += grad_output[i * out_features + k] * weight_data[j * out_features + k];
                            }
                            let idx = if input_shape.len() == 1 { j } else { i * in_features + j };
                            input_grad[idx] += sum;
                        }
                    }
                }
            }
        }

        // dL/dW = x^T @ dL/dy
        if let Some(weight) = self.get_node_mut(weight_id) {
            if weight.requires_grad {
                if let Some(ref mut weight_grad) = weight.grad {
                    for i in 0..in_features {
                        for j in 0..out_features {
                            let mut sum = 0.0;
                            for k in 0..batch {
                                let input_val = if input_shape.len() == 1 {
                                    input_data[i]
                                } else {
                                    input_data[k * in_features + i]
                                };
                                sum += input_val * grad_output[k * out_features + j];
                            }
                            weight_grad[i * out_features + j] += sum;
                        }
                    }
                }
            }
        }

        // dL/db = sum(dL/dy, axis=0)
        if let Some(bias) = self.get_node_mut(bias_id) {
            if bias.requires_grad {
                if let Some(ref mut bias_grad) = bias.grad {
                    for j in 0..out_features {
                        let mut sum = 0.0;
                        for i in 0..batch {
                            sum += grad_output[i * out_features + j];
                        }
                        bias_grad[j] += sum;
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_relu(
        &mut self,
        input_id: usize,
        grad_output: &[f64],
    ) -> Result<(), String> {
        // Forward: y = max(0, x)
        // Backward: dL/dx = dL/dy * (x > 0 ? 1 : 0)

        let input_data = self.get_node(input_id).unwrap().data.clone();

        if let Some(input) = self.get_node_mut(input_id) {
            if input.requires_grad {
                if let Some(ref mut input_grad) = input.grad {
                    for (i, (&x, &g)) in input_data.iter().zip(grad_output.iter()).enumerate() {
                        input_grad[i] += if x > 0.0 { g } else { 0.0 };
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_sigmoid(
        &mut self,
        output_id: usize,
        input_id: usize,
        grad_output: &[f64],
    ) -> Result<(), String> {
        // Forward: y = 1 / (1 + e^-x)
        // Backward: dL/dx = dL/dy * y * (1 - y)

        let output_data = self.get_node(output_id).unwrap().data.clone();

        if let Some(input) = self.get_node_mut(input_id) {
            if input.requires_grad {
                if let Some(ref mut input_grad) = input.grad {
                    for (i, (&y, &g)) in output_data.iter().zip(grad_output.iter()).enumerate() {
                        input_grad[i] += g * y * (1.0 - y);
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_tanh(
        &mut self,
        output_id: usize,
        input_id: usize,
        grad_output: &[f64],
    ) -> Result<(), String> {
        // Forward: y = tanh(x)
        // Backward: dL/dx = dL/dy * (1 - y^2)

        let output_data = self.get_node(output_id).unwrap().data.clone();

        if let Some(input) = self.get_node_mut(input_id) {
            if input.requires_grad {
                if let Some(ref mut input_grad) = input.grad {
                    for (i, (&y, &g)) in output_data.iter().zip(grad_output.iter()).enumerate() {
                        input_grad[i] += g * (1.0 - y * y);
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_softmax(
        &mut self,
        output_id: usize,
        input_id: usize,
        grad_output: &[f64],
    ) -> Result<(), String> {
        // Forward: y_i = e^x_i / sum(e^x_j)
        // Backward: dL/dx_i = sum_j(dL/dy_j * (delta_ij * y_i - y_i * y_j))
        //                    = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))

        let output_data = self.get_node(output_id).unwrap().data.clone();
        let n = output_data.len();

        // Compute sum_j(dL/dy_j * y_j)
        let mut dot_product = 0.0;
        for i in 0..n {
            dot_product += grad_output[i] * output_data[i];
        }

        if let Some(input) = self.get_node_mut(input_id) {
            if input.requires_grad {
                if let Some(ref mut input_grad) = input.grad {
                    for i in 0..n {
                        input_grad[i] += output_data[i] * (grad_output[i] - dot_product);
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_embedding(
        &mut self,
        embedding_id: usize,
        indices: &[i64],
        grad_output: &[f64],
    ) -> Result<(), String> {
        // Forward: output[i] = embedding_matrix[indices[i]]
        // Backward: embedding_matrix.grad[indices[i]] += grad_output[i]

        let embedding = self.get_node(embedding_id).unwrap();
        let embedding_dim = embedding.shape[1];

        if let Some(embedding_mut) = self.get_node_mut(embedding_id) {
            if embedding_mut.requires_grad {
                if let Some(ref mut embedding_grad) = embedding_mut.grad {
                    // Scatter gradients back to the embedding matrix
                    for (i, &idx) in indices.iter().enumerate() {
                        let idx = idx as usize;
                        let grad_start = i * embedding_dim;
                        let emb_start = idx * embedding_dim;

                        for j in 0..embedding_dim {
                            embedding_grad[emb_start + j] += grad_output[grad_start + j];
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_concat(
        &mut self,
        input_ids: &[usize],
        dim: usize,
        grad_output: &[f64],
    ) -> Result<(), String> {
        // Forward: concat([t1, t2, ...], dim=1) concatenates along feature dimension
        // Backward: split grad_output and send chunks to each input

        if dim != 1 {
            return Err(format!("backward_concat(): only dim=1 supported, got dim={}", dim));
        }

        // Get shapes of input tensors
        let mut input_shapes = Vec::new();
        for &id in input_ids {
            let tensor = self.get_node(id).unwrap();
            input_shapes.push(tensor.shape.clone());
        }

        let batch_size = input_shapes[0][0];

        // For each input tensor, extract its portion of the gradient
        let mut offset = 0;
        for (i, &input_id) in input_ids.iter().enumerate() {
            let features = input_shapes[i][1];

            if let Some(input) = self.get_node_mut(input_id) {
                if input.requires_grad {
                    if let Some(ref mut input_grad) = input.grad {
                        // Extract the gradient chunk for this tensor
                        for batch_idx in 0..batch_size {
                            let total_features: usize = input_shapes.iter().map(|s| s[1]).sum();
                            let grad_row_start = batch_idx * total_features;
                            let input_row_start = batch_idx * features;

                            for feat_idx in 0..features {
                                let grad_idx = grad_row_start + offset + feat_idx;
                                let input_idx = input_row_start + feat_idx;
                                input_grad[input_idx] += grad_output[grad_idx];
                            }
                        }
                    }
                }
            }

            offset += features;
        }

        Ok(())
    }

    fn backward_mse_loss(
        &mut self,
        pred_id: usize,
        target_id: usize,
        _grad_output: &[f64],
    ) -> Result<(), String> {
        // Forward: L = mean((pred - target)^2)
        // Backward: dL/dpred = 2/n * (pred - target)

        let pred_data = self.get_node(pred_id).unwrap().data.clone();
        let target_data = self.get_node(target_id).unwrap().data.clone();
        let n = pred_data.len() as f64;

        if let Some(pred) = self.get_node_mut(pred_id) {
            if pred.requires_grad {
                if let Some(ref mut pred_grad) = pred.grad {
                    for (i, (&p, &t)) in pred_data.iter().zip(target_data.iter()).enumerate() {
                        pred_grad[i] += 2.0 / n * (p - t);
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_cross_entropy_loss(
        &mut self,
        pred_id: usize,
        target_id: usize,
        _grad_output: &[f64],
    ) -> Result<(), String> {
        // Forward: L = -mean(target * log(pred))
        // Backward: dL/dpred = -target / pred / n
        // With clipping to avoid division by zero

        let pred_data = self.get_node(pred_id).unwrap().data.clone();
        let target_data = self.get_node(target_id).unwrap().data.clone();
        let n = pred_data.len() as f64;
        let epsilon = 1e-10;

        if let Some(pred) = self.get_node_mut(pred_id) {
            if pred.requires_grad {
                if let Some(ref mut pred_grad) = pred.grad {
                    for (i, (&p, &t)) in pred_data.iter().zip(target_data.iter()).enumerate() {
                        // Clip prediction to avoid division by zero
                        let clipped_p = p.max(epsilon).min(1.0 - epsilon);
                        // Gradient: -target / pred / n
                        pred_grad[i] += -t / clipped_p / n;
                    }
                }
            }
        }

        Ok(())
    }

    fn backward_cross_entropy_logits_loss(
        &mut self,
        logits_id: usize,
        target_id: usize,
        _grad_output: &[f64],
    ) -> Result<(), String> {
        // Forward: L = -mean(target * log(softmax(logits)))
        // Backward: dL/dlogits = (softmax(logits) - target) / n
        //
        // This is the combined gradient of softmax + cross_entropy
        // Much simpler than computing them separately!

        let logits_data = self.get_node(logits_id).unwrap().data.clone();
        let target_data = self.get_node(target_id).unwrap().data.clone();
        let n = logits_data.len() as f64;

        // Compute softmax(logits) for gradient
        let max = logits_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f64> = logits_data.iter().map(|&x| (x - max).exp()).collect();
        let sum: f64 = exp_values.iter().sum();
        let softmax: Vec<f64> = exp_values.iter().map(|&x| x / sum).collect();

        if let Some(logits) = self.get_node_mut(logits_id) {
            if logits.requires_grad {
                if let Some(ref mut logits_grad) = logits.grad {
                    for (i, (&s, &t)) in softmax.iter().zip(target_data.iter()).enumerate() {
                        // Gradient: (softmax - target) / n
                        logits_grad[i] += (s - t) / n;
                    }
                }
            }
        }

        Ok(())
    }
}

// Forward operations that create computation graph nodes

pub fn add(graph: &mut ComputationGraph, a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if !a.same_shape(b) {
        return Err(format!("Shape mismatch: {:?} vs {:?}", a.shape, b.shape));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x + y)
        .collect();
    let requires_grad = a.requires_grad || b.requires_grad;

    let mut result = Tensor::new(data, a.shape.clone());
    result.requires_grad = requires_grad;
    if requires_grad {
        result.grad = Some(vec![0.0; result.data.len()]);
    }
    result.op = Op::Add(a.id, b.id);

    graph.add_node(result.clone());
    Ok(result)
}

pub fn sub(graph: &mut ComputationGraph, a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if !a.same_shape(b) {
        return Err(format!("Shape mismatch: {:?} vs {:?}", a.shape, b.shape));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x - y)
        .collect();
    let requires_grad = a.requires_grad || b.requires_grad;

    let mut result = Tensor::new(data, a.shape.clone());
    result.requires_grad = requires_grad;
    if requires_grad {
        result.grad = Some(vec![0.0; result.data.len()]);
    }
    result.op = Op::Sub(a.id, b.id);

    graph.add_node(result.clone());
    Ok(result)
}

pub fn mul(graph: &mut ComputationGraph, a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if !a.same_shape(b) {
        return Err(format!("Shape mismatch: {:?} vs {:?}", a.shape, b.shape));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x * y)
        .collect();
    let requires_grad = a.requires_grad || b.requires_grad;

    let mut result = Tensor::new(data, a.shape.clone());
    result.requires_grad = requires_grad;
    if requires_grad {
        result.grad = Some(vec![0.0; result.data.len()]);
    }
    result.op = Op::Mul(a.id, b.id);

    graph.add_node(result.clone());
    Ok(result)
}

pub fn div(graph: &mut ComputationGraph, a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if !a.same_shape(b) {
        return Err(format!("Shape mismatch: {:?} vs {:?}", a.shape, b.shape));
    }

    // Check for division by zero
    for &val in &b.data {
        if val == 0.0 {
            return Err("Division by zero".to_string());
        }
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x / y)
        .collect();
    let requires_grad = a.requires_grad || b.requires_grad;

    let mut result = Tensor::new(data, a.shape.clone());
    result.requires_grad = requires_grad;
    if requires_grad {
        result.grad = Some(vec![0.0; result.data.len()]);
    }
    result.op = Op::Div(a.id, b.id);

    graph.add_node(result.clone());
    Ok(result)
}

pub fn neg(graph: &mut ComputationGraph, a: &Tensor) -> Result<Tensor, String> {
    let data: Vec<f64> = a.data.iter().map(|x| -x).collect();
    let requires_grad = a.requires_grad;

    let mut result = Tensor::new(data, a.shape.clone());
    result.requires_grad = requires_grad;
    if requires_grad {
        result.grad = Some(vec![0.0; result.data.len()]);
    }
    result.op = Op::Neg(a.id);

    graph.add_node(result.clone());
    Ok(result)
}

pub fn pow(graph: &mut ComputationGraph, a: &Tensor, exponent: f64) -> Result<Tensor, String> {
    let data: Vec<f64> = a.data.iter().map(|x| x.powf(exponent)).collect();
    let requires_grad = a.requires_grad;

    let mut result = Tensor::new(data, a.shape.clone());
    result.requires_grad = requires_grad;
    if requires_grad {
        result.grad = Some(vec![0.0; result.data.len()]);
    }
    result.op = Op::Pow(a.id, exponent);

    graph.add_node(result.clone());
    Ok(result)
}

pub fn sum(graph: &mut ComputationGraph, a: &Tensor) -> Result<Tensor, String> {
    let sum_value: f64 = a.data.iter().sum();
    let requires_grad = a.requires_grad;

    let mut result = Tensor::scalar(sum_value);
    result.requires_grad = requires_grad;
    if requires_grad {
        result.grad = Some(vec![0.0]);
    }
    result.op = Op::Sum(a.id);

    graph.add_node(result.clone());
    Ok(result)
}

/// Linear layer: output = input @ weight + bias
/// Registers Linear operation in the computation graph for backprop
pub fn linear(graph: &mut ComputationGraph, input: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor, String> {
    // Handle 1D input: reshape to (1, in_features)
    let (input_data, input_shape) = if input.shape.len() == 1 {
        (input.data.clone(), vec![1, input.shape[0]])
    } else {
        (input.data.clone(), input.shape.clone())
    };

    if input_shape.len() != 2 || weight.shape.len() != 2 || bias.shape.len() != 1 {
        return Err(format!(
            "linear(): invalid shapes - input: {:?}, weight: {:?}, bias: {:?}",
            input_shape, weight.shape, bias.shape
        ));
    }

    let (batch, in_features) = (input_shape[0], input_shape[1]);
    let (weight_in, out_features) = (weight.shape[0], weight.shape[1]);

    if in_features != weight_in {
        return Err(format!(
            "linear(): input features {} doesn't match weight input {}",
            in_features, weight_in
        ));
    }

    if bias.shape[0] != out_features {
        return Err(format!(
            "linear(): bias size {} doesn't match output features {}",
            bias.shape[0], out_features
        ));
    }

    // Matrix multiplication: input @ weight + bias
    let mut result_data = vec![0.0; batch * out_features];
    for i in 0..batch {
        for j in 0..out_features {
            let mut sum = 0.0;
            for k in 0..in_features {
                sum += input_data[i * in_features + k] * weight.data[k * out_features + j];
            }
            result_data[i * out_features + j] = sum + bias.data[j];
        }
    }

    // Return shape matches input: 1D input -> 1D output
    let result_shape = if input.shape.len() == 1 {
        vec![out_features]
    } else {
        vec![batch, out_features]
    };

    let requires_grad = input.requires_grad || weight.requires_grad || bias.requires_grad;

    let mut result = Tensor::new(result_data, result_shape);
    result.requires_grad = requires_grad;
    if requires_grad {
        result.grad = Some(vec![0.0; result.data.len()]);
    }
    result.op = Op::Linear(input.id, weight.id, bias.id);

    graph.add_node(result.clone());
    Ok(result)
}

/// Embedding lookup: embedding_matrix[indices]
pub fn embedding(graph: &mut ComputationGraph, embedding_matrix: &Tensor, indices: &[i64]) -> Result<Tensor, String> {
    // Validate embedding matrix shape: [vocab_size, embedding_dim]
    if embedding_matrix.shape.len() != 2 {
        return Err(format!(
            "embedding(): embedding matrix must be 2D, got shape {:?}",
            embedding_matrix.shape
        ));
    }

    let vocab_size = embedding_matrix.shape[0];
    let embedding_dim = embedding_matrix.shape[1];

    // Validate indices
    for &idx in indices {
        if idx < 0 || idx >= vocab_size as i64 {
            return Err(format!(
                "embedding(): index {} out of range [0, {})",
                idx, vocab_size
            ));
        }
    }

    // Perform embedding lookup
    let num_indices = indices.len();
    let mut output_data = Vec::with_capacity(num_indices * embedding_dim);

    for &idx in indices {
        let idx_usize = idx as usize;
        let start = idx_usize * embedding_dim;
        let end = start + embedding_dim;
        output_data.extend_from_slice(&embedding_matrix.data[start..end]);
    }

    // Output shape: [num_indices, embedding_dim]
    let output_shape = vec![num_indices, embedding_dim];

    let requires_grad = embedding_matrix.requires_grad;

    let mut result = Tensor::new(output_data, output_shape);
    result.requires_grad = requires_grad;
    if requires_grad {
        result.grad = Some(vec![0.0; result.data.len()]);
    }
    result.op = Op::Embedding {
        input_id: embedding_matrix.id,
        indices: indices.to_vec(),
    };

    graph.add_node(result.clone());
    Ok(result)
}

/// Concatenate tensors along a dimension (only dim=1 supported for now)
pub fn concat(graph: &mut ComputationGraph, tensors: Vec<&Tensor>, dim: usize) -> Result<Tensor, String> {
    if tensors.is_empty() {
        return Err("concat(): need at least one tensor".to_string());
    }

    if dim != 1 {
        return Err(format!("concat(): only dim=1 supported, got dim={}", dim));
    }

    // All tensors must have same shape except in concat dimension
    let first_shape = &tensors[0].shape;
    if first_shape.len() != 2 {
        return Err(format!("concat(): only 2D tensors supported, got shape {:?}", first_shape));
    }

    let batch_size = first_shape[0];
    for t in &tensors[1..] {
        if t.shape.len() != 2 {
            return Err(format!("concat(): all tensors must have same rank, got {:?}", t.shape));
        }
        if t.shape[0] != batch_size {
            return Err(format!("concat(): batch size mismatch {} vs {}", t.shape[0], batch_size));
        }
    }

    // Calculate output shape
    let total_features: usize = tensors.iter().map(|t| t.shape[1]).sum();
    let output_shape = vec![batch_size, total_features];

    // Concatenate data
    let mut output_data = Vec::with_capacity(batch_size * total_features);
    for i in 0..batch_size {
        for tensor in &tensors {
            let features = tensor.shape[1];
            let start = i * features;
            let end = start + features;
            output_data.extend_from_slice(&tensor.data[start..end]);
        }
    }

    let requires_grad = tensors.iter().any(|t| t.requires_grad);
    let input_ids: Vec<usize> = tensors.iter().map(|t| t.id).collect();

    let mut result = Tensor::new(output_data, output_shape);
    result.requires_grad = requires_grad;
    if requires_grad {
        result.grad = Some(vec![0.0; result.data.len()]);
    }
    result.op = Op::Concat { input_ids, dim };

    graph.add_node(result.clone());
    Ok(result)
}

/// ReLU activation: max(0, x)
pub fn relu(graph: &mut ComputationGraph, input: &Tensor) -> Result<Tensor, String> {
    let data: Vec<f64> = input.data.iter().map(|&x| x.max(0.0)).collect();
    let requires_grad = input.requires_grad;

    let mut result = Tensor::new(data, input.shape.clone());
    result.requires_grad = requires_grad;
    if requires_grad {
        result.grad = Some(vec![0.0; result.data.len()]);
    }
    result.op = Op::ReLU(input.id);

    graph.add_node(result.clone());
    Ok(result)
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(graph: &mut ComputationGraph, input: &Tensor) -> Result<Tensor, String> {
    let data: Vec<f64> = input.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    let requires_grad = input.requires_grad;

    let mut result = Tensor::new(data, input.shape.clone());
    result.requires_grad = requires_grad;
    if requires_grad {
        result.grad = Some(vec![0.0; result.data.len()]);
    }
    result.op = Op::Sigmoid(input.id);

    graph.add_node(result.clone());
    Ok(result)
}

/// Mean Squared Error loss: mean((pred - target)^2)
pub fn mse_loss(graph: &mut ComputationGraph, pred: &Tensor, target: &Tensor) -> Result<Tensor, String> {
    if pred.shape != target.shape {
        return Err(format!(
            "mse_loss(): shape mismatch - pred: {:?}, target: {:?}",
            pred.shape, target.shape
        ));
    }

    let mse: f64 = pred
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(&p, &t)| {
            let diff = p - t;
            diff * diff
        })
        .sum::<f64>()
        / pred.data.len() as f64;

    let requires_grad = pred.requires_grad;

    let mut result = Tensor::scalar(mse);
    result.requires_grad = requires_grad;
    if requires_grad {
        result.grad = Some(vec![1.0]); // Gradient of loss is 1.0
    }
    result.op = Op::MSELoss(pred.id, target.id);

    graph.add_node(result.clone());
    Ok(result)
}

pub fn cross_entropy_loss(graph: &mut ComputationGraph, pred: &Tensor, target: &Tensor) -> Result<Tensor, String> {
    if pred.shape != target.shape {
        return Err(format!(
            "cross_entropy_loss(): shape mismatch - pred: {:?}, target: {:?}",
            pred.shape, target.shape
        ));
    }

    // Cross Entropy Loss: -mean(target * log(pred))
    // With clipping to avoid log(0)
    let epsilon = 1e-10;

    let ce: f64 = pred
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(&p, &t)| {
            let clipped_p = p.max(epsilon).min(1.0 - epsilon);
            -t * clipped_p.ln()
        })
        .sum::<f64>()
        / pred.data.len() as f64;

    let requires_grad = pred.requires_grad;

    let mut result = Tensor::scalar(ce);
    result.requires_grad = requires_grad;
    if requires_grad {
        result.grad = Some(vec![1.0]); // Gradient of loss is 1.0
    }
    result.op = Op::CrossEntropyLoss(pred.id, target.id);

    graph.add_node(result.clone());
    Ok(result)
}

pub fn cross_entropy_loss_with_logits(graph: &mut ComputationGraph, logits: &Tensor, target: &Tensor) -> Result<Tensor, String> {
    if logits.shape != target.shape {
        return Err(format!(
            "cross_entropy_loss_with_logits(): shape mismatch - logits: {:?}, target: {:?}",
            logits.shape, target.shape
        ));
    }

    // Cross Entropy Loss with Logits: -mean(target * log(softmax(logits)))
    // This combines softmax + cross_entropy for numerical stability and efficiency

    // Compute softmax(logits)
    let max = logits
        .data
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let exp_values: Vec<f64> = logits.data.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exp_values.iter().sum();
    let softmax: Vec<f64> = exp_values.iter().map(|&x| x / sum).collect();

    // Compute cross entropy loss
    let epsilon = 1e-10;
    let ce: f64 = softmax
        .iter()
        .zip(target.data.iter())
        .map(|(&s, &t)| {
            let clipped_s = s.max(epsilon).min(1.0 - epsilon);
            -t * clipped_s.ln()
        })
        .sum::<f64>()
        / logits.data.len() as f64;

    let requires_grad = logits.requires_grad;

    let mut result = Tensor::scalar(ce);
    result.requires_grad = requires_grad;
    if requires_grad {
        result.grad = Some(vec![1.0]); // Gradient of loss is 1.0
    }
    result.op = Op::CrossEntropyLogitsLoss(logits.id, target.id);

    graph.add_node(result.clone());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(t.shape, vec![3]);
        assert!(!t.requires_grad);
    }

    #[test]
    fn test_tensor_with_grad() {
        let t = Tensor::with_grad(vec![1.0, 2.0], vec![2]);
        assert!(t.requires_grad);
        assert_eq!(t.grad, Some(vec![0.0, 0.0]));
    }

    #[test]
    fn test_scalar_tensor() {
        let t = Tensor::scalar(5.0);
        assert_eq!(t.data, vec![5.0]);
        assert_eq!(t.shape, vec![1]);
        assert_eq!(t.item().unwrap(), 5.0);
    }

    #[test]
    fn test_forward_add() {
        let mut graph = ComputationGraph::new();
        let a = Tensor::with_grad(vec![1.0, 2.0], vec![2]);
        let b = Tensor::with_grad(vec![3.0, 4.0], vec![2]);

        graph.add_node(a.clone());
        graph.add_node(b.clone());

        let c = add(&mut graph, &a, &b).unwrap();
        assert_eq!(c.data, vec![4.0, 6.0]);
        assert!(c.requires_grad);
    }

    #[test]
    fn test_forward_mul() {
        let mut graph = ComputationGraph::new();
        let a = Tensor::with_grad(vec![2.0, 3.0], vec![2]);
        let b = Tensor::with_grad(vec![4.0, 5.0], vec![2]);

        graph.add_node(a.clone());
        graph.add_node(b.clone());

        let c = mul(&mut graph, &a, &b).unwrap();
        assert_eq!(c.data, vec![8.0, 15.0]);
    }

    #[test]
    fn test_backward_add() {
        let mut graph = ComputationGraph::new();
        let a = Tensor::with_grad(vec![2.0], vec![1]);
        let b = Tensor::with_grad(vec![3.0], vec![1]);

        let a_id = graph.add_node(a.clone());
        let b_id = graph.add_node(b.clone());

        let c = add(&mut graph, &a, &b).unwrap();
        let c_id = c.id;

        graph.backward(c_id).unwrap();

        let a_grad = graph.get_node(a_id).unwrap().grad.as_ref().unwrap();
        let b_grad = graph.get_node(b_id).unwrap().grad.as_ref().unwrap();

        assert_eq!(a_grad, &vec![1.0]);
        assert_eq!(b_grad, &vec![1.0]);
    }

    #[test]
    fn test_backward_mul() {
        let mut graph = ComputationGraph::new();
        let a = Tensor::scalar_with_grad(3.0);
        let b = Tensor::scalar_with_grad(4.0);

        let a_id = graph.add_node(a.clone());
        let b_id = graph.add_node(b.clone());

        let c = mul(&mut graph, &a, &b).unwrap();
        assert_eq!(c.item().unwrap(), 12.0);

        graph.backward(c.id).unwrap();

        let a_grad = graph.get_node(a_id).unwrap().grad.as_ref().unwrap()[0];
        let b_grad = graph.get_node(b_id).unwrap().grad.as_ref().unwrap()[0];

        assert_eq!(a_grad, 4.0); // dc/da = b = 4
        assert_eq!(b_grad, 3.0); // dc/db = a = 3
    }

    #[test]
    fn test_backward_chain() {
        // Test: f(x) = (x + 2) * 3, x = 5
        // f'(x) = 3
        let mut graph = ComputationGraph::new();
        let x = Tensor::scalar_with_grad(5.0);
        let two = Tensor::scalar(2.0);
        let three = Tensor::scalar(3.0);

        let x_id = graph.add_node(x.clone());
        graph.add_node(two.clone());
        graph.add_node(three.clone());

        let a = add(&mut graph, &x, &two).unwrap(); // x + 2 = 7
        let b = mul(&mut graph, &a, &three).unwrap(); // (x + 2) * 3 = 21

        assert_eq!(b.item().unwrap(), 21.0);

        graph.backward(b.id).unwrap();

        let x_grad = graph.get_node(x_id).unwrap().grad.as_ref().unwrap()[0];
        assert_eq!(x_grad, 3.0);
    }

    #[test]
    fn test_backward_complex() {
        // Test: f(x, y) = x * y + x, x = 3, y = 4
        // df/dx = y + 1 = 5
        // df/dy = x = 3
        let mut graph = ComputationGraph::new();
        let x = Tensor::scalar_with_grad(3.0);
        let y = Tensor::scalar_with_grad(4.0);

        let x_id = graph.add_node(x.clone());
        let y_id = graph.add_node(y.clone());

        let xy = mul(&mut graph, &x, &y).unwrap(); // x * y = 12
        let result = add(&mut graph, &xy, &x).unwrap(); // x * y + x = 15

        assert_eq!(result.item().unwrap(), 15.0);

        graph.backward(result.id).unwrap();

        let x_grad = graph.get_node(x_id).unwrap().grad.as_ref().unwrap()[0];
        let y_grad = graph.get_node(y_id).unwrap().grad.as_ref().unwrap()[0];

        assert_eq!(x_grad, 5.0);
        assert_eq!(y_grad, 3.0);
    }

    #[test]
    fn test_sum_operation() {
        let mut graph = ComputationGraph::new();
        let x = Tensor::with_grad(vec![1.0, 2.0, 3.0], vec![3]);
        let x_id = graph.add_node(x.clone());

        let s = sum(&mut graph, &x).unwrap();
        assert_eq!(s.item().unwrap(), 6.0);

        graph.backward(s.id).unwrap();

        let x_grad = graph.get_node(x_id).unwrap().grad.as_ref().unwrap();
        assert_eq!(x_grad, &vec![1.0, 1.0, 1.0]); // gradient broadcasts to all elements
    }

    #[test]
    fn test_pow_operation() {
        let mut graph = ComputationGraph::new();
        let x = Tensor::scalar_with_grad(3.0);
        let x_id = graph.add_node(x.clone());

        let y = pow(&mut graph, &x, 2.0).unwrap(); // x^2 = 9
        assert_eq!(y.item().unwrap(), 9.0);

        graph.backward(y.id).unwrap();

        let x_grad = graph.get_node(x_id).unwrap().grad.as_ref().unwrap()[0];
        assert_eq!(x_grad, 6.0); // d(x^2)/dx = 2x = 6
    }

    #[test]
    fn test_neg_operation() {
        let mut graph = ComputationGraph::new();
        let x = Tensor::scalar_with_grad(5.0);
        let x_id = graph.add_node(x.clone());

        let y = neg(&mut graph, &x).unwrap();
        assert_eq!(y.item().unwrap(), -5.0);

        graph.backward(y.id).unwrap();

        let x_grad = graph.get_node(x_id).unwrap().grad.as_ref().unwrap()[0];
        assert_eq!(x_grad, -1.0);
    }

    #[test]
    fn test_div_operation() {
        let mut graph = ComputationGraph::new();
        let a = Tensor::scalar_with_grad(12.0);
        let b = Tensor::scalar_with_grad(3.0);

        let a_id = graph.add_node(a.clone());
        let b_id = graph.add_node(b.clone());

        let c = div(&mut graph, &a, &b).unwrap();
        assert_eq!(c.item().unwrap(), 4.0);

        graph.backward(c.id).unwrap();

        let a_grad = graph.get_node(a_id).unwrap().grad.as_ref().unwrap()[0];
        let b_grad = graph.get_node(b_id).unwrap().grad.as_ref().unwrap()[0];

        assert_eq!(a_grad, 1.0 / 3.0);
        assert!((b_grad - (-12.0 / 9.0)).abs() < 1e-6);
    }
}

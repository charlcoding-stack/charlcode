// Automatic Differentiation System
// Phase 4: Computational graph and gradient computation

use std::collections::HashMap;

// Unique ID generator for graph nodes
static mut NEXT_ID: usize = 0;

fn next_id() -> usize {
    unsafe {
        let id = NEXT_ID;
        NEXT_ID += 1;
        id
    }
}

// Operation types in the computational graph
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Leaf,                           // Leaf node (input/constant)
    Add(usize, usize),              // Addition
    Sub(usize, usize),              // Subtraction
    Mul(usize, usize),              // Multiplication
    Div(usize, usize),              // Division
    Neg(usize),                     // Negation
    Pow(usize, f64),                // Power (x^n)
    Sum(usize),                     // Sum all elements
    MatMul(usize, usize),           // Matrix multiplication
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
            Err(format!("item() only works on 1-element tensors, got {} elements", self.data.len()))
        }
    }
}

// Computational graph for tracking operations
pub struct ComputationGraph {
    nodes: HashMap<usize, Tensor>,
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
        let output = self.get_node(output_id)
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

            let node_grad = node.grad.clone().unwrap_or_else(|| vec![0.0; node.data.len()]);

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

        let node = self.get_node(node_id)
            .ok_or_else(|| format!("Node {} not found", node_id))?;

        // Visit dependencies based on operation
        match &node.op {
            Op::Leaf => {}
            Op::Add(l, r) | Op::Sub(l, r) | Op::Mul(l, r) | Op::Div(l, r) | Op::MatMul(l, r) => {
                self.visit(*l, visited, temp_mark, sorted)?;
                self.visit(*r, visited, temp_mark, sorted)?;
            }
            Op::Neg(i) | Op::Pow(i, _) | Op::Sum(i) => {
                self.visit(*i, visited, temp_mark, sorted)?;
            }
        }

        temp_mark.insert(node_id, false);
        visited.insert(node_id, true);
        sorted.push(node_id);

        Ok(())
    }

    // Backward pass implementations for each operation

    fn backward_add(&mut self, left_id: usize, right_id: usize, grad: &[f64]) -> Result<(), String> {
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

    fn backward_sub(&mut self, left_id: usize, right_id: usize, grad: &[f64]) -> Result<(), String> {
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

    fn backward_mul(&mut self, left_id: usize, right_id: usize, grad: &[f64]) -> Result<(), String> {
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

    fn backward_div(&mut self, left_id: usize, right_id: usize, grad: &[f64]) -> Result<(), String> {
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

    fn backward_matmul(&mut self, _left_id: usize, _right_id: usize, _grad: &[f64]) -> Result<(), String> {
        // TODO: Implement matrix multiplication gradient
        // This is complex and will be implemented in a later phase
        Err("Matrix multiplication gradients not yet implemented".to_string())
    }
}

// Forward operations that create computation graph nodes

pub fn add(graph: &mut ComputationGraph, a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if !a.same_shape(b) {
        return Err(format!("Shape mismatch: {:?} vs {:?}", a.shape, b.shape));
    }

    let data: Vec<f64> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x + y).collect();
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

    let data: Vec<f64> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x - y).collect();
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

    let data: Vec<f64> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).collect();
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

    let data: Vec<f64> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x / y).collect();
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

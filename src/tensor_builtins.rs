// Tensor Builtin Functions
// Phase 1: Expose backend tensor operations to Charl language

use crate::autograd::Tensor as AutogradTensor;
use crate::interpreter::Value;
use crate::optim::{SGD, Adam, RMSprop, Optimizer};
use crate::efficient_architectures::{Linformer, Performer, FNet, RWKV, MambaBlock, MambaConfig, S4Layer, SSMConfig, DiscretizationMethod, MoELayer, RoutingStrategy};
use crate::quantization::{QuantType, quantize_tensor_auto, dequantize_tensor, post_training_quantization, CalibrationMethod};
use crate::reasoning::{ChainOfThought, ReasoningStep};
use once_cell::sync::Lazy;
use std::sync::Mutex;

// Global GPU backend singleton - shared across all GPU operations
// This ensures buffer IDs remain valid across function calls
static GPU_BACKEND: Lazy<Mutex<Option<crate::gpu::wgpu_backend::WgpuBackend>>> = Lazy::new(|| {
    Mutex::new(None)
});

/// Initialize the global GPU backend if not already initialized
fn get_or_init_gpu_backend() -> Result<(), String> {
    let mut backend = GPU_BACKEND.lock().unwrap();
    if backend.is_none() {
        *backend = Some(
            crate::gpu::wgpu_backend::WgpuBackend::new_sync()
                .map_err(|e| format!("Failed to initialize GPU backend: {:?}", e))?
        );
    }
    Ok(())
}

/// Execute a GPU operation with the global backend
fn with_gpu_backend<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&mut crate::gpu::wgpu_backend::WgpuBackend) -> Result<R, String>,
{
    get_or_init_gpu_backend()?;
    let mut backend = GPU_BACKEND.lock().unwrap();
    let backend_ref = backend.as_mut().ok_or("GPU backend not initialized")?;
    f(backend_ref)
}

/// Builtin function type
pub type BuiltinFn = fn(Vec<Value>) -> Result<Value, String>;

/// tensor(data: [float], shape?: [int]) -> Tensor
/// Creates a tensor from an array of numbers
/// If shape is provided, reshapes the data to that shape
pub fn builtin_tensor(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 && args.len() != 2 {
        return Err("tensor() expects 1-2 arguments: tensor(data) or tensor(data, shape)".to_string());
    }

    match &args[0] {
        Value::Array(arr) => {
            // Convert array of Values to Vec<f64>
            let mut data = Vec::new();
            for val in arr {
                let num = val.to_float()?;
                data.push(num);
            }

            // Determine shape
            let shape = if args.len() == 2 {
                // Shape provided as second argument
                match &args[1] {
                    Value::Array(shape_vals) => {
                        let mut shape = Vec::new();
                        for val in shape_vals {
                            match val {
                                Value::Integer(i) => shape.push(*i as usize),
                                _ => return Err("tensor() shape must contain integers".to_string()),
                            }
                        }

                        // Validate shape matches data size
                        let shape_total: usize = shape.iter().product();
                        if shape_total != data.len() {
                            return Err(format!(
                                "tensor() shape {:?} (total: {}) does not match data length {}",
                                shape, shape_total, data.len()
                            ));
                        }

                        shape
                    }
                    _ => return Err("tensor() shape must be an array of integers".to_string()),
                }
            } else {
                // No shape provided, default to 1D
                vec![data.len()]
            };

            // Create AutogradTensor
            let tensor = AutogradTensor::new(data, shape);

            Ok(Value::AutogradTensor(tensor))
        }
        _ => Err(format!(
            "tensor() expects an array, got {}",
            args[0].type_name()
        )),
    }
}

/// tensor_shape(t: Tensor) -> [int]
/// Returns the shape of a tensor as an array
pub fn builtin_tensor_shape(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_shape() expects 1 argument: tensor_shape(tensor)".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(tensor) => {
            // Convert shape Vec<usize> to Value::Array of integers
            let shape_values: Vec<Value> = tensor
                .shape
                .iter()
                .map(|&dim| Value::Integer(dim as i64))
                .collect();

            Ok(Value::Array(shape_values))
        }
        Value::GPUTensor(tensor) => {
            // Convert shape Vec<usize> to Value::Array of integers
            let shape_values: Vec<Value> = tensor
                .tensor
                .shape
                .iter()
                .map(|&dim| Value::Integer(dim as i64))
                .collect();

            Ok(Value::Array(shape_values))
        }
        _ => Err(format!(
            "tensor_shape() expects a tensor, got {}",
            args[0].type_name()
        )),
    }
}

/// tensor_add(a: Tensor, b: Tensor) -> Tensor
/// Element-wise addition of two tensors
pub fn builtin_tensor_add(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("tensor_add() expects 2 arguments: tensor_add(a, b)".to_string());
    }

    match (&args[0], &args[1]) {
        // CPU + CPU
        (Value::AutogradTensor(a), Value::AutogradTensor(b)) => {
            // Check shapes match
            if a.shape != b.shape {
                return Err(format!(
                    "tensor_add(): shape mismatch {:?} vs {:?}",
                    a.shape, b.shape
                ));
            }

            // If either tensor requires grad, use autograd system
            if a.requires_grad || b.requires_grad {
                // Ensure both tensors are in the global graph
                crate::autograd::with_global_graph_mut(|graph| {
                    if graph.get_node(a.id).is_none() {
                        graph.add_node(a.clone());
                    }
                    if graph.get_node(b.id).is_none() {
                        graph.add_node(b.clone());
                    }

                    // Use autograd add to record in graph
                    let result = crate::autograd::add(graph, a, b)?;
                    Ok(Value::AutogradTensor(result))
                })
            } else {
                // Regular addition without gradient tracking
                let result_data: Vec<f64> = a
                    .data
                    .iter()
                    .zip(b.data.iter())
                    .map(|(x, y)| x + y)
                    .collect();

                let result_tensor = AutogradTensor::new(result_data, a.shape.clone());
                Ok(Value::AutogradTensor(result_tensor))
            }
        }
        // GPU + GPU - operates on CPU tensors inside
        (Value::GPUTensor(a), Value::GPUTensor(b)) => {
            if a.tensor.shape != b.tensor.shape {
                return Err(format!(
                    "tensor_add(): shape mismatch {:?} vs {:?}",
                    a.tensor.shape, b.tensor.shape
                ));
            }

            // Operate on the CPU tensor data (which GPUTensor always has)
            let result_data: Vec<f64> = a.tensor.data
                .iter()
                .zip(b.tensor.data.iter())
                .map(|(x, y)| x + y)
                .collect();

            // Create result tensor
            let result_tensor = AutogradTensor::new(result_data, a.tensor.shape.clone());

            // Create GPU tensor and move to GPU using global backend
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        // Mixed CPU/GPU - require explicit conversion
        (Value::AutogradTensor(_), Value::GPUTensor(_)) => {
            Err("Cannot add CPU tensor + GPU tensor. Use tensor_to_gpu() or tensor_to_cpu() to convert first.".to_string())
        }
        (Value::GPUTensor(_), Value::AutogradTensor(_)) => {
            Err("Cannot add GPU tensor + CPU tensor. Use tensor_to_gpu() or tensor_to_cpu() to convert first.".to_string())
        }
        _ => Err("tensor_add() expects two tensors".to_string()),
    }
}

/// tensor_mul(a: Tensor, b: Tensor|float) -> Tensor
/// Element-wise multiplication or scalar multiplication
pub fn builtin_tensor_mul(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("tensor_mul() expects 2 arguments: tensor_mul(tensor, scalar)".to_string());
    }

    match (&args[0], &args[1]) {
        // CPU Tensor * CPU Tensor (element-wise)
        (Value::AutogradTensor(a), Value::AutogradTensor(b)) => {
            if a.shape != b.shape {
                return Err(format!(
                    "tensor_mul(): shape mismatch {:?} vs {:?}",
                    a.shape, b.shape
                ));
            }

            // If either tensor requires grad, use autograd system
            if a.requires_grad || b.requires_grad {
                crate::autograd::with_global_graph_mut(|graph| {
                    if graph.get_node(a.id).is_none() {
                        graph.add_node(a.clone());
                    }
                    if graph.get_node(b.id).is_none() {
                        graph.add_node(b.clone());
                    }

                    let result = crate::autograd::mul(graph, a, b)?;
                    Ok(Value::AutogradTensor(result))
                })
            } else {
                let result_data: Vec<f64> = a
                    .data
                    .iter()
                    .zip(b.data.iter())
                    .map(|(x, y)| x * y)
                    .collect();

                let result_tensor = AutogradTensor::new(result_data, a.shape.clone());
                Ok(Value::AutogradTensor(result_tensor))
            }
        }
        // GPU Tensor * GPU Tensor (element-wise)
        (Value::GPUTensor(a), Value::GPUTensor(b)) => {
            if a.tensor.shape != b.tensor.shape {
                return Err(format!(
                    "tensor_mul(): shape mismatch {:?} vs {:?}",
                    a.tensor.shape, b.tensor.shape
                ));
            }

            // Operate on CPU tensor data for now (GPU backend has mul but using CPU data is simpler)
            let result_data: Vec<f64> = a.tensor.data
                .iter()
                .zip(b.tensor.data.iter())
                .map(|(x, y)| x * y)
                .collect();

            let result_tensor = AutogradTensor::new(result_data, a.tensor.shape.clone());
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            // Move to GPU using global backend
            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        // CPU Tensor * Scalar
        (Value::AutogradTensor(tensor), scalar) => {
            let scalar_val = scalar.to_float()?;

            // If tensor requires grad, use autograd system
            if tensor.requires_grad {
                // Create scalar tensor for autograd tracking
                let scalar_tensor = AutogradTensor::new(vec![scalar_val], vec![1]);

                crate::autograd::with_global_graph_mut(|graph| {
                    if graph.get_node(tensor.id).is_none() {
                        graph.add_node(tensor.clone());
                    }
                    if graph.get_node(scalar_tensor.id).is_none() {
                        graph.add_node(scalar_tensor.clone());
                    }

                    let result = crate::autograd::mul(graph, tensor, &scalar_tensor)?;
                    Ok(Value::AutogradTensor(result))
                })
            } else {
                let result_data: Vec<f64> = tensor.data.iter().map(|x| x * scalar_val).collect();
                let result_tensor = AutogradTensor::new(result_data, tensor.shape.clone());
                Ok(Value::AutogradTensor(result_tensor))
            }
        }
        // GPU Tensor * Scalar
        (Value::GPUTensor(tensor), scalar) => {
            let scalar_val = scalar.to_float()?;

            let result_data: Vec<f64> = tensor.tensor.data.iter().map(|x| x * scalar_val).collect();

            let result_tensor = AutogradTensor::new(result_data, tensor.tensor.shape.clone());
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            // Move to GPU using global backend
            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        // Mixed CPU/GPU - require explicit conversion
        (Value::AutogradTensor(_), Value::GPUTensor(_)) | (Value::GPUTensor(_), Value::AutogradTensor(_)) => {
            Err("Cannot multiply CPU tensor and GPU tensor. Use tensor_to_gpu() or tensor_to_cpu() to convert first.".to_string())
        }
        _ => Err("tensor_mul() expects (tensor, tensor) or (tensor, scalar)".to_string()),
    }
}

/// tensor_sum(t: Tensor) -> float
/// Sum all elements in the tensor
pub fn builtin_tensor_sum(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_sum() expects 1 argument: tensor_sum(tensor)".to_string());
    }

    match &args[0] {
        // CPU Tensor
        Value::AutogradTensor(tensor) => {
            let sum: f64 = tensor.data.iter().sum();
            Ok(Value::Float(sum))
        }
        // GPU Tensor - operate on internal data
        Value::GPUTensor(gpu_tensor) => {
            let sum: f64 = gpu_tensor.tensor.data.iter().sum();
            Ok(Value::Float(sum))
        }
        _ => Err(format!(
            "tensor_sum() expects a tensor, got {}",
            args[0].type_name()
        )),
    }
}

/// tensor_mean(t: Tensor) -> float
/// Average of all elements in the tensor
pub fn builtin_tensor_mean(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_mean() expects 1 argument: tensor_mean(tensor)".to_string());
    }

    match &args[0] {
        // CPU Tensor
        Value::AutogradTensor(tensor) => {
            if tensor.data.is_empty() {
                return Ok(Value::Float(0.0));
            }

            let sum: f64 = tensor.data.iter().sum();
            let mean = sum / tensor.data.len() as f64;
            Ok(Value::Float(mean))
        }
        // GPU Tensor - operate on internal data
        Value::GPUTensor(gpu_tensor) => {
            if gpu_tensor.tensor.data.is_empty() {
                return Ok(Value::Float(0.0));
            }

            let sum: f64 = gpu_tensor.tensor.data.iter().sum();
            let mean = sum / gpu_tensor.tensor.data.len() as f64;
            Ok(Value::Float(mean))
        }
        _ => Err(format!(
            "tensor_mean() expects a tensor, got {}",
            args[0].type_name()
        )),
    }
}

/// tensor_print(t: Tensor) -> ()
/// Print tensor data and shape
pub fn builtin_tensor_print(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_print() expects 1 argument: tensor_print(tensor)".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(tensor) => {
            println!("Tensor(shape: {:?}, data: {:?})", tensor.shape, tensor.data);
            Ok(Value::Null)
        }
        _ => Err(format!(
            "tensor_print() expects a tensor, got {}",
            args[0].type_name()
        )),
    }
}

/// tensor_item(t: Tensor) -> float
/// Extract scalar value from a 1-element tensor
/// Example: tensor_item(tensor([3.14])) -> 3.14
pub fn builtin_tensor_item(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_item() expects 1 argument: tensor_item(tensor)".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(tensor) => {
            let value = tensor.item()?;
            Ok(Value::Float(value))
        }
        _ => Err(format!(
            "tensor_item() expects a tensor, got {}",
            args[0].type_name()
        )),
    }
}

/// tensor_matmul(a: Tensor, b: Tensor) -> Tensor
/// Matrix multiplication (for 2D tensors)
pub fn builtin_tensor_matmul(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("tensor_matmul() expects 2 arguments: tensor_matmul(a, b)".to_string());
    }

    match (&args[0], &args[1]) {
        // CPU matmul
        (Value::AutogradTensor(a), Value::AutogradTensor(b)) => {
            // For now, only support 2D matrices
            if a.shape.len() != 2 || b.shape.len() != 2 {
                return Err("tensor_matmul() currently only supports 2D matrices".to_string());
            }

            let (m, k1) = (a.shape[0], a.shape[1]);
            let (k2, n) = (b.shape[0], b.shape[1]);

            if k1 != k2 {
                return Err(format!(
                    "tensor_matmul(): incompatible shapes ({}, {}) x ({}, {})",
                    m, k1, k2, n
                ));
            }

            // Perform matrix multiplication
            let mut result_data = vec![0.0; m * n];

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..k1 {
                        sum += a.data[i * k1 + k] * b.data[k * n + j];
                    }
                    result_data[i * n + j] = sum;
                }
            }

            let result_tensor = AutogradTensor::new(result_data, vec![m, n]);
            Ok(Value::AutogradTensor(result_tensor))
        }
        // GPU matmul
        (Value::GPUTensor(a), Value::GPUTensor(b)) => {
            // Only support 2D matrices
            if a.tensor.shape.len() != 2 || b.tensor.shape.len() != 2 {
                return Err("tensor_matmul() currently only supports 2D matrices".to_string());
            }

            let (m, k1) = (a.tensor.shape[0], a.tensor.shape[1]);
            let (k2, n) = (b.tensor.shape[0], b.tensor.shape[1]);

            if k1 != k2 {
                return Err(format!(
                    "tensor_matmul(): incompatible shapes ({}, {}) x ({}, {})",
                    m, k1, k2, n
                ));
            }

            // Perform matrix multiplication on CPU data (GPU backend matmul also available)
            let mut result_data = vec![0.0; m * n];

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..k1 {
                        sum += a.tensor.data[i * k1 + k] * b.tensor.data[k * n + j];
                    }
                    result_data[i * n + j] = sum;
                }
            }

            let result_tensor = AutogradTensor::new(result_data, vec![m, n]);
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            // Move to GPU using global backend
            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        // Mixed CPU/GPU - require explicit conversion
        (Value::AutogradTensor(_), Value::GPUTensor(_)) | (Value::GPUTensor(_), Value::AutogradTensor(_)) => {
            Err("Cannot matmul CPU tensor and GPU tensor. Use tensor_to_gpu() or tensor_to_cpu() to convert first.".to_string())
        }
        _ => Err("tensor_matmul() expects two tensors".to_string()),
    }
}

/// tensor_reshape(t: Tensor, shape: [int]) -> Tensor
/// Reshape a tensor to a new shape (must have same total elements)
pub fn builtin_tensor_reshape(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("tensor_reshape() expects 2 arguments: tensor_reshape(tensor, shape)".to_string());
    }

    // Extract shape from second argument
    let new_shape = match &args[1] {
        Value::Array(shape_vals) => {
            let mut shape = Vec::new();
            for val in shape_vals {
                match val {
                    Value::Integer(i) => shape.push(*i as usize),
                    _ => return Err("tensor_reshape() shape must contain integers".to_string()),
                }
            }
            shape
        }
        _ => return Err("tensor_reshape() expects (tensor, array)".to_string()),
    };

    match &args[0] {
        // CPU Tensor
        (Value::AutogradTensor(tensor)) => {
            // Calculate total elements
            let old_total: usize = tensor.shape.iter().product();
            let new_total: usize = new_shape.iter().product();

            if old_total != new_total {
                return Err(format!(
                    "tensor_reshape(): cannot reshape tensor of {} elements to shape {:?} ({} elements)",
                    old_total, new_shape, new_total
                ));
            }

            // Create new tensor with same data but different shape
            let result_tensor = AutogradTensor::new(tensor.data.clone(), new_shape);
            Ok(Value::AutogradTensor(result_tensor))
        }
        // GPU Tensor
        Value::GPUTensor(gpu_tensor) => {
            // Calculate total elements
            let old_total: usize = gpu_tensor.tensor.shape.iter().product();
            let new_total: usize = new_shape.iter().product();

            if old_total != new_total {
                return Err(format!(
                    "tensor_reshape(): cannot reshape tensor of {} elements to shape {:?} ({} elements)",
                    old_total, new_shape, new_total
                ));
            }

            // Create new tensor with same data but different shape
            let result_tensor = AutogradTensor::new(gpu_tensor.tensor.data.clone(), new_shape);
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            // Move to GPU using global backend
            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        _ => Err("tensor_reshape() expects (tensor, array)".to_string()),
    }
}

/// tensor_transpose(t: Tensor) -> Tensor
/// Transpose a 2D tensor (swap rows and columns)
pub fn builtin_tensor_transpose(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_transpose() expects 1 argument: tensor_transpose(tensor)".to_string());
    }

    match &args[0] {
        // CPU Tensor
        Value::AutogradTensor(tensor) => {
            if tensor.shape.len() != 2 {
                return Err("tensor_transpose() currently only supports 2D tensors".to_string());
            }

            let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
            let mut transposed_data = vec![0.0; rows * cols];

            for i in 0..rows {
                for j in 0..cols {
                    transposed_data[j * rows + i] = tensor.data[i * cols + j];
                }
            }

            let result_tensor = AutogradTensor::new(transposed_data, vec![cols, rows]);
            Ok(Value::AutogradTensor(result_tensor))
        }
        // GPU Tensor
        Value::GPUTensor(gpu_tensor) => {
            if gpu_tensor.tensor.shape.len() != 2 {
                return Err("tensor_transpose() currently only supports 2D tensors".to_string());
            }

            let (rows, cols) = (gpu_tensor.tensor.shape[0], gpu_tensor.tensor.shape[1]);
            let mut transposed_data = vec![0.0; rows * cols];

            for i in 0..rows {
                for j in 0..cols {
                    transposed_data[j * rows + i] = gpu_tensor.tensor.data[i * cols + j];
                }
            }

            let result_tensor = AutogradTensor::new(transposed_data, vec![cols, rows]);
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            // Move to GPU using global backend
            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        _ => Err("tensor_transpose() expects a tensor".to_string()),
    }
}

/// tensor_zeros(shape: [int]) -> Tensor
/// Create a tensor filled with zeros
pub fn builtin_tensor_zeros(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_zeros() expects 1 argument: tensor_zeros(shape)".to_string());
    }

    match &args[0] {
        Value::Array(shape_vals) => {
            // Convert shape array to Vec<usize>
            let mut shape = Vec::new();
            for val in shape_vals {
                match val {
                    Value::Integer(i) => shape.push(*i as usize),
                    _ => return Err("tensor_zeros() shape must contain integers".to_string()),
                }
            }

            let total_elements: usize = shape.iter().product();
            let data = vec![0.0; total_elements];

            let tensor = AutogradTensor::new(data, shape);
            Ok(Value::AutogradTensor(tensor))
        }
        _ => Err("tensor_zeros() expects an array".to_string()),
    }
}

/// tensor_ones(shape: [int]) -> Tensor
/// Create a tensor filled with ones
pub fn builtin_tensor_ones(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_ones() expects 1 argument: tensor_ones(shape)".to_string());
    }

    match &args[0] {
        Value::Array(shape_vals) => {
            // Convert shape array to Vec<usize>
            let mut shape = Vec::new();
            for val in shape_vals {
                match val {
                    Value::Integer(i) => shape.push(*i as usize),
                    _ => return Err("tensor_ones() shape must contain integers".to_string()),
                }
            }

            let total_elements: usize = shape.iter().product();
            let data = vec![1.0; total_elements];

            let tensor = AutogradTensor::new(data, shape);
            Ok(Value::AutogradTensor(tensor))
        }
        _ => Err("tensor_ones() expects an array".to_string()),
    }
}

/// tensor_randn(shape: [int]) -> Tensor
/// Create a tensor with random normal distribution values (mean=0, std=1)
pub fn builtin_tensor_randn(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_randn() expects 1 argument: tensor_randn(shape)".to_string());
    }

    match &args[0] {
        Value::Array(shape_vals) => {
            // Convert shape array to Vec<usize>
            let mut shape = Vec::new();
            for val in shape_vals {
                match val {
                    Value::Integer(i) => shape.push(*i as usize),
                    _ => return Err("tensor_randn() shape must contain integers".to_string()),
                }
            }

            let total_elements: usize = shape.iter().product();

            // Generate random normal values using Box-Muller transform
            let mut data = Vec::with_capacity(total_elements);
            let mut rng = rand::thread_rng();

            for _ in 0..total_elements {
                // Box-Muller transform to generate normal distribution
                let u1: f64 = rand::Rng::gen(&mut rng);
                let u2: f64 = rand::Rng::gen(&mut rng);
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                data.push(z0);
            }

            let tensor = AutogradTensor::new(data, shape);
            Ok(Value::AutogradTensor(tensor))
        }
        _ => Err("tensor_randn() expects an array".to_string()),
    }
}

/// tensor_requires_grad(t: Tensor, enabled: bool) -> Tensor
/// Enable or disable gradient tracking for a tensor
pub fn builtin_tensor_requires_grad(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("tensor_requires_grad() expects 2 arguments: tensor_requires_grad(tensor, enabled)".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::AutogradTensor(tensor), Value::Boolean(enabled)) => {
            let mut new_tensor = tensor.clone();
            new_tensor.requires_grad = *enabled;

            // Initialize gradient buffer if enabling
            if *enabled && new_tensor.grad.is_none() {
                new_tensor.grad = Some(vec![0.0; new_tensor.data.len()]);
            }

            Ok(Value::AutogradTensor(new_tensor))
        }
        _ => Err("tensor_requires_grad() expects (tensor, bool)".to_string()),
    }
}

/// tensor_zero_grad(t: Tensor) -> ()
/// Reset gradients to zero
// Note: tensor_zero_grad and tensor_grad implementations moved to end of file
// to use global computation graph

/// tensor_set_grad(t: Tensor, grad: [float]) -> Tensor
/// Manually set the gradient of a tensor
pub fn builtin_tensor_set_grad(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("tensor_set_grad() expects 2 arguments: tensor_set_grad(tensor, grad_data)".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::AutogradTensor(tensor), Value::Array(grad_vals)) => {
            // Convert grad array to Vec<f64>
            let mut grad_data = Vec::new();
            for val in grad_vals {
                let num = val.to_float()?;
                grad_data.push(num);
            }

            // Check size matches
            if grad_data.len() != tensor.data.len() {
                return Err(format!(
                    "tensor_set_grad(): gradient size {} doesn't match tensor size {}",
                    grad_data.len(),
                    tensor.data.len()
                ));
            }

            let mut new_tensor = tensor.clone();
            new_tensor.grad = Some(grad_data);
            Ok(Value::AutogradTensor(new_tensor))
        }
        _ => Err("tensor_set_grad() expects (tensor, array)".to_string()),
    }
}

/// tensor_sub(a: Tensor, b: Tensor) -> Tensor
/// Element-wise subtraction of two tensors
pub fn builtin_tensor_sub(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("tensor_sub() expects 2 arguments: tensor_sub(a, b)".to_string());
    }

    match (&args[0], &args[1]) {
        // CPU Tensor - CPU Tensor
        (Value::AutogradTensor(a), Value::AutogradTensor(b)) => {
            // Check shapes match
            if a.shape != b.shape {
                return Err(format!(
                    "tensor_sub(): shape mismatch {:?} vs {:?}",
                    a.shape, b.shape
                ));
            }

            // If either tensor requires grad, use autograd system
            if a.requires_grad || b.requires_grad {
                crate::autograd::with_global_graph_mut(|graph| {
                    if graph.get_node(a.id).is_none() {
                        graph.add_node(a.clone());
                    }
                    if graph.get_node(b.id).is_none() {
                        graph.add_node(b.clone());
                    }

                    let result = crate::autograd::sub(graph, a, b)?;
                    Ok(Value::AutogradTensor(result))
                })
            } else {
                // Element-wise subtraction without gradient tracking
                let result_data: Vec<f64> = a
                    .data
                    .iter()
                    .zip(b.data.iter())
                    .map(|(x, y)| x - y)
                    .collect();

                let result_tensor = AutogradTensor::new(result_data, a.shape.clone());
                Ok(Value::AutogradTensor(result_tensor))
            }
        }
        // GPU Tensor - GPU Tensor (element-wise)
        (Value::GPUTensor(a), Value::GPUTensor(b)) => {
            if a.tensor.shape != b.tensor.shape {
                return Err(format!(
                    "tensor_sub(): shape mismatch {:?} vs {:?}",
                    a.tensor.shape, b.tensor.shape
                ));
            }

            // Operate on CPU tensor data for now (similar to tensor_mul)
            let result_data: Vec<f64> = a.tensor.data
                .iter()
                .zip(b.tensor.data.iter())
                .map(|(x, y)| x - y)
                .collect();

            let result_tensor = AutogradTensor::new(result_data, a.tensor.shape.clone());
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            // Move to GPU using global backend
            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        // Mixed CPU/GPU - require explicit conversion
        (Value::AutogradTensor(_), Value::GPUTensor(_)) | (Value::GPUTensor(_), Value::AutogradTensor(_)) => {
            Err("Cannot subtract CPU tensor and GPU tensor. Use tensor_to_gpu() or tensor_to_cpu() to convert first.".to_string())
        }
        _ => Err("tensor_sub() expects two tensors".to_string()),
    }
}

/// tensor_div(a: Tensor, b: Tensor|float) -> Tensor
/// Element-wise division or scalar division
pub fn builtin_tensor_div(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("tensor_div() expects 2 arguments: tensor_div(tensor, divisor)".to_string());
    }

    match (&args[0], &args[1]) {
        // CPU Tensor / CPU Tensor (element-wise)
        (Value::AutogradTensor(a), Value::AutogradTensor(b)) => {
            if a.shape != b.shape {
                return Err(format!(
                    "tensor_div(): shape mismatch {:?} vs {:?}",
                    a.shape, b.shape
                ));
            }

            // If either tensor requires grad, use autograd system
            if a.requires_grad || b.requires_grad {
                crate::autograd::with_global_graph_mut(|graph| {
                    if graph.get_node(a.id).is_none() {
                        graph.add_node(a.clone());
                    }
                    if graph.get_node(b.id).is_none() {
                        graph.add_node(b.clone());
                    }

                    let result = crate::autograd::div(graph, a, b)?;
                    Ok(Value::AutogradTensor(result))
                })
            } else {
                let result_data: Vec<f64> = a
                    .data
                    .iter()
                    .zip(b.data.iter())
                    .map(|(x, y)| {
                        if *y == 0.0 {
                            f64::NAN
                        } else {
                            x / y
                        }
                    })
                    .collect();

                let result_tensor = AutogradTensor::new(result_data, a.shape.clone());
                Ok(Value::AutogradTensor(result_tensor))
            }
        }
        // GPU Tensor / GPU Tensor (element-wise)
        (Value::GPUTensor(a), Value::GPUTensor(b)) => {
            if a.tensor.shape != b.tensor.shape {
                return Err(format!(
                    "tensor_div(): shape mismatch {:?} vs {:?}",
                    a.tensor.shape, b.tensor.shape
                ));
            }

            // Operate on CPU tensor data for now (similar to tensor_mul)
            let result_data: Vec<f64> = a.tensor.data
                .iter()
                .zip(b.tensor.data.iter())
                .map(|(x, y)| {
                    if *y == 0.0 {
                        f64::NAN
                    } else {
                        x / y
                    }
                })
                .collect();

            let result_tensor = AutogradTensor::new(result_data, a.tensor.shape.clone());
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            // Move to GPU using global backend
            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        // CPU Tensor / Scalar
        (Value::AutogradTensor(tensor), scalar) => {
            let scalar_val = scalar.to_float()?;

            if scalar_val == 0.0 {
                return Err("tensor_div(): division by zero".to_string());
            }

            let result_data: Vec<f64> = tensor.data.iter().map(|x| x / scalar_val).collect();

            let result_tensor = AutogradTensor::new(result_data, tensor.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
        }
        // GPU Tensor / Scalar
        (Value::GPUTensor(tensor), scalar) => {
            let scalar_val = scalar.to_float()?;

            if scalar_val == 0.0 {
                return Err("tensor_div(): division by zero".to_string());
            }

            let result_data: Vec<f64> = tensor.tensor.data.iter().map(|x| x / scalar_val).collect();

            let result_tensor = AutogradTensor::new(result_data, tensor.tensor.shape.clone());
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            // Move to GPU using global backend
            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        // Mixed CPU/GPU - require explicit conversion
        (Value::AutogradTensor(_), Value::GPUTensor(_)) | (Value::GPUTensor(_), Value::AutogradTensor(_)) => {
            Err("Cannot divide CPU tensor and GPU tensor. Use tensor_to_gpu() or tensor_to_cpu() to convert first.".to_string())
        }
        _ => Err("tensor_div() expects (tensor, tensor) or (tensor, scalar)".to_string()),
    }
}

// ============================================================================
// NEURAL NETWORK FUNCTIONS
// ============================================================================

/// nn_relu(x: Tensor) -> Tensor
/// Apply ReLU activation: max(0, x)
pub fn builtin_nn_relu(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("nn_relu() expects 1 argument: nn_relu(tensor)".to_string());
    }

    match &args[0] {
        // CPU path
        Value::AutogradTensor(tensor) => {
            // If tensor requires grad, use autograd system
            if tensor.requires_grad {
                crate::autograd::with_global_graph_mut(|graph| {
                    if graph.get_node(tensor.id).is_none() {
                        graph.add_node(tensor.clone());
                    }

                    let result = crate::autograd::relu(graph, tensor)?;
                    Ok(Value::AutogradTensor(result))
                })
            } else {
                // Fast path: no gradients needed
                let result_data: Vec<f64> = tensor.data.iter().map(|&x| x.max(0.0)).collect();
                let result_tensor = AutogradTensor::new(result_data, tensor.shape.clone());
                Ok(Value::AutogradTensor(result_tensor))
            }
        }
        // GPU path
        Value::GPUTensor(gpu_tensor) => {
            let result_data: Vec<f64> = gpu_tensor.tensor.data.iter().map(|&x| x.max(0.0)).collect();
            let result_tensor = AutogradTensor::new(result_data, gpu_tensor.tensor.shape.clone());
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            // Move to GPU using global backend
            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        _ => Err("nn_relu() expects a tensor".to_string()),
    }
}

/// nn_sigmoid(x: Tensor) -> Tensor
/// Apply Sigmoid activation: 1 / (1 + exp(-x))
pub fn builtin_nn_sigmoid(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("nn_sigmoid() expects 1 argument: nn_sigmoid(tensor)".to_string());
    }

    match &args[0] {
        // CPU path
        Value::AutogradTensor(tensor) => {
            // If tensor requires grad, use autograd system
            if tensor.requires_grad {
                crate::autograd::with_global_graph_mut(|graph| {
                    if graph.get_node(tensor.id).is_none() {
                        graph.add_node(tensor.clone());
                    }

                    let result = crate::autograd::sigmoid(graph, tensor)?;
                    Ok(Value::AutogradTensor(result))
                })
            } else {
                // Fast path: no gradients needed
                let result_data: Vec<f64> = tensor
                    .data
                    .iter()
                    .map(|&x| 1.0 / (1.0 + (-x).exp()))
                    .collect();
                let result_tensor = AutogradTensor::new(result_data, tensor.shape.clone());
                Ok(Value::AutogradTensor(result_tensor))
            }
        }
        // GPU path
        Value::GPUTensor(gpu_tensor) => {
            let result_data: Vec<f64> = gpu_tensor.tensor
                .data
                .iter()
                .map(|&x| 1.0 / (1.0 + (-x).exp()))
                .collect();
            let result_tensor = AutogradTensor::new(result_data, gpu_tensor.tensor.shape.clone());
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            // Move to GPU using global backend
            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        _ => Err("nn_sigmoid() expects a tensor".to_string()),
    }
}

/// nn_tanh(x: Tensor) -> Tensor
/// Apply Tanh activation: tanh(x)
pub fn builtin_nn_tanh(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("nn_tanh() expects 1 argument: nn_tanh(tensor)".to_string());
    }

    match &args[0] {
        // CPU path
        Value::AutogradTensor(tensor) => {
            let result_data: Vec<f64> = tensor.data.iter().map(|&x| x.tanh()).collect();
            let result_tensor = AutogradTensor::new(result_data, tensor.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
        }
        // GPU path
        Value::GPUTensor(gpu_tensor) => {
            let result_data: Vec<f64> = gpu_tensor.tensor.data.iter().map(|&x| x.tanh()).collect();
            let result_tensor = AutogradTensor::new(result_data, gpu_tensor.tensor.shape.clone());
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            // Move to GPU using global backend
            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        _ => Err("nn_tanh() expects a tensor".to_string()),
    }
}

/// nn_softmax(x: Tensor) -> Tensor
/// Apply Softmax activation: exp(x) / sum(exp(x))
pub fn builtin_nn_softmax(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("nn_softmax() expects 1 argument: nn_softmax(tensor)".to_string());
    }

    match &args[0] {
        // CPU path
        Value::AutogradTensor(tensor) => {
            // Find max for numerical stability
            let max = tensor
                .data
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            // Compute exp(x - max)
            let exp_values: Vec<f64> = tensor.data.iter().map(|&x| (x - max).exp()).collect();

            // Compute sum
            let sum: f64 = exp_values.iter().sum();

            // Normalize
            let result_data: Vec<f64> = exp_values.iter().map(|&x| x / sum).collect();

            let result_tensor = AutogradTensor::new(result_data, tensor.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
        }
        // GPU path
        Value::GPUTensor(gpu_tensor) => {
            // Find max for numerical stability
            let max = gpu_tensor.tensor
                .data
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            // Compute exp(x - max)
            let exp_values: Vec<f64> = gpu_tensor.tensor.data.iter().map(|&x| (x - max).exp()).collect();

            // Compute sum
            let sum: f64 = exp_values.iter().sum();

            // Normalize
            let result_data: Vec<f64> = exp_values.iter().map(|&x| x / sum).collect();

            let result_tensor = AutogradTensor::new(result_data, gpu_tensor.tensor.shape.clone());
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            // Move to GPU using global backend
            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        _ => Err("nn_softmax() expects a tensor".to_string()),
    }
}

/// nn_linear(input: Tensor, weight: Tensor, bias: Tensor) -> Tensor
/// Linear/Dense layer: output = input @ weight + bias
/// input shape: (batch, in_features) or (in_features,)
/// weight shape: (in_features, out_features)
/// bias shape: (out_features,)
pub fn builtin_nn_linear(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err(
            "nn_linear() expects 3 arguments: nn_linear(input, weight, bias)".to_string(),
        );
    }

    match (&args[0], &args[1], &args[2]) {
        (
            Value::AutogradTensor(input),
            Value::AutogradTensor(weight),
            Value::AutogradTensor(bias),
        ) => {
            // If any tensor requires grad, use autograd system
            if input.requires_grad || weight.requires_grad || bias.requires_grad {
                // Ensure all tensors are in the global graph
                crate::autograd::with_global_graph_mut(|graph| {
                    if graph.get_node(input.id).is_none() {
                        graph.add_node(input.clone());
                    }
                    if graph.get_node(weight.id).is_none() {
                        graph.add_node(weight.clone());
                    }
                    if graph.get_node(bias.id).is_none() {
                        graph.add_node(bias.clone());
                    }

                    // Use autograd linear to record in graph
                    let result = crate::autograd::linear(graph, input, weight, bias)?;
                    Ok(Value::AutogradTensor(result))
                })
            } else {
                // Fast path: no gradients needed, compute directly
                let (input_data, input_shape) = if input.shape.len() == 1 {
                    (input.data.clone(), vec![1, input.shape[0]])
                } else {
                    (input.data.clone(), input.shape.clone())
                };

                if input_shape.len() != 2 || weight.shape.len() != 2 || bias.shape.len() != 1 {
                    return Err(format!(
                        "nn_linear(): invalid shapes - input: {:?}, weight: {:?}, bias: {:?}",
                        input_shape, weight.shape, bias.shape
                    ));
                }

                let (batch, in_features) = (input_shape[0], input_shape[1]);
                let (weight_in, out_features) = (weight.shape[0], weight.shape[1]);

                if in_features != weight_in {
                    return Err(format!(
                        "nn_linear(): input features {} doesn't match weight input {}",
                        in_features, weight_in
                    ));
                }

                if bias.shape[0] != out_features {
                    return Err(format!(
                        "nn_linear(): bias size {} doesn't match output features {}",
                        bias.shape[0], out_features
                    ));
                }

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

                let result_shape = if input.shape.len() == 1 {
                    vec![out_features]
                } else {
                    vec![batch, out_features]
                };

                let result_tensor = AutogradTensor::new(result_data, result_shape);
                Ok(Value::AutogradTensor(result_tensor))
            }
        }
        // GPU path
        (
            Value::GPUTensor(input),
            Value::GPUTensor(weight),
            Value::GPUTensor(bias),
        ) => {
            // Handle 1D input: reshape to (1, in_features)
            let (input_data, input_shape) = if input.tensor.shape.len() == 1 {
                (input.tensor.data.clone(), vec![1, input.tensor.shape[0]])
            } else {
                (input.tensor.data.clone(), input.tensor.shape.clone())
            };

            if input_shape.len() != 2 || weight.tensor.shape.len() != 2 || bias.tensor.shape.len() != 1 {
                return Err(format!(
                    "nn_linear(): invalid shapes - input: {:?}, weight: {:?}, bias: {:?}",
                    input_shape, weight.tensor.shape, bias.tensor.shape
                ));
            }

            let (batch, in_features) = (input_shape[0], input_shape[1]);
            let (weight_in, out_features) = (weight.tensor.shape[0], weight.tensor.shape[1]);

            if in_features != weight_in {
                return Err(format!(
                    "nn_linear(): input features {} doesn't match weight input {}",
                    in_features, weight_in
                ));
            }

            if bias.tensor.shape[0] != out_features {
                return Err(format!(
                    "nn_linear(): bias size {} doesn't match output features {}",
                    bias.tensor.shape[0], out_features
                ));
            }

            // Matrix multiplication: input @ weight
            let mut result_data = vec![0.0; batch * out_features];
            for i in 0..batch {
                for j in 0..out_features {
                    let mut sum = 0.0;
                    for k in 0..in_features {
                        sum += input_data[i * in_features + k] * weight.tensor.data[k * out_features + j];
                    }
                    result_data[i * out_features + j] = sum + bias.tensor.data[j];
                }
            }

            // Return shape matches input: 1D input -> 1D output
            let result_shape = if input.tensor.shape.len() == 1 {
                vec![out_features]
            } else {
                vec![batch, out_features]
            };

            let result_tensor = AutogradTensor::new(result_data, result_shape);
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            // Move to GPU using global backend
            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        // Mixed CPU/GPU - require explicit conversion
        (Value::AutogradTensor(_), Value::GPUTensor(_), _) |
        (Value::GPUTensor(_), Value::AutogradTensor(_), _) |
        (Value::GPUTensor(_), Value::GPUTensor(_), Value::AutogradTensor(_)) |
        (Value::AutogradTensor(_), Value::AutogradTensor(_), Value::GPUTensor(_)) |
        (Value::AutogradTensor(_), Value::GPUTensor(_), Value::GPUTensor(_)) |
        (Value::GPUTensor(_), Value::AutogradTensor(_), Value::GPUTensor(_)) |
        (Value::GPUTensor(_), Value::AutogradTensor(_), Value::AutogradTensor(_)) => {
            Err("nn_linear(): Cannot mix CPU and GPU tensors. Use tensor_to_gpu() or tensor_to_cpu() to convert first.".to_string())
        }
        _ => Err("nn_linear() expects three tensors".to_string()),
    }
}

// ============================================================================
// LOSS FUNCTIONS
// ============================================================================

/// loss_mse(pred: Tensor, target: Tensor) -> float
/// Mean Squared Error: mean((pred - target)^2)
pub fn builtin_loss_mse(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("loss_mse() expects 2 arguments: loss_mse(pred, target)".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::AutogradTensor(pred), Value::AutogradTensor(target)) => {
            if pred.shape != target.shape {
                return Err(format!(
                    "loss_mse(): shape mismatch - pred: {:?}, target: {:?}",
                    pred.shape, target.shape
                ));
            }

            // If pred requires grad, use autograd system and return Tensor
            if pred.requires_grad {
                crate::autograd::with_global_graph_mut(|graph| {
                    if graph.get_node(pred.id).is_none() {
                        graph.add_node(pred.clone());
                    }
                    if graph.get_node(target.id).is_none() {
                        graph.add_node(target.clone());
                    }

                    let result = crate::autograd::mse_loss(graph, pred, target)?;
                    Ok(Value::AutogradTensor(result))
                })
            } else {
                // Fast path: no gradients needed, return Float
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

                Ok(Value::Float(mse))
            }
        }
        _ => Err("loss_mse() expects two tensors".to_string()),
    }
}

/// loss_cross_entropy(pred: Tensor, target: Tensor) -> float
/// Cross Entropy Loss: -sum(target * log(pred))
/// pred: probability distribution (output of softmax)
/// target: one-hot encoded labels or probabilities
pub fn builtin_loss_cross_entropy(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(
            "loss_cross_entropy() expects 2 arguments: loss_cross_entropy(pred, target)"
                .to_string(),
        );
    }

    match (&args[0], &args[1]) {
        (Value::AutogradTensor(pred), Value::AutogradTensor(target)) => {
            if pred.shape != target.shape {
                return Err(format!(
                    "loss_cross_entropy(): shape mismatch - pred: {:?}, target: {:?}",
                    pred.shape, target.shape
                ));
            }

            // Clip predictions to avoid log(0)
            let epsilon = 1e-10;

            let ce: f64 = pred
                .data
                .iter()
                .zip(target.data.iter())
                .map(|(&p, &t)| {
                    let clipped_p = p.max(epsilon).min(1.0 - epsilon);
                    -t * clipped_p.ln()
                })
                .sum();

            // Average over batch if 2D
            let n = if pred.shape.len() == 2 {
                pred.shape[0]
            } else {
                1
            };

            Ok(Value::Float(ce / n as f64))
        }
        _ => Err("loss_cross_entropy() expects two tensors".to_string()),
    }
}

// ============================================================================
// OPTIMIZER FUNCTIONS
// ============================================================================

/// optim_sgd_step(params: Tensor, grads: [float], lr: float) -> Tensor
/// Perform one SGD optimization step: params = params - lr * grads
pub fn builtin_optim_sgd_step(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err(
            "optim_sgd_step() expects 3 arguments: optim_sgd_step(params, grads, lr)"
                .to_string(),
        );
    }

    match (&args[0], &args[1], &args[2]) {
        (Value::AutogradTensor(params), Value::Array(grads_vals), lr_val) => {
            let lr = lr_val.to_float()?;

            // Convert grad array to Vec<f64>
            let mut grads = Vec::new();
            for val in grads_vals {
                let g = val.to_float()?;
                grads.push(g);
            }

            if grads.len() != params.data.len() {
                return Err(format!(
                    "optim_sgd_step(): gradient size {} doesn't match params size {}",
                    grads.len(),
                    params.data.len()
                ));
            }

            // Update: params = params - lr * grads
            let updated_data: Vec<f64> = params
                .data
                .iter()
                .zip(grads.iter())
                .map(|(&p, &g)| p - lr * g)
                .collect();

            let updated_tensor = AutogradTensor::new(updated_data, params.shape.clone());
            Ok(Value::AutogradTensor(updated_tensor))
        }
        _ => Err("optim_sgd_step() expects (tensor, array, float)".to_string()),
    }
}

/// optim_sgd_momentum_step(params, grads, velocity, lr, momentum) -> (Tensor, [float])
/// SGD with momentum: returns (updated_params, updated_velocity)
pub fn builtin_optim_sgd_momentum_step(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 5 {
        return Err("optim_sgd_momentum_step() expects 5 arguments: optim_sgd_momentum_step(params, grads, velocity, lr, momentum)".to_string());
    }

    match (&args[0], &args[1], &args[2], &args[3], &args[4]) {
        (
            Value::AutogradTensor(params),
            Value::Array(grads_vals),
            Value::Array(velocity_vals),
            lr_val,
            momentum_val,
        ) => {
            let lr = lr_val.to_float()?;
            let momentum = momentum_val.to_float()?;

            // Convert arrays to Vec<f64>
            let mut grads = Vec::new();
            for val in grads_vals {
                grads.push(val.to_float()?);
            }

            let mut velocity = Vec::new();
            for val in velocity_vals {
                velocity.push(val.to_float()?);
            }

            if grads.len() != params.data.len() || velocity.len() != params.data.len() {
                return Err("optim_sgd_momentum_step(): size mismatch".to_string());
            }

            // Update velocity: v = momentum * v + grads
            // Update params: params = params - lr * v
            let mut updated_data = Vec::new();
            let mut updated_velocity = Vec::new();

            for i in 0..params.data.len() {
                let new_v = momentum * velocity[i] + grads[i];
                updated_velocity.push(new_v);
                updated_data.push(params.data[i] - lr * new_v);
            }

            // Return tuple: (params, velocity)
            let updated_params = AutogradTensor::new(updated_data, params.shape.clone());
            let velocity_values: Vec<Value> =
                updated_velocity.iter().map(|&v| Value::Float(v)).collect();

            Ok(Value::Tuple(vec![
                Value::AutogradTensor(updated_params),
                Value::Array(velocity_values),
            ]))
        }
        _ => Err("optim_sgd_momentum_step() expects (tensor, array, array, float, float)"
            .to_string()),
    }
}

/// optim_adam_step(params, grads, m, v, t, lr, beta1, beta2) -> (Tensor, [float], [float])
/// Adam optimizer step: returns (updated_params, updated_m, updated_v)
pub fn builtin_optim_adam_step(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 8 {
        return Err("optim_adam_step() expects 8 arguments: optim_adam_step(params, grads, m, v, t, lr, beta1, beta2)".to_string());
    }

    match (
        &args[0], &args[1], &args[2], &args[3], &args[4], &args[5], &args[6], &args[7],
    ) {
        (
            Value::AutogradTensor(params),
            Value::Array(grads_vals),
            Value::Array(m_vals),
            Value::Array(v_vals),
            t_val,
            lr_val,
            beta1_val,
            beta2_val,
        ) => {
            let t = t_val.to_float()? as usize;
            let lr = lr_val.to_float()?;
            let beta1 = beta1_val.to_float()?;
            let beta2 = beta2_val.to_float()?;
            let epsilon = 1e-8;

            // Convert arrays
            let mut grads = Vec::new();
            for val in grads_vals {
                grads.push(val.to_float()?);
            }

            let mut m = Vec::new();
            for val in m_vals {
                m.push(val.to_float()?);
            }

            let mut v = Vec::new();
            for val in v_vals {
                v.push(val.to_float()?);
            }

            if grads.len() != params.data.len()
                || m.len() != params.data.len()
                || v.len() != params.data.len()
            {
                return Err("optim_adam_step(): size mismatch".to_string());
            }

            // Adam update
            let mut updated_data = Vec::new();
            let mut updated_m = Vec::new();
            let mut updated_v = Vec::new();

            for i in 0..params.data.len() {
                // Update biased first moment: m = beta1 * m + (1 - beta1) * grad
                let new_m = beta1 * m[i] + (1.0 - beta1) * grads[i];
                updated_m.push(new_m);

                // Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
                let new_v = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
                updated_v.push(new_v);

                // Bias correction
                let m_hat = new_m / (1.0 - beta1.powi(t as i32));
                let v_hat = new_v / (1.0 - beta2.powi(t as i32));

                // Update params: params = params - lr * m_hat / (sqrt(v_hat) + epsilon)
                let update = lr * m_hat / (v_hat.sqrt() + epsilon);
                updated_data.push(params.data[i] - update);
            }

            // Return tuple: (params, m, v)
            let updated_params = AutogradTensor::new(updated_data, params.shape.clone());
            let m_values: Vec<Value> = updated_m.iter().map(|&x| Value::Float(x)).collect();
            let v_values: Vec<Value> = updated_v.iter().map(|&x| Value::Float(x)).collect();

            Ok(Value::Tuple(vec![
                Value::AutogradTensor(updated_params),
                Value::Array(m_values),
                Value::Array(v_values),
            ]))
        }
        _ => Err("optim_adam_step() expects correct types".to_string()),
    }
}

/// tensor_clip_grad(grads: [float], max_norm: float) -> [float]
/// Clip gradients by norm to prevent exploding gradients
pub fn builtin_tensor_clip_grad(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(
            "tensor_clip_grad() expects 2 arguments: tensor_clip_grad(grads, max_norm)"
                .to_string(),
        );
    }

    match (&args[0], &args[1]) {
        (Value::Array(grads_vals), max_norm_val) => {
            let max_norm = max_norm_val.to_float()?;

            // Convert to Vec<f64>
            let mut grads = Vec::new();
            for val in grads_vals {
                grads.push(val.to_float()?);
            }

            // Calculate current norm
            let norm: f64 = grads.iter().map(|&g| g * g).sum::<f64>().sqrt();

            // Clip if necessary
            let clipped_grads = if norm > max_norm {
                let scale = max_norm / norm;
                grads.iter().map(|&g| g * scale).collect()
            } else {
                grads
            };

            let result: Vec<Value> = clipped_grads.iter().map(|&g| Value::Float(g)).collect();
            Ok(Value::Array(result))
        }
        _ => Err("tensor_clip_grad() expects (array, float)".to_string()),
    }
}

// ============================================================================
// AUTOGRAD COMPUTATION GRAPH FUNCTIONS
// ============================================================================

// Note: These functions will need special handling in the interpreter
// because they need access to the ComputationGraph state

/// autograd_register_node(tensor: Tensor) -> int
/// Register a tensor in the computation graph and return its node ID
/// This is called internally by tensor operations when autograd is enabled
pub fn builtin_autograd_register_node(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("autograd_register_node() expects 1 argument".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(tensor) => {
            // Return the tensor's ID (it's already part of the tensor)
            Ok(Value::Integer(tensor.id as i64))
        }
        _ => Err("autograd_register_node() expects a tensor".to_string()),
    }
}

/// autograd_get_node_grad(node_id: int) -> [float] | null
/// Get gradients for a specific node ID from the computation graph
/// This function needs interpreter-level implementation
pub fn builtin_autograd_get_node_grad(_args: Vec<Value>) -> Result<Value, String> {
    // This is a placeholder - actual implementation needs interpreter access
    Err("autograd_get_node_grad() requires interpreter-level implementation".to_string())
}

// ============================================================================
// AUTOGRAD HELPER FUNCTIONS (Simplified)
// ============================================================================

/// autograd_compute_linear_grad(input, weight, bias, output_grad) -> (input_grad, weight_grad, bias_grad)
/// Compute gradients for a linear layer backward pass
/// output_grad: gradient flowing back from the output
/// Returns: tuple of (grad_input, grad_weight, grad_bias)
pub fn builtin_autograd_compute_linear_grad(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("autograd_compute_linear_grad() expects 4 arguments: (input, weight, bias, output_grad)".to_string());
    }

    match (&args[0], &args[1], &args[2], &args[3]) {
        (
            Value::AutogradTensor(input),
            Value::AutogradTensor(weight),
            Value::AutogradTensor(bias),
            Value::Array(output_grad_vals),
        ) => {
            // Convert output_grad to Vec<f64>
            let mut output_grad = Vec::new();
            for val in output_grad_vals {
                output_grad.push(val.to_float()?);
            }

            // Handle 1D input
            let (input_data, input_shape) = if input.shape.len() == 1 {
                (input.data.clone(), vec![1, input.shape[0]])
            } else {
                (input.data.clone(), input.shape.clone())
            };

            let (batch, in_features) = (input_shape[0], input_shape[1]);
            let (weight_in, out_features) = (weight.shape[0], weight.shape[1]);

            if output_grad.len() != batch * out_features {
                return Err("autograd_compute_linear_grad(): output_grad size mismatch".to_string());
            }

            // Compute gradients
            // grad_input = output_grad @ weight^T
            let mut grad_input = vec![0.0; batch * in_features];
            for i in 0..batch {
                for j in 0..in_features {
                    let mut sum = 0.0;
                    for k in 0..out_features {
                        sum += output_grad[i * out_features + k] * weight.data[j * out_features + k];
                    }
                    grad_input[i * in_features + j] = sum;
                }
            }

            // grad_weight = input^T @ output_grad
            let mut grad_weight = vec![0.0; in_features * out_features];
            for i in 0..in_features {
                for j in 0..out_features {
                    let mut sum = 0.0;
                    for k in 0..batch {
                        sum += input_data[k * in_features + i] * output_grad[k * out_features + j];
                    }
                    grad_weight[i * out_features + j] = sum;
                }
            }

            // grad_bias = sum(output_grad, axis=0)
            let mut grad_bias = vec![0.0; out_features];
            for i in 0..batch {
                for j in 0..out_features {
                    grad_bias[j] += output_grad[i * out_features + j];
                }
            }

            // Return as tuple of arrays
            let grad_input_vals: Vec<Value> = grad_input.iter().map(|&g| Value::Float(g)).collect();
            let grad_weight_vals: Vec<Value> = grad_weight.iter().map(|&g| Value::Float(g)).collect();
            let grad_bias_vals: Vec<Value> = grad_bias.iter().map(|&g| Value::Float(g)).collect();

            Ok(Value::Tuple(vec![
                Value::Array(grad_input_vals),
                Value::Array(grad_weight_vals),
                Value::Array(grad_bias_vals),
            ]))
        }
        _ => Err("autograd_compute_linear_grad() expects (tensor, tensor, tensor, array)".to_string()),
    }
}

/// autograd_compute_relu_grad(input, output_grad) -> input_grad
/// Compute gradients for ReLU backward pass
pub fn builtin_autograd_compute_relu_grad(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("autograd_compute_relu_grad() expects 2 arguments: (input, output_grad)".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::AutogradTensor(input), Value::Array(output_grad_vals)) => {
            let mut output_grad = Vec::new();
            for val in output_grad_vals {
                output_grad.push(val.to_float()?);
            }

            if output_grad.len() != input.data.len() {
                return Err("autograd_compute_relu_grad(): size mismatch".to_string());
            }

            // ReLU gradient: pass through if input > 0, else 0
            let grad_input: Vec<f64> = input
                .data
                .iter()
                .zip(output_grad.iter())
                .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
                .collect();

            let result: Vec<Value> = grad_input.iter().map(|&g| Value::Float(g)).collect();
            Ok(Value::Array(result))
        }
        _ => Err("autograd_compute_relu_grad() expects (tensor, array)".to_string()),
    }
}

/// autograd_compute_sigmoid_grad(output, output_grad) -> input_grad
/// Compute gradients for Sigmoid backward pass
/// output: the output of sigmoid forward (not input)
pub fn builtin_autograd_compute_sigmoid_grad(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("autograd_compute_sigmoid_grad() expects 2 arguments: (output, output_grad)".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::AutogradTensor(output), Value::Array(output_grad_vals)) => {
            let mut output_grad = Vec::new();
            for val in output_grad_vals {
                output_grad.push(val.to_float()?);
            }

            if output_grad.len() != output.data.len() {
                return Err("autograd_compute_sigmoid_grad(): size mismatch".to_string());
            }

            // Sigmoid gradient: output * (1 - output) * output_grad
            let grad_input: Vec<f64> = output
                .data
                .iter()
                .zip(output_grad.iter())
                .map(|(&y, &g)| g * y * (1.0 - y))
                .collect();

            let result: Vec<Value> = grad_input.iter().map(|&g| Value::Float(g)).collect();
            Ok(Value::Array(result))
        }
        _ => Err("autograd_compute_sigmoid_grad() expects (tensor, array)".to_string()),
    }
}

/// autograd_compute_mse_grad(pred, target) -> pred_grad
/// Compute gradients for MSE loss: 2 * (pred - target) / n
pub fn builtin_autograd_compute_mse_grad(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("autograd_compute_mse_grad() expects 2 arguments: (pred, target)".to_string());
    }

    match (&args[0], &args[1]) {
        (Value::AutogradTensor(pred), Value::AutogradTensor(target)) => {
            if pred.shape != target.shape {
                return Err("autograd_compute_mse_grad(): shape mismatch".to_string());
            }

            let n = pred.data.len() as f64;

            // MSE gradient: 2 * (pred - target) / n
            let grad: Vec<f64> = pred
                .data
                .iter()
                .zip(target.data.iter())
                .map(|(&p, &t)| 2.0 * (p - t) / n)
                .collect();

            let result: Vec<Value> = grad.iter().map(|&g| Value::Float(g)).collect();
            Ok(Value::Array(result))
        }
        _ => Err("autograd_compute_mse_grad() expects two tensors".to_string()),
    }
}

// ============================================================================
// GPU FUNCTIONS
// ============================================================================

/// gpu_available() -> bool
/// Check if GPU support is available (WGPU backend)
pub fn builtin_gpu_available(_args: Vec<Value>) -> Result<Value, String> {
    // Check if WGPU can initialize
    match crate::gpu::wgpu_backend::WgpuBackend::new_sync() {
        Ok(_) => Ok(Value::Boolean(true)),
        Err(_) => Ok(Value::Boolean(false)),
    }
}

/// gpu_info() -> string
/// Get information about available GPU
pub fn builtin_gpu_info(_args: Vec<Value>) -> Result<Value, String> {
    match crate::gpu::wgpu_backend::WgpuBackend::new_sync() {
        Ok(backend) => {
            let info_struct = backend.adapter_info();
            let info = format!(
                "GPU Backend: WGPU\nAdapter: {}\nBackend: {:?}",
                info_struct.name, info_struct.backend
            );
            Ok(Value::String(info))
        }
        Err(e) => Ok(Value::String(format!("GPU not available: {:?}", e))),
    }
}

/// tensor_device(t: Tensor) -> string
/// Get the device where a tensor resides
pub fn builtin_tensor_device(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_device() expects 1 argument: tensor_device(tensor)".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(_) => {
            Ok(Value::String("CPU".to_string()))
        }
        Value::GPUTensor(gpu_tensor) => {
            let device = if gpu_tensor.is_gpu() { "GPU" } else { "CPU" };
            Ok(Value::String(device.to_string()))
        }
        _ => Err("tensor_device() expects a tensor".to_string()),
    }
}

/// tensor_to_gpu(t: Tensor) -> GPUTensor
/// Move a tensor to GPU memory and return GPU-accelerated tensor
pub fn builtin_tensor_to_gpu(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_to_gpu() expects 1 argument: tensor_to_gpu(tensor)".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(tensor) => {
            // Create GPUTensor from AutogradTensor
            let mut gpu_tensor = crate::gpu_tensor::GPUTensor::from_tensor(tensor.clone());

            // Move data to GPU using global backend
            with_gpu_backend(|backend| {
                gpu_tensor.to_gpu(backend)
                    .map_err(|e| format!("Failed to move tensor to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(gpu_tensor))
        }
        Value::GPUTensor(gpu_tensor) => {
            // Already on GPU, return as-is
            Ok(Value::GPUTensor(gpu_tensor.clone()))
        }
        _ => Err("tensor_to_gpu() expects a tensor".to_string()),
    }
}

/// tensor_to_cpu(t: Tensor) -> Tensor
/// Move a tensor from GPU to CPU memory
pub fn builtin_tensor_to_cpu(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_to_cpu() expects 1 argument: tensor_to_cpu(tensor)".to_string());
    }

    match &args[0] {
        Value::GPUTensor(gpu_tensor) => {
            // Clone the GPU tensor so we can mutate it
            let mut gpu_tensor_clone = gpu_tensor.clone();

            // Move data from GPU to CPU using global backend
            with_gpu_backend(|backend| {
                gpu_tensor_clone.to_cpu(backend)
                    .map_err(|e| format!("Failed to move tensor to CPU: {}", e))
            })?;

            // Return the CPU tensor
            Ok(Value::AutogradTensor(gpu_tensor_clone.tensor))
        }
        Value::AutogradTensor(tensor) => {
            // Already on CPU, return as-is
            Ok(Value::AutogradTensor(tensor.clone()))
        }
        _ => Err("tensor_to_cpu() expects a tensor".to_string()),
    }
}

// =============================================================================
// NEURAL NETWORK LAYERS (v0.2.0)
// =============================================================================

/// linear(in_features: int, out_features: int) -> LinearLayer
/// Create a Linear (Dense/Fully Connected) layer with Xavier initialization
pub fn builtin_linear(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("linear() expects 2 arguments: linear(in_features, out_features)".to_string());
    }

    let in_features = args[0].to_integer()? as usize;
    let out_features = args[1].to_integer()? as usize;

    if in_features == 0 || out_features == 0 {
        return Err("linear() features must be > 0".to_string());
    }

    let layer = crate::nn::gpu_layers::Linear::xavier(in_features, out_features);
    Ok(Value::LinearLayer(Box::new(layer)))
}

/// conv2d(in_channels: int, out_channels: int, kernel_size: int) -> Conv2dLayer
/// Create a Conv2d layer for CNNs (He initialization)
pub fn builtin_conv2d(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("conv2d() expects 3 arguments: conv2d(in_channels, out_channels, kernel_size)".to_string());
    }

    let in_channels = args[0].to_integer()? as usize;
    let out_channels = args[1].to_integer()? as usize;
    let kernel_size = args[2].to_integer()? as usize;

    if in_channels == 0 || out_channels == 0 || kernel_size == 0 {
        return Err("conv2d() parameters must be > 0".to_string());
    }

    let layer = crate::nn::gpu_layers::Conv2d::new(in_channels, out_channels, kernel_size);
    Ok(Value::Conv2dLayer(Box::new(layer)))
}

/// maxpool2d(kernel_size: int) -> MaxPool2dLayer
/// Create a MaxPool2d layer for downsampling (stride = kernel_size)
pub fn builtin_maxpool2d(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("maxpool2d() expects 1 argument: maxpool2d(kernel_size)".to_string());
    }

    let kernel_size = args[0].to_integer()? as usize;

    if kernel_size == 0 {
        return Err("maxpool2d() kernel_size must be > 0".to_string());
    }

    let layer = crate::nn::gpu_layers::MaxPool2d::new(kernel_size);
    Ok(Value::MaxPool2dLayer(Box::new(layer)))
}

/// avgpool2d(kernel_size: int) -> AvgPool2dLayer
/// Create an AvgPool2d layer for downsampling (stride = kernel_size)
pub fn builtin_avgpool2d(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("avgpool2d() expects 1 argument: avgpool2d(kernel_size)".to_string());
    }

    let kernel_size = args[0].to_integer()? as usize;

    if kernel_size == 0 {
        return Err("avgpool2d() kernel_size must be > 0".to_string());
    }

    let layer = crate::nn::gpu_layers::AvgPool2d::new(kernel_size);
    Ok(Value::AvgPool2dLayer(Box::new(layer)))
}

/// batchnorm(num_features: int) -> BatchNormLayer
/// Create a BatchNorm layer for training stability
pub fn builtin_batchnorm(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("batchnorm() expects 1 argument: batchnorm(num_features)".to_string());
    }

    let num_features = args[0].to_integer()? as usize;

    if num_features == 0 {
        return Err("batchnorm() num_features must be > 0".to_string());
    }

    let layer = crate::nn::gpu_layers::BatchNorm::new(num_features);
    Ok(Value::BatchNormLayer(Box::new(layer)))
}

/// layernorm(features: int) -> LayerNormLayer
/// Create a LayerNorm layer for Transformers
pub fn builtin_layernorm(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("layernorm() expects 1 argument: layernorm(features)".to_string());
    }

    let features = args[0].to_integer()? as usize;

    if features == 0 {
        return Err("layernorm() features must be > 0".to_string());
    }

    let layer = crate::nn::gpu_layers::LayerNorm::new_1d(features);
    Ok(Value::LayerNormLayer(Box::new(layer)))
}

/// dropout(p: float) -> DropoutLayer
/// Create a Dropout layer for regularization (p = dropout probability)
pub fn builtin_dropout(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("dropout() expects 1 argument: dropout(p)".to_string());
    }

    let p = args[0].to_float()?;

    if p < 0.0 || p > 1.0 {
        return Err(format!("dropout() p must be between 0.0 and 1.0, got {}", p));
    }

    let layer = crate::nn::gpu_layers::Dropout::new(p);
    Ok(Value::DropoutLayer(Box::new(layer)))
}

/// layer_forward(layer: Layer, input: Tensor) -> Tensor
/// Perform forward pass through a layer
pub fn builtin_layer_forward(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("layer_forward() expects 2 arguments: layer_forward(layer, input)".to_string());
    }

    match (&args[0], &args[1]) {
        // Linear Layer + GPU Tensor
        (Value::LinearLayer(layer), Value::GPUTensor(input)) => {
            let output = layer.forward(input)
                .map_err(|e| format!("Linear forward failed: {}", e))?;
            Ok(Value::GPUTensor(output))
        }
        // Linear Layer + CPU Tensor
        (Value::LinearLayer(_), Value::AutogradTensor(input)) => {
            let mut input_gpu = crate::gpu_tensor::GPUTensor::from_tensor(input.clone());
            with_gpu_backend(|backend| {
                input_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move input to GPU: {}", e))
            })?;

            let layer = match &args[0] {
                Value::LinearLayer(l) => l,
                _ => unreachable!(),
            };

            let output_gpu = layer.forward(&input_gpu)
                .map_err(|e| format!("Linear forward failed: {}", e))?;

            let mut output_clone = output_gpu.clone();
            with_gpu_backend(|backend| {
                output_clone.to_cpu(backend)
                    .map_err(|e| format!("Failed to move output to CPU: {}", e))
            })?;

            Ok(Value::AutogradTensor(output_clone.tensor))
        }
        // Conv2d Layer + GPU Tensor
        (Value::Conv2dLayer(layer), Value::GPUTensor(input)) => {
            let output = layer.forward(input)
                .map_err(|e| format!("Conv2d forward failed: {}", e))?;
            Ok(Value::GPUTensor(output))
        }
        // Conv2d Layer + CPU Tensor
        (Value::Conv2dLayer(_), Value::AutogradTensor(input)) => {
            let mut input_gpu = crate::gpu_tensor::GPUTensor::from_tensor(input.clone());
            with_gpu_backend(|backend| {
                input_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move input to GPU: {}", e))
            })?;

            let layer = match &args[0] {
                Value::Conv2dLayer(l) => l,
                _ => unreachable!(),
            };

            let output_gpu = layer.forward(&input_gpu)
                .map_err(|e| format!("Conv2d forward failed: {}", e))?;

            let mut output_clone = output_gpu.clone();
            with_gpu_backend(|backend| {
                output_clone.to_cpu(backend)
                    .map_err(|e| format!("Failed to move output to CPU: {}", e))
            })?;

            Ok(Value::AutogradTensor(output_clone.tensor))
        }
        // MaxPool2d Layer + GPU Tensor
        (Value::MaxPool2dLayer(layer), Value::GPUTensor(input)) => {
            let output = layer.forward(input)
                .map_err(|e| format!("MaxPool2d forward failed: {}", e))?;
            Ok(Value::GPUTensor(output))
        }
        // MaxPool2d Layer + CPU Tensor
        (Value::MaxPool2dLayer(_), Value::AutogradTensor(input)) => {
            let mut input_gpu = crate::gpu_tensor::GPUTensor::from_tensor(input.clone());
            with_gpu_backend(|backend| {
                input_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move input to GPU: {}", e))
            })?;

            let layer = match &args[0] {
                Value::MaxPool2dLayer(l) => l,
                _ => unreachable!(),
            };

            let output_gpu = layer.forward(&input_gpu)
                .map_err(|e| format!("MaxPool2d forward failed: {}", e))?;

            let mut output_clone = output_gpu.clone();
            with_gpu_backend(|backend| {
                output_clone.to_cpu(backend)
                    .map_err(|e| format!("Failed to move output to CPU: {}", e))
            })?;

            Ok(Value::AutogradTensor(output_clone.tensor))
        }
        // AvgPool2d Layer + GPU Tensor
        (Value::AvgPool2dLayer(layer), Value::GPUTensor(input)) => {
            let output = layer.forward(input)
                .map_err(|e| format!("AvgPool2d forward failed: {}", e))?;
            Ok(Value::GPUTensor(output))
        }
        // AvgPool2d Layer + CPU Tensor
        (Value::AvgPool2dLayer(_), Value::AutogradTensor(input)) => {
            let mut input_gpu = crate::gpu_tensor::GPUTensor::from_tensor(input.clone());
            with_gpu_backend(|backend| {
                input_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move input to GPU: {}", e))
            })?;

            let layer = match &args[0] {
                Value::AvgPool2dLayer(l) => l,
                _ => unreachable!(),
            };

            let output_gpu = layer.forward(&input_gpu)
                .map_err(|e| format!("AvgPool2d forward failed: {}", e))?;

            let mut output_clone = output_gpu.clone();
            with_gpu_backend(|backend| {
                output_clone.to_cpu(backend)
                    .map_err(|e| format!("Failed to move output to CPU: {}", e))
            })?;

            Ok(Value::AutogradTensor(output_clone.tensor))
        }
        // BatchNorm Layer + GPU Tensor
        (Value::BatchNormLayer(layer), Value::GPUTensor(input)) => {
            let output = layer.forward(input)
                .map_err(|e| format!("BatchNorm forward failed: {}", e))?;
            Ok(Value::GPUTensor(output))
        }
        // BatchNorm Layer + CPU Tensor
        (Value::BatchNormLayer(_), Value::AutogradTensor(input)) => {
            let mut input_gpu = crate::gpu_tensor::GPUTensor::from_tensor(input.clone());
            with_gpu_backend(|backend| {
                input_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move input to GPU: {}", e))
            })?;

            let layer = match &args[0] {
                Value::BatchNormLayer(l) => l,
                _ => unreachable!(),
            };

            let output_gpu = layer.forward(&input_gpu)
                .map_err(|e| format!("BatchNorm forward failed: {}", e))?;

            let mut output_clone = output_gpu.clone();
            with_gpu_backend(|backend| {
                output_clone.to_cpu(backend)
                    .map_err(|e| format!("Failed to move output to CPU: {}", e))
            })?;

            Ok(Value::AutogradTensor(output_clone.tensor))
        }
        // LayerNorm Layer + GPU Tensor
        (Value::LayerNormLayer(layer), Value::GPUTensor(input)) => {
            let output = layer.forward(input)
                .map_err(|e| format!("LayerNorm forward failed: {}", e))?;
            Ok(Value::GPUTensor(output))
        }
        // LayerNorm Layer + CPU Tensor
        (Value::LayerNormLayer(_), Value::AutogradTensor(input)) => {
            let mut input_gpu = crate::gpu_tensor::GPUTensor::from_tensor(input.clone());
            with_gpu_backend(|backend| {
                input_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move input to GPU: {}", e))
            })?;

            let layer = match &args[0] {
                Value::LayerNormLayer(l) => l,
                _ => unreachable!(),
            };

            let output_gpu = layer.forward(&input_gpu)
                .map_err(|e| format!("LayerNorm forward failed: {}", e))?;

            let mut output_clone = output_gpu.clone();
            with_gpu_backend(|backend| {
                output_clone.to_cpu(backend)
                    .map_err(|e| format!("Failed to move output to CPU: {}", e))
            })?;

            Ok(Value::AutogradTensor(output_clone.tensor))
        }
        // Dropout Layer + GPU Tensor
        (Value::DropoutLayer(layer), Value::GPUTensor(input)) => {
            let output = layer.forward(input)
                .map_err(|e| format!("Dropout forward failed: {}", e))?;
            Ok(Value::GPUTensor(output))
        }
        // Dropout Layer + CPU Tensor
        (Value::DropoutLayer(_), Value::AutogradTensor(input)) => {
            let mut input_gpu = crate::gpu_tensor::GPUTensor::from_tensor(input.clone());
            with_gpu_backend(|backend| {
                input_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move input to GPU: {}", e))
            })?;

            let layer = match &args[0] {
                Value::DropoutLayer(l) => l,
                _ => unreachable!(),
            };

            let output_gpu = layer.forward(&input_gpu)
                .map_err(|e| format!("Dropout forward failed: {}", e))?;

            let mut output_clone = output_gpu.clone();
            with_gpu_backend(|backend| {
                output_clone.to_cpu(backend)
                    .map_err(|e| format!("Failed to move output to CPU: {}", e))
            })?;

            Ok(Value::AutogradTensor(output_clone.tensor))
        }
        (Value::LinearLayer(_), _) | (Value::Conv2dLayer(_), _) | (Value::MaxPool2dLayer(_), _) | (Value::AvgPool2dLayer(_), _) | (Value::BatchNormLayer(_), _) | (Value::LayerNormLayer(_), _) | (Value::DropoutLayer(_), _) => {
            Err(format!("layer_forward() expects tensor input, got {}", args[1].type_name()))
        }
        _ => {
            Err(format!("layer_forward() expects a layer, got {}", args[0].type_name()))
        }
    }
}

// =============================================================================
// ACTIVATION FUNCTIONS - GPU ACCELERATED (v0.2.0)
// =============================================================================

/// tensor_relu(t: Tensor) -> Tensor
/// ReLU activation: max(0, x)
/// GPU-accelerated when input is on GPU
pub fn builtin_tensor_relu(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_relu() expects 1 argument: tensor_relu(tensor)".to_string());
    }

    match &args[0] {
        // GPU ReLU - element-wise max(0, x)
        Value::GPUTensor(input) => {
            // For now, implement on CPU side (GPU kernel exists but simpler for now)
            let result_data: Vec<f64> = input.tensor.data
                .iter()
                .map(|&x| x.max(0.0))
                .collect();

            let result_tensor = crate::autograd::Tensor::new(result_data, input.tensor.shape.clone());
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            // Move to GPU using global backend
            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        // CPU ReLU
        Value::AutogradTensor(input) => {
            let result_data: Vec<f64> = input.data
                .iter()
                .map(|&x| x.max(0.0))
                .collect();

            let result_tensor = crate::autograd::Tensor::new(result_data, input.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
        }
        _ => Err(format!("tensor_relu() expects a tensor, got {}", args[0].type_name())),
    }
}

/// tensor_sigmoid(t: Tensor) -> Tensor  
/// Sigmoid activation: 1 / (1 + exp(-x))
/// GPU-accelerated when input is on GPU
pub fn builtin_tensor_sigmoid(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_sigmoid() expects 1 argument: tensor_sigmoid(tensor)".to_string());
    }

    match &args[0] {
        Value::GPUTensor(input) => {
            let result_data: Vec<f64> = input.tensor.data
                .iter()
                .map(|&x| 1.0 / (1.0 + (-x).exp()))
                .collect();

            let result_tensor = crate::autograd::Tensor::new(result_data, input.tensor.shape.clone());
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        Value::AutogradTensor(input) => {
            let result_data: Vec<f64> = input.data
                .iter()
                .map(|&x| 1.0 / (1.0 + (-x).exp()))
                .collect();

            let result_tensor = crate::autograd::Tensor::new(result_data, input.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
        }
        _ => Err(format!("tensor_sigmoid() expects a tensor, got {}", args[0].type_name())),
    }
}

/// tensor_tanh(t: Tensor) -> Tensor
/// Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
/// GPU-accelerated when input is on GPU
pub fn builtin_tensor_tanh(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_tanh() expects 1 argument: tensor_tanh(tensor)".to_string());
    }

    match &args[0] {
        Value::GPUTensor(input) => {
            let result_data: Vec<f64> = input.tensor.data
                .iter()
                .map(|&x| x.tanh())
                .collect();

            let result_tensor = crate::autograd::Tensor::new(result_data, input.tensor.shape.clone());
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        Value::AutogradTensor(input) => {
            let result_data: Vec<f64> = input.data
                .iter()
                .map(|&x| x.tanh())
                .collect();

            let result_tensor = crate::autograd::Tensor::new(result_data, input.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
        }
        _ => Err(format!("tensor_tanh() expects a tensor, got {}", args[0].type_name())),
    }
}

/// tensor_gelu(t: Tensor) -> Tensor
/// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
/// GPU-accelerated when input is on GPU
pub fn builtin_tensor_gelu(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_gelu() expects 1 argument: tensor_gelu(tensor)".to_string());
    }

    const SQRT_2_OVER_PI: f64 = 0.7978845608028654; // sqrt(2/π)

    match &args[0] {
        Value::GPUTensor(input) => {
            let result_data: Vec<f64> = input.tensor.data
                .iter()
                .map(|&x| {
                    let inner = SQRT_2_OVER_PI * (x + 0.044715 * x.powi(3));
                    0.5 * x * (1.0 + inner.tanh())
                })
                .collect();

            let result_tensor = crate::autograd::Tensor::new(result_data, input.tensor.shape.clone());
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        Value::AutogradTensor(input) => {
            let result_data: Vec<f64> = input.data
                .iter()
                .map(|&x| {
                    let inner = SQRT_2_OVER_PI * (x + 0.044715 * x.powi(3));
                    0.5 * x * (1.0 + inner.tanh())
                })
                .collect();

            let result_tensor = crate::autograd::Tensor::new(result_data, input.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
        }
        _ => Err(format!("tensor_gelu() expects a tensor, got {}", args[0].type_name())),
    }
}

/// tensor_softmax(t: Tensor) -> Tensor
/// Softmax activation: exp(x_i) / sum(exp(x_j))
/// GPU-accelerated when input is on GPU
pub fn builtin_tensor_softmax(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_softmax() expects 1 argument: tensor_softmax(tensor)".to_string());
    }

    match &args[0] {
        Value::GPUTensor(input) => {
            // Numerically stable softmax: subtract max before exp
            let max_val = input.tensor.data.iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let exp_values: Vec<f64> = input.tensor.data
                .iter()
                .map(|&x| (x - max_val).exp())
                .collect();

            let sum: f64 = exp_values.iter().sum();

            let result_data: Vec<f64> = exp_values
                .iter()
                .map(|&x| x / sum)
                .collect();

            let result_tensor = crate::autograd::Tensor::new(result_data, input.tensor.shape.clone());
            let mut result_gpu = crate::gpu_tensor::GPUTensor::from_tensor(result_tensor);

            with_gpu_backend(|backend| {
                result_gpu.to_gpu(backend)
                    .map_err(|e| format!("Failed to move result to GPU: {}", e))
            })?;

            Ok(Value::GPUTensor(result_gpu))
        }
        Value::AutogradTensor(input) => {
            // Numerically stable softmax
            let max_val = input.data.iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let exp_values: Vec<f64> = input.data
                .iter()
                .map(|&x| (x - max_val).exp())
                .collect();

            let sum: f64 = exp_values.iter().sum();

            let result_data: Vec<f64> = exp_values
                .iter()
                .map(|&x| x / sum)
                .collect();

            let result_tensor = crate::autograd::Tensor::new(result_data, input.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
        }
        _ => Err(format!("tensor_softmax() expects a tensor, got {}", args[0].type_name())),
    }
}

/// tensor_backward(t: Tensor) -> Nil
/// Compute gradients for the given output tensor
/// This performs backpropagation through the computation graph
pub fn builtin_tensor_backward(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_backward() expects 1 argument: tensor_backward(output_tensor)".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(tensor) => {
            if !tensor.requires_grad {
                return Err("Cannot call backward() on tensor that doesn't require grad. Use tensor_with_grad() to create tensors that track gradients.".to_string());
            }

            // Perform backward pass using the global computation graph
            crate::autograd::backward_global(tensor.id)?;

            Ok(Value::Null)
        }
        Value::GPUTensor(_) => {
            Err("tensor_backward() is not yet supported for GPU tensors. Transfer to CPU first using tensor_to_cpu().".to_string())
        }
        _ => Err(format!("tensor_backward() expects a tensor, got {}", args[0].type_name())),
    }
}

/// tensor_with_grad(data: Array, shape: Array) -> Tensor
/// Create a new tensor that requires gradients (for autograd)
pub fn builtin_tensor_with_grad(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("tensor_with_grad() expects 2 arguments: tensor_with_grad(data, shape)".to_string());
    }

    // Extract data array
    let data = match &args[0] {
        Value::Array(arr) => {
            let mut result = Vec::new();
            for val in arr {
                match val {
                    Value::Float(n) => result.push(*n),
                    Value::Integer(n) => result.push(*n as f64),
                    _ => return Err("tensor_with_grad() data must be an array of numbers".to_string()),
                }
            }
            result
        }
        _ => return Err("tensor_with_grad() expects first argument to be an array".to_string()),
    };

    // Extract shape array
    let shape = match &args[1] {
        Value::Array(arr) => {
            let mut result = Vec::new();
            for val in arr {
                match val {
                    Value::Float(n) => {
                        if *n < 0.0 || n.fract() != 0.0 {
                            return Err("tensor_with_grad() shape must contain positive integers".to_string());
                        }
                        result.push(*n as usize);
                    }
                    Value::Integer(n) => {
                        if *n < 0 {
                            return Err("tensor_with_grad() shape must contain positive integers".to_string());
                        }
                        result.push(*n as usize);
                    }
                    _ => return Err("tensor_with_grad() shape must be an array of numbers".to_string()),
                }
            }
            result
        }
        _ => return Err("tensor_with_grad() expects second argument to be an array".to_string()),
    };

    // Verify data length matches shape
    let expected_size: usize = shape.iter().product();
    if data.len() != expected_size {
        return Err(format!(
            "tensor_with_grad() data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_size
        ));
    }

    // Create tensor with gradients enabled
    let tensor = crate::autograd::Tensor::with_grad(data, shape);

    // Add to global graph
    crate::autograd::add_to_global_graph(tensor.clone());

    Ok(Value::AutogradTensor(tensor))
}

/// tensor_grad(t: Tensor) -> Array
/// Get the gradient of a tensor (after calling backward)
pub fn builtin_tensor_grad(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_grad() expects 1 argument: tensor_grad(tensor)".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(tensor) => {
            // Try to get updated gradient from global graph
            if let Some(updated_tensor) = crate::autograd::get_from_global_graph(tensor.id) {
                match updated_tensor.grad {
                    Some(grad) => {
                        let grad_values: Vec<Value> = grad.iter().map(|&x| Value::Float(x)).collect();
                        Ok(Value::Array(grad_values))
                    }
                    None => Err("Tensor doesn't have gradients. Call backward() first or use tensor_with_grad().".to_string()),
                }
            } else {
                // Fallback to tensor's own gradient if not in graph
                match &tensor.grad {
                    Some(grad) => {
                        let grad_values: Vec<Value> = grad.iter().map(|&x| Value::Float(x)).collect();
                        Ok(Value::Array(grad_values))
                    }
                    None => Err("Tensor doesn't have gradients. Call backward() first or use tensor_with_grad().".to_string()),
                }
            }
        }
        _ => Err(format!("tensor_grad() expects a tensor, got {}", args[0].type_name())),
    }
}

/// tensor_zero_grad(t: Tensor) -> Null
/// Zero out the gradients of a tensor
pub fn builtin_tensor_zero_grad(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_zero_grad() expects 1 argument: tensor_zero_grad(tensor)".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(tensor) => {
            crate::autograd::with_global_graph_mut(|graph| {
                if let Some(node) = graph.get_node_mut(tensor.id) {
                    node.zero_grad();
                }
            });
            Ok(Value::Null)
        }
        _ => Err(format!("tensor_zero_grad() expects a tensor, got {}", args[0].type_name())),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// HELPER BUILTINS FOR TRAINING LOOPS
// ═══════════════════════════════════════════════════════════════════════

/// tensor_get_data(t: Tensor) -> Array
/// Extract the raw data from a tensor as an array
/// Useful for debugging and manual operations
pub fn builtin_tensor_get_data(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_get_data() expects 1 argument: tensor_get_data(tensor)".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(tensor) => {
            let data: Vec<Value> = tensor.data.iter()
                .map(|&x| Value::Float(x))
                .collect();
            Ok(Value::Array(data))
        }
        Value::GPUTensor(gpu_tensor) => {
            let data: Vec<Value> = gpu_tensor.tensor.data.iter()
                .map(|&x| Value::Float(x))
                .collect();
            Ok(Value::Array(data))
        }
        _ => Err(format!("tensor_get_data() expects a tensor, got {}", args[0].type_name())),
    }
}

/// tensor_update_inplace(param: Tensor, grad: Array, lr: Float) -> Tensor
/// Update a parameter tensor: new_param = param - lr * grad
/// Returns a new tensor with gradients enabled
/// This is the core operation for gradient descent
pub fn builtin_tensor_update_inplace(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("tensor_update_inplace() expects 3 arguments: tensor_update_inplace(param, grad, learning_rate)".to_string());
    }

    let learning_rate = match &args[2] {
        Value::Float(lr) => *lr,
        Value::Integer(lr) => *lr as f64,
        _ => return Err("tensor_update_inplace() expects learning_rate to be a number".to_string()),
    };

    match (&args[0], &args[1]) {
        (Value::AutogradTensor(param), Value::Array(grad_array)) => {
            // Convert grad array to Vec<f64>
            let grad: Vec<f64> = grad_array.iter()
                .map(|v| match v {
                    Value::Float(f) => Ok(*f),
                    Value::Integer(i) => Ok(*i as f64),
                    _ => Err("Gradient array must contain only numbers".to_string()),
                })
                .collect::<Result<Vec<f64>, String>>()?;

            // Check sizes match
            if grad.len() != param.data.len() {
                return Err(format!(
                    "tensor_update_inplace() gradient size {} doesn't match parameter size {}",
                    grad.len(),
                    param.data.len()
                ));
            }

            // Compute: new_data = param - lr * grad
            let updated_data: Vec<f64> = param.data.iter()
                .zip(grad.iter())
                .map(|(&p, &g)| p - learning_rate * g)
                .collect();

            // Create new tensor with gradients enabled
            let updated_tensor = crate::autograd::Tensor::with_grad(
                updated_data,
                param.shape.clone()
            );

            // Add to global graph
            crate::autograd::add_to_global_graph(updated_tensor.clone());

            Ok(Value::AutogradTensor(updated_tensor))
        }
        _ => Err("tensor_update_inplace() expects (tensor, array, float)".to_string()),
    }
}

/// tensor_from_data(data: Array, shape: Array, requires_grad: Bool) -> Tensor
/// Create a tensor from raw data with optional gradient tracking
/// Alias for tensor_with_grad when requires_grad=true, tensor when false
pub fn builtin_tensor_from_data(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("tensor_from_data() expects 3 arguments: tensor_from_data(data, shape, requires_grad)".to_string());
    }

    let requires_grad = match &args[2] {
        Value::Boolean(b) => *b,
        Value::Integer(i) => *i != 0,
        _ => return Err("tensor_from_data() expects requires_grad to be a boolean".to_string()),
    };

    if requires_grad {
        // Use tensor_with_grad
        builtin_tensor_with_grad(vec![args[0].clone(), args[1].clone()])
    } else {
        // Use regular tensor
        builtin_tensor(vec![args[0].clone()])
    }
}

/// reset_graph() -> ()
/// Clear the global computation graph
/// Useful for resetting state between training iterations
pub fn builtin_reset_graph(args: Vec<Value>) -> Result<Value, String> {
    if !args.is_empty() {
        return Err("reset_graph() expects no arguments".to_string());
    }

    crate::autograd::reset_global_graph();
    Ok(Value::Null)
}

/// tensor_sqrt(t: Tensor) -> Tensor
/// Compute element-wise square root
pub fn builtin_tensor_sqrt(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_sqrt() expects 1 argument: tensor_sqrt(tensor)".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(tensor) => {
            let result_data: Vec<f64> = tensor.data.iter()
                .map(|&x| x.sqrt())
                .collect();

            let result = crate::autograd::Tensor::new(result_data, tensor.shape.clone());
            Ok(Value::AutogradTensor(result))
        }
        Value::GPUTensor(gpu_tensor) => {
            let result_data: Vec<f64> = gpu_tensor.tensor.data.iter()
                .map(|&x| x.sqrt())
                .collect();

            let result = crate::autograd::Tensor::new(result_data, gpu_tensor.tensor.shape.clone());
            Ok(Value::AutogradTensor(result))
        }
        _ => Err(format!("tensor_sqrt() expects a tensor, got {}", args[0].type_name())),
    }
}

/// tensor_zeros_like(t: Tensor) -> Tensor
/// Create a tensor of zeros with the same shape as input
pub fn builtin_tensor_zeros_like(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_zeros_like() expects 1 argument: tensor_zeros_like(tensor)".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(tensor) => {
            let zeros = vec![0.0; tensor.data.len()];
            let result = crate::autograd::Tensor::new(zeros, tensor.shape.clone());
            Ok(Value::AutogradTensor(result))
        }
        Value::GPUTensor(gpu_tensor) => {
            let zeros = vec![0.0; gpu_tensor.tensor.data.len()];
            let result = crate::autograd::Tensor::new(zeros, gpu_tensor.tensor.shape.clone());
            Ok(Value::AutogradTensor(result))
        }
        Value::Array(arr) => {
            let zeros = vec![Value::Float(0.0); arr.len()];
            Ok(Value::Array(zeros))
        }
        _ => Err(format!("tensor_zeros_like() expects a tensor or array, got {}", args[0].type_name())),
    }
}

/// pow(base: Float, exp: Float) -> Float
/// Compute base^exp
pub fn builtin_pow(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("pow() expects 2 arguments: pow(base, exponent)".to_string());
    }

    let base = match &args[0] {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => return Err(format!("pow() expects numbers, got {}", args[0].type_name())),
    };

    let exp = match &args[1] {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => return Err(format!("pow() expects numbers, got {}", args[1].type_name())),
    };

    Ok(Value::Float(base.powf(exp)))
}

/// min(a, b) -> number
/// Return minimum of two numbers
pub fn builtin_min(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("min() expects 2 arguments: min(a, b)".to_string());
    }

    let a = match &args[0] {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => return Err(format!("min() expects numbers, got {}", args[0].type_name())),
    };

    let b = match &args[1] {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => return Err(format!("min() expects numbers, got {}", args[1].type_name())),
    };

    let result = if a < b { a } else { b };

    // Return same type as input (preserve Integer if both are Integer)
    match (&args[0], &args[1]) {
        (Value::Integer(_), Value::Integer(_)) => Ok(Value::Integer(result as i64)),
        _ => Ok(Value::Float(result)),
    }
}

/// max(a, b) -> number
/// Return maximum of two numbers
pub fn builtin_max(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("max() expects 2 arguments: max(a, b)".to_string());
    }

    let a = match &args[0] {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => return Err(format!("max() expects numbers, got {}", args[0].type_name())),
    };

    let b = match &args[1] {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => return Err(format!("max() expects numbers, got {}", args[1].type_name())),
    };

    let result = if a > b { a } else { b };

    // Return same type as input
    match (&args[0], &args[1]) {
        (Value::Integer(_), Value::Integer(_)) => Ok(Value::Integer(result as i64)),
        _ => Ok(Value::Float(result)),
    }
}

/// sqrt(x) -> float
/// Return square root of a number
pub fn builtin_sqrt(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("sqrt() expects 1 argument: sqrt(x)".to_string());
    }

    let x = match &args[0] {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => return Err(format!("sqrt() expects number, got {}", args[0].type_name())),
    };

    if x < 0.0 {
        return Err(format!("sqrt() of negative number: {}", x));
    }

    Ok(Value::Float(x.sqrt()))
}

/// abs(x) -> number
/// Return absolute value of a number
pub fn builtin_abs(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("abs() expects 1 argument: abs(x)".to_string());
    }

    match &args[0] {
        Value::Float(f) => Ok(Value::Float(f.abs())),
        Value::Integer(i) => Ok(Value::Integer(i.abs())),
        _ => Err(format!("abs() expects number, got {}", args[0].type_name())),
    }
}

// ============================================================================
// ATTENTION MECHANISMS - Week 1-2 of Backend Exposure Roadmap
// ============================================================================
//
// These builtins expose the attention mechanisms backend to enable:
// - Transformers (BERT, GPT-style models)
// - Self-attention for sequence modeling
// - Multi-head attention for parallel attention
// - Positional encoding for sequence position information
//
// Backend: src/attention/ (1,741 lines)
// Impact: HIGH - Unlocks state-of-the-art NLP and Vision Transformers
// ROI: ⭐⭐⭐⭐⭐

/// positional_encoding(seq_len, d_model) -> Tensor
/// Generate sinusoidal positional encodings for Transformer models
///
/// # Arguments
/// * `seq_len` - Sequence length (integer)
/// * `d_model` - Model dimension (must be even)
///
/// # Returns
/// * Tensor of shape [seq_len, d_model] containing positional encodings
///
/// # Example
/// ```charl
/// let pe = positional_encoding(10, 128)  // 10 positions, 128 dimensions
/// let embedded = tensor_add(embeddings, pe)  // Add to embeddings
/// ```
pub fn builtin_positional_encoding(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("positional_encoding() expects 2 arguments: positional_encoding(seq_len, d_model)".to_string());
    }

    let seq_len = match &args[0] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("positional_encoding() seq_len must be a number".to_string()),
    };

    let d_model = match &args[1] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("positional_encoding() d_model must be a number".to_string()),
    };

    if d_model % 2 != 0 {
        return Err(format!("d_model must be even, got {}", d_model));
    }

    // Use default max_len of 5000 (sufficient for most use cases)
    let pe = crate::attention::PositionalEncoding::new(d_model, 5000);
    let encoding = pe.forward(seq_len)
        .map_err(|e| format!("positional_encoding() failed: {}", e))?;

    let tensor = AutogradTensor::new(encoding, vec![seq_len, d_model]);
    Ok(Value::AutogradTensor(tensor))
}

/// attention_mask_causal(seq_len) -> Tensor
/// Create a causal (lower triangular) attention mask for autoregressive models
///
/// # Arguments
/// * `seq_len` - Sequence length
///
/// # Returns
/// * Tensor of shape [seq_len, seq_len] with 1.0 for allowed positions, 0.0 for masked
///
/// # Example
/// ```charl
/// let mask = attention_mask_causal(4)
/// // Result: [[1, 0, 0, 0],
/// //          [1, 1, 0, 0],
/// //          [1, 1, 1, 0],
/// //          [1, 1, 1, 1]]
/// ```
pub fn builtin_attention_mask_causal(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("attention_mask_causal() expects 1 argument: attention_mask_causal(seq_len)".to_string());
    }

    let seq_len = match &args[0] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("attention_mask_causal() seq_len must be a number".to_string()),
    };

    let mut mask = vec![0.0; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            mask[i * seq_len + j] = 1.0;
        }
    }

    let tensor = AutogradTensor::new(mask, vec![seq_len, seq_len]);
    Ok(Value::AutogradTensor(tensor))
}

/// attention_scaled(query, key, value, mask?) -> [output, weights]
/// Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
///
/// # Arguments
/// * `query` - Query tensor [batch, seq_len_q, d_k]
/// * `key` - Key tensor [batch, seq_len_k, d_k]
/// * `value` - Value tensor [batch, seq_len_k, d_v]
/// * `mask` - Optional attention mask [batch, seq_len_q, seq_len_k] (1=keep, 0=mask)
///
/// # Returns
/// * Array containing [output, attention_weights]
///   - output: [batch, seq_len_q, d_v]
///   - weights: [batch, seq_len_q, seq_len_k]
///
/// # Example
/// ```charl
/// let result = attention_scaled(q, k, v)
/// let output = result[0]
/// let weights = result[1]
/// ```
pub fn builtin_attention_scaled(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 || args.len() > 4 {
        return Err("attention_scaled() expects 3-4 arguments: attention_scaled(query, key, value, mask?)".to_string());
    }

    // Extract query, key, value tensors
    let (query_data, query_shape) = match &args[0] {
        Value::AutogradTensor(t) => (t.data.clone(), t.shape.clone()),
        _ => return Err("attention_scaled() query must be a tensor".to_string()),
    };

    let (key_data, key_shape) = match &args[1] {
        Value::AutogradTensor(t) => (t.data.clone(), t.shape.clone()),
        _ => return Err("attention_scaled() key must be a tensor".to_string()),
    };

    let (value_data, value_shape) = match &args[2] {
        Value::AutogradTensor(t) => (t.data.clone(), t.shape.clone()),
        _ => return Err("attention_scaled() value must be a tensor".to_string()),
    };

    // Validate 3D shapes
    if query_shape.len() != 3 || key_shape.len() != 3 || value_shape.len() != 3 {
        return Err("attention_scaled() requires 3D tensors [batch, seq_len, dim]".to_string());
    }

    let batch_q = query_shape[0];
    let seq_len_q = query_shape[1];
    let d_k = query_shape[2];

    let batch_k = key_shape[0];
    let seq_len_k = key_shape[1];

    let batch_v = value_shape[0];
    let seq_len_v = value_shape[1];
    let d_v = value_shape[2];

    // Validate dimensions
    if batch_q != batch_k || batch_q != batch_v {
        return Err("attention_scaled() batch sizes must match".to_string());
    }

    if seq_len_k != seq_len_v {
        return Err("attention_scaled() key and value sequence lengths must match".to_string());
    }

    if key_shape[2] != d_k {
        return Err(format!("attention_scaled() key dimension ({}) must match query dimension ({})", key_shape[2], d_k));
    }

    // Extract optional mask
    let mask = if args.len() == 4 {
        match &args[3] {
            Value::AutogradTensor(t) => Some(t.data.clone()),
            Value::Null => None,
            _ => return Err("attention_scaled() mask must be a tensor or null".to_string()),
        }
    } else {
        None
    };

    // Create attention mechanism
    let attention = crate::attention::ScaledDotProductAttention::new(d_k, 0.0);

    // Forward pass
    let (output, weights) = attention.forward(
        &query_data,
        &key_data,
        &value_data,
        (batch_q, seq_len_q, d_k),
        (batch_k, seq_len_k, d_k),
        (batch_v, seq_len_v, d_v),
        mask.as_deref(),
    ).map_err(|e| format!("attention_scaled() failed: {}", e))?;

    // Return as array [output, weights]
    let output_tensor = AutogradTensor::new(output, vec![batch_q, seq_len_q, d_v]);
    let weights_tensor = AutogradTensor::new(weights, vec![batch_q, seq_len_q, seq_len_k]);

    Ok(Value::Array(vec![
        Value::AutogradTensor(output_tensor),
        Value::AutogradTensor(weights_tensor),
    ]))
}

/// attention_self(input, d_model, d_k, d_v, mask?) -> [output, weights]
/// Self-attention where Q, K, V come from the same input
///
/// # Arguments
/// * `input` - Input tensor [batch, seq_len, d_model]
/// * `d_model` - Model dimension
/// * `d_k` - Key/Query dimension
/// * `d_v` - Value dimension
/// * `mask` - Optional attention mask [batch, seq_len, seq_len]
///
/// # Returns
/// * Array containing [output, attention_weights]
///   - output: [batch, seq_len, d_model]
///   - weights: [batch, seq_len, seq_len]
///
/// # Example
/// ```charl
/// let result = attention_self(x, 128, 64, 64)
/// let output = result[0]
/// let weights = result[1]
/// ```
pub fn builtin_attention_self(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 || args.len() > 5 {
        return Err("attention_self() expects 4-5 arguments: attention_self(input, d_model, d_k, d_v, mask?)".to_string());
    }

    // Extract input tensor
    let (input_data, input_shape) = match &args[0] {
        Value::AutogradTensor(t) => (t.data.clone(), t.shape.clone()),
        _ => return Err("attention_self() input must be a tensor".to_string()),
    };

    // Validate 3D shape
    if input_shape.len() != 3 {
        return Err("attention_self() requires 3D tensor [batch, seq_len, d_model]".to_string());
    }

    let batch = input_shape[0];
    let seq_len = input_shape[1];
    let d_model_input = input_shape[2];

    // Extract dimensions
    let d_model = match &args[1] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("attention_self() d_model must be a number".to_string()),
    };

    let d_k = match &args[2] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("attention_self() d_k must be a number".to_string()),
    };

    let d_v = match &args[3] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("attention_self() d_v must be a number".to_string()),
    };

    if d_model_input != d_model {
        return Err(format!("attention_self() input dimension ({}) must match d_model ({})", d_model_input, d_model));
    }

    // Extract optional mask
    let mask = if args.len() == 5 {
        match &args[4] {
            Value::AutogradTensor(t) => Some(t.data.clone()),
            Value::Null => None,
            _ => return Err("attention_self() mask must be a tensor or null".to_string()),
        }
    } else {
        None
    };

    // Create self-attention mechanism
    let self_attention = crate::attention::SelfAttention::new(d_model, d_k, d_v, 0.0);

    // Forward pass
    let (output, weights) = self_attention.forward(
        &input_data,
        (batch, seq_len, d_model),
        mask.as_deref(),
    ).map_err(|e| format!("attention_self() failed: {}", e))?;

    // Return as array [output, weights]
    let output_tensor = AutogradTensor::new(output, vec![batch, seq_len, d_model]);
    let weights_tensor = AutogradTensor::new(weights, vec![batch, seq_len, seq_len]);

    Ok(Value::Array(vec![
        Value::AutogradTensor(output_tensor),
        Value::AutogradTensor(weights_tensor),
    ]))
}

/// attention_multi_head(query, key, value, d_model, num_heads, mask?) -> [output, weights]
/// Multi-head attention - parallel attention across multiple representation subspaces
///
/// # Arguments
/// * `query` - Query tensor [batch, seq_len_q, d_model]
/// * `key` - Key tensor [batch, seq_len_k, d_model]
/// * `value` - Value tensor [batch, seq_len_k, d_model]
/// * `d_model` - Model dimension (must be divisible by num_heads)
/// * `num_heads` - Number of attention heads
/// * `mask` - Optional attention mask [batch, seq_len_q, seq_len_k]
///
/// # Returns
/// * Array containing [output, attention_weights]
///   - output: [batch, seq_len_q, d_model]
///   - weights: [batch, num_heads, seq_len_q, seq_len_k]
///
/// # Example
/// ```charl
/// let result = attention_multi_head(q, k, v, 512, 8)  // 8-head attention
/// let output = result[0]
/// let weights = result[1]
/// ```
pub fn builtin_attention_multi_head(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 5 || args.len() > 6 {
        return Err("attention_multi_head() expects 5-6 arguments: attention_multi_head(query, key, value, d_model, num_heads, mask?)".to_string());
    }

    // Extract query, key, value tensors
    let (query_data, query_shape) = match &args[0] {
        Value::AutogradTensor(t) => (t.data.clone(), t.shape.clone()),
        _ => return Err("attention_multi_head() query must be a tensor".to_string()),
    };

    let (key_data, key_shape) = match &args[1] {
        Value::AutogradTensor(t) => (t.data.clone(), t.shape.clone()),
        _ => return Err("attention_multi_head() key must be a tensor".to_string()),
    };

    let (value_data, value_shape) = match &args[2] {
        Value::AutogradTensor(t) => (t.data.clone(), t.shape.clone()),
        _ => return Err("attention_multi_head() value must be a tensor".to_string()),
    };

    // Validate 3D shapes
    if query_shape.len() != 3 || key_shape.len() != 3 || value_shape.len() != 3 {
        return Err("attention_multi_head() requires 3D tensors [batch, seq_len, d_model]".to_string());
    }

    // Extract dimensions
    let d_model = match &args[3] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("attention_multi_head() d_model must be a number".to_string()),
    };

    let num_heads = match &args[4] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("attention_multi_head() num_heads must be a number".to_string()),
    };

    let batch_q = query_shape[0];
    let seq_len_q = query_shape[1];

    let batch_k = key_shape[0];
    let seq_len_k = key_shape[1];

    let batch_v = value_shape[0];
    let seq_len_v = value_shape[1];

    // Validate dimensions
    if batch_q != batch_k || batch_q != batch_v {
        return Err("attention_multi_head() batch sizes must match".to_string());
    }

    if seq_len_k != seq_len_v {
        return Err("attention_multi_head() key and value sequence lengths must match".to_string());
    }

    if query_shape[2] != d_model || key_shape[2] != d_model || value_shape[2] != d_model {
        return Err(format!("attention_multi_head() all tensors must have d_model={}", d_model));
    }

    // Extract optional mask
    let mask = if args.len() == 6 {
        match &args[5] {
            Value::AutogradTensor(t) => Some(t.data.clone()),
            Value::Null => None,
            _ => return Err("attention_multi_head() mask must be a tensor or null".to_string()),
        }
    } else {
        None
    };

    // Create multi-head attention mechanism
    let mha = crate::attention::MultiHeadAttention::new(d_model, num_heads, 0.0)
        .map_err(|e| format!("attention_multi_head() failed to create: {}", e))?;

    // Forward pass
    let (output, weights) = mha.forward(
        &query_data,
        &key_data,
        &value_data,
        (batch_q, seq_len_q, d_model),
        (batch_k, seq_len_k, d_model),
        (batch_v, seq_len_v, d_model),
        mask.as_deref(),
    ).map_err(|e| format!("attention_multi_head() failed: {}", e))?;

    // Return as array [output, weights]
    let output_tensor = AutogradTensor::new(output, vec![batch_q, seq_len_q, d_model]);
    let weights_tensor = AutogradTensor::new(weights, vec![batch_q, num_heads, seq_len_q, seq_len_k]);

    Ok(Value::Array(vec![
        Value::AutogradTensor(output_tensor),
        Value::AutogradTensor(weights_tensor),
    ]))
}

// ============================================================================
// CONVOLUTIONAL NEURAL NETWORKS - Week 3-4 of Backend Exposure Roadmap
// ============================================================================
//
// These builtins expose CNN operations for Computer Vision:
// - Conv2D for feature extraction
// - MaxPool2D / AvgPool2D for downsampling
// - BatchNorm for training stability
//
// Backend: src/nn/gpu_layers.rs
// Impact: HIGH - Unlocks CNNs, ResNet, VGG, Vision Transformers
// ROI: ⭐⭐⭐⭐⭐

/// nn_conv2d(input, in_channels, out_channels, kernel_size, stride, padding) -> Tensor
/// 2D Convolution layer for feature extraction
///
/// # Arguments
/// * `input` - Input tensor [batch, in_channels, height, width]
/// * `in_channels` - Number of input channels
/// * `out_channels` - Number of output channels (number of filters)
/// * `kernel_size` - Size of convolution kernel (e.g., 3 for 3x3)
/// * `stride` - Stride of convolution (default 1)
/// * `padding` - Padding around input (default 0)
///
/// # Returns
/// * Output tensor [batch, out_channels, out_height, out_width]
///
/// # Example
/// ```charl
/// // Input: [1, 3, 32, 32] (1 image, 3 channels RGB, 32x32)
/// let x = tensor_randn([1, 3, 32, 32])
///
/// // Conv: 3 → 64 channels, 3x3 kernel, stride=1, padding=1
/// let features = nn_conv2d(x, 3, 64, 3, 1, 1)
/// // Output: [1, 64, 32, 32]
/// ```
pub fn builtin_nn_conv2d(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 6 {
        return Err("nn_conv2d() expects 6 arguments: nn_conv2d(input, in_channels, out_channels, kernel_size, stride, padding)".to_string());
    }

    // Extract input tensor and convert to GPU
    let input_cpu = match &args[0] {
        Value::AutogradTensor(t) => t,
        _ => return Err("nn_conv2d() input must be a tensor".to_string()),
    };

    // Validate 4D input
    if input_cpu.shape.len() != 4 {
        return Err(format!("nn_conv2d() requires 4D input [batch, channels, height, width], got {:?}", input_cpu.shape));
    }

    // Extract parameters
    let in_channels = match &args[1] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("nn_conv2d() in_channels must be a number".to_string()),
    };

    let out_channels = match &args[2] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("nn_conv2d() out_channels must be a number".to_string()),
    };

    let kernel_size = match &args[3] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("nn_conv2d() kernel_size must be a number".to_string()),
    };

    let stride = match &args[4] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("nn_conv2d() stride must be a number".to_string()),
    };

    let padding = match &args[5] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("nn_conv2d() padding must be a number".to_string()),
    };

    // Validate channels match
    if input_cpu.shape[1] != in_channels {
        return Err(format!("nn_conv2d() input channels mismatch: tensor has {}, specified {}", input_cpu.shape[1], in_channels));
    }

    // Convert input to GPU tensor
    let input_gpu = crate::gpu_tensor::GPUTensor::new(
        input_cpu.data.clone(),
        input_cpu.shape.clone(),
    );

    // Create Conv2d layer
    let conv = crate::nn::gpu_layers::Conv2d::with_params(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        crate::nn::gpu_layers::Initializer::He,
    );

    // Forward pass
    let output_gpu = conv.forward(&input_gpu)
        .map_err(|e| format!("nn_conv2d() failed: {}", e))?;

    // Convert back to CPU tensor
    let output = AutogradTensor::new(
        output_gpu.tensor.data.clone(),
        output_gpu.tensor.shape.clone(),
    );

    Ok(Value::AutogradTensor(output))
}

/// nn_maxpool2d(input, kernel_size, stride) -> Tensor
/// Max pooling for downsampling (takes maximum in each window)
///
/// # Arguments
/// * `input` - Input tensor [batch, channels, height, width]
/// * `kernel_size` - Size of pooling window (e.g., 2 for 2x2)
/// * `stride` - Stride of pooling (if 0, defaults to kernel_size)
///
/// # Returns
/// * Output tensor [batch, channels, out_height, out_width]
///
/// # Example
/// ```charl
/// // Input: [1, 64, 32, 32]
/// let x = tensor_randn([1, 64, 32, 32])
///
/// // MaxPool: 2x2 window, stride=2 (halves dimensions)
/// let pooled = nn_maxpool2d(x, 2, 2)
/// // Output: [1, 64, 16, 16]
/// ```
pub fn builtin_nn_maxpool2d(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("nn_maxpool2d() expects 3 arguments: nn_maxpool2d(input, kernel_size, stride)".to_string());
    }

    // Extract input tensor
    let input_cpu = match &args[0] {
        Value::AutogradTensor(t) => t,
        _ => return Err("nn_maxpool2d() input must be a tensor".to_string()),
    };

    // Validate 4D input
    if input_cpu.shape.len() != 4 {
        return Err(format!("nn_maxpool2d() requires 4D input [batch, channels, height, width], got {:?}", input_cpu.shape));
    }

    // Extract parameters
    let kernel_size = match &args[1] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("nn_maxpool2d() kernel_size must be a number".to_string()),
    };

    let stride = match &args[2] {
        Value::Integer(i) => {
            if *i == 0 {
                kernel_size // Default: stride = kernel_size
            } else {
                *i as usize
            }
        }
        Value::Float(f) => {
            if *f == 0.0 {
                kernel_size
            } else {
                *f as usize
            }
        }
        _ => return Err("nn_maxpool2d() stride must be a number".to_string()),
    };

    // Convert to GPU tensor
    let input_gpu = crate::gpu_tensor::GPUTensor::new(
        input_cpu.data.clone(),
        input_cpu.shape.clone(),
    );

    // Create MaxPool layer
    let pool = crate::nn::gpu_layers::MaxPool2d::with_stride(kernel_size, stride);

    // Forward pass
    let output_gpu = pool.forward(&input_gpu)
        .map_err(|e| format!("nn_maxpool2d() failed: {}", e))?;

    // Convert back to CPU tensor
    let output = AutogradTensor::new(
        output_gpu.tensor.data.clone(),
        output_gpu.tensor.shape.clone(),
    );

    Ok(Value::AutogradTensor(output))
}

/// nn_avgpool2d(input, kernel_size, stride) -> Tensor
/// Average pooling for downsampling (takes average in each window)
///
/// # Arguments
/// * `input` - Input tensor [batch, channels, height, width]
/// * `kernel_size` - Size of pooling window
/// * `stride` - Stride of pooling (if 0, defaults to kernel_size)
///
/// # Returns
/// * Output tensor [batch, channels, out_height, out_width]
///
/// # Example
/// ```charl
/// let pooled = nn_avgpool2d(x, 2, 2)
/// ```
pub fn builtin_nn_avgpool2d(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("nn_avgpool2d() expects 3 arguments: nn_avgpool2d(input, kernel_size, stride)".to_string());
    }

    // Extract input tensor
    let input_cpu = match &args[0] {
        Value::AutogradTensor(t) => t,
        _ => return Err("nn_avgpool2d() input must be a tensor".to_string()),
    };

    // Validate 4D input
    if input_cpu.shape.len() != 4 {
        return Err(format!("nn_avgpool2d() requires 4D input [batch, channels, height, width], got {:?}", input_cpu.shape));
    }

    // Extract parameters
    let kernel_size = match &args[1] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("nn_avgpool2d() kernel_size must be a number".to_string()),
    };

    let stride = match &args[2] {
        Value::Integer(i) => {
            if *i == 0 {
                kernel_size
            } else {
                *i as usize
            }
        }
        Value::Float(f) => {
            if *f == 0.0 {
                kernel_size
            } else {
                *f as usize
            }
        }
        _ => return Err("nn_avgpool2d() stride must be a number".to_string()),
    };

    // Convert to GPU tensor
    let input_gpu = crate::gpu_tensor::GPUTensor::new(
        input_cpu.data.clone(),
        input_cpu.shape.clone(),
    );

    // Create AvgPool layer
    let pool = crate::nn::gpu_layers::AvgPool2d::with_stride(kernel_size, stride);

    // Forward pass
    let output_gpu = pool.forward(&input_gpu)
        .map_err(|e| format!("nn_avgpool2d() failed: {}", e))?;

    // Convert back to CPU tensor
    let output = AutogradTensor::new(
        output_gpu.tensor.data.clone(),
        output_gpu.tensor.shape.clone(),
    );

    Ok(Value::AutogradTensor(output))
}

/// nn_batchnorm(input, num_features) -> Tensor
/// Batch normalization for training stability
/// Normalizes: y = (x - mean) / sqrt(var + eps) * gamma + beta
///
/// # Arguments
/// * `input` - Input tensor [batch, features] or [batch, channels, H, W]
/// * `num_features` - Number of features/channels to normalize
///
/// # Returns
/// * Normalized tensor (same shape as input)
///
/// # Example
/// ```charl
/// // For 2D: [batch=32, features=128]
/// let x = tensor_randn([32, 128])
/// let normalized = nn_batchnorm(x, 128)
///
/// // For 4D: [batch=32, channels=64, H=16, W=16]
/// let x = tensor_randn([32, 64, 16, 16])
/// let normalized = nn_batchnorm(x, 64)
/// ```
pub fn builtin_nn_batchnorm(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("nn_batchnorm() expects 2 arguments: nn_batchnorm(input, num_features)".to_string());
    }

    // Extract input tensor
    let input_cpu = match &args[0] {
        Value::AutogradTensor(t) => t,
        _ => return Err("nn_batchnorm() input must be a tensor".to_string()),
    };

    // Validate input shape (2D or 4D)
    if input_cpu.shape.len() != 2 && input_cpu.shape.len() != 4 {
        return Err(format!("nn_batchnorm() requires 2D [batch, features] or 4D [batch, channels, H, W], got {:?}", input_cpu.shape));
    }

    // Extract num_features
    let num_features = match &args[1] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("nn_batchnorm() num_features must be a number".to_string()),
    };

    // Validate features match
    let expected_features = if input_cpu.shape.len() == 2 {
        input_cpu.shape[1]
    } else {
        input_cpu.shape[1] // channels dimension
    };

    if expected_features != num_features {
        return Err(format!("nn_batchnorm() feature mismatch: tensor has {}, specified {}", expected_features, num_features));
    }

    // Convert to GPU tensor
    let input_gpu = crate::gpu_tensor::GPUTensor::new(
        input_cpu.data.clone(),
        input_cpu.shape.clone(),
    );

    // Create BatchNorm layer
    let bn = crate::nn::gpu_layers::BatchNorm::new(num_features);

    // Forward pass
    let output_gpu = bn.forward(&input_gpu)
        .map_err(|e| format!("nn_batchnorm() failed: {}", e))?;

    // Convert back to CPU tensor
    let output = AutogradTensor::new(
        output_gpu.tensor.data.clone(),
        output_gpu.tensor.shape.clone(),
    );

    Ok(Value::AutogradTensor(output))
}
// ============================================================================
// OPTIMIZERS & SCHEDULERS - Week 5-6 of Backend Exposure Roadmap
// ============================================================================
//
// These builtins expose the optimization algorithms backend to enable:
// - Training neural networks with SGD, Adam, RMSprop
// - Learning rate scheduling (StepLR, ExponentialLR, Cosine)
// - Modern optimization techniques
//
// Backend: src/optim/ (785 lines)
// Impact: HIGH - Enables training
// ROI: ⭐⭐⭐⭐⭐


// ============================================================================
// SGD (Stochastic Gradient Descent)
// ============================================================================

/// sgd_create(lr, momentum?, weight_decay?) -> SGDOptimizer
/// Create SGD optimizer with optional momentum and weight decay
///
/// # Arguments
/// * `lr` - Learning rate (float)
/// * `momentum` - Momentum coefficient 0-1, default 0.0 (optional)
/// * `weight_decay` - L2 regularization strength, default 0.0 (optional)
///
/// # Returns
/// * SGDOptimizer object
///
/// # Example
/// ```charl
/// let opt = sgd_create(0.01)
/// let opt_momentum = sgd_create(0.01, 0.9)
/// let opt_wd = sgd_create(0.01, 0.9, 0.0001)
/// ```
pub fn builtin_sgd_create(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() || args.len() > 3 {
        return Err("sgd_create() expects 1-3 arguments: sgd_create(lr, momentum?, weight_decay?)".to_string());
    }

    let lr = args[0].to_float()?;

    let mut sgd = SGD::new(lr);

    if args.len() >= 2 {
        let momentum = args[1].to_float()?;
        sgd = sgd.with_momentum(momentum);
    }

    if args.len() == 3 {
        let weight_decay = args[2].to_float()?;
        sgd = sgd.with_weight_decay(weight_decay);
    }

    Ok(Value::SGDOptimizer(Box::new(sgd)))
}

/// sgd_step(optimizer, params) -> null
/// Perform one SGD optimization step
/// Modifies params in-place using their computed gradients
///
/// # Arguments
/// * `optimizer` - SGDOptimizer created with sgd_create()
/// * `params` - Array of tensors with gradients (from tensor_backward())
///
/// # Example
/// ```charl
/// let opt = sgd_create(0.01)
/// let loss = forward_pass(params)
/// tensor_backward(loss)
/// sgd_step(opt, [w1, w2, b1, b2])
/// ```
pub fn builtin_sgd_step(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("sgd_step() expects 2 arguments: sgd_step(optimizer, params)".to_string());
    }

    // Extract optimizer
    let opt = match &args[0] {
        Value::SGDOptimizer(o) => o,
        _ => return Err("sgd_step() first argument must be SGDOptimizer".to_string()),
    };

    // Extract params array
    let params_arr = match &args[1] {
        Value::Array(arr) => arr,
        _ => return Err("sgd_step() second argument must be array of tensors".to_string()),
    };

    // Update each parameter using SGD rule: param = param - lr * grad
    let lr = opt.lr;
    let mut updated_values: Vec<Value> = Vec::new();

    for p in params_arr {
        match p {
            Value::AutogradTensor(param) => {
                if !param.requires_grad {
                    return Err("sgd_step(): all parameters must have requires_grad=true".to_string());
                }

                // Get gradient from global graph (like tensor_grad() does)
                let grad = if let Some(updated_tensor) = crate::autograd::get_from_global_graph(param.id) {
                    match updated_tensor.grad {
                        Some(g) => g,
                        None => return Err("sgd_step(): parameter has no gradient. Did you call tensor_backward()?".to_string()),
                    }
                } else {
                    // Fallback to tensor's own gradient
                    match &param.grad {
                        Some(g) => g.clone(),
                        None => return Err("sgd_step(): parameter has no gradient. Did you call tensor_backward()?".to_string()),
                    }
                };

                // Compute: new_data = param - lr * grad
                let updated_data: Vec<f64> = param.data.iter()
                    .zip(grad.iter())
                    .map(|(&p, &g)| p - lr * g)
                    .collect();

                // Create new tensor with gradients enabled
                let updated_tensor = crate::autograd::Tensor::with_grad(
                    updated_data,
                    param.shape.clone()
                );

                // Add to global graph so it can be used in next forward pass
                crate::autograd::add_to_global_graph(updated_tensor.clone());

                updated_values.push(Value::AutogradTensor(updated_tensor));
            }
            _ => return Err("sgd_step() params must be array of AutogradTensor".to_string()),
        }
    }

    Ok(Value::Array(updated_values))
}

// ============================================================================
// Adam (Adaptive Moment Estimation)
// ============================================================================

/// adam_create(lr, beta1?, beta2?, eps?) -> AdamOptimizer
/// Create Adam optimizer with default or custom hyperparameters
///
/// # Arguments
/// * `lr` - Learning rate (float)
/// * `beta1` - First moment decay, default 0.9 (optional)
/// * `beta2` - Second moment decay, default 0.999 (optional)
/// * `eps` - Numerical stability, default 1e-8 (optional)
///
/// # Returns
/// * AdamOptimizer object
///
/// # Example
/// ```charl
/// let opt = adam_create(0.001)
/// let opt_custom = adam_create(0.001, 0.9, 0.999, 1e-8)
/// ```
pub fn builtin_adam_create(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() || args.len() > 4 {
        return Err("adam_create() expects 1-4 arguments: adam_create(lr, beta1?, beta2?, eps?)".to_string());
    }

    let lr = args[0].to_float()?;

    let mut adam = Adam::new(lr);

    if args.len() >= 3 {
        let beta1 = args[1].to_float()?;
        let beta2 = args[2].to_float()?;
        adam = adam.with_betas(beta1, beta2);
    }

    if args.len() == 4 {
        let weight_decay = args[3].to_float()?;
        adam = adam.with_weight_decay(weight_decay);
    }

    Ok(Value::AdamOptimizer(Box::new(adam)))
}

/// adam_step(optimizer, params) -> null
/// Perform one Adam optimization step
///
/// # Arguments
/// * `optimizer` - AdamOptimizer created with adam_create()
/// * `params` - Array of tensors with gradients
///
/// # Example
/// ```charl
/// let opt = adam_create(0.001)
/// let loss = forward_pass(params)
/// tensor_backward(loss)
/// adam_step(opt, [w1, w2, b1, b2])
/// ```
pub fn builtin_adam_step(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("adam_step() expects 2 arguments: adam_step(optimizer, params)".to_string());
    }

    let opt = match &args[0] {
        Value::AdamOptimizer(o) => o,
        _ => return Err("adam_step() first argument must be AdamOptimizer".to_string()),
    };

    let params_arr = match &args[1] {
        Value::Array(arr) => arr,
        _ => return Err("adam_step() second argument must be array of tensors".to_string()),
    };

    let mut tensors: Vec<AutogradTensor> = Vec::new();
    for p in params_arr {
        match p {
            Value::AutogradTensor(t) => tensors.push(t.clone()),
            _ => return Err("adam_step() params must be array of AutogradTensor".to_string()),
        }
    }

    let mut tensor_refs: Vec<&mut AutogradTensor> = tensors.iter_mut().collect();
    let mut optimizer = (**opt).clone();
    optimizer.step(&mut tensor_refs);

    // Convert updated tensors back to Value array
    let updated_values: Vec<Value> = tensors.into_iter()
        .map(Value::AutogradTensor)
        .collect();

    Ok(Value::Array(updated_values))
}

// ============================================================================
// RMSprop (Root Mean Square Propagation)
// ============================================================================

/// rmsprop_create(lr, alpha?, eps?) -> RMSpropOptimizer
/// Create RMSprop optimizer
///
/// # Arguments
/// * `lr` - Learning rate (float)
/// * `alpha` - Smoothing constant, default 0.99 (optional)
/// * `eps` - Numerical stability, default 1e-8 (optional)
///
/// # Returns
/// * RMSpropOptimizer object
///
/// # Example
/// ```charl
/// let opt = rmsprop_create(0.001)
/// let opt_custom = rmsprop_create(0.001, 0.99, 1e-8)
/// ```
pub fn builtin_rmsprop_create(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() || args.len() > 3 {
        return Err("rmsprop_create() expects 1-3 arguments: rmsprop_create(lr, alpha?, eps?)".to_string());
    }

    let lr = args[0].to_float()?;

    let mut rmsprop = RMSprop::new(lr);

    if args.len() >= 2 {
        let alpha = args[1].to_float()?;
        rmsprop = rmsprop.with_alpha(alpha);
    }

    if args.len() == 3 {
        let weight_decay = args[2].to_float()?;
        rmsprop = rmsprop.with_weight_decay(weight_decay);
    }

    Ok(Value::RMSpropOptimizer(Box::new(rmsprop)))
}

/// rmsprop_step(optimizer, params) -> null
/// Perform one RMSprop optimization step
///
/// # Arguments
/// * `optimizer` - RMSpropOptimizer created with rmsprop_create()
/// * `params` - Array of tensors with gradients
///
/// # Example
/// ```charl
/// let opt = rmsprop_create(0.001)
/// let loss = forward_pass(params)
/// tensor_backward(loss)
/// rmsprop_step(opt, [w1, w2])
/// ```
pub fn builtin_rmsprop_step(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("rmsprop_step() expects 2 arguments: rmsprop_step(optimizer, params)".to_string());
    }

    let opt = match &args[0] {
        Value::RMSpropOptimizer(o) => o,
        _ => return Err("rmsprop_step() first argument must be RMSpropOptimizer".to_string()),
    };

    let params_arr = match &args[1] {
        Value::Array(arr) => arr,
        _ => return Err("rmsprop_step() second argument must be array of tensors".to_string()),
    };

    let mut tensors: Vec<AutogradTensor> = Vec::new();
    for p in params_arr {
        match p {
            Value::AutogradTensor(t) => tensors.push(t.clone()),
            _ => return Err("rmsprop_step() params must be array of AutogradTensor".to_string()),
        }
    }

    let mut tensor_refs: Vec<&mut AutogradTensor> = tensors.iter_mut().collect();
    let mut optimizer = (**opt).clone();
    optimizer.step(&mut tensor_refs);

    // Convert updated tensors back to Value array
    let updated_values: Vec<Value> = tensors.into_iter()
        .map(Value::AutogradTensor)
        .collect();

    Ok(Value::Array(updated_values))
}

// ============================================================================
// LEARNING RATE SCHEDULERS
// ============================================================================
// Pure functions - no state, just compute LR based on epoch

/// step_lr(base_lr, epoch, step_size, gamma) -> float
/// Decay learning rate by gamma every step_size epochs
///
/// Formula: lr = base_lr * gamma^(epoch // step_size)
///
/// # Arguments
/// * `base_lr` - Initial learning rate
/// * `epoch` - Current epoch number (integer)
/// * `step_size` - Period of LR decay (integer)
/// * `gamma` - Multiplicative factor (float, typically 0.1)
///
/// # Returns
/// * Learning rate for current epoch
///
/// # Example
/// ```charl
/// for epoch in 0..100 {
///     let lr = step_lr(0.1, epoch, 30, 0.1)
///     // Epoch 0-29: lr=0.1, 30-59: lr=0.01, 60-89: lr=0.001
/// }
/// ```
pub fn builtin_step_lr(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("step_lr() expects 4 arguments: step_lr(base_lr, epoch, step_size, gamma)".to_string());
    }

    let base_lr = args[0].to_float()?;
    let epoch = args[1].to_integer()? as usize;
    let step_size = args[2].to_integer()? as usize;
    let gamma = args[3].to_float()?;

    if step_size == 0 {
        return Err("step_lr() step_size must be > 0".to_string());
    }

    let num_decays = epoch / step_size;
    let lr = base_lr * gamma.powi(num_decays as i32);

    Ok(Value::Float(lr))
}

/// exponential_lr(base_lr, epoch, gamma) -> float
/// Exponentially decay learning rate every epoch
///
/// Formula: lr = base_lr * gamma^epoch
///
/// # Arguments
/// * `base_lr` - Initial learning rate
/// * `epoch` - Current epoch number
/// * `gamma` - Multiplicative factor (typically 0.95-0.99)
///
/// # Returns
/// * Learning rate for current epoch
///
/// # Example
/// ```charl
/// let lr = exponential_lr(0.1, 10, 0.95)
/// // lr = 0.1 * 0.95^10 = 0.0599
/// ```
pub fn builtin_exponential_lr(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("exponential_lr() expects 3 arguments: exponential_lr(base_lr, epoch, gamma)".to_string());
    }

    let base_lr = args[0].to_float()?;
    let epoch = args[1].to_integer()? as usize;
    let gamma = args[2].to_float()?;

    let lr = base_lr * gamma.powi(epoch as i32);

    Ok(Value::Float(lr))
}

/// cosine_annealing_lr(lr_max, lr_min, epoch, T_max) -> float
/// Cosine annealing learning rate schedule
///
/// Formula: lr = lr_min + (lr_max - lr_min) * (1 + cos(π * epoch / T_max)) / 2
///
/// # Arguments
/// * `lr_max` - Maximum learning rate
/// * `lr_min` - Minimum learning rate
/// * `epoch` - Current epoch
/// * `T_max` - Maximum number of epochs
///
/// # Returns
/// * Learning rate for current epoch
///
/// # Example
/// ```charl
/// let lr = cosine_annealing_lr(0.1, 0.001, 50, 100)
/// // Smoothly decreases from 0.1 to 0.001 following cosine curve
/// ```
pub fn builtin_cosine_annealing_lr(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("cosine_annealing_lr() expects 4 arguments: cosine_annealing_lr(lr_max, lr_min, epoch, T_max)".to_string());
    }

    let lr_max = args[0].to_float()?;
    let lr_min = args[1].to_float()?;
    let epoch = args[2].to_integer()? as usize;
    let t_max = args[3].to_integer()? as usize;

    if t_max == 0 {
        return Err("cosine_annealing_lr() T_max must be > 0".to_string());
    }

    let t = epoch.min(t_max);
    let pi = std::f64::consts::PI;
    let cosine_arg = pi * (t as f64) / (t_max as f64);
    let lr = lr_min + (lr_max - lr_min) * (1.0 + cosine_arg.cos()) / 2.0;

    Ok(Value::Float(lr))
}

// ============================================================================
// EFFICIENT ARCHITECTURES (Week 7-8 Part 1)
// ============================================================================
// Linear-time alternatives to O(n²) transformer attention
// State Space Models, Mixture of Experts, and efficient attention variants

// ============================================================================
// LINEAR ATTENTION - O(n) variants
// ============================================================================

/// linformer_create(d_model, num_heads, seq_len, k) -> LinformerLayer
/// Create Linformer layer with low-rank attention approximation
///
/// # Arguments
/// * `d_model` - Model dimension
/// * `num_heads` - Number of attention heads
/// * `seq_len` - Maximum sequence length
/// * `k` - Projection dimension (k << seq_len for efficiency)
///
/// # Returns
/// * LinformerLayer object
///
/// # Complexity
/// * O(ndk) where k << n (vs O(n²d) for standard attention)
///
/// # Example
/// ```charl
/// let layer = linformer_create(512, 8, 1024, 256)
/// ```
pub fn builtin_linformer_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("linformer_create() expects 4 arguments: linformer_create(d_model, num_heads, seq_len, k)".to_string());
    }

    let d_model = args[0].to_integer()? as usize;
    let num_heads = args[1].to_integer()? as usize;
    let seq_len = args[2].to_integer()? as usize;
    let k = args[3].to_integer()? as usize;

    let layer = Linformer::new(d_model, num_heads, seq_len, k);
    Ok(Value::LinformerLayer(Box::new(layer)))
}

/// linformer_forward(layer, Q, K, V) -> Tensor
/// Apply Linformer attention
///
/// # Arguments
/// * `layer` - LinformerLayer created with linformer_create()
/// * `Q` - Query tensor [seq_len, d_model]
/// * `K` - Key tensor [seq_len, d_model]
/// * `V` - Value tensor [seq_len, d_model]
///
/// # Returns
/// * Output tensor [seq_len, d_model]
///
/// # Example
/// ```charl
/// let output = linformer_forward(layer, Q, K, V)
/// ```
pub fn builtin_linformer_forward(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("linformer_forward() expects 4 arguments: linformer_forward(layer, Q, K, V)".to_string());
    }

    let layer = match &args[0] {
        Value::LinformerLayer(l) => l,
        _ => return Err("linformer_forward() first argument must be LinformerLayer".to_string()),
    };

    // Extract Q, K, V as Vec<Vec<f32>>
    let Q = tensor_to_vec2d(&args[1])?;
    let K = tensor_to_vec2d(&args[2])?;
    let V = tensor_to_vec2d(&args[3])?;

    // Forward pass
    let output = layer.forward(&Q, &K, &V);

    // Convert back to tensor
    vec2d_to_tensor(output)
}

/// performer_create(d_model, num_features) -> PerformerLayer
/// Create Performer layer with FAVOR+ (Fast Attention Via Orthogonal Random features)
///
/// # Arguments
/// * `d_model` - Model dimension
/// * `num_features` - Number of random features (typically d_model // 2)
///
/// # Returns
/// * PerformerLayer object
///
/// # Complexity
/// * O(nd²) where d is typically small
///
/// # Example
/// ```charl
/// let layer = performer_create(512, 256)
/// ```
pub fn builtin_performer_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("performer_create() expects 2 arguments: performer_create(d_model, num_features)".to_string());
    }

    let d_model = args[0].to_integer()? as usize;
    let num_features = args[1].to_integer()? as usize;

    let layer = Performer::new(d_model, num_features);
    Ok(Value::PerformerLayer(Box::new(layer)))
}

/// performer_forward(layer, Q, K, V) -> Tensor
/// Apply Performer attention
///
/// # Arguments
/// * `layer` - PerformerLayer created with performer_create()
/// * `Q` - Query tensor [seq_len, d_model]
/// * `K` - Key tensor [seq_len, d_model]
/// * `V` - Value tensor [seq_len, d_model]
///
/// # Returns
/// * Output tensor [seq_len, d_model]
///
/// # Example
/// ```charl
/// let output = performer_forward(layer, Q, K, V)
/// ```
pub fn builtin_performer_forward(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("performer_forward() expects 4 arguments: performer_forward(layer, Q, K, V)".to_string());
    }

    let layer = match &args[0] {
        Value::PerformerLayer(l) => l,
        _ => return Err("performer_forward() first argument must be PerformerLayer".to_string()),
    };

    let Q = tensor_to_vec2d(&args[1])?;
    let K = tensor_to_vec2d(&args[2])?;
    let V = tensor_to_vec2d(&args[3])?;

    let output = layer.forward(&Q, &K, &V);
    vec2d_to_tensor(output)
}

// Helper functions for tensor conversion
fn tensor_to_vec2d(value: &Value) -> Result<Vec<Vec<f32>>, String> {
    match value {
        Value::AutogradTensor(t) => {
            if t.shape.len() != 2 {
                return Err(format!("Expected 2D tensor, got shape {:?}", t.shape));
            }
            let rows = t.shape[0];
            let cols = t.shape[1];
            let mut result = vec![vec![0.0f32; cols]; rows];
            for i in 0..rows {
                for j in 0..cols {
                    result[i][j] = t.data[i * cols + j] as f32;  // Convert f64 to f32
                }
            }
            Ok(result)
        }
        _ => Err(format!("Expected AutogradTensor, got {}", value.type_name())),
    }
}

fn vec2d_to_tensor(data: Vec<Vec<f32>>) -> Result<Value, String> {
    if data.is_empty() {
        return Err("Cannot convert empty Vec<Vec<f32>> to tensor".to_string());
    }
    let rows = data.len();
    let cols = data[0].len();
    let flat: Vec<f64> = data.into_iter().flatten().map(|x| x as f64).collect(); // Convert f32 to f64

    let tensor = AutogradTensor::new(flat, vec![rows, cols]);
    Ok(Value::AutogradTensor(tensor))
}

/// fnet_create(d_model) -> FNetLayer
/// Create FNet layer that replaces attention with Fourier Transform
///
/// # Arguments
/// * `d_model` - Model dimension
///
/// # Returns
/// * FNetLayer object
///
/// # Complexity
/// * O(n log n) via FFT
///
/// # Example
/// ```charl
/// let layer = fnet_create(512)
/// ```
pub fn builtin_fnet_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("fnet_create() expects 1 argument: fnet_create(d_model)".to_string());
    }

    let d_model = args[0].to_integer()? as usize;
    let layer = FNet::new(d_model);
    Ok(Value::FNetLayer(Box::new(layer)))
}

/// fnet_forward(layer, input) -> Tensor
/// Apply FNet mixing (Fourier Transform)
///
/// # Arguments
/// * `layer` - FNetLayer created with fnet_create()
/// * `input` - Input tensor [seq_len, d_model]
///
/// # Returns
/// * Output tensor [seq_len, d_model]
///
/// # Example
/// ```charl
/// let output = fnet_forward(layer, input)
/// ```
pub fn builtin_fnet_forward(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("fnet_forward() expects 2 arguments: fnet_forward(layer, input)".to_string());
    }

    let layer = match &args[0] {
        Value::FNetLayer(l) => l,
        _ => return Err("fnet_forward() first argument must be FNetLayer".to_string()),
    };

    let input = tensor_to_vec2d(&args[1])?;
    let output = layer.forward(&input);
    vec2d_to_tensor(output)
}

/// rwkv_create(d_model) -> RWKVLayer
/// Create RWKV (Receptance Weighted Key Value) layer
///
/// # Arguments
/// * `d_model` - Model dimension
///
/// # Returns
/// * RWKVLayer object
///
/// # Complexity
/// * O(n) - RNN-like processing
///
/// # Example
/// ```charl
/// let layer = rwkv_create(512)
/// ```
pub fn builtin_rwkv_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("rwkv_create() expects 1 argument: rwkv_create(d_model)".to_string());
    }

    let d_model = args[0].to_integer()? as usize;
    let layer = RWKV::new(d_model);
    Ok(Value::RWKVLayer(Box::new(layer)))
}

/// rwkv_forward(layer, input) -> Tensor
/// Apply RWKV mixing
///
/// # Arguments
/// * `layer` - RWKVLayer created with rwkv_create()
/// * `input` - Input tensor [seq_len, d_model]
///
/// # Returns
/// * Output tensor [seq_len, d_model]
///
/// # Example
/// ```charl
/// let output = rwkv_forward(layer, input)
/// ```
pub fn builtin_rwkv_forward(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("rwkv_forward() expects 2 arguments: rwkv_forward(layer, input)".to_string());
    }

    let layer = match &args[0] {
        Value::RWKVLayer(l) => l,
        _ => return Err("rwkv_forward() first argument must be RWKVLayer".to_string()),
    };

    let input = tensor_to_vec2d(&args[1])?;
    let output = layer.forward_sequence(&input);
    vec2d_to_tensor(output)
}

// ============================================================================
// MAMBA - Selective State Space Models
// ============================================================================

/// mamba_create(d_model, d_state?) -> MambaLayer
/// Create Mamba block with selective SSM
///
/// # Arguments
/// * `d_model` - Model dimension
/// * `d_state` - State dimension (optional, default 16)
///
/// # Returns
/// * MambaLayer object
///
/// # Complexity
/// * O(n) - Linear in sequence length
///
/// # Example
/// ```charl
/// let layer = mamba_create(512, 16)
/// ```
pub fn builtin_mamba_create(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() || args.len() > 2 {
        return Err("mamba_create() expects 1-2 arguments: mamba_create(d_model, d_state?)".to_string());
    }

    let d_model = args[0].to_integer()? as usize;
    let mut config = MambaConfig::new(d_model);

    if args.len() == 2 {
        let d_state = args[1].to_integer()? as usize;
        config = config.with_state_size(d_state);
    }

    let layer = MambaBlock::new(config);
    Ok(Value::MambaLayer(Box::new(layer)))
}

/// mamba_forward(layer, input) -> Tensor
/// Apply Mamba selective SSM
///
/// # Arguments
/// * `layer` - MambaLayer created with mamba_create()
/// * `input` - Input tensor [seq_len, d_model]
///
/// # Returns
/// * Output tensor [seq_len, d_model]
///
/// # Example
/// ```charl
/// let output = mamba_forward(layer, input)
/// ```
pub fn builtin_mamba_forward(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("mamba_forward() expects 2 arguments: mamba_forward(layer, input)".to_string());
    }

    let layer = match &args[0] {
        Value::MambaLayer(l) => l,
        _ => return Err("mamba_forward() first argument must be MambaLayer".to_string()),
    };

    let input = tensor_to_vec2d(&args[1])?;
    let output = layer.forward_sequence(&input);
    vec2d_to_tensor(output)
}

// ============================================================================
// S4 - Structured State Spaces
// ============================================================================

/// s4_create(state_size, hidden_size, dt?) -> S4Layer
/// Create S4 layer with structured state space
///
/// # Arguments
/// * `state_size` - State dimension (N)
/// * `hidden_size` - Hidden dimension (H)
/// * `dt` - Discretization timestep (optional, default 0.001)
///
/// # Returns
/// * S4Layer object
///
/// # Complexity
/// * O(n) - Linear in sequence length
///
/// # Example
/// ```charl
/// let layer = s4_create(64, 128, 0.01)
/// ```
pub fn builtin_s4_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 || args.len() > 3 {
        return Err("s4_create() expects 2-3 arguments: s4_create(state_size, hidden_size, dt?)".to_string());
    }

    let state_size = args[0].to_integer()? as usize;
    let hidden_size = args[1].to_integer()? as usize;
    
    let mut config = SSMConfig::new(state_size, hidden_size);

    if args.len() == 3 {
        let dt = args[2].to_float()? as f32;
        config = config.with_dt(dt);
    }

    let mut layer = S4Layer::new(config);
    // Discretize with Zero-Order Hold (default)
    layer.discretize(DiscretizationMethod::ZeroOrderHold);

    Ok(Value::S4Layer(Box::new(layer)))
}

/// s4_forward(layer, input) -> Tensor
/// Apply S4 state space model
///
/// # Arguments
/// * `layer` - S4Layer created with s4_create()
/// * `input` - Input tensor [seq_len, hidden_size]
///
/// # Returns
/// * Output tensor [seq_len, hidden_size]
///
/// # Example
/// ```charl
/// let output = s4_forward(layer, input)
/// ```
pub fn builtin_s4_forward(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("s4_forward() expects 2 arguments: s4_forward(layer, input)".to_string());
    }

    let layer = match &args[0] {
        Value::S4Layer(l) => l,
        _ => return Err("s4_forward() first argument must be S4Layer".to_string()),
    };

    let input = tensor_to_vec2d(&args[1])?;
    let output = layer.forward_sequence(&input);
    vec2d_to_tensor(output)
}

// ============================================================================
// MOE - Mixture of Experts
// ============================================================================

/// moe_create(d_model, d_ff, num_experts, top_k) -> MoELayer
/// Create Mixture of Experts layer with sparse routing
///
/// # Arguments
/// * `d_model` - Model dimension
/// * `d_ff` - Feedforward dimension
/// * `num_experts` - Total number of experts
/// * `top_k` - Number of experts to use per token
///
/// # Returns
/// * MoELayer object
///
/// # Complexity
/// * O(nd) but sparse - only top_k/num_experts fraction of computation
///
/// # Example
/// ```charl
/// let layer = moe_create(512, 2048, 8, 2)  // 8 experts, use top 2
/// ```
pub fn builtin_moe_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("moe_create() expects 4 arguments: moe_create(d_model, d_ff, num_experts, top_k)".to_string());
    }

    let d_model = args[0].to_integer()? as usize;
    let d_ff = args[1].to_integer()? as usize;
    let num_experts = args[2].to_integer()? as usize;
    let top_k = args[3].to_integer()? as usize;

    let layer = MoELayer::new(d_model, d_ff, num_experts, top_k)
        .with_strategy(RoutingStrategy::TopK);

    Ok(Value::MoELayer(Box::new(layer)))
}

/// moe_forward(layer, input) -> Tensor
/// Apply Mixture of Experts
///
/// # Arguments
/// * `layer` - MoELayer created with moe_create()
/// * `input` - Input tensor [batch, d_model] or [d_model]
///
/// # Returns
/// * Output tensor (same shape as input)
///
/// # Example
/// ```charl
/// let output = moe_forward(layer, input)
/// ```
pub fn builtin_moe_forward(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("moe_forward() expects 2 arguments: moe_forward(layer, input)".to_string());
    }

    let layer = match &args[0] {
        Value::MoELayer(l) => l,
        _ => return Err("moe_forward() first argument must be MoELayer".to_string()),
    };

    // Handle both 1D and 2D inputs
    match &args[1] {
        Value::AutogradTensor(t) => {
            if t.shape.len() == 1 {
                // Single vector input - convert f64 to f32
                let input_f32: Vec<f32> = t.data.iter().map(|&x| x as f32).collect();
                let output_f32 = layer.forward(&input_f32);
                let output_f64: Vec<f64> = output_f32.iter().map(|&x| x as f64).collect();
                let tensor = AutogradTensor::new(output_f64, t.shape.clone());
                Ok(Value::AutogradTensor(tensor))
            } else if t.shape.len() == 2 {
                // Batch input
                let input_2d = tensor_to_vec2d(&args[1])?;
                let mut outputs = Vec::new();
                for row in input_2d {
                    outputs.push(layer.forward(&row));
                }
                vec2d_to_tensor(outputs)
            } else {
                Err(format!("moe_forward() expects 1D or 2D tensor, got shape {:?}", t.shape))
            }
        }
        _ => Err(format!("moe_forward() expects AutogradTensor, got {}", args[1].type_name())),
    }
}

// ============================================================================
// QUANTIZATION (Week 7-8 Part 2)
// ============================================================================
// Model compression: 4-8x memory reduction, 2-4x speedup

/// quantize_tensor_int8(tensor) -> QuantizedTensor
/// Quantize tensor to INT8 (4x memory reduction)
///
/// # Arguments
/// * `tensor` - AutogradTensor to quantize
///
/// # Returns
/// * QuantizedTensor with INT8 quantization
///
/// # Example
/// ```charl
/// let qtensor = quantize_tensor_int8(tensor)
/// ```
pub fn builtin_quantize_tensor_int8(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("quantize_tensor_int8() expects 1 argument: quantize_tensor_int8(tensor)".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(t) => {
            // Convert f64 to f32 for quantization
            let data_f32: Vec<f32> = t.data.iter().map(|&x| x as f32).collect();
            
            // Quantize to INT8
            let qtensor = quantize_tensor_auto(&data_f32, t.shape.clone(), QuantType::INT8)
                .map_err(|e| format!("Quantization failed: {}", e))?;
            
            Ok(Value::QuantizedTensor(Box::new(qtensor)))
        }
        _ => Err(format!("quantize_tensor_int8() expects AutogradTensor, got {}", args[0].type_name())),
    }
}

/// quantize_tensor_int4(tensor) -> QuantizedTensor
/// Quantize tensor to INT4 (8x memory reduction)
///
/// # Arguments
/// * `tensor` - AutogradTensor to quantize
///
/// # Returns
/// * QuantizedTensor with INT4 quantization
///
/// # Example
/// ```charl
/// let qtensor = quantize_tensor_int4(tensor)
/// ```
pub fn builtin_quantize_tensor_int4(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("quantize_tensor_int4() expects 1 argument: quantize_tensor_int4(tensor)".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(t) => {
            let data_f32: Vec<f32> = t.data.iter().map(|&x| x as f32).collect();
            let qtensor = quantize_tensor_auto(&data_f32, t.shape.clone(), QuantType::INT4)
                .map_err(|e| format!("Quantization failed: {}", e))?;
            
            Ok(Value::QuantizedTensor(Box::new(qtensor)))
        }
        _ => Err(format!("quantize_tensor_int4() expects AutogradTensor, got {}", args[0].type_name())),
    }
}

/// dequantize_tensor(qtensor) -> Tensor
/// Dequantize back to f32 tensor
///
/// # Arguments
/// * `qtensor` - QuantizedTensor to dequantize
///
/// # Returns
/// * AutogradTensor with f32 data
///
/// # Example
/// ```charl
/// let tensor = dequantize_tensor(qtensor)
/// ```
pub fn builtin_dequantize_tensor(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("dequantize_tensor() expects 1 argument: dequantize_tensor(qtensor)".to_string());
    }

    match &args[0] {
        Value::QuantizedTensor(qt) => {
            // Dequantize to f32
            let data_f32 = dequantize_tensor(&qt.data, &qt.params);
            
            // Convert f32 to f64 for AutogradTensor
            let data_f64: Vec<f64> = data_f32.iter().map(|&x| x as f64).collect();
            
            let tensor = AutogradTensor::new(data_f64, qt.shape.clone());
            Ok(Value::AutogradTensor(tensor))
        }
        _ => Err(format!("dequantize_tensor() expects QuantizedTensor, got {}", args[0].type_name())),
    }
}

/// quantized_tensor_info(qtensor) -> Array[type, shape, reduction]
/// Get information about quantized tensor
///
/// # Arguments
/// * `qtensor` - QuantizedTensor
///
/// # Returns
/// * Array with [type_str, shape, reduction_factor]
///
/// # Example
/// ```charl
/// let info = quantized_tensor_info(qtensor)
/// print("Type: " + info[0])
/// ```
pub fn builtin_quantized_tensor_info(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("quantized_tensor_info() expects 1 argument: quantized_tensor_info(qtensor)".to_string());
    }

    match &args[0] {
        Value::QuantizedTensor(qt) => {
            let type_str = match qt.params.quant_type {
                QuantType::INT8 => "INT8",
                QuantType::INT4 => "INT4",
                QuantType::FP16 => "FP16",
                QuantType::BF16 => "BF16",
            };
            
            let reduction = qt.params.quant_type.reduction_factor();
            
            // Return array: [type, shape_string, reduction]
            let info = vec![
                Value::String(type_str.to_string()),
                Value::String(format!("{:?}", qt.shape)),
                Value::Integer(reduction as i64),
            ];
            
            Ok(Value::Array(info))
        }
        _ => Err(format!("quantized_tensor_info() expects QuantizedTensor, got {}", args[0].type_name())),
    }
}

/// quantize_model_weights(weights, calibration_data) -> QuantizedTensor
/// Post-Training Quantization for model weights
///
/// # Arguments
/// * `weights` - Model weights tensor
/// * `calibration_data` - Array of calibration tensors
///
/// # Returns
/// * QuantizedTensor with calibrated quantization
///
/// # Example
/// ```charl
/// let qweights = quantize_model_weights(weights, [batch1, batch2, batch3])
/// ```
pub fn builtin_quantize_model_weights(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("quantize_model_weights() expects 2 arguments: quantize_model_weights(weights, calibration_data)".to_string());
    }

    // Extract weights
    let weights_f32: Vec<f32> = match &args[0] {
        Value::AutogradTensor(t) => t.data.iter().map(|&x| x as f32).collect(),
        _ => return Err(format!("quantize_model_weights() expects AutogradTensor for weights, got {}", args[0].type_name())),
    };

    // Extract calibration data
    let calibration: Vec<Vec<f32>> = match &args[1] {
        Value::Array(arr) => {
            let mut result = Vec::new();
            for val in arr {
                match val {
                    Value::AutogradTensor(t) => {
                        let batch_f32: Vec<f32> = t.data.iter().map(|&x| x as f32).collect();
                        result.push(batch_f32);
                    }
                    _ => return Err("quantize_model_weights() calibration_data must be array of tensors".to_string()),
                }
            }
            result
        }
        _ => return Err("quantize_model_weights() second argument must be array".to_string()),
    };

    // Perform PTQ with MinMax calibration
    let qtensor = post_training_quantization(&weights_f32, &calibration, QuantType::INT8, CalibrationMethod::MinMax)
        .map_err(|e| format!("PTQ failed: {}", e))?;

    Ok(Value::QuantizedTensor(Box::new(qtensor)))
}

// =============================================================================
// WEEK 7-8 PART 3: Advanced Reasoning - Chain-of-Thought
// =============================================================================

/// Create a new Chain-of-Thought reasoning chain
///
/// # Arguments
/// * `problem` - The problem or question to reason about
///
/// # Returns
/// ChainOfThought value
///
/// # Example
/// ```charl
/// let cot = cot_create("How many balls does Roger have?")
/// ```
pub fn builtin_cot_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("cot_create() expects 1 argument: cot_create(problem)".to_string());
    }

    let problem = match &args[0] {
        Value::String(s) => s.clone(),
        v => return Err(format!("cot_create() expects String, got {}", v.type_name())),
    };

    let cot = ChainOfThought::new(problem);
    Ok(Value::ChainOfThought(Box::new(cot)))
}

/// Add a reasoning step to a Chain-of-Thought
///
/// # Arguments
/// * `cot` - ChainOfThought to extend
/// * `step_number` - Step number (for ordering)
/// * `thought` - The thought/reasoning at this step
///
/// # Returns
/// New ChainOfThought with step added
///
/// # Example
/// ```charl
/// let cot = cot_add_step(cot, 1, "Initial: 5 balls")
/// ```
pub fn builtin_cot_add_step(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("cot_add_step() expects 3 arguments: cot_add_step(cot, step_number, thought)".to_string());
    }

    let cot = match &args[0] {
        Value::ChainOfThought(c) => c.as_ref().clone(),
        v => return Err(format!("cot_add_step() expects ChainOfThought, got {}", v.type_name())),
    };

    let step_number = args[1].to_integer()? as usize;

    let thought = match &args[2] {
        Value::String(s) => s.clone(),
        v => return Err(format!("cot_add_step() expects String for thought, got {}", v.type_name())),
    };

    let step = ReasoningStep::new(step_number, thought);
    let new_cot = cot.add_step(step);

    Ok(Value::ChainOfThought(Box::new(new_cot)))
}

/// Add a reasoning step with confidence to a Chain-of-Thought
///
/// # Arguments
/// * `cot` - ChainOfThought to extend
/// * `step_number` - Step number (for ordering)
/// * `thought` - The thought/reasoning at this step
/// * `confidence` - Confidence in this step [0, 1]
///
/// # Returns
/// New ChainOfThought with step added
///
/// # Example
/// ```charl
/// let cot = cot_add_step_conf(cot, 1, "Initial: 5 balls", 0.95)
/// ```
pub fn builtin_cot_add_step_conf(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("cot_add_step_conf() expects 4 arguments: cot_add_step_conf(cot, step_number, thought, confidence)".to_string());
    }

    let cot = match &args[0] {
        Value::ChainOfThought(c) => c.as_ref().clone(),
        v => return Err(format!("cot_add_step_conf() expects ChainOfThought, got {}", v.type_name())),
    };

    let step_number = args[1].to_integer()? as usize;

    let thought = match &args[2] {
        Value::String(s) => s.clone(),
        v => return Err(format!("cot_add_step_conf() expects String for thought, got {}", v.type_name())),
    };

    let confidence = match &args[3] {
        Value::Float(f) => *f as f32,
        Value::Integer(i) => *i as f32,
        v => return Err(format!("cot_add_step_conf() expects number for confidence, got {}", v.type_name())),
    };

    let step = ReasoningStep::new(step_number, thought).with_confidence(confidence);
    let new_cot = cot.add_step(step);

    Ok(Value::ChainOfThought(Box::new(new_cot)))
}

/// Set the final answer for a Chain-of-Thought
///
/// # Arguments
/// * `cot` - ChainOfThought to complete
/// * `answer` - The final answer
///
/// # Returns
/// New ChainOfThought with final answer set
///
/// # Example
/// ```charl
/// let cot = cot_with_answer(cot, "11 balls")
/// ```
pub fn builtin_cot_with_answer(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("cot_with_answer() expects 2 arguments: cot_with_answer(cot, answer)".to_string());
    }

    let cot = match &args[0] {
        Value::ChainOfThought(c) => c.as_ref().clone(),
        v => return Err(format!("cot_with_answer() expects ChainOfThought, got {}", v.type_name())),
    };

    let answer = match &args[1] {
        Value::String(s) => s.clone(),
        v => return Err(format!("cot_with_answer() expects String, got {}", v.type_name())),
    };

    let new_cot = cot.with_final_answer(answer);

    Ok(Value::ChainOfThought(Box::new(new_cot)))
}

/// Compute overall confidence for a Chain-of-Thought
/// (minimum confidence across all steps)
///
/// # Arguments
/// * `cot` - ChainOfThought to analyze
///
/// # Returns
/// New ChainOfThought with computed confidence
///
/// # Example
/// ```charl
/// let cot = cot_compute_confidence(cot)
/// ```
pub fn builtin_cot_compute_confidence(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("cot_compute_confidence() expects 1 argument: cot_compute_confidence(cot)".to_string());
    }

    let mut cot = match &args[0] {
        Value::ChainOfThought(c) => c.as_ref().clone(),
        v => return Err(format!("cot_compute_confidence() expects ChainOfThought, got {}", v.type_name())),
    };

    cot.compute_confidence();

    Ok(Value::ChainOfThought(Box::new(cot)))
}

/// Get information about a Chain-of-Thought
///
/// # Arguments
/// * `cot` - ChainOfThought to inspect
///
/// # Returns
/// Array: [problem, num_steps, confidence, final_answer]
///
/// # Example
/// ```charl
/// let info = cot_get_info(cot)
/// print("Problem: " + info[0])
/// print("Steps: " + str(info[1]))
/// print("Confidence: " + str(info[2]))
/// print("Answer: " + info[3])
/// ```
pub fn builtin_cot_get_info(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("cot_get_info() expects 1 argument: cot_get_info(cot)".to_string());
    }

    let cot = match &args[0] {
        Value::ChainOfThought(c) => c.as_ref(),
        v => return Err(format!("cot_get_info() expects ChainOfThought, got {}", v.type_name())),
    };

    Ok(Value::Array(vec![
        Value::String(cot.problem.clone()),
        Value::Integer(cot.num_steps() as i64),
        Value::Float(cot.confidence as f64),
        Value::String(cot.final_answer.clone()),
    ]))
}

/// Get details of a specific step in a Chain-of-Thought
///
/// # Arguments
/// * `cot` - ChainOfThought to inspect
/// * `step_number` - Step number to retrieve
///
/// # Returns
/// Array: [step_number, thought, confidence, verified]
///
/// # Example
/// ```charl
/// let step = cot_get_step(cot, 1)
/// print("Step " + str(step[0]) + ": " + step[1])
/// ```
pub fn builtin_cot_get_step(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("cot_get_step() expects 2 arguments: cot_get_step(cot, step_number)".to_string());
    }

    let cot = match &args[0] {
        Value::ChainOfThought(c) => c.as_ref(),
        v => return Err(format!("cot_get_step() expects ChainOfThought, got {}", v.type_name())),
    };

    let step_num = args[1].to_integer()? as usize;

    // Find step by step_number
    let step = cot.steps.iter()
        .find(|s| s.step_number == step_num)
        .ok_or_else(|| format!("Step {} not found in chain", step_num))?;

    Ok(Value::Array(vec![
        Value::Integer(step.step_number as i64),
        Value::String(step.thought.clone()),
        Value::Float(step.confidence as f64),
        Value::Boolean(step.verified),
    ]))
}

// ============================================================================
// WEEK 15-16: OPERATOR FUSION
// ============================================================================

/// Create a fusion optimizer with specified configuration
///
/// Usage:
/// ```charl
/// let optimizer = fusion_create("default")  // or "aggressive", "conservative"
/// ```
pub fn builtin_fusion_create(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("fusion_create requires 1 argument: config_type".to_string());
    }

    let config_type = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => return Err("fusion_create: config_type must be a string (\"default\", \"aggressive\", or \"conservative\")".to_string()),
    };

    use crate::fusion::{FusionConfig, FusionOptimizer};

    let config = match config_type {
        "default" => FusionConfig::default(),
        "aggressive" => FusionConfig::aggressive(),
        "conservative" => FusionConfig::conservative(),
        _ => return Err(format!("fusion_create: unknown config type '{}'. Use \"default\", \"aggressive\", or \"conservative\"", config_type)),
    };

    let optimizer = FusionOptimizer::new(config);

    Ok(Value::FusionOptimizer(Box::new(optimizer)))
}

/// Analyze a computation graph and find fusion opportunities
///
/// Usage:
/// ```charl
/// let optimizer = fusion_create("aggressive")
/// let opportunities = fusion_analyze(optimizer, graph)
/// ```
///
/// Returns: Array of fusion opportunities with estimated speedup
pub fn builtin_fusion_analyze(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("fusion_analyze requires 2 arguments: optimizer, graph".to_string());
    }

    let optimizer = match &args[0] {
        Value::FusionOptimizer(opt) => opt,
        _ => return Err("fusion_analyze: first argument must be a FusionOptimizer".to_string()),
    };

    // For now, we'll return a simplified version
    // In a full implementation, we would analyze the actual computation graph

    // Clone the optimizer so we can modify it
    let mut opt_clone = (**optimizer).clone();

    // For demonstration, create empty opportunities list
    // In real implementation, this would analyze the graph passed as args[1]
    let opportunities: Vec<String> = Vec::new();

    // Update stats
    use crate::fusion::optimizer::FusionStats;
    let stats = FusionStats {
        opportunities_found: opportunities.len(),
        fusions_applied: 0,
        total_memory_saved: 0,
        average_speedup: 0.0,
        nodes_eliminated: 0,
    };

    // Return the updated optimizer with stats
    Ok(Value::FusionOptimizer(Box::new(opt_clone)))
}

/// Get fusion statistics from an optimizer
///
/// Usage:
/// ```charl
/// let stats = fusion_get_stats(optimizer)
/// print("Opportunities found: " + str(stats[0]))
/// print("Fusions applied: " + str(stats[1]))
/// print("Memory saved (bytes): " + str(stats[2]))
/// print("Average speedup: " + str(stats[3]))
/// ```
///
/// Returns: [opportunities_found, fusions_applied, total_memory_saved, average_speedup, nodes_eliminated]
pub fn builtin_fusion_get_stats(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("fusion_get_stats requires 1 argument: optimizer".to_string());
    }

    let optimizer = match &args[0] {
        Value::FusionOptimizer(opt) => opt,
        _ => return Err("fusion_get_stats: argument must be a FusionOptimizer".to_string()),
    };

    // For now, return default stats
    // In full implementation, we would extract actual stats from the optimizer
    Ok(Value::Array(vec![
        Value::Integer(0),  // opportunities_found
        Value::Integer(0),  // fusions_applied
        Value::Integer(0),  // total_memory_saved
        Value::Float(0.0),  // average_speedup
        Value::Integer(0),  // nodes_eliminated
    ]))
}

/// Enable or disable fusion globally
///
/// Usage:
/// ```charl
/// fusion_enable(true)   // Enable fusion
/// fusion_enable(false)  // Disable fusion
/// ```
pub fn builtin_fusion_enable(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("fusion_enable requires 1 argument: enabled (bool)".to_string());
    }

    let enabled = match &args[0] {
        Value::Boolean(b) => *b,
        _ => return Err("fusion_enable: argument must be a boolean".to_string()),
    };

    // For now, this is a no-op
    // In full implementation, this would set a global fusion flag

    Ok(Value::Null)
}

/// Set fusion strategy
///
/// Usage:
/// ```charl
/// fusion_set_strategy("aggressive")  // More fusions, higher risk
/// fusion_set_strategy("conservative")  // Fewer fusions, safer
/// fusion_set_strategy("default")  // Balanced approach
/// ```
pub fn builtin_fusion_set_strategy(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("fusion_set_strategy requires 1 argument: strategy".to_string());
    }

    let strategy = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => return Err("fusion_set_strategy: argument must be a string".to_string()),
    };

    match strategy {
        "default" | "aggressive" | "conservative" => {
            // For now, this is a no-op
            // In full implementation, this would update global fusion config
            Ok(Value::Null)
        }
        _ => Err(format!("fusion_set_strategy: unknown strategy '{}'. Use \"default\", \"aggressive\", or \"conservative\"", strategy)),
    }
}

// ============================================================================
// Week 17-19: MULTIMODAL AI
// ============================================================================

// VISION-LANGUAGE BUILTINS

/// clip_encoder_create(embedding_dim) -> CLIPEncoder
/// Create a CLIP encoder for vision-language embeddings
pub fn builtin_clip_encoder_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("clip_encoder_create requires 1 argument: embedding_dim".to_string());
    }

    let embedding_dim = match &args[0] {
        Value::Integer(i) => *i as usize,
        _ => return Err("clip_encoder_create: embedding_dim must be an integer".to_string()),
    };

    use crate::multimodal::vision_language::CLIPEncoder;
    let encoder = CLIPEncoder::new(embedding_dim);
    Ok(Value::CLIPEncoder(Box::new(encoder)))
}

/// clip_encode_image(encoder, image) -> MultimodalEmbedding
/// Encode an image to an embedding vector
pub fn builtin_clip_encode_image(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("clip_encode_image requires 2 arguments: encoder, image".to_string());
    }

    let encoder = match &args[0] {
        Value::CLIPEncoder(e) => e,
        _ => return Err("clip_encode_image: first argument must be a CLIPEncoder".to_string()),
    };

    let image = match &args[1] {
        Value::Image(img) => img,
        _ => return Err("clip_encode_image: second argument must be an Image".to_string()),
    };

    let embedding = encoder.encode_image(image);
    Ok(Value::MultimodalEmbedding(Box::new(embedding)))
}

/// clip_encode_text(encoder, text) -> MultimodalEmbedding
/// Encode text to an embedding vector
pub fn builtin_clip_encode_text(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("clip_encode_text requires 2 arguments: encoder, text".to_string());
    }

    let encoder = match &args[0] {
        Value::CLIPEncoder(e) => e,
        _ => return Err("clip_encode_text: first argument must be a CLIPEncoder".to_string()),
    };

    let text = match &args[1] {
        Value::String(s) => s,
        _ => return Err("clip_encode_text: second argument must be a string".to_string()),
    };

    let embedding = encoder.encode_text(text);
    Ok(Value::MultimodalEmbedding(Box::new(embedding)))
}

/// embedding_cosine_similarity(emb1, emb2) -> float
/// Compute cosine similarity between two embeddings
pub fn builtin_embedding_cosine_similarity(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("embedding_cosine_similarity requires 2 arguments: emb1, emb2".to_string());
    }

    let emb1 = match &args[0] {
        Value::MultimodalEmbedding(e) => e,
        _ => return Err("embedding_cosine_similarity: first argument must be a MultimodalEmbedding".to_string()),
    };

    let emb2 = match &args[1] {
        Value::MultimodalEmbedding(e) => e,
        _ => return Err("embedding_cosine_similarity: second argument must be a MultimodalEmbedding".to_string()),
    };

    let similarity = emb1.cosine_similarity(emb2);
    Ok(Value::Float(similarity as f64))
}

/// image_create(id, width, height, pixels_array) -> Image
/// Create an image from pixel data
pub fn builtin_image_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("image_create requires 4 arguments: id, width, height, pixels_array".to_string());
    }

    let id = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("image_create: id must be a string".to_string()),
    };

    let width = match &args[1] {
        Value::Integer(i) => *i as usize,
        _ => return Err("image_create: width must be an integer".to_string()),
    };

    let height = match &args[2] {
        Value::Integer(i) => *i as usize,
        _ => return Err("image_create: height must be an integer".to_string()),
    };

    let pixels = match &args[3] {
        Value::Array(arr) => {
            let mut pixel_vec = Vec::new();
            for val in arr {
                match val {
                    Value::Integer(i) => pixel_vec.push(*i as f32),
                    Value::Float(f) => pixel_vec.push(*f as f32),
                    _ => return Err("image_create: pixels array must contain numbers".to_string()),
                }
            }
            pixel_vec
        }
        _ => return Err("image_create: pixels must be an array".to_string()),
    };

    use crate::multimodal::vision_language::Image;
    let image = Image::new(id, width, height, pixels);
    Ok(Value::Image(Box::new(image)))
}

/// image_with_caption(image, caption) -> Image
/// Add a caption to an image
pub fn builtin_image_with_caption(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("image_with_caption requires 2 arguments: image, caption".to_string());
    }

    let image = match &args[0] {
        Value::Image(img) => img.as_ref().clone(),
        _ => return Err("image_with_caption: first argument must be an Image".to_string()),
    };

    let caption = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("image_with_caption: caption must be a string".to_string()),
    };

    let image_with_cap = image.with_caption(caption);
    Ok(Value::Image(Box::new(image_with_cap)))
}

/// vqa_create(encoder) -> VQASystem
/// Create a Visual Question Answering system
pub fn builtin_vqa_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("vqa_create requires 1 argument: encoder".to_string());
    }

    let encoder = match &args[0] {
        Value::CLIPEncoder(e) => e.as_ref().clone(),
        _ => return Err("vqa_create: argument must be a CLIPEncoder".to_string()),
    };

    use crate::multimodal::vision_language::VQASystem;
    let vqa = VQASystem::new(encoder);
    Ok(Value::VQASystem(Box::new(vqa)))
}

/// vqa_add_qa(vqa, image_id, question, answer) -> Null
/// Add a question-answer pair to the VQA database
pub fn builtin_vqa_add_qa(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("vqa_add_qa requires 4 arguments: vqa, image_id, question, answer".to_string());
    }

    let mut vqa = match &args[0] {
        Value::VQASystem(v) => v.as_ref().clone(),
        _ => return Err("vqa_add_qa: first argument must be a VQASystem".to_string()),
    };

    let image_id = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("vqa_add_qa: image_id must be a string".to_string()),
    };

    let question = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err("vqa_add_qa: question must be a string".to_string()),
    };

    let answer = match &args[3] {
        Value::String(s) => s.clone(),
        _ => return Err("vqa_add_qa: answer must be a string".to_string()),
    };

    vqa.add_qa(image_id, question, answer);

    // Note: In Charl, we can't mutate the original VQASystem, so we return it
    // The user needs to reassign: vqa = vqa_add_qa(vqa, ...)
    // For now, return Null (later we could return the updated VQASystem)
    Ok(Value::Null)
}

/// vqa_answer(vqa, image, question) -> [answer, confidence, reasoning] or null
/// Answer a question about an image
pub fn builtin_vqa_answer(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("vqa_answer requires 3 arguments: vqa, image, question".to_string());
    }

    let vqa = match &args[0] {
        Value::VQASystem(v) => v,
        _ => return Err("vqa_answer: first argument must be a VQASystem".to_string()),
    };

    let image = match &args[1] {
        Value::Image(img) => img,
        _ => return Err("vqa_answer: second argument must be an Image".to_string()),
    };

    let question = match &args[2] {
        Value::String(s) => s,
        _ => return Err("vqa_answer: question must be a string".to_string()),
    };

    match vqa.answer_question(image, question) {
        Some(answer) => Ok(Value::Array(vec![
            Value::String(answer.answer.clone()),
            Value::Float(answer.confidence as f64),
            Value::String(answer.reasoning.clone()),
        ])),
        None => Ok(Value::Null),
    }
}

/// cross_modal_create(encoder) -> CrossModalRetrieval
/// Create a cross-modal retrieval system
pub fn builtin_cross_modal_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("cross_modal_create requires 1 argument: encoder".to_string());
    }

    let encoder = match &args[0] {
        Value::CLIPEncoder(e) => e.as_ref().clone(),
        _ => return Err("cross_modal_create: argument must be a CLIPEncoder".to_string()),
    };

    use crate::multimodal::vision_language::CrossModalRetrieval;
    let retrieval = CrossModalRetrieval::new(encoder);
    Ok(Value::CrossModalRetrieval(Box::new(retrieval)))
}

// SCENE UNDERSTANDING BUILTINS

/// scene_object_create(id, class, bbox_array) -> SceneObject
/// Create a scene object with bounding box [x, y, width, height]
pub fn builtin_scene_object_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("scene_object_create requires 3 arguments: id, class, bbox_array".to_string());
    }

    let id = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("scene_object_create: id must be a string".to_string()),
    };

    let class = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("scene_object_create: class must be a string".to_string()),
    };

    let bbox = match &args[2] {
        Value::Array(arr) => {
            if arr.len() != 4 {
                return Err("scene_object_create: bbox must have 4 elements [x, y, w, h]".to_string());
            }
            let x = arr[0].to_float()? as f32;
            let y = arr[1].to_float()? as f32;
            let w = arr[2].to_float()? as f32;
            let h = arr[3].to_float()? as f32;
            (x, y, w, h)
        }
        _ => return Err("scene_object_create: bbox must be an array".to_string()),
    };

    use crate::multimodal::scene_understanding::SceneObject;
    let obj = SceneObject::new(id, class, bbox);
    Ok(Value::SceneObject(Box::new(obj)))
}

/// scene_object_with_attribute(obj, attribute) -> SceneObject
/// Add an attribute to a scene object
pub fn builtin_scene_object_with_attribute(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("scene_object_with_attribute requires 2 arguments: obj, attribute".to_string());
    }

    let obj = match &args[0] {
        Value::SceneObject(o) => o.as_ref().clone(),
        _ => return Err("scene_object_with_attribute: first argument must be a SceneObject".to_string()),
    };

    let attribute = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("scene_object_with_attribute: attribute must be a string".to_string()),
    };

    let obj_with_attr = obj.with_attribute(attribute);
    Ok(Value::SceneObject(Box::new(obj_with_attr)))
}

/// scene_graph_create() -> SceneGraph
/// Create an empty scene graph
pub fn builtin_scene_graph_create(args: Vec<Value>) -> Result<Value, String> {
    if !args.is_empty() {
        return Err("scene_graph_create takes no arguments".to_string());
    }

    use crate::multimodal::scene_understanding::SceneGraph;
    let graph = SceneGraph::new();
    Ok(Value::SceneGraph(Box::new(graph)))
}

/// scene_graph_add_object(graph, object) -> SceneGraph
/// Add an object to a scene graph (returns updated graph)
pub fn builtin_scene_graph_add_object(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("scene_graph_add_object requires 2 arguments: graph, object".to_string());
    }

    let mut graph = match &args[0] {
        Value::SceneGraph(g) => g.as_ref().clone(),
        _ => return Err("scene_graph_add_object: first argument must be a SceneGraph".to_string()),
    };

    let object = match &args[1] {
        Value::SceneObject(o) => o.as_ref().clone(),
        _ => return Err("scene_graph_add_object: second argument must be a SceneObject".to_string()),
    };

    graph.add_object(object);
    Ok(Value::SceneGraph(Box::new(graph)))
}

/// scene_graph_add_relation(graph, subject_id, relation, object_id) -> SceneGraph
/// Add a spatial relation between two objects (returns updated graph)
/// relation: "on", "above", "below", "left_of", "right_of", "near", "contains", "inside"
pub fn builtin_scene_graph_add_relation(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("scene_graph_add_relation requires 4 arguments: graph, subject_id, relation, object_id".to_string());
    }

    let mut graph = match &args[0] {
        Value::SceneGraph(g) => g.as_ref().clone(),
        _ => return Err("scene_graph_add_relation: first argument must be a SceneGraph".to_string()),
    };

    let subject_id = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("scene_graph_add_relation: subject_id must be a string".to_string()),
    };

    let relation_str = match &args[2] {
        Value::String(s) => s.as_str(),
        _ => return Err("scene_graph_add_relation: relation must be a string".to_string()),
    };

    let object_id = match &args[3] {
        Value::String(s) => s.clone(),
        _ => return Err("scene_graph_add_relation: object_id must be a string".to_string()),
    };

    use crate::multimodal::scene_understanding::{ObjectRelation, SpatialRelation};
    let relation = match relation_str {
        "on" => SpatialRelation::On,
        "above" => SpatialRelation::Above,
        "below" => SpatialRelation::Below,
        "left_of" => SpatialRelation::LeftOf,
        "right_of" => SpatialRelation::RightOf,
        "near" => SpatialRelation::Near,
        "contains" => SpatialRelation::Contains,
        "inside" => SpatialRelation::Inside,
        _ => return Err(format!("scene_graph_add_relation: unknown relation '{}'. Valid: on, above, below, left_of, right_of, near, contains, inside", relation_str)),
    };

    let rel = ObjectRelation::new(subject_id, relation, object_id);
    graph.add_relation(rel)?;
    Ok(Value::SceneGraph(Box::new(graph)))
}

/// scene_graph_get_relations(graph, object_id) -> array of relation strings
/// Get all relations for an object (as subject)
pub fn builtin_scene_graph_get_relations(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("scene_graph_get_relations requires 2 arguments: graph, object_id".to_string());
    }

    let graph = match &args[0] {
        Value::SceneGraph(g) => g,
        _ => return Err("scene_graph_get_relations: first argument must be a SceneGraph".to_string()),
    };

    let object_id = match &args[1] {
        Value::String(s) => s,
        _ => return Err("scene_graph_get_relations: object_id must be a string".to_string()),
    };

    let relations = graph.get_relations_for(object_id);
    let relation_strs: Vec<Value> = relations
        .iter()
        .map(|r| Value::String(r.to_string()))
        .collect();

    Ok(Value::Array(relation_strs))
}

/// scene_graph_to_description(graph) -> string
/// Convert scene graph to natural language description
pub fn builtin_scene_graph_to_description(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("scene_graph_to_description requires 1 argument: graph".to_string());
    }

    let graph = match &args[0] {
        Value::SceneGraph(g) => g,
        _ => return Err("scene_graph_to_description: argument must be a SceneGraph".to_string()),
    };

    let description = graph.to_description();
    Ok(Value::String(description))
}

/// scene_graph_generator_create() -> SceneGraphGenerator
/// Create a scene graph generator
pub fn builtin_scene_graph_generator_create(args: Vec<Value>) -> Result<Value, String> {
    if !args.is_empty() {
        return Err("scene_graph_generator_create takes no arguments".to_string());
    }

    use crate::multimodal::scene_understanding::SceneGraphGenerator;
    let generator = SceneGraphGenerator::new();
    Ok(Value::SceneGraphGenerator(Box::new(generator)))
}

/// scene_graph_generate(generator, objects_array) -> SceneGraph
/// Generate a scene graph from detected objects
pub fn builtin_scene_graph_generate(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("scene_graph_generate requires 2 arguments: generator, objects_array".to_string());
    }

    let generator = match &args[0] {
        Value::SceneGraphGenerator(g) => g,
        _ => return Err("scene_graph_generate: first argument must be a SceneGraphGenerator".to_string()),
    };

    let objects = match &args[1] {
        Value::Array(arr) => {
            let mut obj_vec = Vec::new();
            for val in arr {
                match val {
                    Value::SceneObject(o) => obj_vec.push(o.as_ref().clone()),
                    _ => return Err("scene_graph_generate: objects array must contain SceneObjects".to_string()),
                }
            }
            obj_vec
        }
        _ => return Err("scene_graph_generate: second argument must be an array of SceneObjects".to_string()),
    };

    let graph = generator.generate(objects);
    Ok(Value::SceneGraph(Box::new(graph)))
}

/// temporal_event_create(id, description, start, end) -> TemporalEvent
/// Create a temporal event
pub fn builtin_temporal_event_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("temporal_event_create requires 4 arguments: id, description, start, end".to_string());
    }

    let id = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("temporal_event_create: id must be a string".to_string()),
    };

    let description = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("temporal_event_create: description must be a string".to_string()),
    };

    let start = args[2].to_float()? as f32;
    let end = args[3].to_float()? as f32;

    use crate::multimodal::scene_understanding::TemporalEvent;
    let event = TemporalEvent::new(id, description, start, end);
    Ok(Value::TemporalEvent(Box::new(event)))
}

/// temporal_event_with_object(event, object_id) -> TemporalEvent
/// Add an object to a temporal event
pub fn builtin_temporal_event_with_object(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("temporal_event_with_object requires 2 arguments: event, object_id".to_string());
    }

    let event = match &args[0] {
        Value::TemporalEvent(e) => e.as_ref().clone(),
        _ => return Err("temporal_event_with_object: first argument must be a TemporalEvent".to_string()),
    };

    let object_id = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("temporal_event_with_object: object_id must be a string".to_string()),
    };

    let event_with_obj = event.with_object(object_id);
    Ok(Value::TemporalEvent(Box::new(event_with_obj)))
}

/// temporal_event_relation(event1, event2) -> string
/// Get temporal relation between two events ("before", "after", "during", "overlaps")
pub fn builtin_temporal_event_relation(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("temporal_event_relation requires 2 arguments: event1, event2".to_string());
    }

    let event1 = match &args[0] {
        Value::TemporalEvent(e) => e,
        _ => return Err("temporal_event_relation: first argument must be a TemporalEvent".to_string()),
    };

    let event2 = match &args[1] {
        Value::TemporalEvent(e) => e,
        _ => return Err("temporal_event_relation: second argument must be a TemporalEvent".to_string()),
    };

    use crate::multimodal::scene_understanding::TemporalRelation;
    let relation = event1.relation_to(event2);
    let relation_str = match relation {
        TemporalRelation::Before => "before",
        TemporalRelation::After => "after",
        TemporalRelation::During => "during",
        TemporalRelation::Overlaps => "overlaps",
    };

    Ok(Value::String(relation_str.to_string()))
}

// CROSS-MODAL REASONING BUILTINS

/// visual_grounding_create(encoder) -> VisualGrounding
/// Create a visual grounding system
pub fn builtin_visual_grounding_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("visual_grounding_create requires 1 argument: encoder".to_string());
    }

    let encoder = match &args[0] {
        Value::CLIPEncoder(e) => e.as_ref().clone(),
        _ => return Err("visual_grounding_create: argument must be a CLIPEncoder".to_string()),
    };

    use crate::multimodal::cross_modal_reasoning::VisualGrounding;
    let grounding = VisualGrounding::new(encoder);
    Ok(Value::VisualGrounding(Box::new(grounding)))
}

/// visual_grounding_ground_phrase(grounding, phrase, scene) -> array of [obj_id, score]
/// Ground a text phrase to objects in a scene
pub fn builtin_visual_grounding_ground_phrase(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("visual_grounding_ground_phrase requires 3 arguments: grounding, phrase, scene".to_string());
    }

    let grounding = match &args[0] {
        Value::VisualGrounding(g) => g,
        _ => return Err("visual_grounding_ground_phrase: first argument must be a VisualGrounding".to_string()),
    };

    let phrase = match &args[1] {
        Value::String(s) => s,
        _ => return Err("visual_grounding_ground_phrase: phrase must be a string".to_string()),
    };

    let scene = match &args[2] {
        Value::SceneGraph(s) => s,
        _ => return Err("visual_grounding_ground_phrase: third argument must be a SceneGraph".to_string()),
    };

    let matches = grounding.ground_phrase(phrase, scene);
    let result: Vec<Value> = matches
        .iter()
        .map(|(obj_id, score)| {
            Value::Array(vec![
                Value::String(obj_id.clone()),
                Value::Float(*score as f64),
            ])
        })
        .collect();

    Ok(Value::Array(result))
}

/// multimodal_reasoner_create(grounding) -> MultimodalReasoner
/// Create a multimodal reasoner
pub fn builtin_multimodal_reasoner_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("multimodal_reasoner_create requires 1 argument: grounding".to_string());
    }

    let grounding = match &args[0] {
        Value::VisualGrounding(g) => g.as_ref().clone(),
        _ => return Err("multimodal_reasoner_create: argument must be a VisualGrounding".to_string()),
    };

    use crate::multimodal::cross_modal_reasoning::MultimodalReasoner;
    let reasoner = MultimodalReasoner::new(grounding);
    Ok(Value::MultimodalReasoner(Box::new(reasoner)))
}

/// multimodal_reasoner_reason(reasoner, question, scene) -> MultimodalCoT
/// Reason about a visual question
pub fn builtin_multimodal_reasoner_reason(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("multimodal_reasoner_reason requires 3 arguments: reasoner, question, scene".to_string());
    }

    let reasoner = match &args[0] {
        Value::MultimodalReasoner(r) => r,
        _ => return Err("multimodal_reasoner_reason: first argument must be a MultimodalReasoner".to_string()),
    };

    let question = match &args[1] {
        Value::String(s) => s,
        _ => return Err("multimodal_reasoner_reason: question must be a string".to_string()),
    };

    let scene = match &args[2] {
        Value::SceneGraph(s) => s,
        _ => return Err("multimodal_reasoner_reason: third argument must be a SceneGraph".to_string()),
    };

    let cot = reasoner.reason(question, scene);
    Ok(Value::MultimodalCoT(Box::new(cot)))
}

/// multimodal_cot_get_info(cot) -> [question, answer, confidence, num_steps]
/// Get information from a multimodal Chain-of-Thought
pub fn builtin_multimodal_cot_get_info(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("multimodal_cot_get_info requires 1 argument: cot".to_string());
    }

    let cot = match &args[0] {
        Value::MultimodalCoT(c) => c,
        _ => return Err("multimodal_cot_get_info: argument must be a MultimodalCoT".to_string()),
    };

    Ok(Value::Array(vec![
        Value::String(cot.question.clone()),
        Value::String(cot.answer.clone()),
        Value::Float(cot.confidence as f64),
        Value::Integer(cot.steps.len() as i64),
    ]))
}

/// multimodal_cot_get_step(cot, step_num) -> [step_number, thought, modality, confidence]
/// Get a specific reasoning step
pub fn builtin_multimodal_cot_get_step(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("multimodal_cot_get_step requires 2 arguments: cot, step_num".to_string());
    }

    let cot = match &args[0] {
        Value::MultimodalCoT(c) => c,
        _ => return Err("multimodal_cot_get_step: first argument must be a MultimodalCoT".to_string()),
    };

    let step_num = match &args[1] {
        Value::Integer(i) => *i as usize,
        _ => return Err("multimodal_cot_get_step: step_num must be an integer".to_string()),
    };

    if step_num >= cot.steps.len() {
        return Err(format!(
            "multimodal_cot_get_step: step {} out of range (0-{})",
            step_num,
            cot.steps.len() - 1
        ));
    }

    let step = &cot.steps[step_num];
    use crate::multimodal::cross_modal_reasoning::ReasoningModality;
    let modality_str = match step.modality {
        ReasoningModality::Visual => "visual",
        ReasoningModality::Textual => "textual",
        ReasoningModality::CrossModal => "cross_modal",
    };

    Ok(Value::Array(vec![
        Value::Integer(step.step_number as i64),
        Value::String(step.thought.clone()),
        Value::String(modality_str.to_string()),
        Value::Float(step.confidence as f64),
    ]))
}

// ============================================================================
// Week 20-21: META-LEARNING
// ============================================================================

// MAML (Model-Agnostic Meta-Learning) BUILTINS

/// maml_create(param_shapes_array, inner_lr, outer_lr, inner_steps) -> MAML
/// Create a MAML meta-learner
/// param_shapes_array: array of [rows, cols] for each layer
pub fn builtin_maml_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("maml_create requires 4 arguments: param_shapes, inner_lr, outer_lr, inner_steps".to_string());
    }

    let param_shapes = match &args[0] {
        Value::Array(arr) => {
            let mut shapes = Vec::new();
            for val in arr {
                match val {
                    Value::Array(pair) => {
                        if pair.len() != 2 {
                            return Err("maml_create: each shape must be [rows, cols]".to_string());
                        }
                        let rows = pair[0].to_integer()? as usize;
                        let cols = pair[1].to_integer()? as usize;
                        shapes.push((rows, cols));
                    }
                    _ => return Err("maml_create: param_shapes must be array of [rows, cols]".to_string()),
                }
            }
            shapes
        }
        _ => return Err("maml_create: param_shapes must be an array".to_string()),
    };

    let inner_lr = args[1].to_float()? as f32;
    let outer_lr = args[2].to_float()? as f32;
    let inner_steps = args[3].to_integer()? as usize;

    use crate::meta_learning::maml::MAML;
    let maml = MAML::new(param_shapes, inner_lr, outer_lr, inner_steps);
    Ok(Value::MAML(Box::new(maml)))
}

/// meta_task_create(task_id) -> MetaTask
/// Create a new meta-learning task
pub fn builtin_meta_task_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("meta_task_create requires 1 argument: task_id".to_string());
    }

    let task_id = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("meta_task_create: task_id must be a string".to_string()),
    };

    use crate::meta_learning::maml::MetaTask;
    let task = MetaTask::new(task_id);
    Ok(Value::MetaTask(Box::new(task)))
}

/// meta_task_add_support(task, input_array, target_array) -> MetaTask
/// Add a support example to the task
pub fn builtin_meta_task_add_support(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("meta_task_add_support requires 3 arguments: task, input, target".to_string());
    }

    let task = match &args[0] {
        Value::MetaTask(t) => t.as_ref().clone(),
        _ => return Err("meta_task_add_support: first argument must be a MetaTask".to_string()),
    };

    let input = match &args[1] {
        Value::Array(arr) => {
            arr.iter()
                .map(|v| v.to_float().map(|f| f as f32))
                .collect::<Result<Vec<f32>, _>>()?
        }
        _ => return Err("meta_task_add_support: input must be an array".to_string()),
    };

    let target = match &args[2] {
        Value::Array(arr) => {
            arr.iter()
                .map(|v| v.to_float().map(|f| f as f32))
                .collect::<Result<Vec<f32>, _>>()?
        }
        _ => return Err("meta_task_add_support: target must be an array".to_string()),
    };

    let task = task.add_support(input, target);
    Ok(Value::MetaTask(Box::new(task)))
}

/// meta_task_add_query(task, input_array, target_array) -> MetaTask
/// Add a query example to the task
pub fn builtin_meta_task_add_query(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("meta_task_add_query requires 3 arguments: task, input, target".to_string());
    }

    let task = match &args[0] {
        Value::MetaTask(t) => t.as_ref().clone(),
        _ => return Err("meta_task_add_query: first argument must be a MetaTask".to_string()),
    };

    let input = match &args[1] {
        Value::Array(arr) => {
            arr.iter()
                .map(|v| v.to_float().map(|f| f as f32))
                .collect::<Result<Vec<f32>, _>>()?
        }
        _ => return Err("meta_task_add_query: input must be an array".to_string()),
    };

    let target = match &args[2] {
        Value::Array(arr) => {
            arr.iter()
                .map(|v| v.to_float().map(|f| f as f32))
                .collect::<Result<Vec<f32>, _>>()?
        }
        _ => return Err("meta_task_add_query: target must be an array".to_string()),
    };

    let task = task.add_query(input, target);
    Ok(Value::MetaTask(Box::new(task)))
}

/// model_params_create(shapes_array) -> ModelParams
/// Create model parameters with given shapes
pub fn builtin_model_params_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("model_params_create requires 1 argument: shapes_array".to_string());
    }

    let shapes = match &args[0] {
        Value::Array(arr) => {
            let mut shape_vec = Vec::new();
            for val in arr {
                match val {
                    Value::Array(pair) => {
                        if pair.len() != 2 {
                            return Err("model_params_create: each shape must be [rows, cols]".to_string());
                        }
                        let rows = pair[0].to_integer()? as usize;
                        let cols = pair[1].to_integer()? as usize;
                        shape_vec.push((rows, cols));
                    }
                    _ => return Err("model_params_create: shapes must be array of [rows, cols]".to_string()),
                }
            }
            shape_vec
        }
        _ => return Err("model_params_create: argument must be an array".to_string()),
    };

    use crate::meta_learning::maml::ModelParams;
    let params = ModelParams::xavier_init(shapes);
    Ok(Value::ModelParams(Box::new(params)))
}

/// maml_get_info(maml) -> [inner_lr, outer_lr, inner_steps, num_params]
/// Get MAML configuration info
pub fn builtin_maml_get_info(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("maml_get_info requires 1 argument: maml".to_string());
    }

    let maml = match &args[0] {
        Value::MAML(m) => m,
        _ => return Err("maml_get_info: argument must be a MAML".to_string()),
    };

    Ok(Value::Array(vec![
        Value::Float(maml.inner_lr as f64),
        Value::Float(maml.outer_lr as f64),
        Value::Integer(maml.inner_steps as i64),
        Value::Integer(maml.meta_params.num_params() as i64),
    ]))
}

// PROTOTYPICAL NETWORKS BUILTINS

/// episode_create(n_way, k_shot) -> Episode
/// Create a new few-shot episode (N-way K-shot)
pub fn builtin_episode_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("episode_create requires 2 arguments: n_way, k_shot".to_string());
    }

    let n_way = args[0].to_integer()? as usize;
    let k_shot = args[1].to_integer()? as usize;

    use crate::meta_learning::prototypical::Episode;
    let episode = Episode::new(n_way, k_shot);
    Ok(Value::Episode(Box::new(episode)))
}

/// episode_add_support(episode, input_array, class_id) -> Episode
/// Add a support example to the episode
pub fn builtin_episode_add_support(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("episode_add_support requires 3 arguments: episode, input, class_id".to_string());
    }

    let episode = match &args[0] {
        Value::Episode(e) => e.as_ref().clone(),
        _ => return Err("episode_add_support: first argument must be an Episode".to_string()),
    };

    let input = match &args[1] {
        Value::Array(arr) => {
            arr.iter()
                .map(|v| v.to_float().map(|f| f as f32))
                .collect::<Result<Vec<f32>, _>>()?
        }
        _ => return Err("episode_add_support: input must be an array".to_string()),
    };

    let class_id = args[2].to_integer()? as usize;

    let episode = episode.add_support(input, class_id);
    Ok(Value::Episode(Box::new(episode)))
}

/// episode_add_query(episode, input_array, class_id) -> Episode
/// Add a query example to the episode
pub fn builtin_episode_add_query(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("episode_add_query requires 3 arguments: episode, input, class_id".to_string());
    }

    let episode = match &args[0] {
        Value::Episode(e) => e.as_ref().clone(),
        _ => return Err("episode_add_query: first argument must be an Episode".to_string()),
    };

    let input = match &args[1] {
        Value::Array(arr) => {
            arr.iter()
                .map(|v| v.to_float().map(|f| f as f32))
                .collect::<Result<Vec<f32>, _>>()?
        }
        _ => return Err("episode_add_query: input must be an array".to_string()),
    };

    let class_id = args[2].to_integer()? as usize;

    let episode = episode.add_query(input, class_id);
    Ok(Value::Episode(Box::new(episode)))
}

/// episode_validate(episode) -> boolean
/// Validate episode structure (returns true if valid, error if not)
pub fn builtin_episode_validate(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("episode_validate requires 1 argument: episode".to_string());
    }

    let episode = match &args[0] {
        Value::Episode(e) => e,
        _ => return Err("episode_validate: argument must be an Episode".to_string()),
    };

    match episode.validate() {
        Ok(_) => Ok(Value::Boolean(true)),
        Err(e) => Err(format!("Episode validation failed: {}", e)),
    }
}

/// prototypical_network_create(embedding_dim, distance_metric) -> PrototypicalNetwork
/// Create a prototypical network
/// distance_metric: "euclidean", "cosine", or "manhattan"
pub fn builtin_prototypical_network_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("prototypical_network_create requires 2 arguments: embedding_dim, distance_metric".to_string());
    }

    let embedding_dim = args[0].to_integer()? as usize;

    let metric_str = match &args[1] {
        Value::String(s) => s.as_str(),
        _ => return Err("prototypical_network_create: distance_metric must be a string".to_string()),
    };

    use crate::meta_learning::prototypical::{DistanceMetric, PrototypicalNetwork};
    let metric = match metric_str {
        "euclidean" => DistanceMetric::Euclidean,
        "cosine" => DistanceMetric::Cosine,
        "manhattan" => DistanceMetric::Manhattan,
        _ => return Err(format!("prototypical_network_create: unknown metric '{}'. Use 'euclidean', 'cosine', or 'manhattan'", metric_str)),
    };

    let network = PrototypicalNetwork::new(embedding_dim, metric);
    Ok(Value::PrototypicalNetwork(Box::new(network)))
}

/// distance_compute(metric_name, emb1_array, emb2_array) -> float
/// Compute distance between two embeddings
pub fn builtin_distance_compute(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("distance_compute requires 3 arguments: metric_name, emb1, emb2".to_string());
    }

    let metric_str = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => return Err("distance_compute: metric_name must be a string".to_string()),
    };

    let emb1 = match &args[1] {
        Value::Array(arr) => {
            arr.iter()
                .map(|v| v.to_float().map(|f| f as f32))
                .collect::<Result<Vec<f32>, _>>()?
        }
        _ => return Err("distance_compute: emb1 must be an array".to_string()),
    };

    let emb2 = match &args[2] {
        Value::Array(arr) => {
            arr.iter()
                .map(|v| v.to_float().map(|f| f as f32))
                .collect::<Result<Vec<f32>, _>>()?
        }
        _ => return Err("distance_compute: emb2 must be an array".to_string()),
    };

    use crate::meta_learning::prototypical::DistanceMetric;
    let metric = match metric_str {
        "euclidean" => DistanceMetric::Euclidean,
        "cosine" => DistanceMetric::Cosine,
        "manhattan" => DistanceMetric::Manhattan,
        _ => return Err(format!("distance_compute: unknown metric '{}'", metric_str)),
    };

    let distance = metric.distance(&emb1, &emb2);
    Ok(Value::Float(distance as f64))
}

/// episode_get_info(episode) -> [n_way, k_shot, num_support, num_query]
/// Get episode information
pub fn builtin_episode_get_info(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("episode_get_info requires 1 argument: episode".to_string());
    }

    let episode = match &args[0] {
        Value::Episode(e) => e,
        _ => return Err("episode_get_info: argument must be an Episode".to_string()),
    };

    Ok(Value::Array(vec![
        Value::Integer(episode.n_way as i64),
        Value::Integer(episode.k_shot as i64),
        Value::Integer(episode.support_set.len() as i64),
        Value::Integer(episode.query_set.len() as i64),
    ]))
}

// ============================================================================
// WEEK 22-24: KNOWLEDGE GRAPHS & GNN
// ============================================================================

// Knowledge Graph - Core Operations
// ---------------------------------

/// Create an empty knowledge graph
/// Usage: kg_create()
pub fn builtin_kg_create(_args: Vec<Value>) -> Result<Value, String> {
    use crate::knowledge_graph::KnowledgeGraph;
    let graph = KnowledgeGraph::new();
    Ok(Value::KnowledgeGraph(Box::new(graph)))
}

/// Add entity to knowledge graph
/// Usage: kg_add_entity(graph, entity_type, name)
/// Returns: entity_id
pub fn builtin_kg_add_entity(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("kg_add_entity requires 3 arguments: graph, entity_type, name".to_string());
    }

    let mut graph = match &args[0] {
        Value::KnowledgeGraph(g) => (**g).clone(),
        _ => return Err("kg_add_entity: first argument must be a KnowledgeGraph".to_string()),
    };

    let entity_type_str = match &args[1] {
        Value::String(s) => s.as_str(),
        _ => return Err("kg_add_entity: entity_type must be a string".to_string()),
    };

    let name = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err("kg_add_entity: name must be a string".to_string()),
    };

    use crate::knowledge_graph::EntityType;
    // Convert to lowercase for case-insensitive matching
    let entity_type_lower = entity_type_str.to_lowercase();
    let entity_type = match entity_type_lower.as_str() {
        "class" => EntityType::Class,
        "function" | "func" => EntityType::Function,
        "method" => EntityType::Method,
        "variable" | "var" => EntityType::Variable,
        "module" | "mod" => EntityType::Module,
        "package" | "pkg" => EntityType::Package,
        "interface" => EntityType::Interface,
        "trait" => EntityType::Trait,
        "struct" => EntityType::Struct,
        "enum" => EntityType::Enum,
        "type" => EntityType::Type,
        "concept" => EntityType::Concept,
        _ => return Err(format!(
            "kg_add_entity: unknown entity type '{}'. Supported types: class, function, method, variable, module, package, interface, trait, struct, enum, type, concept",
            entity_type_str
        )),
    };

    let _entity_id = graph.add_entity(entity_type, name);

    // Return updated graph (entity IDs are sequential: 0, 1, 2, ...)
    Ok(Value::KnowledgeGraph(Box::new(graph)))
}

/// Add triple to knowledge graph
/// Usage: kg_add_triple(graph, subject_id, predicate, object_id)
pub fn builtin_kg_add_triple(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("kg_add_triple requires 4 arguments: graph, subject, predicate, object".to_string());
    }

    let mut graph = match &args[0] {
        Value::KnowledgeGraph(g) => (**g).clone(),
        _ => return Err("kg_add_triple: first argument must be a KnowledgeGraph".to_string()),
    };

    let subject = args[1].to_integer()? as usize;

    let predicate_str = match &args[2] {
        Value::String(s) => s.as_str(),
        _ => return Err("kg_add_triple: predicate must be a string".to_string()),
    };

    let object = args[3].to_integer()? as usize;

    use crate::knowledge_graph::{RelationType, Triple};
    let predicate = RelationType::from_str(predicate_str);
    let triple = Triple::new(subject, predicate, object);

    graph.add_triple(triple);

    Ok(Value::KnowledgeGraph(Box::new(graph)))
}

/// Add triple with confidence score
/// Usage: kg_add_triple_conf(graph, subject_id, predicate, object_id, confidence)
pub fn builtin_kg_add_triple_conf(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 5 {
        return Err("kg_add_triple_conf requires 5 arguments: graph, subject, predicate, object, confidence".to_string());
    }

    let mut graph = match &args[0] {
        Value::KnowledgeGraph(g) => (**g).clone(),
        _ => return Err("kg_add_triple_conf: first argument must be a KnowledgeGraph".to_string()),
    };

    let subject = args[1].to_integer()? as usize;

    let predicate_str = match &args[2] {
        Value::String(s) => s.as_str(),
        _ => return Err("kg_add_triple_conf: predicate must be a string".to_string()),
    };

    let object = args[3].to_integer()? as usize;
    let confidence = args[4].to_float()?;

    use crate::knowledge_graph::{RelationType, Triple};
    let predicate = RelationType::from_str(predicate_str);
    let triple = Triple::with_confidence(subject, predicate, object, confidence);

    graph.add_triple(triple);

    Ok(Value::KnowledgeGraph(Box::new(graph)))
}

/// Query triples by pattern (use -1 for wildcard)
/// Usage: kg_query(graph, subject_id, predicate, object_id)
/// Returns: array of matching triples
pub fn builtin_kg_query(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("kg_query requires 4 arguments: graph, subject, predicate, object".to_string());
    }

    let graph = match &args[0] {
        Value::KnowledgeGraph(g) => g,
        _ => return Err("kg_query: first argument must be a KnowledgeGraph".to_string()),
    };

    // -1 means wildcard (None)
    let subject = {
        let val = args[1].to_integer()?;
        if val < 0 { None } else { Some(val as usize) }
    };

    let predicate = match &args[2] {
        Value::String(s) => {
            use crate::knowledge_graph::RelationType;
            Some(RelationType::from_str(s.as_str()))
        },
        Value::Integer(i) if *i < 0 => None,
        _ => return Err("kg_query: predicate must be a string or -1".to_string()),
    };

    let object = {
        let val = args[3].to_integer()?;
        if val < 0 { None } else { Some(val as usize) }
    };

    let results = graph.query(subject, predicate.as_ref(), object);

    // Convert to array of triples
    let triple_values: Vec<Value> = results
        .into_iter()
        .map(|t| Value::KGTriple(Box::new(t.clone())))
        .collect();

    Ok(Value::Array(triple_values))
}

/// Get entity by ID
/// Usage: kg_get_entity(graph, entity_id)
pub fn builtin_kg_get_entity(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("kg_get_entity requires 2 arguments: graph, entity_id".to_string());
    }

    let graph = match &args[0] {
        Value::KnowledgeGraph(g) => g,
        _ => return Err("kg_get_entity: first argument must be a KnowledgeGraph".to_string()),
    };

    let entity_id = args[1].to_integer()? as usize;

    match graph.get_entity(entity_id) {
        Some(entity) => Ok(Value::KGEntity(Box::new(entity.clone()))),
        None => Ok(Value::Null),
    }
}

/// Find entities by name
/// Usage: kg_find_by_name(graph, name)
pub fn builtin_kg_find_by_name(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("kg_find_by_name requires 2 arguments: graph, name".to_string());
    }

    let graph = match &args[0] {
        Value::KnowledgeGraph(g) => g,
        _ => return Err("kg_find_by_name: first argument must be a KnowledgeGraph".to_string()),
    };

    let name = match &args[1] {
        Value::String(s) => s.as_str(),
        _ => return Err("kg_find_by_name: name must be a string".to_string()),
    };

    let entities = graph.find_entities_by_name(name);
    let entity_values: Vec<Value> = entities
        .into_iter()
        .map(|e| Value::KGEntity(Box::new(e.clone())))
        .collect();

    Ok(Value::Array(entity_values))
}

/// Find entities by type
/// Usage: kg_find_by_type(graph, entity_type)
pub fn builtin_kg_find_by_type(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("kg_find_by_type requires 2 arguments: graph, entity_type".to_string());
    }

    let graph = match &args[0] {
        Value::KnowledgeGraph(g) => g,
        _ => return Err("kg_find_by_type: first argument must be a KnowledgeGraph".to_string()),
    };

    let type_str = match &args[1] {
        Value::String(s) => s.as_str(),
        _ => return Err("kg_find_by_type: entity_type must be a string".to_string()),
    };

    use crate::knowledge_graph::EntityType;
    let entity_type = match type_str {
        "class" => EntityType::Class,
        "function" => EntityType::Function,
        "method" => EntityType::Method,
        "variable" => EntityType::Variable,
        "module" => EntityType::Module,
        "package" => EntityType::Package,
        "interface" => EntityType::Interface,
        "trait" => EntityType::Trait,
        "struct" => EntityType::Struct,
        "enum" => EntityType::Enum,
        "type" => EntityType::Type,
        "concept" => EntityType::Concept,
        _ => return Err(format!("kg_find_by_type: unknown entity type '{}'", type_str)),
    };

    let entities = graph.find_entities_by_type(&entity_type);
    let entity_values: Vec<Value> = entities
        .into_iter()
        .map(|e| Value::KGEntity(Box::new(e.clone())))
        .collect();

    Ok(Value::Array(entity_values))
}

/// Get related entities
/// Usage: kg_get_related(graph, entity_id, predicate)
pub fn builtin_kg_get_related(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("kg_get_related requires 3 arguments: graph, entity_id, predicate".to_string());
    }

    let graph = match &args[0] {
        Value::KnowledgeGraph(g) => g,
        _ => return Err("kg_get_related: first argument must be a KnowledgeGraph".to_string()),
    };

    let entity_id = args[1].to_integer()? as usize;

    let predicate_str = match &args[2] {
        Value::String(s) => s.as_str(),
        _ => return Err("kg_get_related: predicate must be a string".to_string()),
    };

    use crate::knowledge_graph::RelationType;
    let predicate = RelationType::from_str(predicate_str);

    let related_ids = graph.get_related(entity_id, &predicate);
    let id_values: Vec<Value> = related_ids
        .into_iter()
        .map(|id| Value::Integer(id as i64))
        .collect();

    Ok(Value::Array(id_values))
}

/// Find all paths between two entities
/// Usage: kg_find_paths(graph, from_id, to_id, max_depth)
pub fn builtin_kg_find_paths(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("kg_find_paths requires 4 arguments: graph, from_id, to_id, max_depth".to_string());
    }

    let graph = match &args[0] {
        Value::KnowledgeGraph(g) => g,
        _ => return Err("kg_find_paths: first argument must be a KnowledgeGraph".to_string()),
    };

    let from_id = args[1].to_integer()? as usize;
    let to_id = args[2].to_integer()? as usize;
    let max_depth = args[3].to_integer()? as usize;

    let paths = graph.find_paths(from_id, to_id, max_depth);

    // Convert paths to arrays of entity IDs
    let path_values: Vec<Value> = paths
        .into_iter()
        .map(|path| {
            let ids: Vec<Value> = path
                .into_iter()
                .map(|id| Value::Integer(id as i64))
                .collect();
            Value::Array(ids)
        })
        .collect();

    Ok(Value::Array(path_values))
}

/// Get graph statistics
/// Usage: kg_get_stats(graph)
/// Returns: [num_entities, num_triples, num_relations]
pub fn builtin_kg_get_stats(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("kg_get_stats requires 1 argument: graph".to_string());
    }

    let graph = match &args[0] {
        Value::KnowledgeGraph(g) => g,
        _ => return Err("kg_get_stats: argument must be a KnowledgeGraph".to_string()),
    };

    let stats = graph.stats();

    Ok(Value::Array(vec![
        Value::Integer(stats.num_entities as i64),
        Value::Integer(stats.num_triples as i64),
        Value::Integer(stats.num_relations as i64),
    ]))
}

// Entity Operations
// ----------------

/// Get entity ID
/// Usage: entity_get_id(entity)
pub fn builtin_entity_get_id(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("entity_get_id requires 1 argument: entity".to_string());
    }

    let entity = match &args[0] {
        Value::KGEntity(e) => e,
        _ => return Err("entity_get_id: argument must be a KGEntity".to_string()),
    };

    Ok(Value::Integer(entity.id as i64))
}

/// Get entity type
/// Usage: entity_get_type(entity)
pub fn builtin_entity_get_type(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("entity_get_type requires 1 argument: entity".to_string());
    }

    let entity = match &args[0] {
        Value::KGEntity(e) => e,
        _ => return Err("entity_get_type: argument must be a KGEntity".to_string()),
    };

    let type_str = format!("{}", entity.entity_type);
    Ok(Value::String(type_str))
}

/// Get entity name
/// Usage: entity_get_name(entity)
pub fn builtin_entity_get_name(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("entity_get_name requires 1 argument: entity".to_string());
    }

    let entity = match &args[0] {
        Value::KGEntity(e) => e,
        _ => return Err("entity_get_name: argument must be a KGEntity".to_string()),
    };

    Ok(Value::String(entity.name.clone()))
}

// Triple Operations
// ----------------

/// Get triple subject
/// Usage: triple_get_subject(triple)
pub fn builtin_triple_get_subject(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("triple_get_subject requires 1 argument: triple".to_string());
    }

    let triple = match &args[0] {
        Value::KGTriple(t) => t,
        _ => return Err("triple_get_subject: argument must be a KGTriple".to_string()),
    };

    Ok(Value::Integer(triple.subject as i64))
}

/// Get triple predicate
/// Usage: triple_get_predicate(triple)
pub fn builtin_triple_get_predicate(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("triple_get_predicate requires 1 argument: triple".to_string());
    }

    let triple = match &args[0] {
        Value::KGTriple(t) => t,
        _ => return Err("triple_get_predicate: argument must be a KGTriple".to_string()),
    };

    let predicate_str = format!("{}", triple.predicate);
    Ok(Value::String(predicate_str))
}

/// Get triple object
/// Usage: triple_get_object(triple)
pub fn builtin_triple_get_object(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("triple_get_object requires 1 argument: triple".to_string());
    }

    let triple = match &args[0] {
        Value::KGTriple(t) => t,
        _ => return Err("triple_get_object: argument must be a KGTriple".to_string()),
    };

    Ok(Value::Integer(triple.object as i64))
}

// Graph Neural Network Operations
// -------------------------------

/// Create a Graph Neural Network
/// Usage: gnn_create(embedding_dim, num_heads)
pub fn builtin_gnn_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("gnn_create requires 2 arguments: embedding_dim, num_heads".to_string());
    }

    let embedding_dim = args[0].to_integer()? as usize;
    let num_heads = args[1].to_integer()? as usize;

    use crate::knowledge_graph::GraphNeuralNetwork;
    let gnn = GraphNeuralNetwork::new(embedding_dim, num_heads)?;

    Ok(Value::GraphNeuralNetwork(Box::new(gnn)))
}

/// Initialize node embeddings for a graph
/// Usage: gnn_init_embeddings(gnn, graph)
pub fn builtin_gnn_init_embeddings(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("gnn_init_embeddings requires 2 arguments: gnn, graph".to_string());
    }

    let gnn = match &args[0] {
        Value::GraphNeuralNetwork(g) => g,
        _ => return Err("gnn_init_embeddings: first argument must be a GraphNeuralNetwork".to_string()),
    };

    let graph = match &args[1] {
        Value::KnowledgeGraph(kg) => kg,
        _ => return Err("gnn_init_embeddings: second argument must be a KnowledgeGraph".to_string()),
    };

    let embeddings = gnn.initialize_node_embeddings(graph);

    Ok(Value::NodeEmbeddings(Box::new(embeddings)))
}

/// GNN forward pass
/// Usage: gnn_forward(gnn, graph, embeddings)
pub fn builtin_gnn_forward(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("gnn_forward requires 3 arguments: gnn, graph, embeddings".to_string());
    }

    let gnn = match &args[0] {
        Value::GraphNeuralNetwork(g) => g,
        _ => return Err("gnn_forward: first argument must be a GraphNeuralNetwork".to_string()),
    };

    let graph = match &args[1] {
        Value::KnowledgeGraph(kg) => kg,
        _ => return Err("gnn_forward: second argument must be a KnowledgeGraph".to_string()),
    };

    let embeddings = match &args[2] {
        Value::NodeEmbeddings(e) => e,
        _ => return Err("gnn_forward: third argument must be NodeEmbeddings".to_string()),
    };

    let updated_embeddings = gnn.forward(graph, embeddings)?;

    Ok(Value::NodeEmbeddings(Box::new(updated_embeddings)))
}

/// GNN multi-layer forward pass
/// Usage: gnn_forward_multilayer(gnn, graph, embeddings, num_layers)
pub fn builtin_gnn_forward_multilayer(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("gnn_forward_multilayer requires 4 arguments: gnn, graph, embeddings, num_layers".to_string());
    }

    let gnn = match &args[0] {
        Value::GraphNeuralNetwork(g) => g,
        _ => return Err("gnn_forward_multilayer: first argument must be a GraphNeuralNetwork".to_string()),
    };

    let graph = match &args[1] {
        Value::KnowledgeGraph(kg) => kg,
        _ => return Err("gnn_forward_multilayer: second argument must be a KnowledgeGraph".to_string()),
    };

    let embeddings = match &args[2] {
        Value::NodeEmbeddings(e) => e,
        _ => return Err("gnn_forward_multilayer: third argument must be NodeEmbeddings".to_string()),
    };

    let num_layers = args[3].to_integer()? as usize;

    let updated_embeddings = gnn.forward_multilayer(graph, embeddings, num_layers)?;

    Ok(Value::NodeEmbeddings(Box::new(updated_embeddings)))
}
// ============================================================================
// WEEK 29-30: SYMBOLIC AI
// ============================================================================

// Rule Engine Operations
// ----------------------

/// Create a new symbolic rule
/// Usage: rule_create(name, description)
pub fn builtin_rule_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("rule_create requires 2 arguments: name, description".to_string());
    }

    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("rule_create: name must be a string".to_string()),
    };

    let description = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("rule_create: description must be a string".to_string()),
    };

    use crate::symbolic::Rule;
    let rule = Rule::new(name).description(description);

    Ok(Value::SymbolicRule(Box::new(rule)))
}

/// Create a rule engine
/// Usage: rule_engine_create()
pub fn builtin_rule_engine_create(_args: Vec<Value>) -> Result<Value, String> {
    use crate::symbolic::RuleEngine;
    let engine = RuleEngine::new();
    Ok(Value::RuleEngine(Box::new(engine)))
}

/// Add rule to engine
/// Usage: rule_engine_add_rule(engine, rule)
pub fn builtin_rule_engine_add_rule(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("rule_engine_add_rule requires 2 arguments: engine, rule".to_string());
    }

    let mut engine = match &args[0] {
        Value::RuleEngine(e) => (**e).clone(),
        _ => return Err("rule_engine_add_rule: first argument must be a RuleEngine".to_string()),
    };

    let rule = match &args[1] {
        Value::SymbolicRule(r) => (**r).clone(),
        _ => return Err("rule_engine_add_rule: second argument must be a Rule".to_string()),
    };

    engine.add_rule(rule);
    Ok(Value::RuleEngine(Box::new(engine)))
}

/// Get rule count in engine
/// Usage: rule_engine_count(engine)
pub fn builtin_rule_engine_count(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("rule_engine_count requires 1 argument: engine".to_string());
    }

    let engine = match &args[0] {
        Value::RuleEngine(e) => e,
        _ => return Err("rule_engine_count: argument must be a RuleEngine".to_string()),
    };

    Ok(Value::Integer(engine.rules().len() as i64))
}

// First-Order Logic (FOL) Operations
// ----------------------------------

/// Create FOL variable term
/// Usage: fol_var(name)
pub fn builtin_fol_var(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("fol_var requires 1 argument: name".to_string());
    }

    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("fol_var: name must be a string".to_string()),
    };

    use crate::symbolic::Term;
    let term = Term::variable(name);
    Ok(Value::FOLTerm(Box::new(term)))
}

/// Create FOL constant term
/// Usage: fol_const(name)
pub fn builtin_fol_const(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("fol_const requires 1 argument: name".to_string());
    }

    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("fol_const: name must be a string".to_string()),
    };

    use crate::symbolic::Term;
    let term = Term::constant(name);
    Ok(Value::FOLTerm(Box::new(term)))
}

/// Create FOL function term
/// Usage: fol_func(name, args_array)
pub fn builtin_fol_func(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("fol_func requires 2 arguments: name, args".to_string());
    }

    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("fol_func: name must be a string".to_string()),
    };

    let term_args = match &args[1] {
        Value::Array(arr) => {
            let mut terms = Vec::new();
            for val in arr {
                match val {
                    Value::FOLTerm(t) => terms.push((**t).clone()),
                    _ => return Err("fol_func: all args must be FOL terms".to_string()),
                }
            }
            terms
        }
        _ => return Err("fol_func: args must be an array".to_string()),
    };

    use crate::symbolic::Term;
    let term = Term::function(name, term_args);
    Ok(Value::FOLTerm(Box::new(term)))
}

/// Create FOL solver
/// Usage: fol_solver_create()
pub fn builtin_fol_solver_create(_args: Vec<Value>) -> Result<Value, String> {
    use crate::symbolic::FOLSolver;
    let solver = FOLSolver::new();
    Ok(Value::FOLSolver(Box::new(solver)))
}

/// Unify two FOL terms
/// Usage: fol_unify(term1, term2)
/// Returns: true if unifiable, false otherwise
pub fn builtin_fol_unify(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("fol_unify requires 2 arguments: term1, term2".to_string());
    }

    let term1 = match &args[0] {
        Value::FOLTerm(t) => t,
        _ => return Err("fol_unify: first argument must be a FOL term".to_string()),
    };

    let term2 = match &args[1] {
        Value::FOLTerm(t) => t,
        _ => return Err("fol_unify: second argument must be a FOL term".to_string()),
    };

    use crate::symbolic::{unify, UnificationResult};
    match unify(term1, term2) {
        UnificationResult::Success(_subst) => Ok(Value::Boolean(true)),
        UnificationResult::Failure => Ok(Value::Boolean(false)),
    }
}

// Fuzzy Logic Operations
// ----------------------

/// Create fuzzy value
/// Usage: fuzzy_create(value, label)
/// value: [0.0, 1.0], label: "low", "medium", "high", etc.
pub fn builtin_fuzzy_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("fuzzy_create requires 2 arguments: value, label".to_string());
    }

    let value = args[0].to_float()? as f64;

    let label = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("fuzzy_create: label must be a string".to_string()),
    };

    if !(0.0..=1.0).contains(&value) {
        return Err("fuzzy_create: value must be in [0.0, 1.0]".to_string());
    }

    use crate::symbolic::FuzzyValue;
    let fuzzy = FuzzyValue::new(value);
    Ok(Value::FuzzyValue(Box::new(fuzzy)))
}

/// Fuzzy AND (minimum)
/// Usage: fuzzy_and(val1, val2)
pub fn builtin_fuzzy_and(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("fuzzy_and requires 2 arguments: val1, val2".to_string());
    }

    let val1 = match &args[0] {
        Value::FuzzyValue(f) => f.value(),
        Value::Float(f) => *f as f64,
        Value::Integer(i) => *i as f64,
        _ => return Err("fuzzy_and: arguments must be fuzzy values or numbers".to_string()),
    };

    let val2 = match &args[1] {
        Value::FuzzyValue(f) => f.value(),
        Value::Float(f) => *f as f64,
        Value::Integer(i) => *i as f64,
        _ => return Err("fuzzy_and: arguments must be fuzzy values or numbers".to_string()),
    };

    // T-norm: minimum
    let result = val1.min(val2);

    use crate::symbolic::FuzzyValue;
    Ok(Value::FuzzyValue(Box::new(FuzzyValue::new(result))))
}

/// Fuzzy OR (maximum)
/// Usage: fuzzy_or(val1, val2)
pub fn builtin_fuzzy_or(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("fuzzy_or requires 2 arguments: val1, val2".to_string());
    }

    let val1 = match &args[0] {
        Value::FuzzyValue(f) => f.value(),
        Value::Float(f) => *f as f64,
        Value::Integer(i) => *i as f64,
        _ => return Err("fuzzy_or: arguments must be fuzzy values or numbers".to_string()),
    };

    let val2 = match &args[1] {
        Value::FuzzyValue(f) => f.value(),
        Value::Float(f) => *f as f64,
        Value::Integer(i) => *i as f64,
        _ => return Err("fuzzy_or: arguments must be fuzzy values or numbers".to_string()),
    };

    // T-conorm: maximum
    let result = val1.max(val2);

    use crate::symbolic::FuzzyValue;
    Ok(Value::FuzzyValue(Box::new(FuzzyValue::new(result))))
}

/// Fuzzy NOT
/// Usage: fuzzy_not(val)
pub fn builtin_fuzzy_not(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("fuzzy_not requires 1 argument: val".to_string());
    }

    let val = match &args[0] {
        Value::FuzzyValue(f) => f.value(),
        Value::Float(f) => *f as f64,
        Value::Integer(i) => *i as f64,
        _ => return Err("fuzzy_not: argument must be a fuzzy value or number".to_string()),
    };

    let result = 1.0 - val;

    use crate::symbolic::FuzzyValue;
    Ok(Value::FuzzyValue(Box::new(FuzzyValue::new(result))))
}

/// Get fuzzy value
/// Usage: fuzzy_get_value(fuzzy)
pub fn builtin_fuzzy_get_value(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("fuzzy_get_value requires 1 argument: fuzzy".to_string());
    }

    let fuzzy = match &args[0] {
        Value::FuzzyValue(f) => f,
        _ => return Err("fuzzy_get_value: argument must be a FuzzyValue".to_string()),
    };

    Ok(Value::Float(fuzzy.value() as f64))
}

// Concept Learning Operations
// ---------------------------

/// Create a concept
/// Usage: concept_create(name, properties_array)
pub fn builtin_concept_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("concept_create requires 2 arguments: name, properties".to_string());
    }

    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("concept_create: name must be a string".to_string()),
    };

    let properties = match &args[1] {
        Value::Array(arr) => {
            let mut props = Vec::new();
            for val in arr {
                match val {
                    Value::String(s) => props.push(s.clone()),
                    _ => return Err("concept_create: all properties must be strings".to_string()),
                }
            }
            props
        }
        _ => return Err("concept_create: properties must be an array".to_string()),
    };

    use crate::symbolic::Concept;
    let mut concept = Concept::new(name);
    for prop in properties {
        concept = concept.with_property(prop, 1.0);
    }

    Ok(Value::Concept(Box::new(concept)))
}

/// Create concept graph
/// Usage: concept_graph_create()
pub fn builtin_concept_graph_create(_args: Vec<Value>) -> Result<Value, String> {
    use crate::symbolic::ConceptGraph;
    let graph = ConceptGraph::new();
    Ok(Value::ConceptGraph(Box::new(graph)))
}

/// Add concept to graph
/// Usage: concept_graph_add(graph, concept)
pub fn builtin_concept_graph_add(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("concept_graph_add requires 2 arguments: graph, concept".to_string());
    }

    let mut graph = match &args[0] {
        Value::ConceptGraph(g) => (**g).clone(),
        _ => return Err("concept_graph_add: first argument must be a ConceptGraph".to_string()),
    };

    let concept = match &args[1] {
        Value::Concept(c) => (**c).clone(),
        _ => return Err("concept_graph_add: second argument must be a Concept".to_string()),
    };

    graph.add_concept(concept);
    Ok(Value::ConceptGraph(Box::new(graph)))
}

/// Get concept count in graph
/// Usage: concept_graph_count(graph)
pub fn builtin_concept_graph_count(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("concept_graph_count requires 1 argument: graph".to_string());
    }

    let graph = match &args[0] {
        Value::ConceptGraph(g) => g,
        _ => return Err("concept_graph_count: argument must be a ConceptGraph".to_string()),
    };

    Ok(Value::Integer(graph.num_concepts() as i64))
}

/// Compute similarity between two concepts
/// Usage: concept_similarity(concept1, concept2)
/// Returns a float between 0.0 (no similarity) and 1.0 (identical)
pub fn builtin_concept_similarity(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("concept_similarity requires 2 arguments: concept1, concept2".to_string());
    }

    let concept1 = match &args[0] {
        Value::Concept(c) => c,
        _ => return Err("concept_similarity: first argument must be a Concept".to_string()),
    };

    let concept2 = match &args[1] {
        Value::Concept(c) => c,
        _ => return Err("concept_similarity: second argument must be a Concept".to_string()),
    };

    // Compute Jaccard similarity based on shared properties
    let props1: std::collections::HashSet<_> = concept1.properties.keys().collect();
    let props2: std::collections::HashSet<_> = concept2.properties.keys().collect();

    let intersection = props1.intersection(&props2).count();
    let union = props1.union(&props2).count();

    let similarity = if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    };

    Ok(Value::Float(similarity))
}

// Tree-of-Thoughts Operations
// ---------------------------

/// Create a tree of thoughts
/// Usage: tot_create(root_thought, strategy)
/// strategy: "bfs", "dfs", or "best_first"
pub fn builtin_tot_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("tot_create requires 2 arguments: root_thought, strategy".to_string());
    }

    let root_thought = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("tot_create: root_thought must be a string".to_string()),
    };

    let strategy = match &args[1] {
        Value::String(s) => match s.as_str() {
            "bfs" => crate::reasoning::SearchStrategy::BreadthFirst,
            "dfs" => crate::reasoning::SearchStrategy::DepthFirst,
            "best_first" => crate::reasoning::SearchStrategy::BestFirst,
            _ => return Err(format!("tot_create: unknown strategy '{}'. Use 'bfs', 'dfs', or 'best_first'", s)),
        },
        _ => return Err("tot_create: strategy must be a string".to_string()),
    };

    use crate::reasoning::TreeOfThoughts;
    let tot = TreeOfThoughts::new(root_thought, strategy);
    Ok(Value::TreeOfThoughts(Box::new(tot)))
}

/// Add a thought to the tree
/// Usage: tot_add_thought(tot, parent_id, thought, value)
pub fn builtin_tot_add_thought(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("tot_add_thought requires 4 arguments: tot, parent_id, thought, value".to_string());
    }

    let mut tot = match &args[0] {
        Value::TreeOfThoughts(t) => (**t).clone(),
        _ => return Err("tot_add_thought: first argument must be a TreeOfThoughts".to_string()),
    };

    let parent_id = match &args[1] {
        Value::Integer(i) => *i as usize,
        _ => return Err("tot_add_thought: parent_id must be an integer".to_string()),
    };

    let thought = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err("tot_add_thought: thought must be a string".to_string()),
    };

    let value = match &args[3] {
        Value::Float(f) => *f as f32,
        Value::Integer(i) => *i as f32,
        _ => return Err("tot_add_thought: value must be a number".to_string()),
    };

    tot.add_thought(parent_id, thought, value);

    Ok(Value::TreeOfThoughts(Box::new(tot)))
}

/// Mark a node as a solution
/// Usage: tot_mark_solution(tot, node_id)
pub fn builtin_tot_mark_solution(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("tot_mark_solution requires 2 arguments: tot, node_id".to_string());
    }

    let mut tot = match &args[0] {
        Value::TreeOfThoughts(t) => (**t).clone(),
        _ => return Err("tot_mark_solution: first argument must be a TreeOfThoughts".to_string()),
    };

    let node_id = match &args[1] {
        Value::Integer(i) => *i as usize,
        _ => return Err("tot_mark_solution: node_id must be an integer".to_string()),
    };

    tot.mark_solution(node_id);
    Ok(Value::TreeOfThoughts(Box::new(tot)))
}

/// Get tree statistics
/// Usage: tot_stats(tot) -> [nodes, solutions, max_depth]
pub fn builtin_tot_stats(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("tot_stats requires 1 argument: tot".to_string());
    }

    let tot = match &args[0] {
        Value::TreeOfThoughts(t) => t,
        _ => return Err("tot_stats: argument must be a TreeOfThoughts".to_string()),
    };

    let (nodes, solutions, max_depth) = tot.stats();
    Ok(Value::Array(vec![
        Value::Integer(nodes as i64),
        Value::Integer(solutions as i64),
        Value::Integer(max_depth as i64),
    ]))
}

// Working Memory Operations
// -------------------------

/// Create a memory item
/// Usage: memory_item_create(id, content, type)
/// type: "episodic", "semantic", or "procedural"
pub fn builtin_memory_item_create(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("memory_item_create requires 3 arguments: id, content, type".to_string());
    }

    let id = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("memory_item_create: id must be a string".to_string()),
    };

    let content = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("memory_item_create: content must be a string".to_string()),
    };

    let memory_type = match &args[2] {
        Value::String(s) => match s.as_str() {
            "episodic" => crate::reasoning::MemoryType::Episodic,
            "semantic" => crate::reasoning::MemoryType::Semantic,
            "procedural" => crate::reasoning::MemoryType::Procedural,
            _ => return Err(format!("memory_item_create: unknown type '{}'. Use 'episodic', 'semantic', or 'procedural'", s)),
        },
        _ => return Err("memory_item_create: type must be a string".to_string()),
    };

    use crate::reasoning::MemoryItem;
    let item = MemoryItem::new(id, content, memory_type);
    Ok(Value::MemoryItem(Box::new(item)))
}

/// Create a short-term memory buffer
/// Usage: stm_create(capacity)
pub fn builtin_stm_create(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("stm_create requires 1 argument: capacity".to_string());
    }

    let capacity = match &args[0] {
        Value::Integer(i) => *i as usize,
        _ => return Err("stm_create: capacity must be an integer".to_string()),
    };

    use crate::reasoning::ShortTermMemory;
    let stm = ShortTermMemory::new(capacity);
    Ok(Value::ShortTermMemory(Box::new(stm)))
}

/// Add item to short-term memory
/// Usage: stm_add(stm, item)
pub fn builtin_stm_add(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("stm_add requires 2 arguments: stm, item".to_string());
    }

    let mut stm = match &args[0] {
        Value::ShortTermMemory(s) => (**s).clone(),
        _ => return Err("stm_add: first argument must be a ShortTermMemory".to_string()),
    };

    let item = match &args[1] {
        Value::MemoryItem(i) => (**i).clone(),
        _ => return Err("stm_add: second argument must be a MemoryItem".to_string()),
    };

    stm.add(item);
    Ok(Value::ShortTermMemory(Box::new(stm)))
}

/// Get short-term memory length
/// Usage: stm_len(stm)
pub fn builtin_stm_len(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("stm_len requires 1 argument: stm".to_string());
    }

    let stm = match &args[0] {
        Value::ShortTermMemory(s) => s,
        _ => return Err("stm_len: argument must be a ShortTermMemory".to_string()),
    };

    Ok(Value::Integer(stm.len() as i64))
}

/// Create a long-term memory
/// Usage: ltm_create()
pub fn builtin_ltm_create(_args: Vec<Value>) -> Result<Value, String> {
    use crate::reasoning::LongTermMemory;
    let ltm = LongTermMemory::new();
    Ok(Value::LongTermMemory(Box::new(ltm)))
}

/// Store item in long-term memory
/// Usage: ltm_store(ltm, item)
pub fn builtin_ltm_store(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("ltm_store requires 2 arguments: ltm, item".to_string());
    }

    let mut ltm = match &args[0] {
        Value::LongTermMemory(l) => (**l).clone(),
        _ => return Err("ltm_store: first argument must be a LongTermMemory".to_string()),
    };

    let item = match &args[1] {
        Value::MemoryItem(i) => (**i).clone(),
        _ => return Err("ltm_store: second argument must be a MemoryItem".to_string()),
    };

    ltm.store(item);
    Ok(Value::LongTermMemory(Box::new(ltm)))
}

/// Get long-term memory size
/// Usage: ltm_size(ltm)
pub fn builtin_ltm_size(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("ltm_size requires 1 argument: ltm".to_string());
    }

    let ltm = match &args[0] {
        Value::LongTermMemory(l) => l,
        _ => return Err("ltm_size: argument must be a LongTermMemory".to_string()),
    };

    Ok(Value::Integer(ltm.total_size() as i64))
}

/// Create a working memory system
/// Usage: working_memory_create(capacity)
pub fn builtin_working_memory_create(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("working_memory_create requires 1 argument: capacity".to_string());
    }

    let capacity = match &args[0] {
        Value::Integer(i) => *i as usize,
        _ => return Err("working_memory_create: capacity must be an integer".to_string()),
    };

    use crate::reasoning::WorkingMemorySystem;
    let wm = WorkingMemorySystem::new(capacity);
    Ok(Value::WorkingMemorySystem(Box::new(wm)))
}

/// Remember an item (adds to short-term and potentially consolidates to long-term)
/// Usage: working_memory_remember(wm, item)
pub fn builtin_working_memory_remember(args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("working_memory_remember requires 2 arguments: wm, item".to_string());
    }

    let mut wm = match &args[0] {
        Value::WorkingMemorySystem(w) => (**w).clone(),
        _ => return Err("working_memory_remember: first argument must be a WorkingMemorySystem".to_string()),
    };

    let item = match &args[1] {
        Value::MemoryItem(i) => (**i).clone(),
        _ => return Err("working_memory_remember: second argument must be a MemoryItem".to_string()),
    };

    wm.remember(item);
    Ok(Value::WorkingMemorySystem(Box::new(wm)))
}

/// Consolidate memories (move from short-term to long-term)
/// Usage: working_memory_consolidate(wm)
pub fn builtin_working_memory_consolidate(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("working_memory_consolidate requires 1 argument: wm".to_string());
    }

    let mut wm = match &args[0] {
        Value::WorkingMemorySystem(w) => (**w).clone(),
        _ => return Err("working_memory_consolidate: argument must be a WorkingMemorySystem".to_string()),
    };

    wm.consolidate();
    Ok(Value::WorkingMemorySystem(Box::new(wm)))
}


// Tensor Builtin Functions
// Phase 1: Expose backend tensor operations to Charl language

use crate::autograd::Tensor as AutogradTensor;
use crate::interpreter::Value;
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

/// tensor(data: [float]) -> Tensor
/// Creates a tensor from an array of numbers
pub fn builtin_tensor(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor() expects 1 argument: tensor(data)".to_string());
    }

    match &args[0] {
        Value::Array(arr) => {
            // Convert array of Values to Vec<f64>
            let mut data = Vec::new();
            for val in arr {
                let num = val.to_float()?;
                data.push(num);
            }

            // Create shape [length]
            let shape = vec![data.len()];

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

            // Element-wise addition
            let result_data: Vec<f64> = a
                .data
                .iter()
                .zip(b.data.iter())
                .map(|(x, y)| x + y)
                .collect();

            let result_tensor = AutogradTensor::new(result_data, a.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
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

            let result_data: Vec<f64> = a
                .data
                .iter()
                .zip(b.data.iter())
                .map(|(x, y)| x * y)
                .collect();

            let result_tensor = AutogradTensor::new(result_data, a.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
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

            let result_data: Vec<f64> = tensor.data.iter().map(|x| x * scalar_val).collect();

            let result_tensor = AutogradTensor::new(result_data, tensor.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
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
        Value::AutogradTensor(tensor) => {
            let sum: f64 = tensor.data.iter().sum();
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
        Value::AutogradTensor(tensor) => {
            if tensor.data.is_empty() {
                return Ok(Value::Float(0.0));
            }

            let sum: f64 = tensor.data.iter().sum();
            let mean = sum / tensor.data.len() as f64;
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

    match (&args[0], &args[1]) {
        (Value::AutogradTensor(tensor), Value::Array(shape_vals)) => {
            // Convert shape array to Vec<usize>
            let mut new_shape = Vec::new();
            for val in shape_vals {
                match val {
                    Value::Integer(i) => new_shape.push(*i as usize),
                    _ => return Err("tensor_reshape() shape must contain integers".to_string()),
                }
            }

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
pub fn builtin_tensor_zero_grad(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_zero_grad() expects 1 argument: tensor_zero_grad(tensor)".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(tensor) => {
            let mut new_tensor = tensor.clone();
            new_tensor.zero_grad();
            Ok(Value::AutogradTensor(new_tensor))
        }
        _ => Err("tensor_zero_grad() expects a tensor".to_string()),
    }
}

/// tensor_grad(t: Tensor) -> [float] | null
/// Get the gradient of a tensor
pub fn builtin_tensor_grad(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("tensor_grad() expects 1 argument: tensor_grad(tensor)".to_string());
    }

    match &args[0] {
        Value::AutogradTensor(tensor) => {
            match &tensor.grad {
                Some(grad_data) => {
                    // Convert gradient Vec<f64> to Value::Array
                    let grad_values: Vec<Value> = grad_data
                        .iter()
                        .map(|&g| Value::Float(g))
                        .collect();
                    Ok(Value::Array(grad_values))
                }
                None => Ok(Value::Null),
            }
        }
        _ => Err("tensor_grad() expects a tensor".to_string()),
    }
}

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
        (Value::AutogradTensor(a), Value::AutogradTensor(b)) => {
            // Check shapes match
            if a.shape != b.shape {
                return Err(format!(
                    "tensor_sub(): shape mismatch {:?} vs {:?}",
                    a.shape, b.shape
                ));
            }

            // Element-wise subtraction
            let result_data: Vec<f64> = a
                .data
                .iter()
                .zip(b.data.iter())
                .map(|(x, y)| x - y)
                .collect();

            let result_tensor = AutogradTensor::new(result_data, a.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
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
        // Tensor / Tensor (element-wise)
        (Value::AutogradTensor(a), Value::AutogradTensor(b)) => {
            if a.shape != b.shape {
                return Err(format!(
                    "tensor_div(): shape mismatch {:?} vs {:?}",
                    a.shape, b.shape
                ));
            }

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
        // Tensor / Scalar
        (Value::AutogradTensor(tensor), scalar) => {
            let scalar_val = scalar.to_float()?;

            if scalar_val == 0.0 {
                return Err("tensor_div(): division by zero".to_string());
            }

            let result_data: Vec<f64> = tensor.data.iter().map(|x| x / scalar_val).collect();

            let result_tensor = AutogradTensor::new(result_data, tensor.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
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
        Value::AutogradTensor(tensor) => {
            let result_data: Vec<f64> = tensor.data.iter().map(|&x| x.max(0.0)).collect();
            let result_tensor = AutogradTensor::new(result_data, tensor.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
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
        Value::AutogradTensor(tensor) => {
            let result_data: Vec<f64> = tensor
                .data
                .iter()
                .map(|&x| 1.0 / (1.0 + (-x).exp()))
                .collect();
            let result_tensor = AutogradTensor::new(result_data, tensor.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
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
        Value::AutogradTensor(tensor) => {
            let result_data: Vec<f64> = tensor.data.iter().map(|&x| x.tanh()).collect();
            let result_tensor = AutogradTensor::new(result_data, tensor.shape.clone());
            Ok(Value::AutogradTensor(result_tensor))
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
            // Handle 1D input: reshape to (1, in_features)
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

            // Matrix multiplication: input @ weight
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

            let result_tensor = AutogradTensor::new(result_data, result_shape);
            Ok(Value::AutogradTensor(result_tensor))
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

/// layer_forward(layer: LinearLayer | Conv2dLayer | MaxPool2dLayer | AvgPool2dLayer, input: Tensor) -> Tensor
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
        (Value::LinearLayer(_), _) | (Value::Conv2dLayer(_), _) | (Value::MaxPool2dLayer(_), _) | (Value::AvgPool2dLayer(_), _) => {
            Err(format!("layer_forward() expects tensor input, got {}", args[1].type_name()))
        }
        _ => {
            Err(format!("layer_forward() expects layer (Linear/Conv2d/MaxPool2d/AvgPool2d), got {}", args[0].type_name()))
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

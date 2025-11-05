// GPU-enabled Tensor wrapper
// Extends autograd Tensor with GPU compute capabilities

use crate::autograd::Tensor;
use crate::gpu::{ComputeBackend, TensorBuffer, wgpu_backend::WgpuBackend, cpu::CPUBackend};

/// Device where tensor data resides
#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    CPU,
    GPU,
}

/// GPU-enabled tensor that can move between CPU and GPU
pub struct GPUTensor {
    // Core tensor data (always on CPU for autograd)
    pub tensor: Tensor,

    // GPU buffer (if tensor is on GPU)
    gpu_buffer: Option<TensorBuffer>,

    // Current device location
    device: Device,
}

impl GPUTensor {
    /// Create a new GPU tensor from a regular tensor
    pub fn from_tensor(tensor: Tensor) -> Self {
        GPUTensor {
            tensor,
            gpu_buffer: None,
            device: Device::CPU,
        }
    }

    /// Create a new GPU tensor with data
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self::from_tensor(Tensor::new(data, shape))
    }

    /// Create a GPU tensor that requires gradients
    pub fn with_grad(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self::from_tensor(Tensor::with_grad(data, shape))
    }

    /// Get current device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if tensor is on GPU
    pub fn is_gpu(&self) -> bool {
        self.device == Device::GPU
    }

    /// Move tensor to GPU
    pub fn to_gpu(&mut self, backend: &mut dyn ComputeBackend) -> Result<(), String> {
        if self.is_gpu() {
            return Ok(()); // Already on GPU
        }

        // Convert f64 to f32 for GPU
        let data_f32: Vec<f32> = self.tensor.data.iter().map(|&x| x as f32).collect();

        // Allocate GPU buffer
        let buffer = backend.allocate(data_f32.len())
            .map_err(|e| format!("Failed to allocate GPU buffer: {:?}", e))?;

        // Copy data to GPU
        backend.copy_to_device(&data_f32, &buffer)
            .map_err(|e| format!("Failed to copy to GPU: {:?}", e))?;

        self.gpu_buffer = Some(buffer);
        self.device = Device::GPU;

        Ok(())
    }

    /// Move tensor back to CPU
    pub fn to_cpu(&mut self, backend: &mut dyn ComputeBackend) -> Result<(), String> {
        if !self.is_gpu() {
            return Ok(()); // Already on CPU
        }

        let buffer = self.gpu_buffer.as_ref()
            .ok_or_else(|| "No GPU buffer available".to_string())?;

        // Allocate temp f32 buffer
        let mut data_f32 = vec![0.0f32; self.tensor.data.len()];

        // Copy from GPU to CPU
        backend.copy_from_device(buffer, &mut data_f32)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        // Convert f32 back to f64
        self.tensor.data = data_f32.iter().map(|&x| x as f64).collect();

        // Deallocate GPU buffer
        backend.deallocate(buffer.clone())
            .map_err(|e| format!("Failed to deallocate GPU buffer: {:?}", e))?;

        self.gpu_buffer = None;
        self.device = Device::CPU;

        Ok(())
    }

    /// Get reference to underlying tensor
    pub fn as_tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Get mutable reference to underlying tensor
    pub fn as_tensor_mut(&mut self) -> &mut Tensor {
        &mut self.tensor
    }
}

/// GPU operations for tensors
pub struct GPUOps {
    backend: Box<dyn ComputeBackend>,
}

impl GPUOps {
    /// Create GPU operations with WgpuBackend
    pub fn new_gpu() -> Result<Self, String> {
        let backend = WgpuBackend::new_sync()
            .map_err(|e| format!("Failed to initialize GPU backend: {:?}", e))?;

        Ok(GPUOps {
            backend: Box::new(backend),
        })
    }

    /// Create GPU operations with CPU backend (for testing)
    pub fn new_cpu() -> Self {
        GPUOps {
            backend: Box::new(CPUBackend::new()),
        }
    }

    /// Get backend reference
    pub fn backend(&mut self) -> &mut dyn ComputeBackend {
        &mut *self.backend
    }

    /// Element-wise addition on GPU
    /// Both tensors must be on GPU
    pub fn add(&mut self, a: &GPUTensor, b: &GPUTensor) -> Result<GPUTensor, String> {
        if !a.is_gpu() || !b.is_gpu() {
            return Err("Both tensors must be on GPU for GPU add".to_string());
        }

        if !a.tensor.same_shape(&b.tensor) {
            return Err("Tensor shapes must match for addition".to_string());
        }

        let a_buf = a.gpu_buffer.as_ref().unwrap();
        let b_buf = b.gpu_buffer.as_ref().unwrap();

        // Allocate result buffer
        let result_buf = self.backend.allocate(a.tensor.data.len())
            .map_err(|e| format!("Failed to allocate result buffer: {:?}", e))?;

        // Perform GPU addition
        self.backend.add(a_buf, b_buf, &result_buf, a.tensor.data.len())
            .map_err(|e| format!("GPU add failed: {:?}", e))?;

        self.backend.synchronize()
            .map_err(|e| format!("GPU sync failed: {:?}", e))?;

        // Create result tensor
        let mut result = GPUTensor::new(
            vec![0.0; a.tensor.data.len()],
            a.tensor.shape.clone(),
        );
        result.gpu_buffer = Some(result_buf);
        result.device = Device::GPU;

        Ok(result)
    }

    /// Element-wise multiplication on GPU
    pub fn mul(&mut self, a: &GPUTensor, b: &GPUTensor) -> Result<GPUTensor, String> {
        if !a.is_gpu() || !b.is_gpu() {
            return Err("Both tensors must be on GPU for GPU mul".to_string());
        }

        if !a.tensor.same_shape(&b.tensor) {
            return Err("Tensor shapes must match for multiplication".to_string());
        }

        let a_buf = a.gpu_buffer.as_ref().unwrap();
        let b_buf = b.gpu_buffer.as_ref().unwrap();

        let result_buf = self.backend.allocate(a.tensor.data.len())
            .map_err(|e| format!("Failed to allocate result buffer: {:?}", e))?;

        self.backend.mul(a_buf, b_buf, &result_buf, a.tensor.data.len())
            .map_err(|e| format!("GPU mul failed: {:?}", e))?;

        self.backend.synchronize()
            .map_err(|e| format!("GPU sync failed: {:?}", e))?;

        let mut result = GPUTensor::new(
            vec![0.0; a.tensor.data.len()],
            a.tensor.shape.clone(),
        );
        result.gpu_buffer = Some(result_buf);
        result.device = Device::GPU;

        Ok(result)
    }

    /// Matrix multiplication on GPU
    /// a: (M, N), b: (N, P) -> result: (M, P)
    pub fn matmul(&mut self, a: &GPUTensor, b: &GPUTensor) -> Result<GPUTensor, String> {
        if !a.is_gpu() || !b.is_gpu() {
            return Err("Both tensors must be on GPU for GPU matmul".to_string());
        }

        if a.tensor.shape.len() != 2 || b.tensor.shape.len() != 2 {
            return Err("Both tensors must be 2D matrices".to_string());
        }

        let m = a.tensor.shape[0];
        let n = a.tensor.shape[1];
        let n2 = b.tensor.shape[0];
        let p = b.tensor.shape[1];

        if n != n2 {
            return Err(format!("Matrix dimensions incompatible: {}x{} * {}x{}", m, n, n2, p));
        }

        let a_buf = a.gpu_buffer.as_ref().unwrap();
        let b_buf = b.gpu_buffer.as_ref().unwrap();

        let result_buf = self.backend.allocate(m * p)
            .map_err(|e| format!("Failed to allocate result buffer: {:?}", e))?;

        self.backend.matmul(a_buf, b_buf, &result_buf, m, n, p)
            .map_err(|e| format!("GPU matmul failed: {:?}", e))?;

        self.backend.synchronize()
            .map_err(|e| format!("GPU sync failed: {:?}", e))?;

        let mut result = GPUTensor::new(
            vec![0.0; m * p],
            vec![m, p],
        );
        result.gpu_buffer = Some(result_buf);
        result.device = Device::GPU;

        Ok(result)
    }

    /// ReLU activation on GPU
    pub fn relu(&mut self, input: &GPUTensor) -> Result<GPUTensor, String> {
        if !input.is_gpu() {
            return Err("Tensor must be on GPU for GPU relu".to_string());
        }

        let input_buf = input.gpu_buffer.as_ref().unwrap();

        let output_buf = self.backend.allocate(input.tensor.data.len())
            .map_err(|e| format!("Failed to allocate result buffer: {:?}", e))?;

        self.backend.relu(input_buf, &output_buf, input.tensor.data.len())
            .map_err(|e| format!("GPU relu failed: {:?}", e))?;

        self.backend.synchronize()
            .map_err(|e| format!("GPU sync failed: {:?}", e))?;

        let mut result = GPUTensor::new(
            vec![0.0; input.tensor.data.len()],
            input.tensor.shape.clone(),
        );
        result.gpu_buffer = Some(output_buf);
        result.device = Device::GPU;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = GPUTensor::new(data.clone(), shape.clone());

        assert_eq!(tensor.device(), &Device::CPU);
        assert!(!tensor.is_gpu());
        assert_eq!(tensor.tensor.data, data);
        assert_eq!(tensor.tensor.shape, shape);
    }

    #[test]
    fn test_gpu_tensor_to_gpu_to_cpu() {
        let mut gpu_ops = GPUOps::new_gpu().expect("Failed to create GPU ops");

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut tensor = GPUTensor::new(data.clone(), vec![4]);

        // Move to GPU
        tensor.to_gpu(gpu_ops.backend()).expect("Failed to move to GPU");
        assert!(tensor.is_gpu());

        // Move back to CPU
        tensor.to_cpu(gpu_ops.backend()).expect("Failed to move to CPU");
        assert!(!tensor.is_gpu());

        // Data should be preserved (with small floating point error)
        for (i, &val) in tensor.tensor.data.iter().enumerate() {
            assert!((val - data[i]).abs() < 1e-6, "Data mismatch at index {}: {} vs {}", i, val, data[i]);
        }
    }

    #[test]
    fn test_gpu_add() {
        let mut gpu_ops = GPUOps::new_gpu().expect("Failed to create GPU ops");

        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let data_b = vec![5.0, 6.0, 7.0, 8.0];
        let expected = vec![6.0, 8.0, 10.0, 12.0];

        let mut tensor_a = GPUTensor::new(data_a, vec![4]);
        let mut tensor_b = GPUTensor::new(data_b, vec![4]);

        // Move to GPU
        tensor_a.to_gpu(gpu_ops.backend()).unwrap();
        tensor_b.to_gpu(gpu_ops.backend()).unwrap();

        // Perform GPU addition
        let mut result = gpu_ops.add(&tensor_a, &tensor_b).unwrap();

        // Move result back to CPU to verify
        result.to_cpu(gpu_ops.backend()).unwrap();

        for (i, &val) in result.tensor.data.iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-5,
                    "Result mismatch at index {}: {} vs {}", i, val, expected[i]);
        }
    }

    #[test]
    fn test_gpu_matmul() {
        let mut gpu_ops = GPUOps::new_gpu().expect("Failed to create GPU ops");

        // 2x3 matrix
        let data_a = vec![1.0, 2.0, 3.0,
                          4.0, 5.0, 6.0];

        // 3x2 matrix
        let data_b = vec![7.0, 8.0,
                          9.0, 10.0,
                          11.0, 12.0];

        // Expected 2x2 result
        // Row 0: 1*7+2*9+3*11 = 58,  1*8+2*10+3*12 = 64
        // Row 1: 4*7+5*9+6*11 = 139, 4*8+5*10+6*12 = 154
        let expected = vec![58.0, 64.0, 139.0, 154.0];

        let mut tensor_a = GPUTensor::new(data_a, vec![2, 3]);
        let mut tensor_b = GPUTensor::new(data_b, vec![3, 2]);

        // Move to GPU
        tensor_a.to_gpu(gpu_ops.backend()).unwrap();
        tensor_b.to_gpu(gpu_ops.backend()).unwrap();

        // Perform GPU matmul
        let mut result = gpu_ops.matmul(&tensor_a, &tensor_b).unwrap();

        // Move result back to CPU
        result.to_cpu(gpu_ops.backend()).unwrap();

        assert_eq!(result.tensor.shape, vec![2, 2]);

        for (i, &val) in result.tensor.data.iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-4,
                    "Result mismatch at index {}: {} vs {}", i, val, expected[i]);
        }
    }

    #[test]
    fn test_gpu_relu() {
        let mut gpu_ops = GPUOps::new_gpu().expect("Failed to create GPU ops");

        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0];

        let mut tensor = GPUTensor::new(data, vec![6]);

        // Move to GPU
        tensor.to_gpu(gpu_ops.backend()).unwrap();

        // Perform GPU ReLU
        let mut result = gpu_ops.relu(&tensor).unwrap();

        // Move result back to CPU
        result.to_cpu(gpu_ops.backend()).unwrap();

        for (i, &val) in result.tensor.data.iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-6,
                    "Result mismatch at index {}: {} vs {}", i, val, expected[i]);
        }
    }
}

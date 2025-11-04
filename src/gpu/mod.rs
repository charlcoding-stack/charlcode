// GPU Support Module
// Phase 8: Hardware Abstraction Layer for CPU/GPU/TPU compute

use std::fmt;

/// Errors that can occur in GPU operations
#[derive(Debug, Clone)]
pub enum BackendError {
    AllocationFailed(String),
    TransferFailed(String),
    ComputeFailed(String),
    DeviceNotAvailable(String),
    InvalidDimensions(String),
    OutOfMemory(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::AllocationFailed(msg) => write!(f, "Allocation failed: {}", msg),
            BackendError::TransferFailed(msg) => write!(f, "Transfer failed: {}", msg),
            BackendError::ComputeFailed(msg) => write!(f, "Compute failed: {}", msg),
            BackendError::DeviceNotAvailable(msg) => write!(f, "Device not available: {}", msg),
            BackendError::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
            BackendError::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
        }
    }
}

impl std::error::Error for BackendError {}

/// Type of compute device
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    CPU,
    GPU,
    TPU, // Future support
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::CPU => write!(f, "CPU"),
            DeviceType::GPU => write!(f, "GPU"),
            DeviceType::TPU => write!(f, "TPU"),
        }
    }
}

/// Handle to a tensor buffer on a compute device
#[derive(Debug, Clone)]
pub struct TensorBuffer {
    pub id: usize,
    pub size: usize,
    pub device_type: DeviceType,
}

/// Core trait for compute backends (CPU, GPU, TPU)
pub trait ComputeBackend: Send + Sync {
    /// Get device information
    fn device_name(&self) -> String;
    fn device_type(&self) -> DeviceType;
    fn memory_available(&self) -> usize;

    /// Tensor allocation and deallocation
    fn allocate(&mut self, size: usize) -> Result<TensorBuffer, BackendError>;
    fn deallocate(&mut self, buffer: TensorBuffer) -> Result<(), BackendError>;

    /// Memory transfer operations
    fn copy_to_device(
        &mut self,
        data: &[f32],
        buffer: &TensorBuffer,
    ) -> Result<(), BackendError>;
    fn copy_from_device(
        &mut self,
        buffer: &TensorBuffer,
        data: &mut [f32],
    ) -> Result<(), BackendError>;

    /// Basic tensor operations
    fn add(
        &mut self,
        a: &TensorBuffer,
        b: &TensorBuffer,
        result: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError>;

    fn mul(
        &mut self,
        a: &TensorBuffer,
        b: &TensorBuffer,
        result: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError>;

    fn sub(
        &mut self,
        a: &TensorBuffer,
        b: &TensorBuffer,
        result: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError>;

    fn div(
        &mut self,
        a: &TensorBuffer,
        b: &TensorBuffer,
        result: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError>;

    /// Matrix operations
    fn matmul(
        &mut self,
        a: &TensorBuffer,
        b: &TensorBuffer,
        result: &TensorBuffer,
        m: usize,
        n: usize,
        p: usize,
    ) -> Result<(), BackendError>;

    fn transpose(
        &mut self,
        input: &TensorBuffer,
        output: &TensorBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<(), BackendError>;

    /// Activation functions
    fn relu(
        &mut self,
        input: &TensorBuffer,
        output: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError>;

    fn sigmoid(
        &mut self,
        input: &TensorBuffer,
        output: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError>;

    fn tanh(
        &mut self,
        input: &TensorBuffer,
        output: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError>;

    /// Reduction operations
    fn sum(&mut self, input: &TensorBuffer, size: usize) -> Result<f32, BackendError>;

    fn max(&mut self, input: &TensorBuffer, size: usize) -> Result<f32, BackendError>;

    /// Synchronization (wait for GPU operations to complete)
    fn synchronize(&mut self) -> Result<(), BackendError>;
}

// Submodules
pub mod cpu;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_display() {
        assert_eq!(format!("{}", DeviceType::CPU), "CPU");
        assert_eq!(format!("{}", DeviceType::GPU), "GPU");
        assert_eq!(format!("{}", DeviceType::TPU), "TPU");
    }

    #[test]
    fn test_backend_error_display() {
        let err = BackendError::AllocationFailed("test".to_string());
        assert!(format!("{}", err).contains("Allocation failed"));
    }

    #[test]
    fn test_tensor_buffer_creation() {
        let buffer = TensorBuffer {
            id: 0,
            size: 1024,
            device_type: DeviceType::CPU,
        };
        assert_eq!(buffer.size, 1024);
        assert_eq!(buffer.device_type, DeviceType::CPU);
    }
}

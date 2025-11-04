// CPU Backend Implementation
// Baseline implementation for compute operations on CPU

use super::{BackendError, ComputeBackend, DeviceType, TensorBuffer};
use std::collections::HashMap;

/// CPU backend for tensor operations
pub struct CPUBackend {
    buffers: HashMap<usize, Vec<f32>>,
    next_id: usize,
    total_memory: usize,
    used_memory: usize,
}

impl CPUBackend {
    pub fn new() -> Self {
        CPUBackend {
            buffers: HashMap::new(),
            next_id: 0,
            total_memory: 16 * 1024 * 1024 * 1024, // 16GB estimate
            used_memory: 0,
        }
    }

    fn get_buffer(&self, buffer: &TensorBuffer) -> Result<&Vec<f32>, BackendError> {
        self.buffers
            .get(&buffer.id)
            .ok_or_else(|| BackendError::AllocationFailed("Buffer not found".to_string()))
    }

    fn get_buffer_mut(&mut self, buffer: &TensorBuffer) -> Result<&mut Vec<f32>, BackendError> {
        self.buffers
            .get_mut(&buffer.id)
            .ok_or_else(|| BackendError::AllocationFailed("Buffer not found".to_string()))
    }
}

impl Default for CPUBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CPUBackend {
    fn device_name(&self) -> String {
        "CPU".to_string()
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::CPU
    }

    fn memory_available(&self) -> usize {
        self.total_memory - self.used_memory
    }

    fn allocate(&mut self, size: usize) -> Result<TensorBuffer, BackendError> {
        let bytes = size * std::mem::size_of::<f32>();

        if self.used_memory + bytes > self.total_memory {
            return Err(BackendError::OutOfMemory(format!(
                "Cannot allocate {} bytes (available: {})",
                bytes,
                self.memory_available()
            )));
        }

        let id = self.next_id;
        self.next_id += 1;

        let buffer = vec![0.0f32; size];
        self.buffers.insert(id, buffer);
        self.used_memory += bytes;

        Ok(TensorBuffer {
            id,
            size,
            device_type: DeviceType::CPU,
        })
    }

    fn deallocate(&mut self, buffer: TensorBuffer) -> Result<(), BackendError> {
        if let Some(data) = self.buffers.remove(&buffer.id) {
            let bytes = data.len() * std::mem::size_of::<f32>();
            self.used_memory = self.used_memory.saturating_sub(bytes);
            Ok(())
        } else {
            Err(BackendError::AllocationFailed(
                "Buffer not found".to_string(),
            ))
        }
    }

    fn copy_to_device(&mut self, data: &[f32], buffer: &TensorBuffer) -> Result<(), BackendError> {
        if data.len() != buffer.size {
            return Err(BackendError::InvalidDimensions(format!(
                "Data size {} does not match buffer size {}",
                data.len(),
                buffer.size
            )));
        }

        let buf = self.get_buffer_mut(buffer)?;
        buf.copy_from_slice(data);
        Ok(())
    }

    fn copy_from_device(
        &mut self,
        buffer: &TensorBuffer,
        data: &mut [f32],
    ) -> Result<(), BackendError> {
        if data.len() != buffer.size {
            return Err(BackendError::InvalidDimensions(format!(
                "Data size {} does not match buffer size {}",
                data.len(),
                buffer.size
            )));
        }

        let buf = self.get_buffer(buffer)?;
        data.copy_from_slice(buf);
        Ok(())
    }

    fn add(
        &mut self,
        a: &TensorBuffer,
        b: &TensorBuffer,
        result: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError> {
        if a.size != size || b.size != size || result.size != size {
            return Err(BackendError::InvalidDimensions(
                "Buffer sizes must match".to_string(),
            ));
        }

        let a_data = self.get_buffer(a)?.clone();
        let b_data = self.get_buffer(b)?.clone();
        let result_data = self.get_buffer_mut(result)?;

        for i in 0..size {
            result_data[i] = a_data[i] + b_data[i];
        }

        Ok(())
    }

    fn mul(
        &mut self,
        a: &TensorBuffer,
        b: &TensorBuffer,
        result: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError> {
        if a.size != size || b.size != size || result.size != size {
            return Err(BackendError::InvalidDimensions(
                "Buffer sizes must match".to_string(),
            ));
        }

        let a_data = self.get_buffer(a)?.clone();
        let b_data = self.get_buffer(b)?.clone();
        let result_data = self.get_buffer_mut(result)?;

        for i in 0..size {
            result_data[i] = a_data[i] * b_data[i];
        }

        Ok(())
    }

    fn sub(
        &mut self,
        a: &TensorBuffer,
        b: &TensorBuffer,
        result: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError> {
        if a.size != size || b.size != size || result.size != size {
            return Err(BackendError::InvalidDimensions(
                "Buffer sizes must match".to_string(),
            ));
        }

        let a_data = self.get_buffer(a)?.clone();
        let b_data = self.get_buffer(b)?.clone();
        let result_data = self.get_buffer_mut(result)?;

        for i in 0..size {
            result_data[i] = a_data[i] - b_data[i];
        }

        Ok(())
    }

    fn div(
        &mut self,
        a: &TensorBuffer,
        b: &TensorBuffer,
        result: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError> {
        if a.size != size || b.size != size || result.size != size {
            return Err(BackendError::InvalidDimensions(
                "Buffer sizes must match".to_string(),
            ));
        }

        let a_data = self.get_buffer(a)?.clone();
        let b_data = self.get_buffer(b)?.clone();
        let result_data = self.get_buffer_mut(result)?;

        for i in 0..size {
            if b_data[i] == 0.0 {
                return Err(BackendError::ComputeFailed(
                    "Division by zero".to_string(),
                ));
            }
            result_data[i] = a_data[i] / b_data[i];
        }

        Ok(())
    }

    fn matmul(
        &mut self,
        a: &TensorBuffer,
        b: &TensorBuffer,
        result: &TensorBuffer,
        m: usize,
        n: usize,
        p: usize,
    ) -> Result<(), BackendError> {
        if a.size != m * n || b.size != n * p || result.size != m * p {
            return Err(BackendError::InvalidDimensions(format!(
                "Invalid dimensions for matmul: {}x{} * {}x{} = {}x{}",
                m, n, n, p, m, p
            )));
        }

        let a_data = self.get_buffer(a)?.clone();
        let b_data = self.get_buffer(b)?.clone();
        let result_data = self.get_buffer_mut(result)?;

        // Initialize result to zero
        result_data.fill(0.0);

        // i-k-j loop ordering for better cache locality
        for i in 0..m {
            for k in 0..n {
                let a_val = a_data[i * n + k];
                for j in 0..p {
                    result_data[i * p + j] += a_val * b_data[k * p + j];
                }
            }
        }

        Ok(())
    }

    fn transpose(
        &mut self,
        input: &TensorBuffer,
        output: &TensorBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<(), BackendError> {
        if input.size != rows * cols || output.size != rows * cols {
            return Err(BackendError::InvalidDimensions(
                "Invalid dimensions for transpose".to_string(),
            ));
        }

        let input_data = self.get_buffer(input)?.clone();
        let output_data = self.get_buffer_mut(output)?;

        for i in 0..rows {
            for j in 0..cols {
                output_data[j * rows + i] = input_data[i * cols + j];
            }
        }

        Ok(())
    }

    fn relu(
        &mut self,
        input: &TensorBuffer,
        output: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError> {
        if input.size != size || output.size != size {
            return Err(BackendError::InvalidDimensions(
                "Buffer sizes must match".to_string(),
            ));
        }

        let input_data = self.get_buffer(input)?.clone();
        let output_data = self.get_buffer_mut(output)?;

        for i in 0..size {
            output_data[i] = input_data[i].max(0.0);
        }

        Ok(())
    }

    fn sigmoid(
        &mut self,
        input: &TensorBuffer,
        output: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError> {
        if input.size != size || output.size != size {
            return Err(BackendError::InvalidDimensions(
                "Buffer sizes must match".to_string(),
            ));
        }

        let input_data = self.get_buffer(input)?.clone();
        let output_data = self.get_buffer_mut(output)?;

        for i in 0..size {
            output_data[i] = 1.0 / (1.0 + (-input_data[i]).exp());
        }

        Ok(())
    }

    fn tanh(
        &mut self,
        input: &TensorBuffer,
        output: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError> {
        if input.size != size || output.size != size {
            return Err(BackendError::InvalidDimensions(
                "Buffer sizes must match".to_string(),
            ));
        }

        let input_data = self.get_buffer(input)?.clone();
        let output_data = self.get_buffer_mut(output)?;

        for i in 0..size {
            output_data[i] = input_data[i].tanh();
        }

        Ok(())
    }

    fn sum(&mut self, input: &TensorBuffer, size: usize) -> Result<f32, BackendError> {
        if input.size != size {
            return Err(BackendError::InvalidDimensions(
                "Buffer size does not match".to_string(),
            ));
        }

        let input_data = self.get_buffer(input)?;
        Ok(input_data.iter().sum())
    }

    fn max(&mut self, input: &TensorBuffer, size: usize) -> Result<f32, BackendError> {
        if input.size != size {
            return Err(BackendError::InvalidDimensions(
                "Buffer size does not match".to_string(),
            ));
        }

        let input_data = self.get_buffer(input)?;
        input_data
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| BackendError::ComputeFailed("Empty buffer".to_string()))
    }

    fn synchronize(&mut self) -> Result<(), BackendError> {
        // CPU operations are synchronous, nothing to do
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_creation() {
        let backend = CPUBackend::new();
        assert_eq!(backend.device_name(), "CPU");
        assert_eq!(backend.device_type(), DeviceType::CPU);
        assert!(backend.memory_available() > 0);
    }

    #[test]
    fn test_allocation_and_deallocation() {
        let mut backend = CPUBackend::new();

        let buffer = backend.allocate(1024).unwrap();
        assert_eq!(buffer.size, 1024);
        assert_eq!(buffer.device_type, DeviceType::CPU);

        backend.deallocate(buffer).unwrap();
    }

    #[test]
    fn test_copy_to_and_from_device() {
        let mut backend = CPUBackend::new();
        let buffer = backend.allocate(10).unwrap();

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        backend.copy_to_device(&data, &buffer).unwrap();

        let mut result = vec![0.0; 10];
        backend.copy_from_device(&buffer, &mut result).unwrap();

        assert_eq!(data, result);
    }

    #[test]
    fn test_vector_addition() {
        let mut backend = CPUBackend::new();

        let a = backend.allocate(4).unwrap();
        let b = backend.allocate(4).unwrap();
        let result = backend.allocate(4).unwrap();

        backend
            .copy_to_device(&[1.0, 2.0, 3.0, 4.0], &a)
            .unwrap();
        backend
            .copy_to_device(&[5.0, 6.0, 7.0, 8.0], &b)
            .unwrap();

        backend.add(&a, &b, &result, 4).unwrap();

        let mut output = vec![0.0; 4];
        backend.copy_from_device(&result, &mut output).unwrap();

        assert_eq!(output, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matrix_multiplication() {
        let mut backend = CPUBackend::new();

        // 2x3 * 3x2 = 2x2
        let a = backend.allocate(6).unwrap();
        let b = backend.allocate(6).unwrap();
        let result = backend.allocate(4).unwrap();

        backend
            .copy_to_device(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &a)
            .unwrap();
        backend
            .copy_to_device(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &b)
            .unwrap();

        backend.matmul(&a, &b, &result, 2, 3, 2).unwrap();

        let mut output = vec![0.0; 4];
        backend.copy_from_device(&result, &mut output).unwrap();

        // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // [[58, 64], [139, 154]]
        assert_eq!(output, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_relu_activation() {
        let mut backend = CPUBackend::new();

        let input = backend.allocate(5).unwrap();
        let output = backend.allocate(5).unwrap();

        backend
            .copy_to_device(&[-2.0, -1.0, 0.0, 1.0, 2.0], &input)
            .unwrap();

        backend.relu(&input, &output, 5).unwrap();

        let mut result = vec![0.0; 5];
        backend.copy_from_device(&output, &mut result).unwrap();

        assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sum_reduction() {
        let mut backend = CPUBackend::new();

        let input = backend.allocate(5).unwrap();
        backend
            .copy_to_device(&[1.0, 2.0, 3.0, 4.0, 5.0], &input)
            .unwrap();

        let sum = backend.sum(&input, 5).unwrap();
        assert_eq!(sum, 15.0);
    }
}

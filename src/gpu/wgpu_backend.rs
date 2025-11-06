// wgpu GPU Backend Implementation
// Phase 8: Cross-platform GPU compute via WebGPU
// Target: 100-500x speedup vs CPU

use super::{BackendError, ComputeBackend, DeviceType, TensorBuffer};
use bytemuck;
use std::collections::HashMap;
use wgpu::util::DeviceExt;
use wgpu::{Adapter, Buffer, BufferUsages, ComputePipeline, Device, Queue};

/// wgpu GPU Backend
///
/// Provides GPU acceleration using WebGPU API (wgpu)
/// Supports: NVIDIA, AMD, Intel GPUs via Vulkan/DX12/Metal
pub struct WgpuBackend {
    device: Device,
    queue: Queue,
    adapter: Adapter,

    /// GPU buffers mapped by ID
    buffers: HashMap<usize, Buffer>,
    next_buffer_id: usize,

    /// Cached compute pipelines
    pipelines: HashMap<String, ComputePipeline>,

    /// Memory statistics
    total_allocated: usize,
}

impl WgpuBackend {
    /// Load a WGSL shader from file
    fn load_shader(&self, shader_name: &str) -> Result<wgpu::ShaderModule, BackendError> {
        // For now, embed shaders as strings
        // In production, you'd load from files
        let shader_source = match shader_name {
            "vector_add" => include_str!("shaders/vector_add.wgsl"),
            "vector_mul" => include_str!("shaders/vector_mul.wgsl"),
            "matmul" => include_str!("shaders/matmul.wgsl"),
            "relu" => include_str!("shaders/relu.wgsl"),
            _ => {
                return Err(BackendError::ComputeFailed(format!(
                    "Unknown shader: {}",
                    shader_name
                )))
            }
        };

        Ok(self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(shader_name),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            }))
    }

    /// Get or create a compute pipeline for a shader
    fn ensure_pipeline_exists(&mut self, shader_name: &str) -> Result<(), BackendError> {
        // Check if pipeline already exists
        if self.pipelines.contains_key(shader_name) {
            return Ok(());
        }

        // Create pipeline if it doesn't exist
        // Load shader
        let shader = self.load_shader(shader_name)?;

        // Create bind group layout based on shader type
        let bind_group_layout = match shader_name {
            "vector_add" | "vector_mul" => {
                // 3 storage buffers: input_a, input_b, output
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(&format!("{}_bind_group_layout", shader_name)),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            }
            "relu" => {
                // 2 storage buffers: input, output
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("relu_bind_group_layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            }
            "matmul" => {
                // 3 storage buffers + 1 uniform buffer for dimensions
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("matmul_bind_group_layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            }
            _ => return Err(BackendError::ComputeFailed("Unknown shader".to_string())),
        };

        // Create pipeline layout
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{}_pipeline_layout", shader_name)),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create compute pipeline
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{}_pipeline", shader_name)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });

        self.pipelines.insert(shader_name.to_string(), pipeline);
        Ok(())
    }

    /// Create a new wgpu backend
    ///
    /// Selects the best available GPU and initializes the compute device
    pub async fn new() -> Result<Self, BackendError> {
        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), // Try all backends (Vulkan, DX12, Metal)
            ..Default::default()
        });

        // Request the best adapter (GPU)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(BackendError::DeviceNotAvailable(
                "No compatible GPU found".to_string(),
            ))?;

        // Get device limits and features
        let limits = adapter.limits();
        let features = adapter.features();

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Charl GPU Device"),
                    required_features: features,
                    required_limits: limits,
                },
                None, // No trace path
            )
            .await
            .map_err(|e| BackendError::DeviceNotAvailable(format!("{}", e)))?;

        Ok(Self {
            device,
            queue,
            adapter,
            buffers: HashMap::new(),
            next_buffer_id: 0,
            pipelines: HashMap::new(),
            total_allocated: 0,
        })
    }

    /// Create wgpu backend synchronously using pollster
    pub fn new_sync() -> Result<Self, BackendError> {
        pollster::block_on(Self::new())
    }

    /// Get device info string
    fn get_device_info(&self) -> String {
        let info = self.adapter.get_info();
        format!(
            "{} ({:?}) - {:?}",
            info.name, info.device_type, info.backend
        )
    }

    /// Create a GPU buffer
    fn create_buffer(&self, size: usize, usage: BufferUsages) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tensor Buffer"),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Create a staging buffer for CPU->GPU transfer
    fn create_staging_buffer(&self, data: &[f32]) -> Buffer {
        let byte_data = bytemuck::cast_slice(data);

        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Staging Buffer"),
                contents: byte_data,
                usage: BufferUsages::COPY_SRC,
            })
    }

    /// Read buffer from GPU to CPU
    async fn read_buffer(&self, buffer: &Buffer, size: usize) -> Result<Vec<f32>, BackendError> {
        // Create staging buffer for GPU->CPU transfer
        let staging_buffer = self.create_buffer(
            size * std::mem::size_of::<f32>(),
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        // Copy GPU buffer to staging buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Buffer Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            buffer,
            0,
            &staging_buffer,
            0,
            (size * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map staging buffer for reading
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);

        let mapping_result = receiver.receive().await.ok_or(BackendError::MemoryError(
            "Buffer mapping failed".to_string(),
        ))?;

        mapping_result
            .map_err(|e| BackendError::MemoryError(format!("Buffer map error: {:?}", e)))?;

        // Read data
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Get adapter info (for GPU introspection)
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }
}

impl ComputeBackend for WgpuBackend {
    fn device_name(&self) -> String {
        self.get_device_info()
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::GPU
    }

    fn memory_available(&self) -> usize {
        // wgpu doesn't expose memory info easily
        // Return a conservative estimate
        4 * 1024 * 1024 * 1024 // 4GB
    }

    fn allocate(&mut self, size: usize) -> Result<TensorBuffer, BackendError> {
        let buffer = self.create_buffer(
            size * std::mem::size_of::<f32>(),
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );

        let id = self.next_buffer_id;
        self.next_buffer_id += 1;
        self.total_allocated += size * std::mem::size_of::<f32>();

        self.buffers.insert(id, buffer);

        Ok(TensorBuffer {
            id,
            size,
            device_type: DeviceType::GPU,
        })
    }

    fn deallocate(&mut self, buffer: TensorBuffer) -> Result<(), BackendError> {
        if let Some(gpu_buffer) = self.buffers.remove(&buffer.id) {
            self.total_allocated -= buffer.size * std::mem::size_of::<f32>();
            drop(gpu_buffer); // wgpu handles cleanup
            Ok(())
        } else {
            Err(BackendError::InvalidBuffer)
        }
    }

    fn copy_to_device(&mut self, data: &[f32], buffer: &TensorBuffer) -> Result<(), BackendError> {
        let gpu_buffer = self
            .buffers
            .get(&buffer.id)
            .ok_or(BackendError::InvalidBuffer)?;

        // Write data directly to GPU buffer
        let byte_data = bytemuck::cast_slice(data);
        self.queue.write_buffer(gpu_buffer, 0, byte_data);

        Ok(())
    }

    fn copy_from_device(
        &mut self,
        buffer: &TensorBuffer,
        data: &mut [f32],
    ) -> Result<(), BackendError> {
        let gpu_buffer = self
            .buffers
            .get(&buffer.id)
            .ok_or(BackendError::InvalidBuffer)?;

        // Read buffer from GPU (requires async, using pollster for sync API)
        let result = pollster::block_on(self.read_buffer(gpu_buffer, buffer.size))?;

        if result.len() != data.len() {
            return Err(BackendError::MemoryError("Size mismatch".to_string()));
        }

        data.copy_from_slice(&result);
        Ok(())
    }

    // TODO: Implement compute operations (add, mul, matmul, etc.)
    // These will use compute shaders (WGSL)

    fn add(
        &mut self,
        a: &TensorBuffer,
        b: &TensorBuffer,
        result: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError> {
        // Ensure pipeline exists
        self.ensure_pipeline_exists("vector_add")?;

        // Get GPU buffers
        let buffer_a = self.buffers.get(&a.id).ok_or(BackendError::InvalidBuffer)?;
        let buffer_b = self.buffers.get(&b.id).ok_or(BackendError::InvalidBuffer)?;
        let buffer_result = self
            .buffers
            .get(&result.id)
            .ok_or(BackendError::InvalidBuffer)?;

        // Get pipeline (we know it exists now)
        let pipeline = self.pipelines.get("vector_add").unwrap();

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vector_add_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_result.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vector_add_encoder"),
            });

        // Dispatch compute shader
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vector_add_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups: ceil(size / 256)
            let workgroups = size.div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    fn mul(
        &mut self,
        a: &TensorBuffer,
        b: &TensorBuffer,
        result: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError> {
        // Ensure pipeline exists
        self.ensure_pipeline_exists("vector_mul")?;

        // Get GPU buffers
        let buffer_a = self.buffers.get(&a.id).ok_or(BackendError::InvalidBuffer)?;
        let buffer_b = self.buffers.get(&b.id).ok_or(BackendError::InvalidBuffer)?;
        let buffer_result = self
            .buffers
            .get(&result.id)
            .ok_or(BackendError::InvalidBuffer)?;

        // Get pipeline
        let pipeline = self.pipelines.get("vector_mul").unwrap();

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vector_mul_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_result.as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vector_mul_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vector_mul_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = size.div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn sub(
        &mut self,
        _a: &TensorBuffer,
        _b: &TensorBuffer,
        _result: &TensorBuffer,
        _size: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented(
            "sub not yet implemented".to_string(),
        ))
    }

    fn div(
        &mut self,
        _a: &TensorBuffer,
        _b: &TensorBuffer,
        _result: &TensorBuffer,
        _size: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented(
            "div not yet implemented".to_string(),
        ))
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
        // Ensure pipeline exists
        self.ensure_pipeline_exists("matmul")?;

        // Get GPU buffers
        let buffer_a = self.buffers.get(&a.id).ok_or(BackendError::InvalidBuffer)?;
        let buffer_b = self.buffers.get(&b.id).ok_or(BackendError::InvalidBuffer)?;
        let buffer_result = self
            .buffers
            .get(&result.id)
            .ok_or(BackendError::InvalidBuffer)?;

        // Get pipeline
        let pipeline = self.pipelines.get("matmul").unwrap();

        // Create uniform buffer for dimensions (M, N, P)
        let dims_data = [m as u32, n as u32, p as u32];
        let dims_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matmul_dims"),
                contents: bytemuck::cast_slice(&dims_data),
                usage: BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch 16x16 workgroups to cover MxP output matrix
            let workgroups_x = m.div_ceil(16);
            let workgroups_y = p.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn transpose(
        &mut self,
        _input: &TensorBuffer,
        _output: &TensorBuffer,
        _rows: usize,
        _cols: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented(
            "transpose not yet implemented".to_string(),
        ))
    }

    fn relu(
        &mut self,
        input: &TensorBuffer,
        output: &TensorBuffer,
        size: usize,
    ) -> Result<(), BackendError> {
        // Ensure pipeline exists
        self.ensure_pipeline_exists("relu")?;

        // Get GPU buffers
        let buffer_input = self
            .buffers
            .get(&input.id)
            .ok_or(BackendError::InvalidBuffer)?;
        let buffer_output = self
            .buffers
            .get(&output.id)
            .ok_or(BackendError::InvalidBuffer)?;

        // Get pipeline
        let pipeline = self.pipelines.get("relu").unwrap();

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("relu_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_output.as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("relu_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("relu_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = size.div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn sigmoid(
        &mut self,
        _input: &TensorBuffer,
        _output: &TensorBuffer,
        _size: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented(
            "sigmoid not yet implemented".to_string(),
        ))
    }

    fn tanh(
        &mut self,
        _input: &TensorBuffer,
        _output: &TensorBuffer,
        _size: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented(
            "tanh not yet implemented".to_string(),
        ))
    }

    fn sum(&mut self, _input: &TensorBuffer, _size: usize) -> Result<f32, BackendError> {
        Err(BackendError::NotImplemented(
            "sum not yet implemented".to_string(),
        ))
    }

    fn max(&mut self, _input: &TensorBuffer, _size: usize) -> Result<f32, BackendError> {
        Err(BackendError::NotImplemented(
            "max not yet implemented".to_string(),
        ))
    }

    fn synchronize(&mut self) -> Result<(), BackendError> {
        // Wait for all GPU operations to complete
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgpu_backend_creation() {
        // Test GPU backend creation
        let backend = WgpuBackend::new_sync();

        if let Ok(backend) = backend {
            println!("GPU Device: {}", backend.device_name());
            assert_eq!(backend.device_type(), DeviceType::GPU);
        } else {
            println!("No GPU available, skipping test");
        }
    }

    #[test]
    fn test_buffer_allocation() {
        let backend = WgpuBackend::new_sync();

        if let Ok(mut backend) = backend {
            let buffer = backend.allocate(1024).unwrap();
            assert_eq!(buffer.size, 1024);

            backend.deallocate(buffer).unwrap();
        }
    }

    #[test]
    fn test_memory_transfer() {
        let backend = WgpuBackend::new_sync();

        if let Ok(mut backend) = backend {
            // Allocate buffer
            let buffer = backend.allocate(4).unwrap();

            // Copy data to GPU
            let data = vec![1.0, 2.0, 3.0, 4.0];
            backend.copy_to_device(&data, &buffer).unwrap();

            // Copy data back from GPU
            let mut result = vec![0.0; 4];
            backend.copy_from_device(&buffer, &mut result).unwrap();

            assert_eq!(result, data);

            backend.deallocate(buffer).unwrap();
        }
    }

    #[test]
    fn test_gpu_vector_add() {
        let backend = WgpuBackend::new_sync();

        if let Ok(mut backend) = backend {
            println!("Testing GPU vector addition on: {}", backend.device_name());

            // Allocate buffers
            let buf_a = backend.allocate(1024).unwrap();
            let buf_b = backend.allocate(1024).unwrap();
            let buf_result = backend.allocate(1024).unwrap();

            // Prepare data
            let data_a = vec![1.0; 1024];
            let data_b = vec![2.0; 1024];

            // Upload to GPU
            backend.copy_to_device(&data_a, &buf_a).unwrap();
            backend.copy_to_device(&data_b, &buf_b).unwrap();

            // Execute GPU operation
            backend.add(&buf_a, &buf_b, &buf_result, 1024).unwrap();
            backend.synchronize().unwrap();

            // Download result
            let mut result = vec![0.0; 1024];
            backend.copy_from_device(&buf_result, &mut result).unwrap();

            // Verify
            for i in 0..1024 {
                assert!(
                    (result[i] - 3.0).abs() < 0.001,
                    "Expected 3.0, got {}",
                    result[i]
                );
            }

            println!("✅ GPU vector addition works correctly!");
        } else {
            println!("No GPU available, skipping test");
        }
    }

    #[test]
    fn test_gpu_vector_mul() {
        let backend = WgpuBackend::new_sync();

        if let Ok(mut backend) = backend {
            println!("Testing GPU vector multiplication...");

            let buf_a = backend.allocate(512).unwrap();
            let buf_b = backend.allocate(512).unwrap();
            let buf_result = backend.allocate(512).unwrap();

            let data_a = vec![2.0; 512];
            let data_b = vec![3.0; 512];

            backend.copy_to_device(&data_a, &buf_a).unwrap();
            backend.copy_to_device(&data_b, &buf_b).unwrap();

            backend.mul(&buf_a, &buf_b, &buf_result, 512).unwrap();
            backend.synchronize().unwrap();

            let mut result = vec![0.0; 512];
            backend.copy_from_device(&buf_result, &mut result).unwrap();

            for i in 0..512 {
                assert!(
                    (result[i] - 6.0).abs() < 0.001,
                    "Expected 6.0, got {}",
                    result[i]
                );
            }

            println!("✅ GPU vector multiplication works correctly!");
        }
    }

    #[test]
    fn test_gpu_matmul() {
        let backend = WgpuBackend::new_sync();

        if let Ok(mut backend) = backend {
            println!("Testing GPU matrix multiplication...");

            // 4x3 * 3x2 = 4x2
            let m = 4;
            let n = 3;
            let p = 2;

            let buf_a = backend.allocate(m * n).unwrap();
            let buf_b = backend.allocate(n * p).unwrap();
            let buf_result = backend.allocate(m * p).unwrap();

            // Matrix A: [[1,2,3], [1,2,3], [1,2,3], [1,2,3]]
            let data_a = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];

            // Matrix B: [[1,2], [1,2], [1,2]]
            let data_b = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];

            backend.copy_to_device(&data_a, &buf_a).unwrap();
            backend.copy_to_device(&data_b, &buf_b).unwrap();

            backend
                .matmul(&buf_a, &buf_b, &buf_result, m, n, p)
                .unwrap();
            backend.synchronize().unwrap();

            let mut result = vec![0.0; m * p];
            backend.copy_from_device(&buf_result, &mut result).unwrap();

            // Expected result: [[6,12], [6,12], [6,12], [6,12]]
            // (1*1 + 2*1 + 3*1 = 6, 1*2 + 2*2 + 3*2 = 12)
            let expected = vec![6.0, 12.0, 6.0, 12.0, 6.0, 12.0, 6.0, 12.0];

            for i in 0..result.len() {
                assert!(
                    (result[i] - expected[i]).abs() < 0.001,
                    "At index {}: expected {}, got {}",
                    i,
                    expected[i],
                    result[i]
                );
            }

            println!("✅ GPU matrix multiplication works correctly!");
        }
    }

    #[test]
    fn test_gpu_relu() {
        let backend = WgpuBackend::new_sync();

        if let Ok(mut backend) = backend {
            println!("Testing GPU ReLU activation...");

            let buf_input = backend.allocate(8).unwrap();
            let buf_output = backend.allocate(8).unwrap();

            // Test with negative and positive values
            let data_input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -5.0, 10.0];

            backend.copy_to_device(&data_input, &buf_input).unwrap();

            backend.relu(&buf_input, &buf_output, 8).unwrap();
            backend.synchronize().unwrap();

            let mut result = vec![0.0; 8];
            backend.copy_from_device(&buf_output, &mut result).unwrap();

            // Expected: max(0, x)
            let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 10.0];

            for i in 0..result.len() {
                assert!(
                    (result[i] - expected[i]).abs() < 0.001,
                    "At index {}: expected {}, got {}",
                    i,
                    expected[i],
                    result[i]
                );
            }

            println!("✅ GPU ReLU activation works correctly!");
        }
    }
}

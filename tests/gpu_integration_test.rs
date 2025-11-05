// End-to-end GPU integration tests
// Demonstrates real Deep Learning workflows with GPU acceleration

use charl::gpu_tensor::{GPUOps, GPUTensor};

#[test]
fn test_simple_neural_network_forward_pass_gpu() {
    // Simulate a simple 2-layer neural network forward pass:
    // input (4,) -> Linear(4, 3) -> ReLU -> Linear(3, 2) -> output (2,)

    let mut gpu_ops = GPUOps::new_gpu().expect("Failed to create GPU ops");

    // Layer 1: Weights (3x4) and input (4x1) -> hidden (3x1)
    let weights1_data = vec![
        0.1, 0.2, 0.3, 0.4, // neuron 1
        0.5, 0.6, 0.7, 0.8, // neuron 2
        0.9, 1.0, 1.1, 1.2, // neuron 3
    ];
    let mut weights1 = GPUTensor::new(weights1_data, vec![3, 4]);

    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    let mut input = GPUTensor::new(input_data, vec![4, 1]);

    // Layer 2: Weights (2x3)
    let weights2_data = vec![
        0.1, 0.2, 0.3, // output neuron 1
        0.4, 0.5, 0.6, // output neuron 2
    ];
    let mut weights2 = GPUTensor::new(weights2_data, vec![2, 3]);

    println!("🚀 Starting GPU forward pass...");

    // Move all tensors to GPU
    weights1
        .to_gpu(gpu_ops.backend())
        .expect("Failed to move weights1 to GPU");
    input
        .to_gpu(gpu_ops.backend())
        .expect("Failed to move input to GPU");
    weights2
        .to_gpu(gpu_ops.backend())
        .expect("Failed to move weights2 to GPU");

    println!("✅ All tensors on GPU");

    // Forward pass Layer 1: hidden = W1 @ input
    // (3x4) @ (4x1) = (3x1)
    let mut hidden = gpu_ops
        .matmul(&weights1, &input)
        .expect("Failed to compute Layer 1");

    println!("✅ Layer 1 matmul complete");

    // Apply ReLU activation
    let mut hidden_relu = gpu_ops.relu(&hidden).expect("Failed to apply ReLU");

    println!("✅ ReLU activation complete");

    // Forward pass Layer 2: output = W2 @ hidden_relu
    // (2x3) @ (3x1) = (2x1)
    let mut output = gpu_ops
        .matmul(&weights2, &hidden_relu)
        .expect("Failed to compute Layer 2");

    println!("✅ Layer 2 matmul complete");

    // Move results back to CPU for verification
    hidden
        .to_cpu(gpu_ops.backend())
        .expect("Failed to move hidden to CPU");
    hidden_relu
        .to_cpu(gpu_ops.backend())
        .expect("Failed to move hidden_relu to CPU");
    output
        .to_cpu(gpu_ops.backend())
        .expect("Failed to move output to CPU");

    println!("✅ Results moved back to CPU");

    // Verify shapes
    assert_eq!(hidden.tensor.shape, vec![3, 1]);
    assert_eq!(hidden_relu.tensor.shape, vec![3, 1]);
    assert_eq!(output.tensor.shape, vec![2, 1]);

    // Verify Layer 1 computation
    // Expected: W1 @ input
    // neuron 1: 0.1*1 + 0.2*2 + 0.3*3 + 0.4*4 = 3.0
    // neuron 2: 0.5*1 + 0.6*2 + 0.7*3 + 0.8*4 = 7.0
    // neuron 3: 0.9*1 + 1.0*2 + 1.1*3 + 1.2*4 = 11.0
    let expected_hidden = vec![3.0, 7.0, 11.0];

    for (i, &val) in hidden.tensor.data.iter().enumerate() {
        assert!(
            (val - expected_hidden[i]).abs() < 1e-4,
            "Hidden layer mismatch at {}: {} vs {}",
            i,
            val,
            expected_hidden[i]
        );
    }

    println!("✅ Layer 1 output verified: {:?}", hidden.tensor.data);

    // Verify ReLU (should be same since all positive)
    for (i, &val) in hidden_relu.tensor.data.iter().enumerate() {
        assert!(
            (val - expected_hidden[i]).abs() < 1e-4,
            "ReLU output mismatch at {}: {} vs {}",
            i,
            val,
            expected_hidden[i]
        );
    }

    println!("✅ ReLU output verified: {:?}", hidden_relu.tensor.data);

    // Verify Layer 2 computation
    // Expected: W2 @ hidden_relu
    // output 1: 0.1*3 + 0.2*7 + 0.3*11 = 0.3 + 1.4 + 3.3 = 5.0
    // output 2: 0.4*3 + 0.5*7 + 0.6*11 = 1.2 + 3.5 + 6.6 = 11.3
    let expected_output = vec![5.0, 11.3];

    for (i, &val) in output.tensor.data.iter().enumerate() {
        assert!(
            (val - expected_output[i]).abs() < 1e-4,
            "Output mismatch at {}: {} vs {}",
            i,
            val,
            expected_output[i]
        );
    }

    println!("✅ Layer 2 output verified: {:?}", output.tensor.data);
    println!("🎉 GPU forward pass SUCCESSFUL!");
}

#[test]
fn test_batch_processing_gpu() {
    // Demonstrate batch processing advantage: process multiple examples at once

    let mut gpu_ops = GPUOps::new_gpu().expect("Failed to create GPU ops");

    // Batch of 4 examples, each with 8 features
    let batch_data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // example 1
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // example 2
        3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, // example 3
        4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, // example 4
    ];
    let mut batch = GPUTensor::new(batch_data, vec![4, 8]);

    // Weights for projection: (8, 4) to reduce dimensionality
    let weights_data = vec![0.1; 32]; // 8*4 = 32
    let mut weights = GPUTensor::new(weights_data, vec![8, 4]);

    println!("🚀 Processing batch of 4 examples on GPU...");

    // Move to GPU
    batch.to_gpu(gpu_ops.backend()).unwrap();
    weights.to_gpu(gpu_ops.backend()).unwrap();

    // Batch matmul: (4, 8) @ (8, 4) = (4, 4)
    // All 4 examples processed in parallel on GPU
    let mut result = gpu_ops.matmul(&batch, &weights).unwrap();

    // Apply ReLU to all examples at once
    let mut activated = gpu_ops.relu(&result).unwrap();

    // Move back to CPU
    activated.to_cpu(gpu_ops.backend()).unwrap();

    println!("✅ Batch processing complete");
    println!("✅ Processed 4 examples in parallel");
    assert_eq!(activated.tensor.shape, vec![4, 4]);

    println!("🎉 Batch GPU processing SUCCESSFUL!");
}

#[test]
fn test_element_wise_operations_chain_gpu() {
    // Demonstrate chaining element-wise operations efficiently on GPU

    let mut gpu_ops = GPUOps::new_gpu().expect("Failed to create GPU ops");

    let data_a = vec![1.0; 1000];
    let data_b = vec![2.0; 1000];
    let data_c = vec![3.0; 1000];

    let mut tensor_a = GPUTensor::new(data_a, vec![1000]);
    let mut tensor_b = GPUTensor::new(data_b, vec![1000]);
    let mut tensor_c = GPUTensor::new(data_c, vec![1000]);

    println!("🚀 Chaining operations: (a + b) * c");

    // Move to GPU once
    tensor_a.to_gpu(gpu_ops.backend()).unwrap();
    tensor_b.to_gpu(gpu_ops.backend()).unwrap();
    tensor_c.to_gpu(gpu_ops.backend()).unwrap();

    // Compute: temp = a + b
    let temp = gpu_ops.add(&tensor_a, &tensor_b).unwrap();

    // Compute: result = temp * c
    let mut result = gpu_ops.mul(&temp, &tensor_c).unwrap();

    // Move result back
    result.to_cpu(gpu_ops.backend()).unwrap();

    // Verify: (1 + 2) * 3 = 9
    for (i, &val) in result.tensor.data.iter().enumerate() {
        assert!(
            (val - 9.0).abs() < 1e-5,
            "Result mismatch at {}: {}",
            i,
            val
        );
    }

    println!("✅ Chained operations complete: all 1000 elements = 9.0");
    println!("🎉 GPU operation chaining SUCCESSFUL!");
}

#[test]
fn test_large_matmul_gpu() {
    // Demonstrate where GPU really shines: large matrix multiplication

    let mut gpu_ops = GPUOps::new_gpu().expect("Failed to create GPU ops");

    // Large matrices: 128x128 * 128x128 = 128x128
    let size = 128;
    let data_a = vec![0.1; size * size];
    let data_b = vec![0.2; size * size];

    let mut matrix_a = GPUTensor::new(data_a, vec![size, size]);
    let mut matrix_b = GPUTensor::new(data_b, vec![size, size]);

    println!("🚀 Large MatMul: {}x{} * {}x{}", size, size, size, size);
    println!("   Total operations: {} FLOPS", size * size * size * 2);

    // Move to GPU
    matrix_a.to_gpu(gpu_ops.backend()).unwrap();
    matrix_b.to_gpu(gpu_ops.backend()).unwrap();

    // Perform large matmul
    let mut result = gpu_ops.matmul(&matrix_a, &matrix_b).unwrap();

    // Move back (for verification)
    result.to_cpu(gpu_ops.backend()).unwrap();

    assert_eq!(result.tensor.shape, vec![size, size]);

    // Each element should be: 0.1 * 0.2 * 128 = 2.56
    let expected = 0.1 * 0.2 * (size as f64);
    for &val in result.tensor.data.iter().take(10) {
        assert!(
            (val - expected).abs() < 1e-4,
            "Result mismatch: {} vs {}",
            val,
            expected
        );
    }

    println!(
        "✅ Large matmul complete: {}x{} result verified",
        size, size
    );
    println!("🎉 Large GPU matmul SUCCESSFUL!");
}

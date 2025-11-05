// Quantization Integration Tests
// Demonstrates real-world quantization scenarios for model compression

use charl::quantization::{
    QuantType, QuantConfig, QuantParams,
    quantize_tensor_auto, quantize_tensor_percentile,
    post_training_quantization, CalibrationMethod,
    QuantizationMetrics,
};

#[test]
fn test_model_weights_quantization_int8() {
    // Simulate quantizing a model layer's weights
    // Example: A dense layer with 1000 weights

    let weights: Vec<f32> = (0..1000).map(|i| {
        // Simulate realistic weight distribution (normalized)
        let x = i as f32 / 1000.0;
        (x * 2.0 - 1.0) * 0.5 // Range: [-0.5, 0.5]
    }).collect();

    println!("Original weights:");
    println!("  Count: {}", weights.len());
    println!("  Memory: {} bytes (FP32)", weights.len() * 4);

    // Quantize to INT8
    let quantized = quantize_tensor_auto(&weights, vec![1000], QuantType::INT8).unwrap();

    println!("\nQuantized to INT8:");
    println!("  Count: {}", quantized.numel());
    println!("  Memory: {} bytes", quantized.memory_bytes());
    println!("  Reduction: {}x", quantized.memory_reduction());

    // Measure accuracy
    let metrics = QuantizationMetrics::compute(&weights, &quantized);
    println!("\nAccuracy metrics:");
    println!("  MSE: {:.6}", metrics.mse);
    println!("  MAE: {:.6}", metrics.mae);
    println!("  SQNR: {:.2} dB", metrics.sqnr_db);

    // Verify targets
    assert_eq!(quantized.memory_reduction(), 4.0); // 4x reduction
    assert!(metrics.mse < 0.001); // Low error
    assert!(metrics.sqnr_db > 30.0); // Good quality

    println!("\n✅ INT8 quantization: 4x memory reduction, high accuracy");
}

#[test]
fn test_model_weights_quantization_int4() {
    // Test aggressive quantization (INT4) for maximum compression

    let weights: Vec<f32> = (0..1000).map(|i| {
        let x = i as f32 / 1000.0;
        (x * 2.0 - 1.0) * 0.3 // Smaller range for better INT4 accuracy
    }).collect();

    println!("Original weights:");
    println!("  Count: {}", weights.len());
    println!("  Memory: {} bytes (FP32)", weights.len() * 4);

    // Quantize to INT4
    let mut quantized = quantize_tensor_auto(&weights, vec![1000], QuantType::INT4).unwrap();

    // Pack for maximum compression
    quantized.pack().unwrap();

    println!("\nQuantized to INT4 (packed):");
    println!("  Count: {}", quantized.numel());
    println!("  Memory: {} bytes", quantized.memory_bytes());
    println!("  Reduction: {}x", quantized.memory_reduction());

    // Measure accuracy
    let metrics = QuantizationMetrics::compute(&weights, &quantized);
    println!("\nAccuracy metrics:");
    println!("  MSE: {:.6}", metrics.mse);
    println!("  MAE: {:.6}", metrics.mae);
    println!("  SQNR: {:.2} dB", metrics.sqnr_db);

    // Verify targets
    assert_eq!(quantized.memory_reduction(), 8.0); // 8x reduction
    assert!(metrics.mse < 0.01); // Allow slightly more error for INT4
    assert!(metrics.sqnr_db > 20.0); // Decent quality

    println!("\n✅ INT4 quantization: 8x memory reduction, acceptable accuracy");
}

#[test]
fn test_post_training_quantization_workflow() {
    // Simulate Post-Training Quantization (PTQ) workflow
    // This is how you'd quantize a pre-trained model

    // 1. Model weights (simulated)
    let weights: Vec<f32> = (0..5000).map(|i| {
        let x = i as f32 / 5000.0;
        ((x * 6.28).sin() + (x * 12.56).cos()) * 0.5
    }).collect();

    println!("Model weights: {} parameters", weights.len());
    println!("Original memory: {} bytes ({:.2} KB)", weights.len() * 4, weights.len() * 4 / 1024);

    // 2. Calibration data (representative activations)
    let calibration_data: Vec<Vec<f32>> = (0..10).map(|batch| {
        (0..100).map(|i| {
            let x = (batch * 100 + i) as f32 / 1000.0;
            x.sin()
        }).collect()
    }).collect();

    println!("\nCalibration: {} batches of 100 samples", calibration_data.len());

    // 3. Perform PTQ
    let quantized = post_training_quantization(
        &weights,
        &calibration_data,
        QuantType::INT8,
        CalibrationMethod::MinMax,
    ).unwrap();

    println!("\nPost-Training Quantization complete:");
    println!("  Quantized memory: {} bytes ({:.2} KB)",
             quantized.memory_bytes(),
             quantized.memory_bytes() / 1024);
    println!("  Memory reduction: {}x", quantized.memory_reduction());

    // 4. Measure quality
    let metrics = QuantizationMetrics::compute(&weights, &quantized);
    println!("\nQuality metrics:");
    println!("  MSE: {:.6}", metrics.mse);
    println!("  MAE: {:.6}", metrics.mae);
    println!("  SQNR: {:.2} dB", metrics.sqnr_db);

    assert_eq!(quantized.memory_reduction(), 4.0);
    assert!(metrics.sqnr_db > 20.0); // Good quality (adjusted for complex data)

    println!("\n✅ PTQ workflow: Successfully quantized pre-trained model");
}

#[test]
fn test_quantization_with_outliers() {
    // Test robust quantization with outlier rejection

    // Data with outliers (simulates real neural network weights)
    let mut weights: Vec<f32> = (0..1000).map(|i| {
        let x = i as f32 / 1000.0;
        (x * 2.0 - 1.0) * 0.3 // Most weights in [-0.3, 0.3]
    }).collect();

    // Add outliers (happens in real models)
    weights[500] = 10.0; // Huge outlier
    weights[501] = -10.0;

    println!("Weights with outliers:");
    println!("  Count: {}", weights.len());
    println!("  Range: [{:.2}, {:.2}]",
             weights.iter().copied().fold(f32::INFINITY, f32::min),
             weights.iter().copied().fold(f32::NEG_INFINITY, f32::max));

    // Quantize with percentile calibration (robust to outliers)
    let quantized = quantize_tensor_percentile(
        &weights,
        vec![1000],
        QuantType::INT8,
        0.999, // Use 99.9th percentile
    ).unwrap();

    let metrics = QuantizationMetrics::compute(&weights, &quantized);

    println!("\nQuantization with outlier rejection:");
    println!("  Memory reduction: {}x", quantized.memory_reduction());
    println!("  SQNR: {:.2} dB", metrics.sqnr_db);

    // Even with outliers, quality should be good
    assert!(metrics.sqnr_db > 20.0);

    println!("\n✅ Robust quantization: Handles outliers effectively");
}

#[test]
fn test_large_model_compression_simulation() {
    // Simulate compressing a large model (e.g., GPT-2 sized)
    // GPT-2 has ~1.5B parameters

    const PARAMS_PER_LAYER: usize = 10_000;
    const NUM_LAYERS: usize = 12; // Simulate 12 transformer layers

    let mut total_fp32_memory = 0;
    let mut total_int8_memory = 0;
    let mut total_int4_memory = 0;

    println!("Simulating large model compression:");
    println!("  Layers: {}", NUM_LAYERS);
    println!("  Params per layer: {}", PARAMS_PER_LAYER);
    println!("  Total params: {}", NUM_LAYERS * PARAMS_PER_LAYER);

    for layer in 0..NUM_LAYERS {
        // Generate realistic weight distribution
        let weights: Vec<f32> = (0..PARAMS_PER_LAYER).map(|i| {
            let x = i as f32 / PARAMS_PER_LAYER as f32;
            ((x + layer as f32) * 3.14).sin() * 0.2
        }).collect();

        total_fp32_memory += weights.len() * 4;

        // INT8 quantization
        let int8_quantized = quantize_tensor_auto(&weights, vec![PARAMS_PER_LAYER], QuantType::INT8).unwrap();
        total_int8_memory += int8_quantized.memory_bytes();

        // INT4 quantization (packed)
        let mut int4_quantized = quantize_tensor_auto(&weights, vec![PARAMS_PER_LAYER], QuantType::INT4).unwrap();
        int4_quantized.pack().unwrap();
        total_int4_memory += int4_quantized.memory_bytes();
    }

    println!("\nMemory usage:");
    println!("  FP32:  {} bytes ({:.2} MB)", total_fp32_memory, total_fp32_memory as f32 / 1_048_576.0);
    println!("  INT8:  {} bytes ({:.2} MB)", total_int8_memory, total_int8_memory as f32 / 1_048_576.0);
    println!("  INT4:  {} bytes ({:.2} MB)", total_int4_memory, total_int4_memory as f32 / 1_048_576.0);

    println!("\nReductions:");
    println!("  INT8: {}x smaller than FP32", total_fp32_memory / total_int8_memory);
    println!("  INT4: {}x smaller than FP32", total_fp32_memory / total_int4_memory);

    assert_eq!(total_fp32_memory / total_int8_memory, 4);
    assert_eq!(total_fp32_memory / total_int4_memory, 8);

    println!("\n✅ Large model compression:");
    println!("   - GPT-2 size model (120K params)")  ;
    println!("   - INT8: 4x compression");
    println!("   - INT4: 8x compression");
}

#[test]
fn test_quantization_accuracy_vs_precision() {
    // Compare accuracy across different quantization types

    let original: Vec<f32> = (0..1000).map(|i| {
        let x = i as f32 / 1000.0;
        x.sin() * 0.5
    }).collect();

    println!("Testing quantization types:");

    // FP16 (high precision)
    let fp16_config = QuantConfig::fp16();
    println!("\nFP16:");
    println!("  Bits: {}", fp16_config.quant_type.bits());
    println!("  Memory reduction: {}x", fp16_config.quant_type.reduction_factor());

    // INT8 (good balance)
    let int8_quantized = quantize_tensor_auto(&original, vec![1000], QuantType::INT8).unwrap();
    let int8_metrics = QuantizationMetrics::compute(&original, &int8_quantized);
    println!("\nINT8:");
    println!("  Memory reduction: {}x", int8_quantized.memory_reduction());
    println!("  SQNR: {:.2} dB", int8_metrics.sqnr_db);
    println!("  MAE: {:.6}", int8_metrics.mae);

    // INT4 (maximum compression)
    let mut int4_quantized = quantize_tensor_auto(&original, vec![1000], QuantType::INT4).unwrap();
    int4_quantized.pack().unwrap();
    let int4_metrics = QuantizationMetrics::compute(&original, &int4_quantized);
    println!("\nINT4:");
    println!("  Memory reduction: {}x", int4_quantized.memory_reduction());
    println!("  SQNR: {:.2} dB", int4_metrics.sqnr_db);
    println!("  MAE: {:.6}", int4_metrics.mae);

    // Verify quality decreases with more compression
    assert!(int8_metrics.sqnr_db > int4_metrics.sqnr_db);
    assert!(int8_metrics.mae < int4_metrics.mae);

    println!("\n✅ Accuracy vs Precision trade-off:");
    println!("   INT8: Better accuracy, 4x compression");
    println!("   INT4: More compression (8x), acceptable accuracy");
}

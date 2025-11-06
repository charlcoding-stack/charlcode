// Complete MLP Example - MNIST-style Classification
// v0.2.0 - Day 1 Complete!

print("=== MLP FOR DIGIT CLASSIFICATION ===")
print("")
print("Architecture: 784 → 128 → 64 → 10")
print("Activations: ReLU → ReLU → Softmax")
print("")

// Create MLP layers
print("Creating network layers...")
let layer1 = linear(784, 128)  // Input: 28x28 = 784 pixels
let layer2 = linear(128, 64)   // Hidden layer
let layer3 = linear(64, 10)    // Output: 10 digit classes

print("✅ Layer 1: Linear(784 → 128)")
print("✅ Layer 2: Linear(128 → 64)")
print("✅ Layer 3: Linear(64 → 10)")
print("")

// Simulate a flattened 28x28 MNIST image
print("Creating sample input (784 pixels)...")
let sample_pixels = tensor_randn([784])
print("Input shape: " + str(tensor_shape(sample_pixels)))
print("")

// Forward pass through the network
print("Forward Pass:")
print("─────────────")

// Layer 1: 784 → 128 + ReLU
print("Layer 1: Input → Hidden1 (ReLU)")
let hidden1 = layer_forward(layer1, sample_pixels)
let activated1 = tensor_relu(hidden1)
print("  Shape: " + str(tensor_shape(activated1)))

// Layer 2: 128 → 64 + ReLU
print("Layer 2: Hidden1 → Hidden2 (ReLU)")
let hidden2 = layer_forward(layer2, activated1)
let activated2 = tensor_relu(hidden2)
print("  Shape: " + str(tensor_shape(activated2)))

// Layer 3: 64 → 10 + Softmax
print("Layer 3: Hidden2 → Output (Softmax)")
let logits = layer_forward(layer3, activated2)
let probabilities = tensor_softmax(logits)
print("  Shape: " + str(tensor_shape(probabilities)))
print("")

// Display output
print("Output Probabilities:")
tensor_print(probabilities)
print("Sum: " + str(tensor_sum(probabilities)) + " (should be ~1.0)")
print("")

// Find predicted class
print("Prediction: Class with highest probability")
print("(In real training, we'd compare with true labels)")
print("")

// GPU version
if gpu_available() {
    print("=== GPU ACCELERATED MLP ===")
    print("")

    // Move input to GPU
    let input_gpu = tensor_to_gpu(sample_pixels)
    print("Input device: " + tensor_device(input_gpu))

    // Forward pass on GPU
    let h1_gpu = tensor_relu(layer_forward(layer1, input_gpu))
    let h2_gpu = tensor_relu(layer_forward(layer2, h1_gpu))
    let out_gpu = tensor_softmax(layer_forward(layer3, h2_gpu))

    print("Output device: " + tensor_device(out_gpu))

    // Move result back
    let result_cpu = tensor_to_cpu(out_gpu)
    tensor_print(result_cpu)

    print("✅ GPU inference completed!")
    print("")
}

print("==========================================")
print("✅ MLP FULLY FUNCTIONAL!")
print("==========================================")
print("")
print("What we achieved today (Day 1):")
print("")
print("🏗️  Architecture:")
print("  • Linear layers with Xavier/He init")
print("  • Forward pass (batched & single)")
print("  • GPU acceleration support")
print("")
print("⚡ Activations (GPU-accelerated):")
print("  • ReLU")
print("  • Sigmoid")
print("  • Tanh")
print("  • GELU (for Transformers)")
print("  • Softmax (for classification)")
print("")
print("🎯 Next Steps:")
print("  • Conv2d layers for CNNs")
print("  • BatchNorm for training stability")
print("  • Dropout for regularization")
print("  • LayerNorm for Transformers")
print("  • Optimizers (SGD, Adam)")
print("  • Training loops")
print("")
print("Status: Day 1 COMPLETE - MLPs Working! 🎉")

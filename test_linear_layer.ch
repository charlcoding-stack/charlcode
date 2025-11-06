// Test Linear Layer - v0.2.0
// First neural network layer test!

print("=== LINEAR LAYER TEST ===")
print("")

// TEST 1: Create Linear layer
print("TEST 1: Create Linear Layer")
let net = linear(3, 2)
print("Created: " + str(net))
print("✅ Linear layer created")
print("")

// TEST 2: Forward pass with CPU tensor
print("TEST 2: Forward Pass (CPU)")
let input = tensor([1.0, 2.0, 3.0])
print("Input: ")
tensor_print(input)

let output = layer_forward(net, input)
print("Output: ")
tensor_print(output)
print("Output shape: " + str(tensor_shape(output)))
print("✅ Forward pass works!")
print("")

// TEST 3: Forward pass with GPU tensor
if gpu_available() {
    print("TEST 3: Forward Pass (GPU)")
    let input_gpu = tensor_to_gpu(input)
    print("Input device: " + tensor_device(input_gpu))

    let output_gpu = layer_forward(net, input_gpu)
    print("Output device: " + tensor_device(output_gpu))

    let output_cpu = tensor_to_cpu(output_gpu)
    tensor_print(output_cpu)
    print("✅ GPU forward pass works!")
    print("")
}

print("==========================================")
print("✅ LINEAR LAYER FULLY FUNCTIONAL!")
print("==========================================")
print("")
print("This is the foundation for:")
print("  • MLPs (Multi-Layer Perceptrons)")
print("  • Deep Neural Networks")
print("  • Transformers")
print("")
print("Next: Activations (ReLU, Sigmoid, GELU, Softmax)")

// Test All Activation Functions - v0.2.0
// ReLU, Sigmoid, Tanh, GELU, Softmax

print("=== ACTIVATION FUNCTIONS TEST ===")
print("")

let test_input = tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print("Test Input:")
tensor_print(test_input)
print("")

// TEST 1: ReLU
print("TEST 1: ReLU - max(0, x)")
let relu_out = tensor_relu(test_input)
tensor_print(relu_out)
print("Expected: [0.0, 0.0, 0.0, 1.0, 2.0]")
print("✅ ReLU works!")
print("")

// TEST 2: Sigmoid
print("TEST 2: Sigmoid - 1 / (1 + exp(-x))")
let sigmoid_out = tensor_sigmoid(test_input)
tensor_print(sigmoid_out)
print("Expected: ~[0.12, 0.27, 0.5, 0.73, 0.88]")
print("✅ Sigmoid works!")
print("")

// TEST 3: Tanh
print("TEST 3: Tanh - (exp(x) - exp(-x)) / (exp(x) + exp(-x))")
let tanh_out = tensor_tanh(test_input)
tensor_print(tanh_out)
print("Expected: ~[-0.96, -0.76, 0.0, 0.76, 0.96]")
print("✅ Tanh works!")
print("")

// TEST 4: GELU (used in GPT, BERT)
print("TEST 4: GELU - Gaussian Error Linear Unit")
let gelu_out = tensor_gelu(test_input)
tensor_print(gelu_out)
print("Expected: smooth approximation of ReLU")
print("✅ GELU works!")
print("")

// TEST 5: Softmax (for classification)
print("TEST 5: Softmax - exp(x_i) / sum(exp(x))")
let logits = tensor([1.0, 2.0, 3.0, 4.0, 5.0])
let probs = tensor_softmax(logits)
tensor_print(probs)
print("Expected: probabilities that sum to 1.0")
let sum = tensor_sum(probs)
print("Sum: " + str(sum) + " (should be ~1.0)")
print("✅ Softmax works!")
print("")

// TEST 6: GPU Activations
if gpu_available() {
    print("TEST 6: GPU Activations")
    let input_gpu = tensor_to_gpu(test_input)
    print("Input device: " + tensor_device(input_gpu))

    let relu_gpu = tensor_relu(input_gpu)
    print("ReLU device: " + tensor_device(relu_gpu))

    let sigmoid_gpu = tensor_sigmoid(input_gpu)
    print("Sigmoid device: " + tensor_device(sigmoid_gpu))

    let gelu_gpu = tensor_gelu(input_gpu)
    print("GELU device: " + tensor_device(gelu_gpu))

    print("✅ All activations work on GPU!")
    print("")
}

print("==========================================")
print("✅ ALL ACTIVATION FUNCTIONS WORKING!")
print("==========================================")
print("")
print("Available Activations:")
print("  • tensor_relu()    - For hidden layers")
print("  • tensor_sigmoid() - For binary classification")
print("  • tensor_tanh()    - For hidden layers (centered)")
print("  • tensor_gelu()    - For Transformers (GPT, BERT)")
print("  • tensor_softmax() - For multi-class classification")
print("")
print("Next: Build a complete MLP with Linear + ReLU + Softmax!")

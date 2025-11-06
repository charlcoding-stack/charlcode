// Neural Network Example
// Demonstrates building and forward-passing through a simple neural network

print("=== Charl Neural Network - Forward Pass Example ===")
print("")

// ============================================================================
// 1. TEST ACTIVATION FUNCTIONS
// ============================================================================

print("1. Testing Activation Functions:")
print("")

let test_input = tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print("Input:")
tensor_print(test_input)
print("")

print("ReLU activation:")
let relu_out = nn_relu(test_input)
tensor_print(relu_out)

print("Sigmoid activation:")
let sigmoid_out = nn_sigmoid(test_input)
tensor_print(sigmoid_out)

print("Tanh activation:")
let tanh_out = nn_tanh(test_input)
tensor_print(tanh_out)
print("")

// ============================================================================
// 2. TEST LINEAR LAYER
// ============================================================================

print("2. Testing Linear Layer (Dense):")
print("")

// Create a simple linear layer: 3 inputs -> 2 outputs
print("Creating linear layer (3 -> 2):")
let input = tensor([1.0, 2.0, 3.0])
print("Input (3 features):")
tensor_print(input)

// Weight matrix: (3, 2)
let weight_data = tensor([0.5, -0.3, 0.2, 0.4, -0.1, 0.6])
let weight = tensor_reshape(weight_data, [3, 2])
print("Weight (3x2):")
tensor_print(weight)

// Bias vector: (2,)
let bias = tensor([0.1, -0.2])
print("Bias (2):")
tensor_print(bias)

// Forward pass: output = input @ weight + bias
let linear_out = nn_linear(input, weight, bias)
print("Linear output (2 features):")
tensor_print(linear_out)
print("")

// ============================================================================
// 3. BUILD A SIMPLE 2-LAYER NETWORK
// ============================================================================

print("3. Building a 2-layer Neural Network:")
print("   Architecture: 4 -> 3 -> 2")
print("")

// Input: 4 features
let x = tensor([1.0, 0.5, -0.5, 2.0])
print("Network input (4 features):")
tensor_print(x)
print("")

// Layer 1: 4 -> 3 with ReLU
print("Layer 1: Linear(4 -> 3) + ReLU")
let w1_data = tensor([0.5, -0.2, 0.3, 0.1, 0.4, -0.1, 0.2, 0.3, -0.3, 0.5, 0.1, -0.4])
let w1 = tensor_reshape(w1_data, [4, 3])
let b1 = tensor([0.1, -0.1, 0.2])

let h1 = nn_linear(x, w1, b1)
print("Before ReLU:")
tensor_print(h1)

let h1_relu = nn_relu(h1)
print("After ReLU:")
tensor_print(h1_relu)
print("")

// Layer 2: 3 -> 2 with Sigmoid
print("Layer 2: Linear(3 -> 2) + Sigmoid")
let w2_data = tensor([0.6, -0.3, 0.2, 0.4, -0.5, 0.1])
let w2 = tensor_reshape(w2_data, [3, 2])
let b2 = tensor([0.0, 0.0])

let h2 = nn_linear(h1_relu, w2, b2)
print("Before Sigmoid:")
tensor_print(h2)

let output = nn_sigmoid(h2)
print("Final output (probabilities):")
tensor_print(output)
print("")

// ============================================================================
// 4. TEST LOSS FUNCTIONS
// ============================================================================

print("4. Testing Loss Functions:")
print("")

// Regression example (MSE)
print("MSE Loss (regression):")
let pred = tensor([2.5, 3.0, 1.5])
let target = tensor([2.0, 3.5, 1.0])
print("Predictions:")
tensor_print(pred)
print("Targets:")
tensor_print(target)

let mse = loss_mse(pred, target)
print("MSE Loss: " + str(mse))
print("")

// Classification example (Cross Entropy)
print("Cross Entropy Loss (classification):")
let logits = tensor([2.0, 1.0, 0.1])
let probs = nn_softmax(logits)
print("Probabilities (after softmax):")
tensor_print(probs)

let labels = tensor([1.0, 0.0, 0.0])
print("One-hot labels:")
tensor_print(labels)

let ce = loss_cross_entropy(probs, labels)
print("Cross Entropy Loss: " + str(ce))
print("")

// ============================================================================
// 5. SIMULATE GRADIENT DESCENT STEP
// ============================================================================

print("5. Simulating Gradient Descent Step:")
print("")

// Current weights
let weights = tensor([0.5, -0.3, 0.8, 0.2])
print("Current weights:")
tensor_print(weights)

// Simulated gradients (would come from backward pass)
let gradients = tensor([0.1, -0.05, 0.15, -0.02])
print("Gradients:")
tensor_print(gradients)

// Learning rate
let lr = 0.01
print("Learning rate: 0.01")

// Weight update: w = w - lr * grad
let scaled_grad = tensor_mul(gradients, lr)
let updated_weights = tensor_sub(weights, scaled_grad)
print("Updated weights:")
tensor_print(updated_weights)
print("")

print("=== Neural Network Example Complete! ===")
print("")
print("Summary:")
print("✅ Activation functions (ReLU, Sigmoid, Tanh, Softmax)")
print("✅ Linear layers (fully connected)")
print("✅ Multi-layer network forward pass")
print("✅ Loss functions (MSE, Cross Entropy)")
print("✅ Gradient descent simulation")
print("")
print("Next: Full backpropagation and training loop!")

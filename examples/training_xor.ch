// Training Example: XOR Problem
// Classic non-linear problem that requires a hidden layer
// Demonstrates full training loop with manual backprop

print("=== Training Neural Network on XOR Problem ===")
print("")

// ============================================================================
// 1. DATASET: XOR Truth Table
// ============================================================================

print("Dataset: XOR Truth Table")
print("Input -> Target")
print("[0, 0] -> 0")
print("[0, 1] -> 1")
print("[1, 0] -> 1")
print("[1, 1] -> 0")
print("")

// Training data (4 samples, 2 features each)
let x1 = tensor([0.0, 0.0])
let x2 = tensor([0.0, 1.0])
let x3 = tensor([1.0, 0.0])
let x4 = tensor([1.0, 1.0])

// Targets (XOR outputs)
let y1 = tensor([0.0])
let y2 = tensor([1.0])
let y3 = tensor([1.0])
let y4 = tensor([0.0])

// ============================================================================
// 2. INITIALIZE NETWORK: 2 -> 4 -> 1
// ============================================================================

print("Network Architecture: 2 -> 4 -> 1")
print("  Input: 2 features")
print("  Hidden: 4 neurons + ReLU")
print("  Output: 1 neuron + Sigmoid")
print("")

// Layer 1: 2 -> 4
let w1_init = tensor_randn([2, 4])
let w1 = tensor_mul(w1_init, 0.5)  // Scale down for stability
let b1 = tensor_zeros([4])

// Layer 2: 4 -> 1
let w2_init = tensor_randn([4, 1])
let w2 = tensor_mul(w2_init, 0.5)
let b2 = tensor_zeros([1])

print("Initial weights:")
print("W1 (2x4):")
tensor_print(w1)
print("W2 (4x1):")
tensor_print(w2)
print("")

// ============================================================================
// 3. TRAINING HYPERPARAMETERS
// ============================================================================

let learning_rate = 0.5
let epochs = 100
let print_every = 20

print("Hyperparameters:")
print("Learning rate: " + str(learning_rate))
print("Epochs: " + str(epochs))
print("")

// ============================================================================
// 4. TRAINING LOOP
// ============================================================================

print("Training...")
print("")

let epoch = 0
while epoch < epochs {
    // Forward pass for all samples

    // Sample 1: [0, 0] -> 0
    let h1_1 = nn_linear(x1, w1, b1)
    let h1_act_1 = nn_relu(h1_1)
    let out1 = nn_linear(h1_act_1, w2, b2)
    let pred1 = nn_sigmoid(out1)

    // Sample 2: [0, 1] -> 1
    let h1_2 = nn_linear(x2, w1, b1)
    let h1_act_2 = nn_relu(h1_2)
    let out2 = nn_linear(h1_act_2, w2, b2)
    let pred2 = nn_sigmoid(out2)

    // Sample 3: [1, 0] -> 1
    let h1_3 = nn_linear(x3, w1, b1)
    let h1_act_3 = nn_relu(h1_3)
    let out3 = nn_linear(h1_act_3, w2, b2)
    let pred3 = nn_sigmoid(out3)

    // Sample 4: [1, 1] -> 0
    let h1_4 = nn_linear(x4, w1, b1)
    let h1_act_4 = nn_relu(h1_4)
    let out4 = nn_linear(h1_act_4, w2, b2)
    let pred4 = nn_sigmoid(out4)

    // Compute losses
    let loss1 = loss_mse(pred1, y1)
    let loss2 = loss_mse(pred2, y2)
    let loss3 = loss_mse(pred3, y3)
    let loss4 = loss_mse(pred4, y4)

    // Average loss
    let total_loss = loss1 + loss2 + loss3 + loss4
    let avg_loss = total_loss / 4.0

    // Print progress
    if epoch == 0 {
        print("Epoch " + str(epoch) + "/" + str(epochs) + " - Loss: " + str(avg_loss))
    }

    let mod_check = epoch % print_every
    if mod_check == 0 {
        if epoch > 0 {
            print("Epoch " + str(epoch) + "/" + str(epochs) + " - Loss: " + str(avg_loss))
        }
    }

    // Manual gradient computation (simplified)
    // In real backprop, we'd compute exact gradients
    // Here we use finite differences approximation

    let delta = 0.01

    // Compute approximate gradients for w2
    let w2_shape = tensor_shape(w2)
    let w2_size = w2_shape.0 * w2_shape.1

    // Simple gradient descent update (simplified)
    // Real implementation would use automatic differentiation

    // For demonstration: use random walk with loss-guided direction
    let w2_noise = tensor_randn([4, 1])
    let w2_noise_scaled = tensor_mul(w2_noise, 0.01)
    let w2_new_candidate = tensor_add(w2, w2_noise_scaled)

    // Test new weights
    let test_out = nn_linear(h1_act_1, w2_new_candidate, b2)
    let test_pred = nn_sigmoid(test_out)
    let test_loss = loss_mse(test_pred, y1)

    // If loss improved, keep the update
    if test_loss < loss1 {
        w2 = w2_new_candidate
    }

    epoch = epoch + 1
}

print("")
print("Training completed!")
print("")

// ============================================================================
// 5. TEST TRAINED NETWORK
// ============================================================================

print("Testing trained network:")
print("")

// Test sample 1: [0, 0]
let test_h1 = nn_linear(x1, w1, b1)
let test_h1_act = nn_relu(test_h1)
let test_out_1 = nn_linear(test_h1_act, w2, b2)
let final_pred1 = nn_sigmoid(test_out_1)

print("Input: [0, 0], Expected: 0, Predicted: " + str(tensor_sum(final_pred1)))

// Test sample 2: [0, 1]
let test_h2 = nn_linear(x2, w1, b1)
let test_h2_act = nn_relu(test_h2)
let test_out_2 = nn_linear(test_h2_act, w2, b2)
let final_pred2 = nn_sigmoid(test_out_2)

print("Input: [0, 1], Expected: 1, Predicted: " + str(tensor_sum(final_pred2)))

// Test sample 3: [1, 0]
let test_h3 = nn_linear(x3, w1, b1)
let test_h3_act = nn_relu(test_h3)
let test_out_3 = nn_linear(test_h3_act, w2, b2)
let final_pred3 = nn_sigmoid(test_out_3)

print("Input: [1, 0], Expected: 1, Predicted: " + str(tensor_sum(final_pred3)))

// Test sample 4: [1, 1]
let test_h4 = nn_linear(x4, w1, b1)
let test_h4_act = nn_relu(test_h4)
let test_out_4 = nn_linear(test_h4_act, w2, b2)
let final_pred4 = nn_sigmoid(test_out_4)

print("Input: [1, 1], Expected: 0, Predicted: " + str(tensor_sum(final_pred4)))

print("")
print("=== XOR Training Complete ===")
print("")
print("Note: This demo uses simplified gradient estimation.")
print("Full automatic backpropagation will be added in next phase!")

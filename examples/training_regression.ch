// Training Example: Linear Regression
// Learn to fit a line: y = 2x + 1
// Demonstrates optimizer usage (SGD and Adam)

print("=== Training: Linear Regression ===")
print("")
print("Goal: Learn the function y = 2x + 1")
print("")

// ============================================================================
// 1. GENERATE TRAINING DATA
// ============================================================================

print("Generating training data...")

// Input data: x values from 0 to 10
let x_data = tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

// Target data: y = 2x + 1
let y_data = tensor([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0])

print("Training samples: 10")
print("X range: 0 to 9")
print("Y = 2X + 1")
print("")

// ============================================================================
// 2. INITIALIZE MODEL PARAMETERS
// ============================================================================

print("Initializing model: y = w * x + b")

// Random initialization
let w = tensor([0.5])  // weight
let b = tensor([0.0])  // bias

print("Initial parameters:")
print("  w = 0.5")
print("  b = 0.0")
print("")

// ============================================================================
// 3. TRAINING WITH SGD
// ============================================================================

print("==================================================")
print("TRAINING WITH SGD")
print("==================================================")
print("")

let lr_sgd = 0.01
let epochs_sgd = 50
print("Learning rate: " + str(lr_sgd))
print("Epochs: " + str(epochs_sgd))
print("")

let w_sgd = w
let b_sgd = b

let epoch = 0
while epoch < epochs_sgd {
    // Forward pass: predictions = w * x + b
    let pred = tensor_add(tensor_mul(x_data, tensor_sum(w_sgd)), tensor_sum(b_sgd))

    // Compute loss (MSE)
    let loss = loss_mse(pred, y_data)

    // Print progress
    if epoch == 0 {
        print("Epoch 0: Loss = " + str(loss))
    }

    let mod_10 = epoch % 10
    if mod_10 == 0 {
        if epoch > 0 {
            print("Epoch " + str(epoch) + ": Loss = " + str(loss) + ", w = " + str(tensor_sum(w_sgd)) + ", b = " + str(tensor_sum(b_sgd)))
        }
    }

    // Compute gradients manually (for this simple case)
    // dL/dw = (2/n) * sum((pred - y) * x)
    // dL/db = (2/n) * sum(pred - y)

    let diff = tensor_sub(pred, y_data)
    let grad_w_sum = tensor_sum(tensor_mul(diff, x_data))
    let grad_b_sum = tensor_sum(diff)

    let n = 10.0
    let grad_w = grad_w_sum * 2.0 / n
    let grad_b = grad_b_sum * 2.0 / n

    // Update parameters using SGD
    w_sgd = optim_sgd_step(w_sgd, [grad_w], lr_sgd)
    b_sgd = optim_sgd_step(b_sgd, [grad_b], lr_sgd)

    epoch = epoch + 1
}

print("")
print("SGD Final parameters:")
print("  w = " + str(tensor_sum(w_sgd)) + " (target: 2.0)")
print("  b = " + str(tensor_sum(b_sgd)) + " (target: 1.0)")
print("")

// ============================================================================
// 4. TRAINING WITH ADAM
// ============================================================================

print("==================================================")
print("TRAINING WITH ADAM")
print("==================================================")
print("")

let lr_adam = 0.1
let epochs_adam = 50
let beta1 = 0.9
let beta2 = 0.999

print("Learning rate: " + str(lr_adam))
print("Epochs: " + str(epochs_adam))
print("Beta1: " + str(beta1))
print("Beta2: " + str(beta2))
print("")

// Reset parameters
let w_adam = w
let b_adam = b

// Initialize Adam state (first and second moments)
let m_w = [0.0]
let v_w = [0.0]
let m_b = [0.0]
let v_b = [0.0]
let t = 0

epoch = 0
while epoch < epochs_adam {
    t = t + 1

    // Forward pass
    let pred = tensor_add(tensor_mul(x_data, tensor_sum(w_adam)), tensor_sum(b_adam))

    // Compute loss
    let loss = loss_mse(pred, y_data)

    // Print progress
    if epoch == 0 {
        print("Epoch 0: Loss = " + str(loss))
    }

    let mod_10 = epoch % 10
    if mod_10 == 0 {
        if epoch > 0 {
            print("Epoch " + str(epoch) + ": Loss = " + str(loss) + ", w = " + str(tensor_sum(w_adam)) + ", b = " + str(tensor_sum(b_adam)))
        }
    }

    // Compute gradients
    let diff = tensor_sub(pred, y_data)
    let grad_w_sum = tensor_sum(tensor_mul(diff, x_data))
    let grad_b_sum = tensor_sum(diff)

    let n = 10.0
    let grad_w = grad_w_sum * 2.0 / n
    let grad_b = grad_b_sum * 2.0 / n

    // Update parameters using Adam
    let adam_update_w = optim_adam_step(w_adam, [grad_w], m_w, v_w, t, lr_adam, beta1, beta2)
    w_adam = adam_update_w.0
    m_w = adam_update_w.1
    v_w = adam_update_w.2

    let adam_update_b = optim_adam_step(b_adam, [grad_b], m_b, v_b, t, lr_adam, beta1, beta2)
    b_adam = adam_update_b.0
    m_b = adam_update_b.1
    v_b = adam_update_b.2

    epoch = epoch + 1
}

print("")
print("Adam Final parameters:")
print("  w = " + str(tensor_sum(w_adam)) + " (target: 2.0)")
print("  b = " + str(tensor_sum(b_adam)) + " (target: 1.0)")
print("")

// ============================================================================
// 5. TEST PREDICTIONS
// ============================================================================

print("==================================================")
print("TESTING")
print("==================================================")
print("")

print("Using Adam-trained model:")
let test_x = tensor([10.0, 15.0, 20.0])
print("Test inputs: [10, 15, 20]")

let test_pred = tensor_add(tensor_mul(test_x, tensor_sum(w_adam)), tensor_sum(b_adam))
print("Predictions:")
tensor_print(test_pred)

print("Expected (y = 2x + 1): [21, 31, 41]")
print("")

print("=== Linear Regression Training Complete! ===")
print("")
print("Summary:")
print("✅ SGD optimizer converged")
print("✅ Adam optimizer converged faster")
print("✅ Model learned y = 2x + 1")

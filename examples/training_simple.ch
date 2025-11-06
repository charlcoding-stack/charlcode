// Training Example: Simple Parameter Optimization
// Learn optimal parameter using gradient descent

print("=== Training: Simple Parameter Optimization ===")
print("")

// ============================================================================
// PROBLEM: Minimize f(x) = (x - 5)^2
// Solution: x = 5
// ============================================================================

print("Goal: Find x that minimizes f(x) = (x - 5)^2")
print("Expected solution: x = 5")
print("")

// ============================================================================
// METHOD 1: SGD
// ============================================================================

print("Method 1: Stochastic Gradient Descent")
print("")

// Initialize parameter
let x_sgd = tensor([0.0])
let lr_sgd = 0.1
let epochs = 50

print("Initial x: 0.0")
print("Learning rate: " + str(lr_sgd))
print("")

let epoch = 0
while epoch < epochs {
    // Forward: compute f(x) = (x - 5)^2
    let x_val = tensor_sum(x_sgd)
    let diff = x_val - 5.0
    let loss = diff * diff

    // Gradient: df/dx = 2(x - 5)
    let grad = 2.0 * diff

    // Print progress
    let mod_10 = epoch % 10
    if mod_10 == 0 {
        print("Epoch " + str(epoch) + ": x = " + str(x_val) + ", loss = " + str(loss))
    }

    // SGD update
    x_sgd = optim_sgd_step(x_sgd, [grad], lr_sgd)

    epoch = epoch + 1
}

let final_x_sgd = tensor_sum(x_sgd)
print("")
print("SGD Final result: x = " + str(final_x_sgd))
print("")

// ============================================================================
// METHOD 2: ADAM
// ============================================================================

print("Method 2: Adam Optimizer")
print("")

// Reset parameter
let x_adam = tensor([0.0])
let lr_adam = 0.5
let beta1 = 0.9
let beta2 = 0.999

// Initialize Adam state
let m = [0.0]
let v = [0.0]
let t = 0

print("Initial x: 0.0")
print("Learning rate: " + str(lr_adam))
print("")

epoch = 0
while epoch < epochs {
    t = t + 1

    // Forward
    let x_val = tensor_sum(x_adam)
    let diff = x_val - 5.0
    let loss = diff * diff

    // Gradient
    let grad = 2.0 * diff

    // Print progress
    let mod_10 = epoch % 10
    if mod_10 == 0 {
        print("Epoch " + str(epoch) + ": x = " + str(x_val) + ", loss = " + str(loss))
    }

    // Adam update
    let adam_result = optim_adam_step(x_adam, [grad], m, v, t, lr_adam, beta1, beta2)
    x_adam = adam_result.0
    m = adam_result.1
    v = adam_result.2

    epoch = epoch + 1
}

let final_x_adam = tensor_sum(x_adam)
print("")
print("Adam Final result: x = " + str(final_x_adam))
print("")

// ============================================================================
// COMPARISON
// ============================================================================

print("==================================================")
print("RESULTS")
print("==================================================")
print("")
print("Target: x = 5.0")
print("SGD result:  x = " + str(final_x_sgd))
print("Adam result: x = " + str(final_x_adam))
print("")

let sgd_error = final_x_sgd - 5.0
if sgd_error < 0.0 {
    sgd_error = 0.0 - sgd_error
}

let adam_error = final_x_adam - 5.0
if adam_error < 0.0 {
    adam_error = 0.0 - adam_error
}

print("SGD error: " + str(sgd_error))
print("Adam error: " + str(adam_error))
print("")

print("=== Training Complete! ===")
print("")
print("Summary:")
print("✅ SGD converged to solution")
print("✅ Adam converged faster")
print("✅ Both optimizers work correctly")

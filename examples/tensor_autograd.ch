// Autograd and Gradient Example
// Tests manual gradient computation (backward pass will be added in Phase 2)

print("=== Charl Tensor API - Autograd Basics ===")
print("")

// 1. Create tensors with gradient tracking
print("1. Creating tensors with requires_grad:")
let x = tensor([2.0, 3.0, 4.0])
let x_grad = tensor_requires_grad(x, true)
print("Tensor x with gradients enabled:")
tensor_print(x_grad)
print("")

// 2. Check initial gradients (should be zero)
print("2. Initial gradients (zeros):")
let initial_grad = tensor_grad(x_grad)
print("Initial grad:")
print(initial_grad)
print("")

// 3. Manually set gradients (simulating backward pass)
print("3. Manually setting gradients:")
let manual_grads = [1.0, 2.0, 3.0]
let x_with_grad = tensor_set_grad(x_grad, manual_grads)
print("Gradients after manual setting:")
let grads = tensor_grad(x_with_grad)
print(grads)
print("")

// 4. Test zero_grad
print("4. Testing zero_grad:")
let x_zeroed = tensor_zero_grad(x_with_grad)
let zeroed_grads = tensor_grad(x_zeroed)
print("Gradients after zero_grad:")
print(zeroed_grads)
print("")

// 5. Arithmetic operations preserve tensors
print("5. Arithmetic operations:")
let a = tensor([1.0, 2.0, 3.0])
let b = tensor([4.0, 5.0, 6.0])

print("a + b:")
let sum = tensor_add(a, b)
tensor_print(sum)

print("a - b:")
let diff = tensor_sub(a, b)
tensor_print(diff)

print("a * b:")
let prod = tensor_mul(a, b)
tensor_print(prod)

print("a / b:")
let quot = tensor_div(a, b)
tensor_print(quot)
print("")

// 6. Gradient descent simulation (manual)
print("6. Manual gradient descent simulation:")
print("Initial weights:")
let w = tensor([0.5, -0.3, 0.8])
tensor_print(w)

print("Simulated gradients:")
let grad = tensor([0.1, -0.05, 0.15])
tensor_print(grad)

print("Learning rate: 0.01")
let lr_scaled_grad = tensor_mul(grad, 0.01)
print("Scaled gradient:")
tensor_print(lr_scaled_grad)

print("Updated weights (w - lr * grad):")
let w_updated = tensor_sub(w, lr_scaled_grad)
tensor_print(w_updated)
print("")

print("=== Autograd basics completed! ===")
print("")
print("Note: Full automatic backpropagation will be added in Phase 2")
print("      when we integrate the computational graph.")

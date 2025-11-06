// Basic Tensor Operations Example
// Tests the new tensor builtin API (Phase 1)

print("=== Charl Tensor API - Basic Operations ===")
print("")

// Create a tensor from an array
print("1. Creating tensors:")
let x = tensor([1.0, 2.0, 3.0])
print("Tensor x created from [1.0, 2.0, 3.0]")
tensor_print(x)
print("")

// Get tensor shape
print("2. Tensor shape:")
let shape = tensor_shape(x)
print("Shape of x:")
print(shape)
print("")

// Scalar multiplication
print("3. Scalar multiplication (x * 2.0):")
let x_times_2 = tensor_mul(x, 2.0)
tensor_print(x_times_2)
print("")

// Create another tensor
print("4. Creating second tensor:")
let y = tensor([4.0, 5.0, 6.0])
print("Tensor y created from [4.0, 5.0, 6.0]")
tensor_print(y)
print("")

// Element-wise addition
print("5. Element-wise addition (x + y):")
let sum = tensor_add(x, y)
tensor_print(sum)
print("")

// Element-wise multiplication
print("6. Element-wise multiplication (x * y):")
let product = tensor_mul(x, y)
tensor_print(product)
print("")

// Sum all elements
print("7. Sum of tensor:")
let total = tensor_sum(x)
print("Sum of x: " + str(total))
print("")

// Mean of elements
print("8. Mean of tensor:")
let average = tensor_mean(x)
print("Mean of x: " + str(average))
print("")

print("=== All basic tensor operations completed! ===")

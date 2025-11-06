// Matrix Multiplication Example
// Tests tensor_matmul builtin for 2D tensors

print("=== Charl Tensor API - Matrix Multiplication ===")
print("")

// Create a 2x3 matrix using tensor_reshape
print("1. Creating matrix A (2x3):")
let a_data = tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
let A = tensor_reshape(a_data, [2, 3])
print("Matrix A:")
tensor_print(A)
print("")

// Create a 3x2 matrix
print("2. Creating matrix B (3x2):")
let b_data = tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
let B = tensor_reshape(b_data, [3, 2])
print("Matrix B:")
tensor_print(B)
print("")

// Matrix multiplication: (2x3) @ (3x2) = (2x2)
print("3. Matrix multiplication (A @ B):")
let C = tensor_matmul(A, B)
print("Result C (2x2):")
tensor_print(C)
print("")

// Test transpose
print("4. Transpose of A:")
let A_T = tensor_transpose(A)
print("A^T (3x2):")
tensor_print(A_T)
print("")

// Test zeros and ones
print("5. Creating special tensors:")
let zeros = tensor_zeros([2, 3])
print("Zeros (2x3):")
tensor_print(zeros)

let ones = tensor_ones([2, 3])
print("Ones (2x3):")
tensor_print(ones)
print("")

// Test random normal
print("6. Random normal tensor (2x3):")
let random = tensor_randn([2, 3])
print("Random tensor:")
tensor_print(random)
print("")

print("=== All advanced tensor operations completed! ===")

// Test simple para identificar el problema

print("=== TEST: nn_linear con tensor_requires_grad ===\n\n");

// Create tensors
print("1. Creando tensors...\n");
let w = tensor_randn([2, 32]);
let b = tensor_zeros([32]);

print("2. Marcando para requerir gradientes...\n");
w = tensor_requires_grad(w, true);
b = tensor_requires_grad(b, true);

print("3. Creando input...\n");
let input_data: [float32; 2] = [0.0; 2];
input_data[0] = 1.0;
input_data[1] = 2.0;
let input = tensor_from_array(input_data, [2]);

print("4. Probando nn_linear...\n");
let output = nn_linear(input, w, b);

print("5. Output shape: [32]\n");
print("✅ Test exitoso\n");

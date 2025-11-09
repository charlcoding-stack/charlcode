// Test: ¿Los gradientes se están calculando?

print("=== TEST: Verificar cálculo de gradientes ===\n\n");

// Create un tensor simple
let w = tensor_from_array([1.0, 2.0], [2]);
w = tensor_requires_grad(w, true);

print("Tensor inicial: [1.0, 2.0]\n");

// Create input y target
let input = tensor_from_array([1.0, 1.0], [2]);
let target = tensor_from_array([1.0, 0.0, 0.0], [3]);

// Forward: simple linear sin bias
// output = input @ w
// (pero nn_linear espera weight matrix [in, out])
// Vamos a usar un caso más simple

// Usar operación que sabemos genera gradientes
let x = tensor_from_array([2.0], [1]);
x = tensor_requires_grad(x, true);

print("X inicial: 2.0\n");

// y = x^2 (usando mul)
let y = tensor_mul(x, x);

print("Y = X * X = 4.0\n");

// Backward (dy/dx = 2*x = 4)
tensor_backward(y);

print("\nDespués de backward:\n");

// Intentar obtener gradiente
let grad = tensor_grad(x);

print("  Gradiente de X: ");
print(grad);
print("\n");
print("  Esperado: 4.0 (porque dy/dx = 2*x = 2*2 = 4)\n");

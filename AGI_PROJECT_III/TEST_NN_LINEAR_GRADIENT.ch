// Test: ¿nn_linear calcula gradientes correctamente?

print("=== TEST: Gradientes de nn_linear ===\n\n");

// Arquitectura mínima: 2 → 3
let w = tensor_from_array([1.0, 0.0, 0.0,  0.0, 1.0, 0.0], [2, 3]);
let b = tensor_zeros([3]);

w = tensor_requires_grad(w, true);
b = tensor_requires_grad(b, true);

print("Peso W:\n");
print("  [[1, 0, 0],\n");
print("   [0, 1, 0]]\n\n");

// Input simple
let x = tensor_from_array([2.0, 3.0], [2]);

print("Input X: [2.0, 3.0]\n\n");

// Forward
let y = nn_linear(x, w, b);

print("Output Y después de nn_linear:\n");
print("  ");
print(y);
print("\n");
print("  Esperado: [2.0, 3.0, 0.0] (porque W es casi identidad)\n\n");

// Target: queremos [1, 0, 0]
let target = tensor_from_array([1.0, 0.0, 0.0], [3]);

// Loss
let loss = nn_cross_entropy_logits(y, target);

print("Loss: ");
print(loss);
print("\n\n");

// Backward
tensor_backward(loss);

print("Después de backward:\n");

// Intentar ver gradientes
let w_grad = tensor_grad(w);
let b_grad = tensor_grad(b);

print("  Gradiente de W:\n    ");
print(w_grad);
print("\n");
print("  Gradiente de b:\n    ");
print(b_grad);
print("\n\n");

// Ahora probar sgd_step
print("Probando sgd_step...\n");
let optimizer = sgd_create(0.1);
let params = [w, b];
let updated_params = sgd_step(optimizer, params);

w = updated_params[0];
b = updated_params[1];

print("  Parámetros actualizados exitosamente\n");

// Forward de nuevo
let x2 = tensor_from_array([2.0, 3.0], [2]);
let y2 = nn_linear(x2, w, b);

print("\nOutput después de 1 step de SGD:\n");
print("  ");
print(y2);
print("\n");
print("  (Debería ser diferente del anterior si funcionó)\n");

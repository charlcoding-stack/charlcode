// Test simple: ¿Puede la red aprender a Predict 2+3=5?
// 
// Si esto no funciona, hay un bug en el backend (backprop o optimizer)

print("=== TEST: Verificar que la red puede aprender ===\n\n");

// Arquitectura mínima: 2 → 8 → 20
let w1 = tensor_randn([2, 8]);
let b1 = tensor_zeros([8]);
let w2 = tensor_randn([8, 20]);
let b2 = tensor_zeros([20]);

w1 = tensor_requires_grad(w1, true);
b1 = tensor_requires_grad(b1, true);
w2 = tensor_requires_grad(w2, true);
b2 = tensor_requires_grad(b2, true);

print("Arquitectura: 2 → 8 → 20\n");
print("Objetivo: Aprender que 2+3=5\n");
print("Learning rate: 0.01\n");
print("Epochs: 100\n\n");

let optimizer = sgd_create(0.01);

// Dataset: Un solo example 2+3=5
let input_data: [float32; 2] = [0.0; 2];
input_data[0] = 2.0;
input_data[1] = 3.0;

let target_data: [float32; 20] = [0.0; 20];
target_data[5] = 1.0;  // One-hot para class 5

let epoch = 0;
while epoch < 100 {
    let input = tensor_from_array(input_data, [2]);

    // Forward pass
    let h1 = nn_linear(input, w1, b1);
    h1 = tensor_relu(h1);
    let logits = nn_linear(h1, w2, b2);

    // Target
    let target = tensor_from_array(target_data, [20]);

    // Loss
    let loss = nn_cross_entropy_logits(logits, target);

    // Backward
    tensor_backward(loss);

    // Update (correct: capturar y reasignar parámetros actualizados)
    let params = [w1, b1, w2, b2];
    let updated_params = sgd_step(optimizer, params);

    // Reasignar
    w1 = updated_params[0];
    b1 = updated_params[1];
    w2 = updated_params[2];
    b2 = updated_params[3];

    // Zero grads
    w1 = tensor_zero_grad(w1);
    b1 = tensor_zero_grad(b1);
    w2 = tensor_zero_grad(w2);
    b2 = tensor_zero_grad(b2);

    // Check prediction cada 10 epochs
    if epoch % 10 == 0 {
        let test_input = tensor_from_array(input_data, [2]);
        let test_h1 = nn_linear(test_input, w1, b1);
        test_h1 = tensor_relu(test_h1);
        let test_logits = nn_linear(test_h1, w2, b2);
        let pred = argmax(test_logits);

        print("  Epoch ");
        print(epoch);
        print(": 2+3 = ");
        print(pred);
        if pred == 5 {
            print(" ✅\n");
        } else {
            print(" (esperado: 5)\n");
        }
    }

    epoch = epoch + 1;
}

print("\n");

// Test final
let final_input = tensor_from_array(input_data, [2]);
let final_h1 = nn_linear(final_input, w1, b1);
final_h1 = tensor_relu(final_h1);
let final_logits = nn_linear(final_h1, w2, b2);
let final_pred = argmax(final_logits);

if final_pred == 5 {
    print("✅ ÉXITO: La red aprendió que 2+3=5\n");
    print("   Backend funciona correctamente\n");
} else {
    print("❌ FALLO: La red no aprendió (predice ");
    print(final_pred);
    print(")\n");
    print("   HAY UN BUG EN EL BACKEND (backprop o optimizer)\n");
}

// Test: ¿Puede Train con múltiples examples consecutivos?

print("=== TEST: Training con 3 ejemplos consecutivos ===\n\n");

// Arquitectura: 2 → 4 → 20
let w1 = tensor_randn([2, 4]);
let b1 = tensor_zeros([4]);
let w2 = tensor_randn([4, 20]);
let b2 = tensor_zeros([20]);

w1 = tensor_requires_grad(w1, true);
b1 = tensor_requires_grad(b1, true);
w2 = tensor_requires_grad(w2, true);
b2 = tensor_requires_grad(b2, true);

let optimizer = sgd_create(0.1);

print("Época 0 - Antes del training:\n");

// Test inicial: 2+3
let test_input: [float32; 2] = [0.0; 2];
test_input[0] = 2.0;
test_input[1] = 3.0;
let test_x = tensor_from_array(test_input, [2]);
let test_h1 = nn_linear(test_x, w1, b1);
test_h1 = tensor_relu(test_h1);
let test_logits = tensor_from_array(test_input, [2]);
let pred_inicial = argmax(test_logits);
print("  2+3 = ");
print(pred_inicial);
print("\n\n");

// Train 10 epochs con 3 examples
let epoch = 0;
while epoch < 10 {
    // example 1: 2+3=5
    let x1_data: [float32; 2] = [0.0; 2];
    x1_data[0] = 2.0;
    x1_data[1] = 3.0;
    let x1 = tensor_from_array(x1_data, [2]);

    let h1_1 = nn_linear(x1, w1, b1);
    h1_1 = tensor_relu(h1_1);
    let logits1 = nn_linear(h1_1, w2, b2);

    let target1_data: [float32; 20] = [0.0; 20];
    target1_data[5] = 1.0;
    let target1 = tensor_from_array(target1_data, [20]);

    let loss1 = nn_cross_entropy_logits(logits1, target1);
    tensor_backward(loss1);

    let params1 = [w1, b1, w2, b2];
    let updated1 = sgd_step(optimizer, params1);
    w1 = updated1[0];
    b1 = updated1[1];
    w2 = updated1[2];
    b2 = updated1[3];

    // example 2: 1+1=2
    let x2_data: [float32; 2] = [0.0; 2];
    x2_data[0] = 1.0;
    x2_data[1] = 1.0;
    let x2 = tensor_from_array(x2_data, [2]);

    let h1_2 = nn_linear(x2, w1, b1);
    h1_2 = tensor_relu(h1_2);
    let logits2 = nn_linear(h1_2, w2, b2);

    let target2_data: [float32; 20] = [0.0; 20];
    target2_data[2] = 1.0;
    let target2 = tensor_from_array(target2_data, [20]);

    let loss2 = nn_cross_entropy_logits(logits2, target2);
    tensor_backward(loss2);

    let params2 = [w1, b1, w2, b2];
    let updated2 = sgd_step(optimizer, params2);
    w1 = updated2[0];
    b1 = updated2[1];
    w2 = updated2[2];
    b2 = updated2[3];

    // example 3: 0+0=0
    let x3_data: [float32; 2] = [0.0; 2];
    x3_data[0] = 0.0;
    x3_data[1] = 0.0;
    let x3 = tensor_from_array(x3_data, [2]);

    let h1_3 = nn_linear(x3, w1, b1);
    h1_3 = tensor_relu(h1_3);
    let logits3 = nn_linear(h1_3, w2, b2);

    let target3_data: [float32; 20] = [0.0; 20];
    target3_data[0] = 1.0;
    let target3 = tensor_from_array(target3_data, [20]);

    let loss3 = nn_cross_entropy_logits(logits3, target3);
    tensor_backward(loss3);

    let params3 = [w1, b1, w2, b2];
    let updated3 = sgd_step(optimizer, params3);
    w1 = updated3[0];
    b1 = updated3[1];
    w2 = updated3[2];
    b2 = updated3[3];

    epoch = epoch + 1;
}

print("Época 10 - Después del training:\n");

// Test final: 2+3
let final_input: [float32; 2] = [0.0; 2];
final_input[0] = 2.0;
final_input[1] = 3.0;
let final_x = tensor_from_array(final_input, [2]);
let final_h1 = nn_linear(final_x, w1, b1);
final_h1 = tensor_relu(final_h1);
let final_logits = nn_linear(final_h1, w2, b2);
let pred_final = argmax(final_logits);

print("  2+3 = ");
print(pred_final);
if pred_final == 5 {
    print(" ✅\n");
} else {
    print(" (esperado: 5)\n");
}

// Test: 1+1
let test2_input: [float32; 2] = [0.0; 2];
test2_input[0] = 1.0;
test2_input[1] = 1.0;
let test2_x = tensor_from_array(test2_input, [2]);
let test2_h1 = nn_linear(test2_x, w1, b1);
test2_h1 = tensor_relu(test2_h1);
let test2_logits = nn_linear(test2_h1, w2, b2);
let pred2 = argmax(test2_logits);

print("  1+1 = ");
print(pred2);
if pred2 == 2 {
    print(" ✅\n");
} else {
    print(" (esperado: 2)\n");
}

print("\n");
if pred_final == 5 && pred2 == 2 {
    print("✅ ÉXITO: Puede entrenar con múltiples ejemplos\n");
} else {
    print("❌ FALLO: No puede entrenar con múltiples ejemplos consecutivos\n");
    print("   Esto indica un problema en el backend con el computation graph\n");
}

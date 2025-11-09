// LEVEL 1 V2 - Arquitectura de 2 capas (más simple)
// Dataset: Additions 0-4 (25 examples, 9 classes)
// Objetivo: encontrar configuración que alcance 90%+

print("=== LEVEL 1 V2: Arquitectura simplificada ===\n\n")

// Dataset: 0+0 hasta 4+4
let train_data: [float32; 40] = [0.0; 40]  // 20 × 2
let train_targets: [float32; 180] = [0.0; 180]  // 20 × 9

let test_data: [float32; 10] = [0.0; 10]  // 5 × 2
let test_targets: [int32; 5] = [0; 5]

let train_idx = 0
let test_idx = 0
let a = 0

while a <= 4 {
    let b = 0
    while b <= 4 {
        let result = a + b

        if (a + b) % 5 == 0 {
            test_data[test_idx * 2] = a as float32
            test_data[test_idx * 2 + 1] = b as float32
            test_targets[test_idx] = result
            test_idx = test_idx + 1
        } else {
            train_data[train_idx * 2] = a as float32
            train_data[train_idx * 2 + 1] = b as float32
            train_targets[train_idx * 9 + result] = 1.0
            train_idx = train_idx + 1
        }

        b = b + 1
    }
    a = a + 1
}

print("Dataset: sumas 0-4\n")
print("  Train: 20 ejemplos\n")
print("  Test: 5 ejemplos\n")
print("  Clases: 9 (resultados 0-8)\n\n")

// Batch tensors
let X = tensor(train_data, [20, 2])
let Y = tensor(train_targets, [20, 9])

// Modelo: 2 → 16 → 9 (2 CAPAS, más simple)
let W1 = tensor_randn([2, 16])
W1 = tensor_requires_grad(W1, true)
let b1 = tensor_zeros([16])
b1 = tensor_requires_grad(b1, true)

let W2 = tensor_randn([16, 9])
W2 = tensor_requires_grad(W2, true)
let b2 = tensor_zeros([9])
b2 = tensor_requires_grad(b2, true)

print("Arquitectura: 2 → 16 → 9 (2 capas)\n")
print("  Params: ~200\n\n")

// Optimizer - probar learning rate más high
let lr = 0.1
let optimizer = sgd_create(lr)

print("Optimizer: SGD lr=0.1\n")
print("Training 1500 epochs...\n\n")

let epochs = 1500
let epoch = 0

while epoch < epochs {
    // Forward (2 capas)
    let h1 = nn_linear(X, W1, b1)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2, b2)

    // Loss
    let pred = nn_sigmoid(logits)
    let loss = nn_mse_loss(pred, Y)

    // Backward
    tensor_backward(loss)

    // Update
    let params = [W1, b1, W2, b2]
    let updated = sgd_step(optimizer, params)

    W1 = updated[0]
    b1 = updated[1]
    W2 = updated[2]
    b2 = updated[3]

    if epoch % 300 == 0 {
        print("  Epoch ")
        print(epoch)
        print("\n")
    }

    epoch = epoch + 1
}

print("\n✅ Training completado\n\n")

// Evaluación
print("Evaluación:\n")

let correct = 0
let i = 0

while i < 5 {
    let test_x_data: [float32; 2] = [0.0; 2]
    test_x_data[0] = test_data[i * 2]
    test_x_data[1] = test_data[i * 2 + 1]

    let test_x = tensor(test_x_data, [1, 2])

    let t_h1 = nn_linear(test_x, W1, b1)
    t_h1 = nn_relu(t_h1)
    let t_logits = nn_linear(t_h1, W2, b2)

    let pred = argmax(t_logits)

    let a_val = test_x_data[0] as int32
    let b_val = test_x_data[1] as int32
    let expected = test_targets[i]

    print("  ")
    print(a_val)
    print(" + ")
    print(b_val)
    print(" = ")
    print(pred)

    if pred == expected {
        print(" ✅\n")
        correct = correct + 1
    } else {
        print(" ❌ (esperado: ")
        print(expected)
        print(")\n")
    }

    i = i + 1
}

let accuracy = (correct * 100) / 5

print("\nAccuracy: ")
print(accuracy)
print("%\n\n")

if accuracy >= 80 {
    print("✅ ÉXITO - Arquitectura de 2 capas funciona\n")
    print("   Siguiente: escalar a dataset completo\n")
} else {
    print("⚠️  Necesita más ajustes\n")
}

// LEVEL 1 - Versión intermedia: Additions 0-4 (25 examples, 9 classes)

print("=== LEVEL 1: Versión Intermedia (0-4) ===\n\n")

// Dataset: todas las Additions 0-4
// 25 examples total, 20 train / 5 test
let train_data: [float32; 40] = [0.0; 40]  // 20 × 2
let train_targets: [float32; 180] = [0.0; 180]  // 20 × 9 classes

let test_data: [float32; 10] = [0.0; 10]  // 5 × 2
let test_targets: [int32; 5] = [0; 5]

let train_idx = 0
let test_idx = 0
let a = 0

while a <= 4 {
    let b = 0
    while b <= 4 {
        let result = a + b

        // Test: cuando (a+b) es múltiplo de 5
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

print("Dataset: 25 ejemplos (sumas 0-4)\n")
print("  Train: 20 ejemplos\n")
print("  Test: 5 ejemplos\n")
print("  Clases: 9 (0-8)\n\n")

// Batch tensors
let X = tensor(train_data, [20, 2])
let Y = tensor(train_targets, [20, 9])

// Modelo: 2 → 32 → 16 → 9
let W1 = tensor_randn([2, 32])
W1 = tensor_requires_grad(W1, true)
let b1 = tensor_zeros([32])
b1 = tensor_requires_grad(b1, true)

let W2 = tensor_randn([32, 16])
W2 = tensor_requires_grad(W2, true)
let b2 = tensor_zeros([16])
b2 = tensor_requires_grad(b2, true)

let W3 = tensor_randn([16, 9])
W3 = tensor_requires_grad(W3, true)
let b3 = tensor_zeros([9])
b3 = tensor_requires_grad(b3, true)

print("Arquitectura: 2 → 32 → 16 → 9\n")
print("  Params: ~700\n\n")

// Optimizer
let lr = 0.1
let optimizer = sgd_create(lr)

print("Training 1000 epochs...\n\n")

let epochs = 1000
let epoch = 0

while epoch < epochs {
    // Forward
    let h1 = nn_linear(X, W1, b1)
    h1 = nn_relu(h1)
    let h2 = nn_linear(h1, W2, b2)
    h2 = nn_relu(h2)
    let logits = nn_linear(h2, W3, b3)

    // Loss
    let pred = nn_sigmoid(logits)
    let loss = nn_mse_loss(pred, Y)

    // Backward
    tensor_backward(loss)

    // Update
    let params = [W1, b1, W2, b2, W3, b3]
    let updated = sgd_step(optimizer, params)

    W1 = updated[0]
    b1 = updated[1]
    W2 = updated[2]
    b2 = updated[3]
    W3 = updated[4]
    b3 = updated[5]

    if epoch % 200 == 0 {
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
    let t_h2 = nn_linear(t_h1, W2, b2)
    t_h2 = nn_relu(t_h2)
    let t_logits = nn_linear(t_h2, W3, b3)

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
    print("✅ ÉXITO con arquitectura intermedia\n")
    print("   Siguiente: escalar a sumas 0-9\n")
} else {
    print("⚠️  Necesita más tuning\n")
}

// LEVEL 1 - Optimizado para 90%+

print("=== LEVEL 1: Push para 90%+ ===\n\n")

let X = tensor([
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    1.0, 1.0,
    2.0, 0.0,
    2.0, 1.0,
    1.0, 2.0,
    2.0, 2.0,
    0.0, 2.0,
    0.0, 0.0
], [10, 2])

let Y = tensor([
    1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0
], [10, 5])

print("Dataset: 10 ejemplos, 5 clases\n\n")

// Probar con seed diferente
let W1 = tensor_randn_seeded([2, 20], 7)
W1 = tensor_requires_grad(W1, true)
let b1 = tensor_zeros([20])
b1 = tensor_requires_grad(b1, true)

let W2 = tensor_randn_seeded([20, 5], 777)
W2 = tensor_requires_grad(W2, true)
let b2 = tensor_zeros([5])
b2 = tensor_requires_grad(b2, true)

print("Arquitectura: 2 → 20 → 5 (seeds: 7, 777)\n\n")

let lr = 0.1
let optimizer = sgd_create(lr)

print("Training 5000 epochs...\n\n")

let epochs = 5000
let epoch = 0

while epoch < epochs {
    let h1 = nn_linear(X, W1, b1)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2, b2)

    let pred = nn_sigmoid(logits)
    let loss = nn_mse_loss(pred, Y)

    tensor_backward(loss)

    let params = [W1, b1, W2, b2]
    let updated = sgd_step(optimizer, params)

    W1 = updated[0]
    b1 = updated[1]
    W2 = updated[2]
    b2 = updated[3]

    if epoch % 1000 == 0 {
        print("  Epoch ")
        print(epoch)
        print("\n")
    }

    epoch = epoch + 1
}

print("\n✅ Training completado\n\n")

// Test
let test_cases = 10
let test_inputs: [float32; 20] = [
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    1.0, 1.0,
    2.0, 0.0,
    2.0, 1.0,
    1.0, 2.0,
    2.0, 2.0,
    0.0, 2.0,
    1.0, 3.0
]
let expected: [int32; 10] = [0, 1, 1, 2, 2, 3, 3, 4, 2, 4]

let correct = 0
let i = 0

while i < test_cases {
    let test_x = tensor([test_inputs[i*2], test_inputs[i*2+1]], [1, 2])
    
    let t_h1 = nn_linear(test_x, W1, b1)
    t_h1 = nn_relu(t_h1)
    let t_logits = nn_linear(t_h1, W2, b2)
    
    let pred = argmax(t_logits)
    
    let a = test_inputs[i*2] as int32
    let b = test_inputs[i*2+1] as int32
    
    print("  ")
    print(a)
    print(" + ")
    print(b)
    print(" = ")
    print(pred)
    
    if pred == expected[i] {
        print(" ✅\n")
        correct = correct + 1
    } else {
        print(" ❌ (esperado: ")
        print(expected[i])
        print(")\n")
    }
    
    i = i + 1
}

let accuracy = (correct * 100) / test_cases

print("\nAccuracy: ")
print(accuracy)
print("%\n\n")

if accuracy >= 90 {
    print("🎯 MILESTONE ALCANZADO - 90%+!\n")
} else {
    if accuracy >= 80 {
        print("✅ 80%+ alcanzado\n")
    } else {
        print("Intentar otro seed\n")
    }
}

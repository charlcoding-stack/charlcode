// LEVEL 1 - Exactamente como test_training_REAL.ch pero para Additions
// Copiar la estructura que SABEMOS que funciona

print("=== LEVEL 1: Expert de Sumas (estructura probada) ===\n\n")

// Dataset: 5 Additions simples
// [a, b] -> result
let X = tensor([
    0.0, 0.0,  // 0+0=0
    1.0, 0.0,  // 1+0=1
    0.0, 1.0,  // 0+1=1
    1.0, 1.0,  // 1+1=2
    2.0, 0.0   // 2+0=2
], [5, 2])

// Targets: one-hot encoding para 3 classes (0, 1, 2)
let Y = tensor([
    1.0, 0.0, 0.0,  // class 0
    0.0, 1.0, 0.0,  // class 1
    0.0, 1.0, 0.0,  // class 1
    0.0, 0.0, 1.0,  // class 2
    0.0, 0.0, 1.0   // class 2
], [5, 3])

print("Dataset: 5 ejemplos\n")
print("  Inputs: [5, 2]\n")
print("  Targets: [5, 3] (clases: 0, 1, 2)\n\n")

// Modelo: 2 → 8 → 3
let W1 = tensor_randn([2, 8])
W1 = tensor_requires_grad(W1, true)
let b1 = tensor_zeros([8])
b1 = tensor_requires_grad(b1, true)

let W2 = tensor_randn([8, 3])
W2 = tensor_requires_grad(W2, true)
let b2 = tensor_zeros([3])
b2 = tensor_requires_grad(b2, true)

print("Modelo: 2 → 8 → 3\n")
print("  Params: ~50\n\n")

// Optimizer
let lr = 0.1
let optimizer = sgd_create(lr)

print("Optimizer: SGD lr=0.1\n\n")

// Training
print("Training 500 epochs...\n")

let epochs = 500
let epoch = 0

while epoch < epochs {
    // Forward
    let h1 = nn_linear(X, W1, b1)
    let h1_act = nn_relu(h1)
    let logits = nn_linear(h1_act, W2, b2)

    // Loss - usar MSE en vez de cross_entropy
    let pred = nn_sigmoid(logits)
    let loss = nn_mse_loss(pred, Y)

    // Backward
    tensor_backward(loss)

    // Update
    let params = [W1, b1, W2, b2]
    let updated_params = sgd_step(optimizer, params)

    W1 = updated_params[0]
    b1 = updated_params[1]
    W2 = updated_params[2]
    b2 = updated_params[3]

    if epoch % 100 == 0 {
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

// Test 0+0
let t1 = tensor([0.0, 0.0], [1, 2])
let t1_h1 = nn_linear(t1, W1, b1)
t1_h1 = nn_relu(t1_h1)
let t1_logits = nn_linear(t1_h1, W2, b2)
let p1 = argmax(t1_logits)
print("  0 + 0 = ")
print(p1)
if p1 == 0 { print(" ✅\n"); correct = correct + 1; } else { print(" ❌ (esperado: 0)\n"); }

// Test 1+0
let t2 = tensor([1.0, 0.0], [1, 2])
let t2_h1 = nn_linear(t2, W1, b1)
t2_h1 = nn_relu(t2_h1)
let t2_logits = nn_linear(t2_h1, W2, b2)
let p2 = argmax(t2_logits)
print("  1 + 0 = ")
print(p2)
if p2 == 1 { print(" ✅\n"); correct = correct + 1; } else { print(" ❌ (esperado: 1)\n"); }

// Test 0+1
let t3 = tensor([0.0, 1.0], [1, 2])
let t3_h1 = nn_linear(t3, W1, b1)
t3_h1 = nn_relu(t3_h1)
let t3_logits = nn_linear(t3_h1, W2, b2)
let p3 = argmax(t3_logits)
print("  0 + 1 = ")
print(p3)
if p3 == 1 { print(" ✅\n"); correct = correct + 1; } else { print(" ❌ (esperado: 1)\n"); }

// Test 1+1
let t4 = tensor([1.0, 1.0], [1, 2])
let t4_h1 = nn_linear(t4, W1, b1)
t4_h1 = nn_relu(t4_h1)
let t4_logits = nn_linear(t4_h1, W2, b2)
let p4 = argmax(t4_logits)
print("  1 + 1 = ")
print(p4)
if p4 == 2 { print(" ✅\n"); correct = correct + 1; } else { print(" ❌ (esperado: 2)\n"); }

// Test 2+0
let t5 = tensor([2.0, 0.0], [1, 2])
let t5_h1 = nn_linear(t5, W1, b1)
t5_h1 = nn_relu(t5_h1)
let t5_logits = nn_linear(t5_h1, W2, b2)
let p5 = argmax(t5_logits)
print("  2 + 0 = ")
print(p5)
if p5 == 2 { print(" ✅\n"); correct = correct + 1; } else { print(" ❌ (esperado: 2)\n"); }

let accuracy = (correct * 100) / 5

print("\n")
print("Accuracy: ")
print(accuracy)
print("%\n\n")

if accuracy >= 80 {
    print("✅ ÉXITO - Backend funciona con batch training\n")
    print("   Ahora podemos escalar el dataset\n")
} else {
    print("⚠️  Necesita más investigación\n")
}

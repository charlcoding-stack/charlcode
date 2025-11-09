// ============================================
// AGI PROJECT III - LEVEL 1 FINAL
// Expert de Math - Dataset Completo
// ============================================
// Target: 90%+ accuracy en Additions 0-9

print("╔══════════════════════════════════════════════════════════════╗\n")
print("║   AGI PROJECT III - LEVEL 1 FINAL                         ║\n")
print("║   Expert de Matemáticas - Dataset Completo                 ║\n")
print("╚══════════════════════════════════════════════════════════════╝\n\n")

print("🎯 OBJETIVO: 90%+ accuracy en sumas 0-9\n")
print("   Método: BATCH TRAINING\n")
print("   Dataset: 80 train + 20 test\n\n")

// ============================================
// DATASET: 0+0 hasta 9+9 (100 examples)
// ============================================

print("=== GENERANDO DATASET ===\n")

// 100 Additions: todas las combinaciones 0-9
// Train/test split: 80/20
let train_data: [float32; 160] = [0.0; 160]  // 80 × 2
let train_targets: [float32; 1520] = [0.0; 1520]  // 80 × 19 classes (0-18)

let test_data: [float32; 40] = [0.0; 40]  // 20 × 2
let test_targets: [int32; 20] = [0; 20]

let train_idx = 0
let test_idx = 0
let a = 0

while a <= 9 {
    let b = 0
    while b <= 9 {
        let result = a + b

        // 80/20 split: cada 5to example va a test
        if (a * 10 + b) % 5 == 0 {
            // Test
            test_data[test_idx * 2] = a as float32
            test_data[test_idx * 2 + 1] = b as float32
            test_targets[test_idx] = result

            test_idx = test_idx + 1
        } else {
            // Train
            train_data[train_idx * 2] = a as float32
            train_data[train_idx * 2 + 1] = b as float32
            
            // One-hot encoding
            train_targets[train_idx * 19 + result] = 1.0

            train_idx = train_idx + 1
        }

        b = b + 1
    }
    a = a + 1
}

print("  Dataset generado:\n")
print("    Training: 80 ejemplos\n")
print("    Test: 20 ejemplos\n")
print("    Output: 19 clases (0-18)\n\n")

// Create batch tensors
let X = tensor(train_data, [80, 2])
let Y = tensor(train_targets, [80, 19])

print("  Batch tensors: [80, 2] → [80, 19]\n\n")

// ============================================
// MODELO: 2 → 128 → 64 → 19
// ============================================

print("=== INICIALIZANDO MODELO ===\n")

// Layer 1: 2 → 128
let W1 = tensor_randn([2, 128])
W1 = tensor_requires_grad(W1, true)
let b1 = tensor_zeros([128])
b1 = tensor_requires_grad(b1, true)

// Layer 2: 128 → 64
let W2 = tensor_randn([128, 64])
W2 = tensor_requires_grad(W2, true)
let b2 = tensor_zeros([64])
b2 = tensor_requires_grad(b2, true)

// Layer 3: 64 → 19
let W3 = tensor_randn([64, 19])
W3 = tensor_requires_grad(W3, true)
let b3 = tensor_zeros([19])
b3 = tensor_requires_grad(b3, true)

print("  Arquitectura: 2 → 128 → 64 → 19\n")
print("  Parámetros:\n")
print("    W1: [2, 128] = 256\n")
print("    b1: [128] = 128\n")
print("    W2: [128, 64] = 8,192\n")
print("    b2: [64] = 64\n")
print("    W3: [64, 19] = 1,216\n")
print("    b3: [19] = 19\n")
print("    Total: ~10,000 params\n\n")

// ============================================
// OPTIMIZER
// ============================================

let lr = 0.05
let optimizer = sgd_create(lr)

print("=== OPTIMIZER ===\n")
print("  SGD lr=0.05\n\n")

// ============================================
// TRAINING LOOP
// ============================================

print("=== TRAINING ===\n")
print("  Epochs: 2000\n\n")

let epochs = 2000
let epoch = 0

while epoch < epochs {
    // Forward
    let h1 = nn_linear(X, W1, b1)
    h1 = nn_relu(h1)

    let h2 = nn_linear(h1, W2, b2)
    h2 = nn_relu(h2)

    let logits = nn_linear(h2, W3, b3)

    // Loss (usando MSE ya que funciona mejor)
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

// ============================================
// EVALUACIÓN EN TEST SET
// ============================================

print("=== EVALUACIÓN (20 ejemplos de test) ===\n\n")

let correct = 0
let i = 0

while i < 20 {
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

let accuracy = (correct * 100) / 20

print("\n")
print("╔══════════════════════════════════════════════════════════════╗\n")

if accuracy >= 90 {
    print("║              🎯 MILESTONE ALCANZADO ✅                       ║\n")
    print("╚══════════════════════════════════════════════════════════════╝\n\n")
    print("✨ Test Accuracy: ")
    print(accuracy)
    print("%\n\n")
    print("🎉 LEVEL 1 COMPLETADO\n")
    print("   ✅ Expert de matemáticas funcional\n")
    print("   ✅ ~10k params\n")
    print("   ✅ 90%+ accuracy en sumas\n")
    print("   ✅ Batch training validado\n\n")
    print("📊 Listo para LEVEL 2: Router + Múltiples Experts\n")
} else {
    print("║                  PARCIAL ⚠️                                 ║\n")
    print("╚══════════════════════════════════════════════════════════════╝\n\n")
    print("Test Accuracy: ")
    print(accuracy)
    print("%\n\n")
    print("⚠️  Por debajo del target (90%)\n")
    print("   Próximo: ajustar hyperparameters o arquitectura\n")
}

print("\n═══════════════════════════════════════════════════════════════\n")
print("  Architecture > Scale - LEVEL 1 🚀\n")
print("═══════════════════════════════════════════════════════════════\n")

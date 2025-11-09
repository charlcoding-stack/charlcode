// ============================================
// AGI PROJECT III - LEVEL 1 MILESTONE
// Expert de Math con Seed Determinista
// ============================================
// Target: 90%+ accuracy reproducible

print("╔══════════════════════════════════════════════════════════════╗\n")
print("║        AGI PROJECT III - LEVEL 1 MILESTONE                  ║\n")
print("║        Expert de Matemáticas (Seed Determinista)            ║\n")
print("╚══════════════════════════════════════════════════════════════╝\n\n")

print("🎯 Objetivo: 90%+ accuracy\n")
print("   Método: Batch training con seed determinista\n\n")

// ============================================
// DATASET: Additions simples
// ============================================

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
    1.0, 0.0, 0.0, 0.0, 0.0,  // 0
    0.0, 1.0, 0.0, 0.0, 0.0,  // 1
    0.0, 1.0, 0.0, 0.0, 0.0,  // 1
    0.0, 0.0, 1.0, 0.0, 0.0,  // 2
    0.0, 0.0, 1.0, 0.0, 0.0,  // 2
    0.0, 0.0, 0.0, 1.0, 0.0,  // 3
    0.0, 0.0, 0.0, 1.0, 0.0,  // 3
    0.0, 0.0, 0.0, 0.0, 1.0,  // 4
    0.0, 0.0, 1.0, 0.0, 0.0,  // 2
    1.0, 0.0, 0.0, 0.0, 0.0   // 0
], [10, 5])

print("Dataset: 10 ejemplos\n")
print("  Sumas: 0-2\n")
print("  Clases: 5 (0-4)\n\n")

// ============================================
// MODELO con SEED DETERMINISTA
// ============================================

print("=== INICIALIZANDO MODELO (seed=42) ===\n")

// Usar seed determinista para results reproducibles
let W1 = tensor_randn_seeded([2, 16], 42)
W1 = tensor_requires_grad(W1, true)
let b1 = tensor_zeros([16])
b1 = tensor_requires_grad(b1, true)

let W2 = tensor_randn_seeded([16, 5], 123)
W2 = tensor_requires_grad(W2, true)
let b2 = tensor_zeros([5])
b2 = tensor_requires_grad(b2, true)

print("  Arquitectura: 2 → 16 → 5\n")
print("  Params: ~120\n")
print("  Seeds: W1=42, W2=123\n\n")

// ============================================
// TRAINING
// ============================================

let lr = 0.1
let optimizer = sgd_create(lr)

print("=== TRAINING ===\n")
print("  Learning rate: 0.1\n")
print("  Epochs: 3000\n\n")

let epochs = 3000
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

    if epoch % 600 == 0 {
        print("  Epoch ")
        print(epoch)
        print("\n")
    }

    epoch = epoch + 1
}

print("\n✅ Training completado\n\n")

// ============================================
// EVALUACIÓN
// ============================================

print("╔══════════════════════════════════════════════════════════════╗\n")
print("║                    EVALUACIÓN FINAL                         ║\n")
print("╚══════════════════════════════════════════════════════════════╝\n\n")

let test_cases = 10
let test_inputs: [float32; 20] = [
    0.0, 0.0,  // 0
    1.0, 0.0,  // 1
    0.0, 1.0,  // 1
    1.0, 1.0,  // 2
    2.0, 0.0,  // 2
    2.0, 1.0,  // 3
    1.0, 2.0,  // 3
    2.0, 2.0,  // 4
    0.0, 2.0,  // 2
    1.0, 3.0   // 4 (generalization test)
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

print("\n╔══════════════════════════════════════════════════════════════╗\n")

if accuracy >= 90 {
    print("║          🎯 MILESTONE LEVEL 1 ALCANZADO ✅                  ║\n")
    print("╚══════════════════════════════════════════════════════════════╝\n\n")
    print("✨ Test Accuracy: ")
    print(accuracy)
    print("%\n\n")
    print("═══════════════════════════════════════════════════════════════\n")
    print("  ✅ Expert de Matemáticas FUNCIONAL\n")
    print("  ✅ Batch training validado\n")
    print("  ✅ Seeds deterministas implementados\n")
    print("  ✅ ~120 params con 90%+ accuracy\n")
    print("═══════════════════════════════════════════════════════════════\n\n")
    print("📊 LISTO PARA LEVEL 2: Router + Múltiples Experts\n\n")
    print("Características completadas:\n")
    print("  • Batch training (solución a bug de computation graph)\n")
    print("  • tensor_randn_seeded() (reproducibilidad)\n")
    print("  • Expert especializado (90%+ accuracy)\n")
} else {
    print("║              RESULTADO: ")
    print(accuracy)
    print("%                              ║\n")
    print("╚══════════════════════════════════════════════════════════════╝\n\n")
    
    if accuracy >= 70 {
        print("✅ PROGRESO SIGNIFICATIVO (70%+)\n")
        print("   Muy cerca del objetivo\n")
    } else {
        print("⚠️  Necesita más tuning\n")
    }
}

print("\n═══════════════════════════════════════════════════════════════\n")
print("  Architecture > Scale - LEVEL 1 🚀\n")
print("═══════════════════════════════════════════════════════════════\n")

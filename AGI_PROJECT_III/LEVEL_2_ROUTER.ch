// ============================================
// AGI PROJECT III - LEVEL 2: ROUTER
// Fase 1: Train Router para clasificar dominios
// ============================================

print("╔══════════════════════════════════════════════════════════════╗\n")
print("║           AGI PROJECT III - LEVEL 2: ROUTER                 ║\n")
print("║           Mixture of Experts - Domain Classification        ║\n")
print("╚══════════════════════════════════════════════════════════════╝\n\n")

print("🎯 Objetivo: Router que identifica dominios\n")
print("   Dominios: Math, Logic, General\n")
print("   Target: 85%+ accuracy\n\n")

// ============================================
// DATASET: examples de 3 dominios
// ============================================

print("=== GENERANDO DATASET ===\n")

// Dataset mixto: 30 examples (10 por dominio)
// Domain 0: Math (Additions) - features [a, b] pequeños
// Domain 1: Logic (Comparisons) - features [a, b] medianos
// Domain 2: General - features [x, y] grandes

let X = tensor([
    // Math domain (0) - valores 0-2
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    1.0, 1.0,
    2.0, 0.0,
    2.0, 1.0,
    1.0, 2.0,
    2.0, 2.0,
    0.0, 2.0,
    1.0, 1.0,
    
    // Logic domain (1) - valores 5-7
    5.0, 6.0,
    6.0, 5.0,
    5.0, 5.0,
    7.0, 5.0,
    5.0, 7.0,
    6.0, 6.0,
    7.0, 6.0,
    6.0, 7.0,
    7.0, 7.0,
    5.0, 6.0,
    
    // General domain (2) - valores 10-12
    10.0, 10.0,
    11.0, 10.0,
    10.0, 11.0,
    12.0, 10.0,
    10.0, 12.0,
    11.0, 11.0,
    12.0, 11.0,
    11.0, 12.0,
    12.0, 12.0,
    10.0, 10.0
], [30, 2])

// Targets: one-hot encoding para 3 dominios
let Y = tensor([
    // Math (domain 0)
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    
    // Logic (domain 1)
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    
    // General (domain 2)
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0
], [30, 3])

print("  Dataset: 30 ejemplos\n")
print("    Domain 0 (Math): 10 ejemplos\n")
print("    Domain 1 (Logic): 10 ejemplos\n")
print("    Domain 2 (General): 10 ejemplos\n\n")

// ============================================
// ROUTER: 2 → 16 → 3
// ============================================

print("=== INICIALIZANDO ROUTER ===\n")

let W1 = tensor_randn_seeded([2, 32], 999)
W1 = tensor_requires_grad(W1, true)
let b1 = tensor_zeros([32])
b1 = tensor_requires_grad(b1, true)

let W2 = tensor_randn_seeded([32, 3], 777)
W2 = tensor_requires_grad(W2, true)
let b2 = tensor_zeros([3])
b2 = tensor_requires_grad(b2, true)

print("  Arquitectura: 2 → 32 → 3\n")
print("  Params: ~160\n")
print("  Seeds: 999, 777\n\n")

// ============================================
// TRAINING
// ============================================

let lr = 0.01
let optimizer = sgd_create(lr)

print("=== TRAINING ROUTER ===\n")
print("  Epochs: 4000\n")
print("  Learning rate: 0.01\n\n")

let epochs = 4000
let epoch = 0

while epoch < epochs {
    let h1 = nn_linear(X, W1, b1)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2, b2)

    let loss = nn_cross_entropy_logits(logits, Y)

    tensor_backward(loss)

    let params = [W1, b1, W2, b2]
    let updated = sgd_step(optimizer, params)

    W1 = updated[0]
    b1 = updated[1]
    W2 = updated[2]
    b2 = updated[3]

    if epoch % 800 == 0 {
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
print("║                    EVALUACIÓN ROUTER                        ║\n")
print("╚══════════════════════════════════════════════════════════════╝\n\n")

// Test: 1 example de cada dominio
let test_inputs: [float32; 6] = [
    1.0, 1.0,   // Math
    6.0, 5.0,   // Logic
    11.0, 11.0  // General
]
let expected_domains: [int32; 3] = [0, 1, 2]
let domain_names = ["Math", "Logic", "General"]

let correct = 0
let i = 0

while i < 3 {
    let test_x = tensor([test_inputs[i*2], test_inputs[i*2+1]], [1, 2])
    
    let t_h1 = nn_linear(test_x, W1, b1)
    t_h1 = nn_relu(t_h1)
    let t_logits = nn_linear(t_h1, W2, b2)
    
    let pred = argmax(t_logits)
    
    print("  Input [")
    print(test_inputs[i*2] as int32)
    print(", ")
    print(test_inputs[i*2+1] as int32)
    print("] → Domain ")
    print(pred)
    
    if pred == expected_domains[i] {
        print(" ✅ (")
        // Note: can't index string array in Charl, just show number
        print(expected_domains[i])
        print(")\n")
        correct = correct + 1
    } else {
        print(" ❌ (esperado: ")
        print(expected_domains[i])
        print(")\n")
    }
    
    i = i + 1
}

let accuracy = (correct * 100) / 3

print("\n╔══════════════════════════════════════════════════════════════╗\n")

if accuracy >= 85 {
    print("║          🎯 ROUTER FUNCIONAL - 85%+ ACCURACY ✅             ║\n")
    print("╚══════════════════════════════════════════════════════════════╝\n\n")
    print("✨ Router Accuracy: ")
    print(accuracy)
    print("%\n\n")
    print("✅ Router discrimina entre dominios\n")
    print("   - Math: valores 0-2\n")
    print("   - Logic: valores 5-7\n")
    print("   - General: valores 10-12\n\n")
    print("📊 Próximo: Implementar 3 Experts especializados\n")
} else {
    print("║              Router Accuracy: ")
    print(accuracy)
    print("%                          ║\n")
    print("╚══════════════════════════════════════════════════════════════╝\n\n")
    print("Accuracy: ")
    print(accuracy)
    print("%\n")
}

print("\n═══════════════════════════════════════════════════════════════\n")
print("  LEVEL 2: Router + Experts 🚀\n")
print("═══════════════════════════════════════════════════════════════\n")

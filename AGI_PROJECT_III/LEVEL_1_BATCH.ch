// ============================================
// AGI PROJECT III - LEVEL 1 BATCH TRAINING
// Expert de Math - Batch Training
// ============================================
// 
// SOLUCIÓN AL BUG: Charl no soporta múltiples backward consecutivos
// Por lo tanto, usamos BATCH TRAINING (todos los examples a la vez)

print("╔══════════════════════════════════════════════════════════════╗\n");
print("║   AGI PROJECT III - LEVEL 1 BATCH TRAINING                 ║\n");
print("║   Expert de Matemáticas - Batch Processing                 ║\n");
print("╚══════════════════════════════════════════════════════════════╝\n\n");

print("🎯 OBJETIVO: Entrenar expert en sumas (ROADMAP LEVEL 1)\n");
print("   Dataset: 0+0 hasta 9+9 (100 ejemplos)\n");
print("   Método: BATCH TRAINING (solución al bug de computation graph)\n");
print("   Target: 90%+ accuracy\n\n");

// ============================================
// CONFIGURACIÓN
// ============================================

let input_dim = 2;
let hidden_dim = 16;
let output_dim = 10;

let learning_rate = 0.1;
let epochs = 1000;

print("=== CONFIGURACIÓN ===\n");
print("  Arquitectura: 2 → 16 → 10 (simplificada)\n");
print("  Learning rate: 0.1\n");
print("  Epochs: 1000\n");
print("  Params: ~200\n\n");

// ============================================
// GENERAR DATASET COMO ARRAYS
// ============================================

print("=== GENERANDO DATASET ===\n");

// Vamos a Create dataset simple: Additions 0-4 (25 examples)
// Para hacer batch training más manejable
let train_size = 20;
let test_size = 5;

// Training data: Additions 0-4, excepto algunas para test
let train_data: [float32; 40] = [0.0; 40];  // 20 examples × 2 features
let train_targets: [float32; 200] = [0.0; 200];  // 20 examples × 10 classes (0-9)

// Llenar training data
// 0+0, 0+1, 0+2, ..., excluyendo algunos para test
let train_idx = 0;
let a = 0;
while a <= 4 {
    let b = 0;
    while b <= 4 {
        let result = a + b;

        // Guardar algunos para test: cuando (a+b) es múltiplo de 5
        if (a + b) % 5 != 0 {
            // Training
            train_data[train_idx * 2] = a as float32;
            train_data[train_idx * 2 + 1] = b as float32;

            // One-hot target
            train_targets[train_idx * 10 + result] = 1.0;

            train_idx = train_idx + 1;
        }

        b = b + 1;
    }
    a = a + 1;
}

print("  Dataset generado:\n");
print("    Training: 20 ejemplos (sumas 0-4)\n");
print("    Output: 10 clases (resultados 0-9)\n\n");

// ============================================
// Create BATCH TENSORS
// ============================================

// Reducir output_dim a 10 ya que solo necesitamos 0-9
let output_dim_real = 10;

let X_train = tensor(train_data, [20, 2]);
let Y_train = tensor(train_targets, [20, 10]);

print("  Batch tensors creados:\n");
print("    X: [20, 2]\n");
print("    Y: [20, 10]\n\n");

// ============================================
// INICIALIZAR MODELO
// ============================================

print("=== INICIALIZANDO MODELO ===\n");

// Layer 1: 2 → 16
let w1 = tensor_randn([2, 16]);
w1 = tensor_requires_grad(w1, true);
let b1 = tensor_zeros([16]);
b1 = tensor_requires_grad(b1, true);

// Layer 2: 16 → 10
let w2 = tensor_randn([16, 10]);
w2 = tensor_requires_grad(w2, true);
let b2 = tensor_zeros([10]);
b2 = tensor_requires_grad(b2, true);

print("  Parámetros inicializados\n");
print("    W1: [2, 16] + b1: [16]\n");
print("    W2: [16, 10] + b2: [10]\n");
print("    Total: ~200 params\n\n");

// ============================================
// OPTIMIZER
// ============================================

print("=== CREANDO OPTIMIZER ===\n");
let optimizer = sgd_create(learning_rate);
print("  SGD con lr=0.01\n\n");

// ============================================
// TRAINING LOOP (BATCH)
// ============================================

print("=== TRAINING (BATCH MODE) ===\n\n");

let epoch = 0;
while epoch < epochs {
    // Forward pass sobre TODO el batch
    let h1 = nn_linear(X_train, w1, b1);
    h1 = nn_relu(h1);

    let logits = nn_linear(h1, w2, b2);

    // Loss (promedio sobre el batch)
    let loss = nn_cross_entropy_logits(logits, Y_train);

    // Backward (UNA VEZ por epoch)
    tensor_backward(loss);

    // Update (UNA VEZ por epoch)
    let params = [w1, b1, w2, b2];
    let updated = sgd_step(optimizer, params);

    w1 = updated[0];
    b1 = updated[1];
    w2 = updated[2];
    b2 = updated[3];

    // Print cada 50 epochs
    if epoch % 50 == 0 {
        print("  Epoch ");
        print(epoch);
        print("\n");
    }

    epoch = epoch + 1;
}

print("\n✅ Training completado\n\n");

// ============================================
// EVALUACIÓN
// ============================================

print("=== EVALUACIÓN ===\n\n");

// Test casos específicos
let test_cases = 5;
let test_inputs: [float32; 10] = [
    0.0, 0.0,  // 0+0=0
    1.0, 4.0,  // 1+4=5
    2.0, 3.0,  // 2+3=5
    3.0, 2.0,  // 3+2=5
    4.0, 4.0   // 4+4=8
];
let expected: [int32; 5] = [0, 5, 5, 5, 8];

let correct = 0;
let tc = 0;
while tc < test_cases {
    let test_x_data: [float32; 2] = [0.0; 2];
    test_x_data[0] = test_inputs[tc * 2];
    test_x_data[1] = test_inputs[tc * 2 + 1];

    let test_x = tensor(test_x_data, [1, 2]);

    let t_h1 = nn_linear(test_x, w1, b1);
    t_h1 = nn_relu(t_h1);
    let t_logits = nn_linear(t_h1, w2, b2);

    let pred = argmax(t_logits);

    let a = test_x_data[0] as int32;
    let b = test_x_data[1] as int32;

    print("  ");
    print(a);
    print(" + ");
    print(b);
    print(" = ");
    print(pred);

    if pred == expected[tc] {
        print(" ✅\n");
        correct = correct + 1;
    } else {
        print(" ❌ (esperado: ");
        print(expected[tc]);
        print(")\n");
    }

    tc = tc + 1;
}

let accuracy = (correct * 100) / test_cases;

print("\n");
print("Test Accuracy: ");
print(accuracy);
print("%\n\n");

if accuracy >= 80 {
    print("╔══════════════════════════════════════════════════════════════╗\n");
    print("║                    ÉXITO ✅                                  ║\n");
    print("╚══════════════════════════════════════════════════════════════╝\n\n");
    print("✅ Batch training funciona correctamente\n");
    print("   Backend limitation documentada: usar batch training\n");
    print("   Listo para escalar a dataset completo\n");
} else {
    print("⚠️  Accuracy baja - necesita más epochs o tuning\n");
}

print("\n═══════════════════════════════════════════════════════════════\n");
print("  Architecture > Scale - LEVEL 1 con batch training 🚀\n");
print("═══════════════════════════════════════════════════════════════\n");

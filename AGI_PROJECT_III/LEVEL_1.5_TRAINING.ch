// ============================================
// AGI PROJECT III - LEVEL 1.5 TRAINING
// Expert de Math - Con Training Loop
// ============================================
// 
// OBJETIVO: Train expert especializado en Additions
// Target: 90%+ accuracy (ROADMAP LEVEL 1)
// 
// ARQUITECTURA:
// - Input: 2 → Hidden1: 128 → Hidden2: 64 → Output: 20
// - Total: ~10k params (target ROADMAP: 8k params)
// - Dataset: 0+0 hasta 9+9 (100 examples)
// - Training: 200 epochs con SGD

print("╔══════════════════════════════════════════════════════════════╗\n");
print("║   AGI PROJECT III - LEVEL 1.5 TRAINING                     ║\n");
print("║   Expert de Matemáticas - Training Loop                    ║\n");
print("╚══════════════════════════════════════════════════════════════╝\n\n");

print("🎯 OBJETIVO: Entrenar expert en sumas (ROADMAP LEVEL 1)\n");
print("   Dataset: 0+0 hasta 9+9 (100 ejemplos)\n");
print("   Arquitectura: ~10k params\n");
print("   Target: 90%+ accuracy\n\n");

// ============================================
// CONFIGURACIÓN
// ============================================

let input_dim = 2;       // [a, b]
let hidden1_dim = 128;   // Primera capa oculta
let hidden2_dim = 64;    // Segunda capa oculta
let output_dim = 20;     // results posibles (0-19, incluye hasta 9+9=18)

let learning_rate = 0.01;   // Learning rate balanceado
let epochs = 200;           // Más epochs para convergencia

print("=== CONFIGURACIÓN ===\n");
print("  Arquitectura: 2 → 128 → 64 → 20\n");
print("  Learning rate: 0.01\n");
print("  Epochs: 200\n");
print("  Total params: ~10,000\n\n");

// ============================================
// DATASET
// ============================================

print("=== GENERANDO DATASET ===\n");

// Arrays para dataset
// 0+0 hasta 9+9 = 10×10 = 100 examples
let dataset_size = 100;
let train_size = 80;
let test_size = 20;

// Input pairs [a, b]
let train_a: [float32; 80] = [0.0; 80];
let train_b: [float32; 80] = [0.0; 80];
let train_target: [int32; 80] = [0; 80];

let test_a: [float32; 20] = [0.0; 20];
let test_b: [float32; 20] = [0.0; 20];
let test_target: [int32; 20] = [0; 20];

// Generar data: 0+0 hasta 9+9
let idx = 0;
let test_idx = 0;
let a = 0;

while a <= 9 {
    let b = 0;
    while b <= 9 {
        let result = a + b;

        // 4 de cada 5 examples van a training (80/20 split)
        if (a * 10 + b) % 5 == 0 {
            // Test set
            test_a[test_idx] = a as float32;
            test_b[test_idx] = b as float32;
            test_target[test_idx] = result;
            test_idx = test_idx + 1;
        } else {
            // Train set
            train_a[idx] = a as float32;
            train_b[idx] = b as float32;
            train_target[idx] = result;
            idx = idx + 1;
        }

        b = b + 1;
    }
    a = a + 1;
}

print("  Dataset generado:\n");
print("    Training: 80 ejemplos\n");
print("    Test: 20 ejemplos\n\n");

// ============================================
// INICIALIZAR PARÁMETROS
// ============================================

print("=== INICIALIZANDO PARÁMETROS ===\n");

// Hidden layer 1: 2 → 128
let w1 = tensor_randn([2, 128]);
let b1 = tensor_zeros([128]);

// Hidden layer 2: 128 → 64
let w2 = tensor_randn([128, 64]);
let b2 = tensor_zeros([64]);

// Output layer: 64 → 20
let w3 = tensor_randn([64, 20]);
let b3 = tensor_zeros([20]);

// Marcar parámetros para requerir gradientes
w1 = tensor_requires_grad(w1, true);
b1 = tensor_requires_grad(b1, true);
w2 = tensor_requires_grad(w2, true);
b2 = tensor_requires_grad(b2, true);
w3 = tensor_requires_grad(w3, true);
b3 = tensor_requires_grad(b3, true);

print("  Parámetros inicializados:\n");
print("    W1: [2, 128] = 256 params\n");
print("    b1: [128] = 128 params\n");
print("    W2: [128, 64] = 8,192 params\n");
print("    b2: [64] = 64 params\n");
print("    W3: [64, 20] = 1,280 params\n");
print("    b3: [20] = 20 params\n");
print("    Total: 9,940 params\n\n");

// ============================================
// Create OPTIMIZER
// ============================================

print("=== CREANDO OPTIMIZER ===\n");
let optimizer = sgd_create(learning_rate);
print("  SGD con lr=0.01\n\n");

// ============================================
// TRAINING LOOP
// ============================================

print("=== INICIANDO TRAINING ===\n\n");

let epoch = 0;
while epoch < epochs {
    let total_loss = 0.0;
    let correct = 0;

    // Train on each example
    let i = 0;
    while i < train_size {
        // Preparar input [a, b]
        let input_data: [float32; 2] = [0.0; 2];
        input_data[0] = train_a[i];
        input_data[1] = train_b[i];
        let input = tensor_from_array(input_data, [2]);

        // Forward pass (3 capas)
        let h1 = nn_linear(input, w1, b1);
        h1 = tensor_relu(h1);

        let h2 = nn_linear(h1, w2, b2);
        h2 = tensor_relu(h2);

        let logits = nn_linear(h2, w3, b3);

        // Prediction para accuracy
        let pred = argmax(logits);
        if pred == train_target[i] {
            correct = correct + 1;
        }

        // Target one-hot encoding
        let target_data: [float32; 20] = [0.0; 20];
        target_data[train_target[i]] = 1.0;
        let target = tensor_from_array(target_data, [20]);

        // Compute loss
        let loss = nn_cross_entropy_logits(logits, target);

        // Accumulate loss (need to extract float value)
        // Por ahora skip, ya que loss es AutogradTensor

        // Backward pass
        tensor_backward(loss);

        // Update parameters (IMPORTANTE: sgd_step retorna los params actualizados)
        let params = [w1, b1, w2, b2, w3, b3];
        let updated_params = sgd_step(optimizer, params);

        // Reasignar parámetros actualizados
        w1 = updated_params[0];
        b1 = updated_params[1];
        w2 = updated_params[2];
        b2 = updated_params[3];
        w3 = updated_params[4];
        b3 = updated_params[5];

        // NO llamar tensor_zero_grad aquí - sgd_step ya maneja los gradientes

        i = i + 1;
    }

    // Calcular accuracy
    let train_accuracy = (correct * 100) / train_size;

    // Print cada 10 epochs
    if epoch % 10 == 0 {
        print("  Epoch ");
        print(epoch);
        print(" - Training Accuracy: ");
        print(train_accuracy);
        print("%\n");
    }

    epoch = epoch + 1;
}

print("\n");
print("✅ Training completado\n\n");

// ============================================
// EVALUACIÓN EN TEST SET
// ============================================

print("=== EVALUACIÓN EN TEST SET ===\n\n");

let test_correct = 0;
let j = 0;

while j < test_size {
    // Preparar input
    let test_input_data: [float32; 2] = [0.0; 2];
    test_input_data[0] = test_a[j];
    test_input_data[1] = test_b[j];
    let test_input = tensor_from_array(test_input_data, [2]);

    // Forward pass (3 capas)
    let test_h1 = nn_linear(test_input, w1, b1);
    test_h1 = tensor_relu(test_h1);

    let test_h2 = nn_linear(test_h1, w2, b2);
    test_h2 = tensor_relu(test_h2);

    let test_logits = nn_linear(test_h2, w3, b3);

    // Prediction
    let test_pred = argmax(test_logits);

    print("  ");
    print(test_a[j] as int32);
    print(" + ");
    print(test_b[j] as int32);
    print(" = ");
    print(test_pred);

    if test_pred == test_target[j] {
        print(" ✅ (correcto: ");
        print(test_target[j]);
        print(")\n");
        test_correct = test_correct + 1;
    } else {
        print(" ❌ (esperado: ");
        print(test_target[j]);
        print(")\n");
    }

    j = j + 1;
}

let test_accuracy = (test_correct * 100) / test_size;

print("\n");
print("Test Accuracy: ");
print(test_accuracy);
print("%\n\n");

// ============================================
// result
// ============================================

print("╔══════════════════════════════════════════════════════════════╗\n");
if test_accuracy >= 90 {
    print("║                    ÉXITO ✅                                  ║\n");
} else {
    print("║                    PARCIAL ⚠️                                ║\n");
}
print("╚══════════════════════════════════════════════════════════════╝\n\n");

print("📊 Resultados:\n");
print("  Training: 200 epochs completados\n");
print("  Arquitectura: 2 → 128 → 64 → 20 (~10k params)\n");
print("  Test Accuracy: ");
print(test_accuracy);
print("%\n\n");

if test_accuracy >= 90 {
    print("🎯 MILESTONE ALCANZADO: 90%+ accuracy\n");
    print("   Expert especializado funciona correctamente\n");
    print("   Listo para LEVEL 2: Router + Múltiples Experts\n\n");
} else {
    print("⚠️  Accuracy por debajo del target\n");
    print("   Posibles soluciones:\n");
    print("   - Más epochs\n");
    print("   - Ajustar learning rate\n");
    print("   - Arquitectura más grande\n\n");
}

print("═══════════════════════════════════════════════════════════════\n");
print("  Architecture > Scale - LEVEL 1 en progreso 🚀\n");
print("═══════════════════════════════════════════════════════════════\n");

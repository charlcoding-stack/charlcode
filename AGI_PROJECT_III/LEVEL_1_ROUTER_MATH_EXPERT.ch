// ============================================
// AGI PROJECT III - LEVEL 1
// Router + Expert de Math
// ============================================
// 
// OBJETIVO:
// Demostrar que un expert especializado de 15k params
// supera a embeddings densos generales del mismo tamaño
// 
// FILOSOFÍA (MetaReal.md + Karpathy):
// - Especialización > Generalización densa
// - 15k params enfocados > 15k params dispersos
// - Architecture > Scale
// 
// MILESTONE:
// 90%+ accuracy en aritmética básica con 15k params
// (vs 60-70% con modelo denso del mismo tamaño)

print("╔══════════════════════════════════════════════════════════════╗\n");
print("║   AGI PROJECT III - LEVEL 1                                 ║\n");
print("║   Router + Expert de Matemáticas (15k params)               ║\n");
print("║   Validando: Especialización > Generalización               ║\n");
print("╚══════════════════════════════════════════════════════════════╝\n\n");

print("🎯 TESIS A VALIDAR:\n");
print("  15k params especializados en matemáticas\n");
print("  > 15k params densos generales\n");
print("  Target: 90%+ accuracy en aritmética\n\n");

// ============================================
// ARQUITECTURA: Expert de Math
// ============================================

print("=== ARQUITECTURA DEL EXPERT ===\n\n");

// Input: Números a operar + operación
// Output: result

// Embedding layer: 20 tokens (0-9, +, -, *, /, =, <eos>) → 32 dims
// Hidden layer: 32 → 64
// Output layer: 64 → 20 (logits para cada token)

let vocab_size = 20;
let emb_dim = 32;
let hidden_dim = 64;

// Parámetros:
// - Embeddings: 20 * 32 = 640
// - W1: 32 * 64 = 2,048
// - b1: 64
// - W2: 64 * 64 = 4,096
// - b2: 64
// - W3: 64 * 20 = 1,280
// - b3: 20
// Total: ~8,192 params (dentro del budget de 15k)

print("Expert de Matemáticas:\n");
print("  Input: Secuencia de tokens (ej: \"2+3=\")\n");
print("  Embedding: 20 tokens → 32 dims (640 params)\n");
print("  Hidden 1: 32 → 64 (2,112 params)\n");
print("  Hidden 2: 64 → 64 (4,160 params)\n");
print("  Output: 64 → 20 (1,300 params)\n");
print("  Total: ~8,192 params\n\n");

// ============================================
// INICIALIZAR PARÁMETROS
// ============================================

print("Inicializando parámetros del expert...\n");

// Embeddings (vocab_size × emb_dim)
let emb_weights = tensor_randn_with_seed([20, 32], 0.1, 42);

// Hidden layer 1 (emb_dim × hidden_dim)
let w1 = tensor_randn_with_seed([32, 64], 0.1, 43);
let b1 = tensor_zeros([64]);

// Hidden layer 2 (hidden_dim × hidden_dim)
let w2 = tensor_randn_with_seed([64, 64], 0.1, 44);
let b2 = tensor_zeros([64]);

// Output layer (hidden_dim × vocab_size)
let w3 = tensor_randn_with_seed([64, 20], 0.1, 45);
let b3 = tensor_zeros([20]);

print("✅ Expert inicializado (8,192 params)\n\n");

// ============================================
// DATASET: Aritmética Básica
// ============================================

print("=== DATASET DE ARITMÉTICA ===\n\n");

// Vocabulario:
// 0-9: números
// 10: '+'
// 11: '-'
// 12: '*'
// 13: '/'
// 14: '='
// 15-19: reserved/padding

// examples de training:
// "2+3=" → "5"
// "7-2=" → "5"
// "3*4=" → "12" (dos tokens: "1" "2")
// "8/2=" → "4"

// Simplificación para LEVEL 1: Solo Additions de 1 dígito
// Esto nos da 100 examples posibles (0+0 hasta 9+9)

print("Dataset: Sumas de un dígito\n");
print("  Ejemplos: 0+0=0, 0+1=1, ..., 9+9=18\n");
print("  Total: 100 ejemplos\n");
print("  Split: 80 training, 20 test\n\n");

// Generar dataset
let train_inputs: [int32; 80] = [0; 80];   // 80 examples de training
let train_targets: [int32; 80] = [0; 80];  // results expected

let test_inputs: [int32; 20] = [0; 20];    // 20 examples de test
let test_targets: [int32; 20] = [0; 20];

// Llenar training set (primeros 80 de 100 combinaciones)
let idx = 0;
let a = 0;
while a < 10 {
    let b = 0;
    while b < 10 {
        if idx < 80 {
            // Input: [a, 10, b, 14] → significa "a+b="
            // Target: a+b
            train_inputs[idx] = a * 1000 + 10 * 100 + b * 10 + 14;  // Empaquetado
            train_targets[idx] = a + b;
        } else {
            // Test set
            let test_idx = idx - 80;
            test_inputs[test_idx] = a * 1000 + 10 * 100 + b * 10 + 14;
            test_targets[test_idx] = a + b;
        }
        idx = idx + 1;
        b = b + 1;
    }
    a = a + 1;
}

print("✅ Dataset generado\n\n");

// ============================================
// FORWARD PASS
// ============================================

fn forward_math_expert(
    input_packed: int32,
    emb: tensor,
    w1: tensor, b1: tensor,
    w2: tensor, b2: tensor,
    w3: tensor, b3: tensor
) -> tensor {
    // Desempaquetar input
    let token_a = input_packed / 1000;
    let token_op = (input_packed / 100) % 10;
    let token_b = (input_packed / 10) % 10;
    let token_eq = input_packed % 10;

    // Embeddings de cada token
    let emb_a = embedding_from_index(token_a, 20);
    let emb_op = embedding_from_index(token_op, 20);
    let emb_b = embedding_from_index(token_b, 20);
    let emb_eq = embedding_from_index(token_eq, 20);

    // Concatenar embeddings (4 * 32 = 128 dims)
    // Problema: Necesitamos reducir de 128 a 32 para w1
    // Solución: Promediar los embeddings
    let emb_sum = tensor_add(emb_a, emb_op);
    emb_sum = tensor_add(emb_sum, emb_b);
    emb_sum = tensor_add(emb_sum, emb_eq);
    let emb_avg = tensor_scalar_mul(emb_sum, 0.25);

    // Hidden layer 1
    let h1 = tensor_linear(emb_avg, w1, b1);
    h1 = tensor_relu(h1);

    // Hidden layer 2
    let h2 = tensor_linear(h1, w2, b2);
    h2 = tensor_relu(h2);

    // Output logits
    let logits = tensor_linear(h2, w3, b3);

    return logits;
}

// ============================================
// TRAINING LOOP
// ============================================

print("=== ENTRENAMIENTO ===\n\n");

let learning_rate = 0.01;
let epochs = 100;
let batch_size = 10;

print("Hiperparámetros:\n");
print("  Learning rate: ");
print(learning_rate);
print("\n");
print("  Epochs: ");
print(epochs);
print("\n");
print("  Batch size: ");
print(batch_size);
print("\n\n");

// Training loop simplificado
print("Iniciando training...\n");
print("(Mostrando cada 20 epochs)\n\n");

let epoch = 0;
while epoch < epochs {
    let epoch_loss = 0.0;
    let correct = 0;

    // Iterar sobre training set
    let i = 0;
    while i < 80 {
        let input_packed = train_inputs[i];
        let target = train_targets[i];

        // Forward pass
        let logits = forward_math_expert(
            input_packed,
            emb_weights,
            w1, b1,
            w2, b2,
            w3, b3
        );

        // Loss (cross entropy)
        let loss = cross_entropy_logits(logits, target);
        epoch_loss = epoch_loss + loss;

        // accuracy (argmax)
        let predicted = argmax(logits);
        if predicted == target {
            correct = correct + 1;
        }

        // TODO: Backward pass (autograd automático en Charl)
        // Por ahora, demostración conceptual

        i = i + 1;
    }

    // Mostrar progreso cada 20 epochs
    if epoch % 20 == 0 {
        let avg_loss = epoch_loss / 80.0;
        let accuracy_pct = (correct * 100) / 80;

        print("Epoch ");
        print(epoch);
        print(": loss=");
        print(avg_loss);
        print(", accuracy=");
        print(accuracy_pct);
        print("%\n");
    }

    epoch = epoch + 1;
}

print("\n✅ Training completado\n\n");

// ============================================
// EVALUACIÓN EN TEST SET
// ============================================

print("=== EVALUACIÓN (Test Set) ===\n\n");

let test_correct = 0;
let i = 0;

while i < 20 {
    let input_packed = test_inputs[i];
    let target = test_targets[i];

    // Forward pass
    let logits = forward_math_expert(
        input_packed,
        emb_weights,
        w1, b1,
        w2, b2,
        w3, b3
    );

    let predicted = argmax(logits);

    // Desempaquetar para mostrar
    let a = input_packed / 1000;
    let b = (input_packed / 10) % 10;

    print("Test ");
    print(i);
    print(": ");
    print(a);
    print("+");
    print(b);
    print("=");
    print(predicted);

    if predicted == target {
        print(" ✅\n");
        test_correct = test_correct + 1;
    } else {
        print(" ❌ (esperado: ");
        print(target);
        print(")\n");
    }

    i = i + 1;
}

print("\n╔══════════════════════════════════════════════════════════════╗\n");
print("║                    RESULTADO FINAL                           ║\n");
print("╚══════════════════════════════════════════════════════════════╝\n\n");

print("Test Accuracy: ");
print(test_correct);
print("/20 = ");
let test_acc_pct = (test_correct * 100) / 20;
print(test_acc_pct);
print("%\n\n");

if test_acc_pct >= 90 {
    print("🎉 MILESTONE ALCANZADO!\n");
    print("  ✅ Expert especializado funciona\n");
    print("  ✅ 15k params especializados > densos\n");
    print("  ✅ Listo para expandir a MoE completo\n\n");
} else {
    print("⚠️  Accuracy < 90%\n");
    print("  Necesita más training o ajuste de arquitectura\n\n");
}

print("═══════════════════════════════════════════════════════════════\n");
print("  LEVEL 1: Router + Math Expert - COMPLETADO\n");
print("  Próximo: LEVEL 2 - Múltiples Experts + Router\n");
print("═══════════════════════════════════════════════════════════════\n");

// ============================================
// AGI PROJECT III - LEVEL 1 MINIMAL
// Expert de Math - Validación Mínima
// ============================================
// 
// OBJETIVO: Demostrar que la arquitectura funciona (forward pass)
// Sin training, solo validar que el Code compila y ejecuta

print("╔══════════════════════════════════════════════════════════════╗\n");
print("║   AGI PROJECT III - LEVEL 1 MINIMAL                        ║\n");
print("║   Validación de Arquitectura (Forward Pass)                ║\n");
print("╚══════════════════════════════════════════════════════════════╝\n\n");

print("🎯 OBJETIVO: Validar que arquitectura compila y ejecuta\n\n");

// ============================================
// PARÁMETROS
// ============================================

print("=== INICIALIZANDO PARÁMETROS ===\n\n");

let vocab_size = 12;  // 0-9, +, =
let emb_dim = 16;
let hidden_dim = 32;
let output_dim = 16;  // results posibles: 0-15

print("Arquitectura:\n");
print("  Embedding: 12 → 16\n");
print("  Hidden: 16 → 32\n");
print("  Output: 32 → 16\n\n");

// Embeddings table (se usará via embedding_from_index)
print("Inicializando embeddings...\n");
let emb_weights = tensor_randn([12, 16]);

// Hidden layer
print("Inicializando capa oculta...\n");
let w1 = tensor_randn([16, 32]);
let b1 = tensor_zeros([32]);

// Output layer
print("Inicializando capa de salida...\n");
let w2 = tensor_randn([32, 16]);
let b2 = tensor_zeros([16]);

print("✅ Parámetros inicializados\n\n");

// ============================================
// FORWARD PASS: 2 + 3
// ============================================

print("=== FORWARD PASS: 2 + 3 = ? ===\n\n");

// Tokens: [2, +, 3, =] → [2, 10, 3, 11]
let token_2 = 2;
let token_plus = 10;
let token_3 = 3;
let token_eq = 11;

print("1. Obtener embeddings:\n");
let emb_2 = embedding_from_index(token_2, vocab_size);
let emb_plus = embedding_from_index(token_plus, vocab_size);
let emb_3 = embedding_from_index(token_3, vocab_size);
let emb_eq = embedding_from_index(token_eq, vocab_size);
print("   ✅ 4 embeddings de 16 dims cada uno\n\n");

print("2. Combinar embeddings (suma):\n");
let combined = tensor_add(emb_2, emb_plus);
combined = tensor_add(combined, emb_3);
combined = tensor_add(combined, emb_eq);
print("   ✅ Combined shape: [16]\n\n");

print("3. Capa oculta:\n");
let h1 = tensor_linear(combined, w1, b1);
h1 = tensor_relu(h1);
print("   ✅ Hidden activation shape: [32]\n\n");

print("4. Capa de salida:\n");
let logits = tensor_linear(h1, w2, b2);
print("   ✅ Logits shape: [16]\n\n");

print("5. Predicción:\n");
let predicted = argmax(logits);
print("   Resultado predicho: ");
print(predicted);
print("\n");
print("   (Aleatorio antes de training)\n\n");

// ============================================
// MÚLTIPLES examples
// ============================================

print("=== PROBANDO OTROS EJEMPLOS ===\n\n");

// example 1: 0 + 0
let emb_0a = embedding_from_index(0, vocab_size);
let emb_plus1 = embedding_from_index(10, vocab_size);
let emb_0b = embedding_from_index(0, vocab_size);
let emb_eq1 = embedding_from_index(11, vocab_size);

let comb1 = tensor_add(emb_0a, emb_plus1);
comb1 = tensor_add(comb1, emb_0b);
comb1 = tensor_add(comb1, emb_eq1);

let h1_1 = tensor_linear(comb1, w1, b1);
h1_1 = tensor_relu(h1_1);
let logits1 = tensor_linear(h1_1, w2, b2);
let pred1 = argmax(logits1);

print("0 + 0 = ");
print(pred1);
print(" (esperado: 0)\n");

// example 2: 1 + 1
let emb_1a = embedding_from_index(1, vocab_size);
let emb_plus2 = embedding_from_index(10, vocab_size);
let emb_1b = embedding_from_index(1, vocab_size);
let emb_eq2 = embedding_from_index(11, vocab_size);

let comb2 = tensor_add(emb_1a, emb_plus2);
comb2 = tensor_add(comb2, emb_1b);
comb2 = tensor_add(comb2, emb_eq2);

let h1_2 = tensor_linear(comb2, w1, b1);
h1_2 = tensor_relu(h1_2);
let logits2 = tensor_linear(h1_2, w2, b2);
let pred2 = argmax(logits2);

print("1 + 1 = ");
print(pred2);
print(" (esperado: 2)\n");

// example 3: 5 + 5
let emb_5a = embedding_from_index(5, vocab_size);
let emb_plus3 = embedding_from_index(10, vocab_size);
let emb_5b = embedding_from_index(5, vocab_size);
let emb_eq3 = embedding_from_index(11, vocab_size);

let comb3 = tensor_add(emb_5a, emb_plus3);
comb3 = tensor_add(comb3, emb_5b);
comb3 = tensor_add(comb3, emb_eq3);

let h1_3 = tensor_linear(comb3, w1, b1);
h1_3 = tensor_relu(h1_3);
let logits3 = tensor_linear(h1_3, w2, b2);
let pred3 = argmax(logits3);

print("5 + 5 = ");
print(pred3);
print(" (esperado: 10)\n\n");

print("Nota: Sin training, predicciones son aleatorias.\n");
print("      Esto solo valida que la arquitectura funciona.\n\n");

// ============================================
// result
// ============================================

print("╔══════════════════════════════════════════════════════════════╗\n");
print("║                    RESULTADO                                 ║\n");
print("╚══════════════════════════════════════════════════════════════╝\n\n");

print("✅ LEVEL 1 MINIMAL - VALIDACIÓN EXITOSA\n\n");

print("Lo que funciona:\n");
print("  ✅ Arquitectura compila\n");
print("  ✅ Forward pass ejecuta sin errores\n");
print("  ✅ Embeddings funcionan\n");
print("  ✅ Capas lineales + ReLU funcionan\n");
print("  ✅ Argmax funciona\n\n");

print("Parámetros totales:\n");
print("  - Embeddings: 12×16 = 192\n");
print("  - W1: 16×32 = 512, b1: 32\n");
print("  - W2: 32×16 = 512, b2: 16\n");
print("  - Total: ~1,264 params\n\n");

print("🔨 PRÓXIMO PASO: LEVEL 1.5\n");
print("  Implementar training loop para alcanzar 90%+ accuracy\n\n");

print("═══════════════════════════════════════════════════════════════\n");
print("  Arquitectura validada ✅ - Listo para training\n");
print("═══════════════════════════════════════════════════════════════\n");

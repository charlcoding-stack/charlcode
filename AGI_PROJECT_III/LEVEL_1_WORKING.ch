// ============================================
// AGI PROJECT III - LEVEL 1 WORKING
// Math Expert - Functional Version
// ============================================
// 
// With correct API:
// - nn_embedding(indices, matrix)
// - nn_linear(input, weight, bias)
// - tensor_relu(x)
// - argmax(logits)

print("╔══════════════════════════════════════════════════════════════╗\n");
print("║   AGI PROJECT III - LEVEL 1 WORKING                        ║\n");
print("║   Math Expert - Correct API                             ║\n");
print("╚══════════════════════════════════════════════════════════════╝\n\n");

print("🎯 OBJECTIVE: Validate architecture with complete forward pass\n\n");

// ============================================
// PARAMETERS
// ============================================

print("=== INITIALIZING PARAMETERS ===\n\n");

let vocab_size = 12;  // 0-9, +, =
let emb_dim = 16;
let hidden_dim = 32;

// Embedding matrix
let emb_matrix = tensor_randn([12, 16]);

// Hidden layer
let w1 = tensor_randn([16, 32]);
let b1 = tensor_zeros([32]);

// Output layer
let w2 = tensor_randn([32, 16]);
let b2 = tensor_zeros([16]);

print("Architecture:\n");
print("  Embeddings: 12×16 (192 params)\n");
print("  Hidden: 16→32 (544 params)\n");
print("  Output: 32→16 (528 params)\n");
print("  Total: ~1,264 params\n\n");

// ============================================
// FORWARD PASS
// ============================================

print("=== FORWARD PASS: 2 + 3 = ? ===\n\n");

// Create index for token '2'
let idx_2: [float32; 1] = [0.0; 1];
idx_2[0] = 2.0;
let idx_2_tensor = tensor_from_array(idx_2, [1]);

print("1. Get embedding for number 2:\n");
let emb_2 = nn_embedding(idx_2_tensor, emb_matrix);
print("   Embedding shape: [1, 16]\n\n");

print("2. Hidden layer:\n");
let h1 = nn_linear(emb_2, w1, b1);
print("   Before ReLU: [1, 32]\n");
h1 = tensor_relu(h1);
print("   After ReLU: [1, 32]\n\n");

print("3. Output layer:\n");
let logits = nn_linear(h1, w2, b2);
print("   Logits: [1, 16]\n\n");

print("4. Prediction:\n");
let pred = argmax(logits);
print("   Predicted number: ");
print(pred);
print("\n");
print("   (Random without training)\n\n");

// ============================================
// TEST OTHER NUMBERS
// ============================================

print("=== TESTING OTHER NUMBERS ===\n\n");

// Number 0
let idx_0: [float32; 1] = [0.0; 1];
idx_0[0] = 0.0;
let idx_0_tensor = tensor_from_array(idx_0, [1]);
let emb_0 = nn_embedding(idx_0_tensor, emb_matrix);
let h_0 = nn_linear(emb_0, w1, b1);
h_0 = tensor_relu(h_0);
let logits_0 = nn_linear(h_0, w2, b2);
let pred_0 = argmax(logits_0);

print("Input: 0 → Output: ");
print(pred_0);
print("\n");

// Número 5
let idx_5: [float32; 1] = [0.0; 1];
idx_5[0] = 5.0;
let idx_5_tensor = tensor_from_array(idx_5, [1]);
let emb_5 = nn_embedding(idx_5_tensor, emb_matrix);
let h_5 = nn_linear(emb_5, w1, b1);
h_5 = tensor_relu(h_5);
let logits_5 = nn_linear(h_5, w2, b2);
let pred_5 = argmax(logits_5);

print("Input: 5 → Output: ");
print(pred_5);
print("\n");

// Number 9
let idx_9: [float32; 1] = [0.0; 1];
idx_9[0] = 9.0;
let idx_9_tensor = tensor_from_array(idx_9, [1]);
let emb_9 = nn_embedding(idx_9_tensor, emb_matrix);
let h_9 = nn_linear(emb_9, w1, b1);
h_9 = tensor_relu(h_9);
let logits_9 = nn_linear(h_9, w2, b2);
let pred_9 = argmax(logits_9);

print("Input: 9 → Output: ");
print(pred_9);
print("\n\n");

print("Note: Random predictions without training.\n");
print("      We only validate that the forward pass works.\n\n");

// ============================================
// RESULT
// ============================================

print("╔══════════════════════════════════════════════════════════════╗\n");
print("║                    SUCCESS ✅                                ║\n");
print("╚══════════════════════════════════════════════════════════════╝\n\n");

print("✅ LEVEL 1 WORKING - Forward Pass Validated\n\n");

print("What works:\n");
print("  ✅ nn_embedding(indices, matrix)\n");
print("  ✅ nn_linear(input, weight, bias)\n");
print("  ✅ tensor_relu(x)\n");
print("  ✅ argmax(logits)\n");
print("  ✅ Complete architecture executes without errors\n\n");

print("Validated architecture:\n");
print("  Input: Number (0-9)\n");
print("  Embedding: 12×16\n");
print("  Hidden: 16→32 + ReLU\n");
print("  Output: 32→16\n");
print("  Prediction: argmax\n\n");

print("🎯 NEXT STEP: LEVEL 1.5 - Training Loop\n");
print("  - Implement addition dataset\n");
print("  - Training with autograd\n");
print("  - Target: 90%+ accuracy\n\n");

print("📊 PROJECT_III Progress:\n");
print("  ✅ Structure created\n");
print("  ✅ API investigated\n");
print("  ✅ Functional forward pass\n");
print("  ⏳ Training pending\n");
print("  ⏳ Evaluation pending\n\n");

print("═══════════════════════════════════════════════════════════════\n");
print("  Architecture > Scale - In progress 🚀\n");
print("═══════════════════════════════════════════════════════════════\n");

// ═══════════════════════════════════════════════════════════
// AGI PROJECT III - LEVEL 7: Dense Baseline Model
// Multi-task Dense: ~1664 params (vs MoE ~1270 params)
// ═══════════════════════════════════════════════════════════

print("╔══════════════════════════════════════════════════════════════╗");
print("║    AGI PROJECT III - LEVEL 7: Dense Baseline               ║");
print("║    Multi-task Dense Model (~1664 params)                   ║");
print("╚══════════════════════════════════════════════════════════════╝");
print("");

print("📊 Architecture:");
print("  Shared layers: 2→64→16 (~1152 params)");
print("  Task heads (7): 16→{10,2,5,3,3,4,5} (~512 params)");
print("  TOTAL: ~1664 params");
print("");

print("🔑 Key Difference from MoE:");
print("  MoE: Sparse activation (~20% params active per query)");
print("  Dense: ALL params active for every query (100%)");
print("");

// ═══════════════════════════════════════════════════════════
// TRAINING DATA (mismo que MoE usó)
// ═══════════════════════════════════════════════════════════

// Usamos el mismo dataset del Router (70 examples, 10 por dominio)
// pero necesitamos las labels específicas de cada Task

// Domain 0: MATH (10 examples) - Valores iguales [a,a]
let X_math_train = tensor([
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
    5, 5, 6, 6, 7, 7, 8, 8, 9, 9
], [10, 2]);

let Y_math_train = tensor([
    // Additions: 0,1,2,3,4,5,6,7,8,9 → classes 0-9
    1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  // 0
    0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  // 2
    0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  // 4
    0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,  // 6
    0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,  // 8
    0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,  // 10 → class 5 (mod 10)
    0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,  // 12 → class 6 (mod 10)
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,  // 14 → class 7 (mod 10)
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,  // 16 → class 8 (mod 10)
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0   // 18 → class 9 (mod 10)
], [10, 10]);

// Domain 1: LOGIC (10 examples) - Valores diferentes [a,b] donde a>b
let X_logic_train = tensor([
    2, 0, 3, 1, 4, 2, 5, 3, 6, 4,
    7, 5, 8, 6, 9, 7, 10, 8, 11, 9
], [10, 2]);

let Y_logic_train = tensor([
    // true (a>b): class 0, false: class 1
    1.0,0.0,  1.0,0.0,  1.0,0.0,  1.0,0.0,  1.0,0.0,
    1.0,0.0,  1.0,0.0,  1.0,0.0,  1.0,0.0,  1.0,0.0
], [10, 2]);

// Domain 2: CODE (10 examples)
let X_code_train = tensor([
    0.6, 0.2, 0.5, 0.1, 0.7, 0.3, 0.8, 0.4, 0.9, 0.5,
    0.55, 0.15, 0.65, 0.25, 0.75, 0.35, 0.85, 0.45, 0.6, 0.3
], [10, 2]);

let Y_code_train = tensor([
    // operators: mixed (simplificado a class 1 = *)
    0.0,1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,0.0,
    0.0,1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,0.0
], [10, 5]);

// Domain 3: LANGUAGE (10 examples)
let X_lang_train = tensor([
    0.7, 0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.75, 0.5, 0.85,
    0.4, 0.9, 0.6, 0.95, 0.3, 0.88, 0.2, 0.78, 0.1, 0.72
], [10, 2]);

let Y_lang_train = tensor([
    // Sentiment: positive (class 0 dominante)
    1.0,0.0,0.0,  1.0,0.0,0.0,  1.0,0.0,0.0,  1.0,0.0,0.0,  1.0,0.0,0.0,
    1.0,0.0,0.0,  1.0,0.0,0.0,  1.0,0.0,0.0,  1.0,0.0,0.0,  1.0,0.0,0.0
], [10, 3]);

// Domain 4: General (10 examples)
let X_gen_train = tensor([
    13, 13, 11, 11, 17, 17, 10, 10, 15, 15,
    20, 20, 12, 12, 18, 18, 14, 14, 16, 16
], [10, 2]);

let Y_gen_train = tensor([
    // Rangos: low/med/high (mixed)
    0.0,1.0,0.0,  1.0,0.0,0.0,  0.0,0.0,1.0,  1.0,0.0,0.0,  0.0,1.0,0.0,
    0.0,0.0,1.0,  1.0,0.0,0.0,  0.0,0.0,1.0,  0.0,1.0,0.0,  0.0,0.0,1.0
], [10, 3]);

// Domain 5: MEMORY (10 examples)
let X_mem_train = tensor([
    0.95, 0.5, 0.92, 0.3, 0.97, 0.6, 0.91, 0.4, 0.99, 0.7,
    0.93, 0.35, 0.96, 0.55, 0.94, 0.45, 0.98, 0.65, 0.90, 0.25
], [10, 2]);

let Y_mem_train = tensor([
    // Facts: fact IDs (mixed)
    1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,  0.0,0.0,1.0,0.0,  0.0,0.0,0.0,1.0,  1.0,0.0,0.0,0.0,
    0.0,1.0,0.0,0.0,  0.0,0.0,1.0,0.0,  0.0,0.0,0.0,1.0,  1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0
], [10, 4]);

// Domain 6: REASONING (10 examples)
let X_reas_train = tensor([
    0.1, 0.6, 0.2, 0.3, 0.3, 0.6, 0.4, 0.4, 0.5, 0.7,
    0.15, 0.5, 0.25, 0.35, 0.35, 0.55, 0.45, 0.45, 0.50, 0.65
], [10, 2]);

let Y_reas_train = tensor([
    // Reasoning types: mixed (class 2 dominante = compuesto)
    0.0,0.0,1.0,0.0,0.0,  0.0,0.0,1.0,0.0,0.0,  0.0,0.0,1.0,0.0,0.0,  0.0,0.0,1.0,0.0,0.0,  0.0,0.0,1.0,0.0,0.0,
    0.0,0.0,1.0,0.0,0.0,  0.0,0.0,1.0,0.0,0.0,  0.0,0.0,1.0,0.0,0.0,  0.0,0.0,1.0,0.0,0.0,  0.0,0.0,1.0,0.0,0.0
], [10, 5]);

// ═══════════════════════════════════════════════════════════
// MODELO DENSE: Shared Layers + Multi-task Heads
// ═══════════════════════════════════════════════════════════

print("═══ BUILDING DENSE MODEL ═══");
print("");

// Shared layers: 2→64→16
let W1_shared = tensor_randn_seeded([2, 64], 3000);
W1_shared = tensor_requires_grad(W1_shared, true);
let b1_shared = tensor_zeros([64]);
b1_shared = tensor_requires_grad(b1_shared, true);

let W2_shared = tensor_randn_seeded([64, 16], 3001);
W2_shared = tensor_requires_grad(W2_shared, true);
let b2_shared = tensor_zeros([16]);
b2_shared = tensor_requires_grad(b2_shared, true);

// Task-specific heads
// Math head: 16→10
let W_math = tensor_randn_seeded([16, 10], 3002);
W_math = tensor_requires_grad(W_math, true);
let b_math = tensor_zeros([10]);
b_math = tensor_requires_grad(b_math, true);

// Logic head: 16→2
let W_logic = tensor_randn_seeded([16, 2], 3003);
W_logic = tensor_requires_grad(W_logic, true);
let b_logic = tensor_zeros([2]);
b_logic = tensor_requires_grad(b_logic, true);

// Code head: 16→5
let W_code = tensor_randn_seeded([16, 5], 3004);
W_code = tensor_requires_grad(W_code, true);
let b_code = tensor_zeros([5]);
b_code = tensor_requires_grad(b_code, true);

// Language head: 16→3
let W_lang = tensor_randn_seeded([16, 3], 3005);
W_lang = tensor_requires_grad(W_lang, true);
let b_lang = tensor_zeros([3]);
b_lang = tensor_requires_grad(b_lang, true);

// General head: 16→3
let W_gen = tensor_randn_seeded([16, 3], 3006);
W_gen = tensor_requires_grad(W_gen, true);
let b_gen = tensor_zeros([3]);
b_gen = tensor_requires_grad(b_gen, true);

// Memory head: 16→4
let W_mem = tensor_randn_seeded([16, 4], 3007);
W_mem = tensor_requires_grad(W_mem, true);
let b_mem = tensor_zeros([4]);
b_mem = tensor_requires_grad(b_mem, true);

// Reasoning head: 16→5
let W_reas = tensor_randn_seeded([16, 5], 3008);
W_reas = tensor_requires_grad(W_reas, true);
let b_reas = tensor_zeros([5]);
b_reas = tensor_requires_grad(b_reas, true);

print("✅ Dense model initialized");
print("  Param count: 2×64 (128) + 64×16 (1024) + task heads (512) = ~1664 params");
print("");

// ═══════════════════════════════════════════════════════════
// TRAINING: Train all tasks jointly
// ═══════════════════════════════════════════════════════════

print("═══ TRAINING DENSE MODEL (5000 epochs) ═══");
print("Training on all 7 domains simultaneously...");
print("");

let lr_dense = 0.01;
let optimizer_shared = sgd_create(lr_dense);
let optimizer_math = sgd_create(lr_dense);
let optimizer_logic = sgd_create(lr_dense);
let optimizer_code = sgd_create(lr_dense);
let optimizer_lang = sgd_create(lr_dense);
let optimizer_gen = sgd_create(lr_dense);
let optimizer_mem = sgd_create(lr_dense);
let optimizer_reas = sgd_create(lr_dense);

for epoch in 0..5000 {
    // Forward pass para cada dominio (todos comparten representación)

    // Math
    let h1_math = nn_linear(X_math_train, W1_shared, b1_shared);
    h1_math = nn_relu(h1_math);
    let h2_math = nn_linear(h1_math, W2_shared, b2_shared);
    h2_math = nn_relu(h2_math);
    let logits_math = nn_linear(h2_math, W_math, b_math);
    let loss_math = nn_cross_entropy_logits(logits_math, Y_math_train);

    // Logic
    let h1_logic = nn_linear(X_logic_train, W1_shared, b1_shared);
    h1_logic = nn_relu(h1_logic);
    let h2_logic = nn_linear(h1_logic, W2_shared, b2_shared);
    h2_logic = nn_relu(h2_logic);
    let logits_logic = nn_linear(h2_logic, W_logic, b_logic);
    let loss_logic = nn_cross_entropy_logits(logits_logic, Y_logic_train);

    // Code
    let h1_code = nn_linear(X_code_train, W1_shared, b1_shared);
    h1_code = nn_relu(h1_code);
    let h2_code = nn_linear(h1_code, W2_shared, b2_shared);
    h2_code = nn_relu(h2_code);
    let logits_code = nn_linear(h2_code, W_code, b_code);
    let loss_code = nn_cross_entropy_logits(logits_code, Y_code_train);

    // Language
    let h1_lang = nn_linear(X_lang_train, W1_shared, b1_shared);
    h1_lang = nn_relu(h1_lang);
    let h2_lang = nn_linear(h1_lang, W2_shared, b2_shared);
    h2_lang = nn_relu(h2_lang);
    let logits_lang = nn_linear(h2_lang, W_lang, b_lang);
    let loss_lang = nn_cross_entropy_logits(logits_lang, Y_lang_train);

    // General
    let h1_gen = nn_linear(X_gen_train, W1_shared, b1_shared);
    h1_gen = nn_relu(h1_gen);
    let h2_gen = nn_linear(h1_gen, W2_shared, b2_shared);
    h2_gen = nn_relu(h2_gen);
    let logits_gen = nn_linear(h2_gen, W_gen, b_gen);
    let loss_gen = nn_cross_entropy_logits(logits_gen, Y_gen_train);

    // Memory
    let h1_mem = nn_linear(X_mem_train, W1_shared, b1_shared);
    h1_mem = nn_relu(h1_mem);
    let h2_mem = nn_linear(h1_mem, W2_shared, b2_shared);
    h2_mem = nn_relu(h2_mem);
    let logits_mem = nn_linear(h2_mem, W_mem, b_mem);
    let loss_mem = nn_cross_entropy_logits(logits_mem, Y_mem_train);

    // Reasoning
    let h1_reas = nn_linear(X_reas_train, W1_shared, b1_shared);
    h1_reas = nn_relu(h1_reas);
    let h2_reas = nn_linear(h1_reas, W2_shared, b2_shared);
    h2_reas = nn_relu(h2_reas);
    let logits_reas = nn_linear(h2_reas, W_reas, b_reas);
    let loss_reas = nn_cross_entropy_logits(logits_reas, Y_reas_train);

    // Backward pass para cada Task
    tensor_backward(loss_math);
    tensor_backward(loss_logic);
    tensor_backward(loss_code);
    tensor_backward(loss_lang);
    tensor_backward(loss_gen);
    tensor_backward(loss_mem);
    tensor_backward(loss_reas);

    // Update shared layers (acumulan gradientes de todas las Tasks)
    let params_shared = [W1_shared, b1_shared, W2_shared, b2_shared];
    let updated_shared = sgd_step(optimizer_shared, params_shared);
    W1_shared = updated_shared[0];
    b1_shared = updated_shared[1];
    W2_shared = updated_shared[2];
    b2_shared = updated_shared[3];

    // Update task-specific heads
    let params_math = [W_math, b_math];
    let updated_math = sgd_step(optimizer_math, params_math);
    W_math = updated_math[0];
    b_math = updated_math[1];

    let params_logic = [W_logic, b_logic];
    let updated_logic = sgd_step(optimizer_logic, params_logic);
    W_logic = updated_logic[0];
    b_logic = updated_logic[1];

    let params_code = [W_code, b_code];
    let updated_code = sgd_step(optimizer_code, params_code);
    W_code = updated_code[0];
    b_code = updated_code[1];

    let params_lang = [W_lang, b_lang];
    let updated_lang = sgd_step(optimizer_lang, params_lang);
    W_lang = updated_lang[0];
    b_lang = updated_lang[1];

    let params_gen = [W_gen, b_gen];
    let updated_gen = sgd_step(optimizer_gen, params_gen);
    W_gen = updated_gen[0];
    b_gen = updated_gen[1];

    let params_mem = [W_mem, b_mem];
    let updated_mem = sgd_step(optimizer_mem, params_mem);
    W_mem = updated_mem[0];
    b_mem = updated_mem[1];

    let params_reas = [W_reas, b_reas];
    let updated_reas = sgd_step(optimizer_reas, params_reas);
    W_reas = updated_reas[0];
    b_reas = updated_reas[1];

    // Zero gradients
    W1_shared = tensor_zero_grad(W1_shared);
    b1_shared = tensor_zero_grad(b1_shared);
    W2_shared = tensor_zero_grad(W2_shared);
    b2_shared = tensor_zero_grad(b2_shared);
    W_math = tensor_zero_grad(W_math);
    b_math = tensor_zero_grad(b_math);
    W_logic = tensor_zero_grad(W_logic);
    b_logic = tensor_zero_grad(b_logic);
    W_code = tensor_zero_grad(W_code);
    b_code = tensor_zero_grad(b_code);
    W_lang = tensor_zero_grad(W_lang);
    b_lang = tensor_zero_grad(b_lang);
    W_gen = tensor_zero_grad(W_gen);
    b_gen = tensor_zero_grad(b_gen);
    W_mem = tensor_zero_grad(W_mem);
    b_mem = tensor_zero_grad(b_mem);
    W_reas = tensor_zero_grad(W_reas);
    b_reas = tensor_zero_grad(b_reas);
}

print("✅ Dense model trained on all 7 domains");
print("");

// ═══════════════════════════════════════════════════════════
// QUICK TEST
// ═══════════════════════════════════════════════════════════

print("═══ QUICK TEST ON TRAINING EXAMPLES ═══");
print("");

// Test Math
print("TEST: Math [2, 2] (2+2=4 → clase 4)");
let test_math = tensor([2, 2], [1, 2]);
let h1_test_math = nn_linear(test_math, W1_shared, b1_shared);
h1_test_math = nn_relu(h1_test_math);
let h2_test_math = nn_linear(h1_test_math, W2_shared, b2_shared);
h2_test_math = nn_relu(h2_test_math);
let logits_test_math = nn_linear(h2_test_math, W_math, b_math);
let pred_test_math = argmax(logits_test_math);
if pred_test_math == 4 {
    print("  Dense Math head → Clase 4 ✅");
} else {
    print("  Dense Math head → ❌");
}
print("");

// Test Logic
print("TEST: Logic [3, 1] (3>1: true → clase 0)");
let test_logic = tensor([3, 1], [1, 2]);
let h1_test_logic = nn_linear(test_logic, W1_shared, b1_shared);
h1_test_logic = nn_relu(h1_test_logic);
let h2_test_logic = nn_linear(h1_test_logic, W2_shared, b2_shared);
h2_test_logic = nn_relu(h2_test_logic);
let logits_test_logic = nn_linear(h2_test_logic, W_logic, b_logic);
let pred_test_logic = argmax(logits_test_logic);
if pred_test_logic == 0 {
    print("  Dense Logic head → Clase 0 (true) ✅");
} else {
    print("  Dense Logic head → ❌");
}
print("");

print("╔══════════════════════════════════════════════════════════════╗");
print("║      ✅ DENSE BASELINE MODEL COMPLETADO                     ║");
print("╚══════════════════════════════════════════════════════════════╝");
print("");

print("📊 Model Summary:");
print("  Architecture: Multi-task Dense");
print("  Shared layers: 2→64→16");
print("  Task heads: 7 (Math, Logic, Code, Language, General, Memory, Reasoning)");
print("  Total params: ~1664");
print("  Activation: 100% (all params active per query)");
print("");

print("📈 Próximo paso:");
print("  - Evaluar Dense en test dataset (70 ejemplos unseen)");
print("  - Comparar con MoE en métricas de accuracy");

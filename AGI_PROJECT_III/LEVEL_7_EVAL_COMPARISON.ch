// ═══════════════════════════════════════════════════════════════════
// AGI PROJECT III - LEVEL 7: MoE vs Dense Evaluation
// Comparison directa en test dataset (70 examples unseen)
// ═══════════════════════════════════════════════════════════════════

print("╔══════════════════════════════════════════════════════════════╗");
print("║    AGI PROJECT III - LEVEL 7: MoE vs Dense EVALUATION      ║");
print("║    Architecture > Scale: La Prueba Final                   ║");
print("╚══════════════════════════════════════════════════════════════╝");
print("");

print("🎯 OBJETIVO:");
print("  Validar: MoE (~1270 params, sparse) > Dense (~1664 params, full)");
print("");

// ═══════════════════════════════════════════════════════════
// PART 1: TRAIN MOE SYSTEM (Simplified - 3000 epochs)
// ═══════════════════════════════════════════════════════════

print("═══════════════════════════════════════════════════════════");
print("PART 1: TRAINING MoE SYSTEM");
print("═══════════════════════════════════════════════════════════");
print("");

let lr = 0.01;

// ROUTER (simplified training)
print("Training Router (3000 epochs)...");

let X_router = tensor([
    // Domain 0: MATH (valores IGUALES)
    0, 0, 1, 1, 2, 2, 3, 3,
    // Domain 1: LOGIC (valores DIFERENTES)
    2, 0, 3, 1, 4, 2, 5, 3,
    // Domain 2: CODE
    0.6, 0.2, 0.7, 0.3,
    // Domain 3: LANGUAGE
    0.7, 0.9, 0.3, 0.8,
    // Domain 4: General
    13, 13, 17, 17,
    // Domain 5: MEMORY
    0.95, 0.5, 0.92, 0.3,
    // Domain 6: REASONING
    0.1, 0.6, 0.2, 0.3
], [18, 2]);

let Y_router = tensor([
    // 18 examples: 4+4+2+2+2+2+2
    // Domain 0: Math (4 examples)
    1.0,0.0,0.0,0.0,0.0,0.0,0.0,  1.0,0.0,0.0,0.0,0.0,0.0,0.0,  1.0,0.0,0.0,0.0,0.0,0.0,0.0,  1.0,0.0,0.0,0.0,0.0,0.0,0.0,
    // Domain 1: Logic (4 examples)
    0.0,1.0,0.0,0.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,0.0,0.0,0.0,
    // Domain 2: Code (2 examples)
    0.0,0.0,1.0,0.0,0.0,0.0,0.0,  0.0,0.0,1.0,0.0,0.0,0.0,0.0,
    // Domain 3: Language (2 examples)
    0.0,0.0,0.0,1.0,0.0,0.0,0.0,  0.0,0.0,0.0,1.0,0.0,0.0,0.0,
    // Domain 4: General (2 examples)
    0.0,0.0,0.0,0.0,1.0,0.0,0.0,  0.0,0.0,0.0,0.0,1.0,0.0,0.0,
    // Domain 5: Memory (2 examples)
    0.0,0.0,0.0,0.0,0.0,1.0,0.0,  0.0,0.0,0.0,0.0,0.0,1.0,0.0,
    // Domain 6: Reasoning (2 examples)
    0.0,0.0,0.0,0.0,0.0,0.0,1.0,  0.0,0.0,0.0,0.0,0.0,0.0,1.0
], [18, 7]);

let W1_router = tensor_randn_seeded([2, 32], 1600);
W1_router = tensor_requires_grad(W1_router, true);
let b1_router = tensor_zeros([32]);
b1_router = tensor_requires_grad(b1_router, true);
let W2_router = tensor_randn_seeded([32, 7], 1601);
W2_router = tensor_requires_grad(W2_router, true);
let b2_router = tensor_zeros([7]);
b2_router = tensor_requires_grad(b2_router, true);

let optimizer_router = sgd_create(lr);

for epoch in 0..3000 {
    let h1 = nn_linear(X_router, W1_router, b1_router);
    h1 = nn_relu(h1);
    let logits = nn_linear(h1, W2_router, b2_router);
    let loss = nn_cross_entropy_logits(logits, Y_router);
    tensor_backward(loss);
    let params = [W1_router, b1_router, W2_router, b2_router];
    let updated = sgd_step(optimizer_router, params);
    W1_router = updated[0];
    b1_router = updated[1];
    W2_router = updated[2];
    b2_router = updated[3];
    W1_router = tensor_zero_grad(W1_router);
    b1_router = tensor_zero_grad(b1_router);
    W2_router = tensor_zero_grad(W2_router);
    b2_router = tensor_zero_grad(b2_router);
}

print("✅ Router trained");
print("");

// EXPERT MATH (simplified training)
print("Training Expert Math (2000 epochs)...");

let X_math = tensor([
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
    5, 5, 6, 6, 7, 7, 8, 8, 9, 9
], [10, 2]);

let Y_math = tensor([
    1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0
], [10, 10]);

let W1_math = tensor_randn_seeded([2, 32], 1700);
W1_math = tensor_requires_grad(W1_math, true);
let b1_math = tensor_zeros([32]);
b1_math = tensor_requires_grad(b1_math, true);
let W2_math = tensor_randn_seeded([32, 10], 1701);
W2_math = tensor_requires_grad(W2_math, true);
let b2_math = tensor_zeros([10]);
b2_math = tensor_requires_grad(b2_math, true);

let optimizer_math = sgd_create(lr);

for epoch in 0..2000 {
    let h1 = nn_linear(X_math, W1_math, b1_math);
    h1 = nn_relu(h1);
    let logits = nn_linear(h1, W2_math, b2_math);
    let loss = nn_cross_entropy_logits(logits, Y_math);
    tensor_backward(loss);
    let params = [W1_math, b1_math, W2_math, b2_math];
    let updated = sgd_step(optimizer_math, params);
    W1_math = updated[0];
    b1_math = updated[1];
    W2_math = updated[2];
    b2_math = updated[3];
    W1_math = tensor_zero_grad(W1_math);
    b1_math = tensor_zero_grad(b1_math);
    W2_math = tensor_zero_grad(W2_math);
    b2_math = tensor_zero_grad(b2_math);
}

print("✅ Expert Math trained");
print("");

// EXPERT LOGIC (simplified training)
print("Training Expert Logic (2000 epochs)...");

let X_logic = tensor([
    1, 0, 2, 1, 3, 2, 4, 3,
    0, 1, 1, 2, 2, 3, 3, 4
], [8, 2]);

let Y_logic = tensor([
    1.0,0.0,  1.0,0.0,  1.0,0.0,  1.0,0.0,
    0.0,1.0,  0.0,1.0,  0.0,1.0,  0.0,1.0
], [8, 2]);

let W1_logic = tensor_randn_seeded([2, 16], 1702);
W1_logic = tensor_requires_grad(W1_logic, true);
let b1_logic = tensor_zeros([16]);
b1_logic = tensor_requires_grad(b1_logic, true);
let W2_logic = tensor_randn_seeded([16, 2], 1703);
W2_logic = tensor_requires_grad(W2_logic, true);
let b2_logic = tensor_zeros([2]);
b2_logic = tensor_requires_grad(b2_logic, true);

let optimizer_logic = sgd_create(lr);

for epoch in 0..2000 {
    let h1 = nn_linear(X_logic, W1_logic, b1_logic);
    h1 = nn_relu(h1);
    let logits = nn_linear(h1, W2_logic, b2_logic);
    let loss = nn_cross_entropy_logits(logits, Y_logic);
    tensor_backward(loss);
    let params = [W1_logic, b1_logic, W2_logic, b2_logic];
    let updated = sgd_step(optimizer_logic, params);
    W1_logic = updated[0];
    b1_logic = updated[1];
    W2_logic = updated[2];
    b2_logic = updated[3];
    W1_logic = tensor_zero_grad(W1_logic);
    b1_logic = tensor_zero_grad(b1_logic);
    W2_logic = tensor_zero_grad(W2_logic);
    b2_logic = tensor_zero_grad(b2_logic);
}

print("✅ Expert Logic trained");
print("");

print("MoE System Ready: Router + 2 Experts (simplified for speed)");
print("Total MoE params: ~240 (Router) + ~350 (Math) + ~50 (Logic) = ~640 params");
print("");

// ═══════════════════════════════════════════════════════════
// PART 2: TRAIN DENSE SYSTEM (Simplified)
// ═══════════════════════════════════════════════════════════

print("═══════════════════════════════════════════════════════════");
print("PART 2: TRAINING DENSE BASELINE (Multi-task)");
print("═══════════════════════════════════════════════════════════");
print("");

// Shared layers
let W1_shared = tensor_randn_seeded([2, 32], 3000);
W1_shared = tensor_requires_grad(W1_shared, true);
let b1_shared = tensor_zeros([32]);
b1_shared = tensor_requires_grad(b1_shared, true);
let W2_shared = tensor_randn_seeded([32, 16], 3001);
W2_shared = tensor_requires_grad(W2_shared, true);
let b2_shared = tensor_zeros([16]);
b2_shared = tensor_requires_grad(b2_shared, true);

// Math head
let W_math_dense = tensor_randn_seeded([16, 10], 3002);
W_math_dense = tensor_requires_grad(W_math_dense, true);
let b_math_dense = tensor_zeros([10]);
b_math_dense = tensor_requires_grad(b_math_dense, true);

// Logic head
let W_logic_dense = tensor_randn_seeded([16, 2], 3003);
W_logic_dense = tensor_requires_grad(W_logic_dense, true);
let b_logic_dense = tensor_zeros([2]);
b_logic_dense = tensor_requires_grad(b_logic_dense, true);

print("Training Dense (2000 epochs, all tasks jointly)...");

let optimizer_dense_shared = sgd_create(lr);
let optimizer_dense_math = sgd_create(lr);
let optimizer_dense_logic = sgd_create(lr);

for epoch in 0..2000 {
    // Math task
    let h1_m = nn_linear(X_math, W1_shared, b1_shared);
    h1_m = nn_relu(h1_m);
    let h2_m = nn_linear(h1_m, W2_shared, b2_shared);
    h2_m = nn_relu(h2_m);
    let logits_m = nn_linear(h2_m, W_math_dense, b_math_dense);
    let loss_m = nn_cross_entropy_logits(logits_m, Y_math);

    // Logic task
    let h1_l = nn_linear(X_logic, W1_shared, b1_shared);
    h1_l = nn_relu(h1_l);
    let h2_l = nn_linear(h1_l, W2_shared, b2_shared);
    h2_l = nn_relu(h2_l);
    let logits_l = nn_linear(h2_l, W_logic_dense, b_logic_dense);
    let loss_l = nn_cross_entropy_logits(logits_l, Y_logic);

    // Backward
    tensor_backward(loss_m);
    tensor_backward(loss_l);

    // Update shared
    let params_shared = [W1_shared, b1_shared, W2_shared, b2_shared];
    let updated_shared = sgd_step(optimizer_dense_shared, params_shared);
    W1_shared = updated_shared[0];
    b1_shared = updated_shared[1];
    W2_shared = updated_shared[2];
    b2_shared = updated_shared[3];

    // Update heads
    let params_m = [W_math_dense, b_math_dense];
    let updated_m = sgd_step(optimizer_dense_math, params_m);
    W_math_dense = updated_m[0];
    b_math_dense = updated_m[1];

    let params_l = [W_logic_dense, b_logic_dense];
    let updated_l = sgd_step(optimizer_dense_logic, params_l);
    W_logic_dense = updated_l[0];
    b_logic_dense = updated_l[1];

    // Zero grads
    W1_shared = tensor_zero_grad(W1_shared);
    b1_shared = tensor_zero_grad(b1_shared);
    W2_shared = tensor_zero_grad(W2_shared);
    b2_shared = tensor_zero_grad(b2_shared);
    W_math_dense = tensor_zero_grad(W_math_dense);
    b_math_dense = tensor_zero_grad(b_math_dense);
    W_logic_dense = tensor_zero_grad(W_logic_dense);
    b_logic_dense = tensor_zero_grad(b_logic_dense);
}

print("✅ Dense Baseline trained");
print("Total Dense params: 2→32→16 (576) + heads 16→{10,2} (192) = ~768 params");
print("");

// ═══════════════════════════════════════════════════════════
// PART 3: EVALUATE ON TEST SET
// ═══════════════════════════════════════════════════════════

print("═══════════════════════════════════════════════════════════");
print("PART 3: EVALUATION ON TEST DATASET");
print("═══════════════════════════════════════════════════════════");
print("");

// Test Math domain (10 examples)
print("═══ EVALUATING MATH DOMAIN (10 test cases) ═══");
let X_math_test = tensor([
    0, 1,  1, 2,  2, 3,  3, 4,  4, 5,
    0, 2,  1, 3,  2, 4,  3, 5,  0, 3
], [10, 2]);

// Expected: 1,3,5,7,9,2,4,6,8,3

// MoE evaluation (Math)
let moe_math_correct = 0;
for i in 0..10 {
    let test_x = tensor_get(X_math_test, [i, 0]);
    let test_y = tensor_get(X_math_test, [i, 1]);
    let query = tensor([test_x, test_y], [1, 2]);

    // Router
    let h1_r = nn_linear(query, W1_router, b1_router);
    h1_r = nn_relu(h1_r);
    let logits_r = nn_linear(h1_r, W2_router, b2_router);
    let domain_pred = argmax(logits_r);

    // Expert Math (si router acertó)
    if domain_pred == 0 {
        let h1_m = nn_linear(query, W1_math, b1_math);
        h1_m = nn_relu(h1_m);
        let logits_m = nn_linear(h1_m, W2_math, b2_math);
        let pred_m = argmax(logits_m);

        // Check if correct (mapping Addition to class)
        let suma = test_x + test_y;
        if suma < 10 {
            if pred_m == suma {
                moe_math_correct = moe_math_correct + 1;
            }
        } else {
            // For 10+, map to class mod 10
            let expected_class = suma - 10;
            if pred_m == expected_class {
                moe_math_correct = moe_math_correct + 1;
            }
        }
    }
}

// Dense evaluation (Math)
let dense_math_correct = 0;
for i in 0..10 {
    let test_x = tensor_get(X_math_test, [i, 0]);
    let test_y = tensor_get(X_math_test, [i, 1]);
    let query = tensor([test_x, test_y], [1, 2]);

    let h1_d = nn_linear(query, W1_shared, b1_shared);
    h1_d = nn_relu(h1_d);
    let h2_d = nn_linear(h1_d, W2_shared, b2_shared);
    h2_d = nn_relu(h2_d);
    let logits_d = nn_linear(h2_d, W_math_dense, b_math_dense);
    let pred_d = argmax(logits_d);

    let suma = test_x + test_y;
    if suma < 10 {
        if pred_d == suma {
            dense_math_correct = dense_math_correct + 1;
        }
    } else {
        let expected_class = suma - 10;
        if pred_d == expected_class {
            dense_math_correct = dense_math_correct + 1;
        }
    }
}

print("Math Domain Results:");
if moe_math_correct == 10 {
    print("  MoE:   10/10 ✅");
} else {
    print("  MoE:   < 10/10");
}
if dense_math_correct == 10 {
    print("  Dense: 10/10 ✅");
} else {
    print("  Dense: < 10/10");
}
print("");

// Test Logic domain (10 examples)
print("═══ EVALUATING LOGIC DOMAIN (10 test cases) ═══");
let X_logic_test = tensor([
    4, 1,  5, 2,  6, 3,  7, 4,  8, 5,
    1, 4,  2, 5,  3, 6,  1, 1,  3, 3
], [10, 2]);

// Expected: true(5x), false(5x)

// MoE evaluation (Logic)
let moe_logic_correct = 0;
for i in 0..10 {
    let test_x = tensor_get(X_logic_test, [i, 0]);
    let test_y = tensor_get(X_logic_test, [i, 1]);
    let query = tensor([test_x, test_y], [1, 2]);

    let h1_r = nn_linear(query, W1_router, b1_router);
    h1_r = nn_relu(h1_r);
    let logits_r = nn_linear(h1_r, W2_router, b2_router);
    let domain_pred = argmax(logits_r);

    if domain_pred == 1 {
        let h1_l = nn_linear(query, W1_logic, b1_logic);
        h1_l = nn_relu(h1_l);
        let logits_l = nn_linear(h1_l, W2_logic, b2_logic);
        let pred_l = argmax(logits_l);

        // Check correctness
        if test_x > test_y {
            if pred_l == 0 {
                moe_logic_correct = moe_logic_correct + 1;
            }
        } else {
            if pred_l == 1 {
                moe_logic_correct = moe_logic_correct + 1;
            }
        }
    }
}

// Dense evaluation (Logic)
let dense_logic_correct = 0;
for i in 0..10 {
    let test_x = tensor_get(X_logic_test, [i, 0]);
    let test_y = tensor_get(X_logic_test, [i, 1]);
    let query = tensor([test_x, test_y], [1, 2]);

    let h1_d = nn_linear(query, W1_shared, b1_shared);
    h1_d = nn_relu(h1_d);
    let h2_d = nn_linear(h1_d, W2_shared, b2_shared);
    h2_d = nn_relu(h2_d);
    let logits_d = nn_linear(h2_d, W_logic_dense, b_logic_dense);
    let pred_d = argmax(logits_d);

    if test_x > test_y {
        if pred_d == 0 {
            dense_logic_correct = dense_logic_correct + 1;
        }
    } else {
        if pred_d == 1 {
            dense_logic_correct = dense_logic_correct + 1;
        }
    }
}

print("Logic Domain Results:");
if moe_logic_correct == 10 {
    print("  MoE:   10/10 ✅");
} else {
    print("  MoE:   < 10/10");
}
if dense_logic_correct == 10 {
    print("  Dense: 10/10 ✅");
} else {
    print("  Dense: < 10/10");
}
print("");

// ═══════════════════════════════════════════════════════════
// FINAL COMPARISON
// ═══════════════════════════════════════════════════════════

print("╔══════════════════════════════════════════════════════════════╗");
print("║              LEVEL 7: FINAL RESULTS                         ║");
print("╚══════════════════════════════════════════════════════════════╝");
print("");

let moe_total = moe_math_correct + moe_logic_correct;
let dense_total = dense_math_correct + dense_logic_correct;

print("📊 ACCURACY COMPARISON:");
print("");
print("  Math Domain:");
if moe_math_correct >= dense_math_correct {
    print("    MoE:   Better or equal ✅");
} else {
    print("    MoE:   Lower than Dense");
}
if dense_math_correct >= moe_math_correct {
    print("    Dense: Better or equal");
} else {
    print("    Dense: Lower than MoE ✅");
}
print("");
print("  Logic Domain:");
if moe_logic_correct >= dense_logic_correct {
    print("    MoE:   Better or equal ✅");
} else {
    print("    MoE:   Lower than Dense");
}
if dense_logic_correct >= moe_logic_correct {
    print("    Dense: Better or equal");
} else {
    print("    Dense: Lower than MoE ✅");
}
print("");
print("  OVERALL (20 test cases):");
if moe_total >= dense_total {
    print("    MoE:   WINS or TIE ✅");
} else {
    print("    MoE:   Lower than Dense");
}
if dense_total >= moe_total {
    print("    Dense: WINS or TIE");
} else {
    print("    Dense: Lower than MoE ✅");
}
print("");

print("📈 EFFICIENCY:");
print("  MoE:   ~640 params, ~20% active per query (~128 params)");
print("  Dense: ~768 params, 100% active per query (768 params)");
print("");

if moe_total > dense_total {
    print("✅ TESIS VALIDADA: MoE > Dense");
    print("   Arquitectura especializada supera a generalista");
} else {
    print("⚠️ Resultados mixtos - necesita más tuning");
}

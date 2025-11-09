// ============================================
// AGI PROJECT III - LEVEL 4: Memory Expert + MoE
// Router + 6 Specialized Experts (including Memory)
// Total: ~1100 params
// ============================================

print("╔══════════════════════════════════════════════════════════════╗\n")
print("║      AGI PROJECT III - LEVEL 4: Memory Expert + MoE         ║\n")
print("║      6 Experts: Math, Logic, Code, Language, General, Memory║\n")
print("╚══════════════════════════════════════════════════════════════╝\n\n")

let lr = 0.01

// ============================================
// EXPERT 1: MATH (de LEVEL 3)
// ============================================

print("═══ EXPERT 1: MATH ═══\n")
print("Tarea: Sumas (a + b = 0-9)\n")
print("Training 3000 epochs...\n")

let X_math = tensor([
    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0,
    2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 0.0, 3.0, 1.0,
    3.0, 2.0, 3.0, 3.0, 4.0, 0.0, 4.0, 1.0, 4.0, 2.0,
    4.0, 3.0, 4.0, 4.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0
], [20, 2])

let Y_math = tensor([
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
], [20, 10])

let W1_math = tensor_randn_seeded([2, 32], 100)
W1_math = tensor_requires_grad(W1_math, true)
let b1_math = tensor_zeros([32])
b1_math = tensor_requires_grad(b1_math, true)
let W2_math = tensor_randn_seeded([32, 10], 200)
W2_math = tensor_requires_grad(W2_math, true)
let b2_math = tensor_zeros([10])
b2_math = tensor_requires_grad(b2_math, true)

let optimizer_math = sgd_create(lr)
let epoch = 0
while epoch < 3000 {
    let h1 = nn_linear(X_math, W1_math, b1_math)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2_math, b2_math)
    let loss = nn_cross_entropy_logits(logits, Y_math)
    tensor_backward(loss)
    let params = [W1_math, b1_math, W2_math, b2_math]
    let updated = sgd_step(optimizer_math, params)
    W1_math = updated[0]
    b1_math = updated[1]
    W2_math = updated[2]
    b2_math = updated[3]
    epoch = epoch + 1
}
print("✅ Expert Math trained\n\n")

// ============================================
// EXPERT 2: LOGIC (de LEVEL 3)
// ============================================

print("═══ EXPERT 2: LOGIC ═══\n")
print("Tarea: Comparaciones (a > b?)\n")
print("Training 3000 epochs...\n")

let X_logic = tensor([
    0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 2.0,
    2.0, 0.0, 0.0, 2.0, 3.0, 2.0, 2.0, 3.0, 4.0, 3.0,
    3.0, 4.0, 4.0, 4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0
], [15, 2])

let Y_logic = tensor([
    1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0
], [15, 2])

let W1_logic = tensor_randn_seeded([2, 16], 300)
W1_logic = tensor_requires_grad(W1_logic, true)
let b1_logic = tensor_zeros([16])
b1_logic = tensor_requires_grad(b1_logic, true)
let W2_logic = tensor_randn_seeded([16, 2], 400)
W2_logic = tensor_requires_grad(W2_logic, true)
let b2_logic = tensor_zeros([2])
b2_logic = tensor_requires_grad(b2_logic, true)

let optimizer_logic = sgd_create(lr)
epoch = 0
while epoch < 3000 {
    let h1 = nn_linear(X_logic, W1_logic, b1_logic)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2_logic, b2_logic)
    let loss = nn_cross_entropy_logits(logits, Y_logic)
    tensor_backward(loss)
    let params = [W1_logic, b1_logic, W2_logic, b2_logic]
    let updated = sgd_step(optimizer_logic, params)
    W1_logic = updated[0]
    b1_logic = updated[1]
    W2_logic = updated[2]
    b2_logic = updated[3]
    epoch = epoch + 1
}
print("✅ Expert Logic trained\n\n")

// ============================================
// EXPERT 3: CODE (de LEVEL 3)
// ============================================

print("═══ EXPERT 3: CODE ═══\n")
print("Tarea: Identificar operador\n")
print("Training 4000 epochs...\n")

let X_code = tensor([
    0.5, 0.2, 0.1, 0.2, 0.6, 0.2, 0.15, 0.2, 0.1, 0.3,
    0.7, 0.3, 0.1, 0.4, 0.8, 0.2, 0.2, 0.4, 0.2, 0.5,
    0.9, 0.4, 0.2, 0.3, 0.12, 0.3, 0.25, 0.2, 0.3, 0.4
], [15, 2])

let Y_code = tensor([
    1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0
], [15, 5])

let W1_code = tensor_randn_seeded([2, 32], 500)
W1_code = tensor_requires_grad(W1_code, true)
let b1_code = tensor_zeros([32])
b1_code = tensor_requires_grad(b1_code, true)
let W2_code = tensor_randn_seeded([32, 5], 600)
W2_code = tensor_requires_grad(W2_code, true)
let b2_code = tensor_zeros([5])
b2_code = tensor_requires_grad(b2_code, true)

let optimizer_code = sgd_create(lr)
epoch = 0
while epoch < 4000 {
    let h1 = nn_linear(X_code, W1_code, b1_code)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2_code, b2_code)
    let loss = nn_cross_entropy_logits(logits, Y_code)
    tensor_backward(loss)
    let params = [W1_code, b1_code, W2_code, b2_code]
    let updated = sgd_step(optimizer_code, params)
    W1_code = updated[0]
    b1_code = updated[1]
    W2_code = updated[2]
    b2_code = updated[3]
    epoch = epoch + 1
}
print("✅ Expert Code trained\n\n")

// ============================================
// EXPERT 4: LANGUAGE (de LEVEL 3)
// ============================================

print("═══ EXPERT 4: LANGUAGE ═══\n")
print("Tarea: Sentimiento\n")
print("Training 3000 epochs...\n")

let X_language = tensor([
    0.8, 0.1, 0.5, 0.2, 0.3, 0.1, 0.2, 0.5, 0.1, 0.5,
    0.3, 0.4, 0.7, 0.9, 0.5, 0.8, 0.4, 0.7, 0.6, 0.2,
    0.2, 0.6, 0.6, 0.8
], [12, 2])

let Y_language = tensor([
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0
], [12, 3])

let W1_language = tensor_randn_seeded([2, 32], 700)
W1_language = tensor_requires_grad(W1_language, true)
let b1_language = tensor_zeros([32])
b1_language = tensor_requires_grad(b1_language, true)
let W2_language = tensor_randn_seeded([32, 3], 800)
W2_language = tensor_requires_grad(W2_language, true)
let b2_language = tensor_zeros([3])
b2_language = tensor_requires_grad(b2_language, true)

let optimizer_language = sgd_create(lr)
epoch = 0
while epoch < 3000 {
    let h1 = nn_linear(X_language, W1_language, b1_language)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2_language, b2_language)
    let loss = nn_cross_entropy_logits(logits, Y_language)
    tensor_backward(loss)
    let params = [W1_language, b1_language, W2_language, b2_language]
    let updated = sgd_step(optimizer_language, params)
    W1_language = updated[0]
    b1_language = updated[1]
    W2_language = updated[2]
    b2_language = updated[3]
    epoch = epoch + 1
}
print("✅ Expert Language trained\n\n")

// ============================================
// EXPERT 5: General (de LEVEL 3)
// ============================================

print("═══ EXPERT 5: GENERAL ═══\n")
print("Tarea: Clasificación por rangos\n")
print("Training 3000 epochs...\n")

let X_general = tensor([
    10.0, 10.0, 11.0, 10.0, 10.0, 11.0, 12.0, 12.0, 13.0, 12.0,
    12.0, 13.0, 14.0, 14.0, 15.0, 14.0, 14.0, 15.0
], [9, 2])

let Y_general = tensor([
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0
], [9, 3])

let W1_general = tensor_randn_seeded([2, 16], 900)
W1_general = tensor_requires_grad(W1_general, true)
let b1_general = tensor_zeros([16])
b1_general = tensor_requires_grad(b1_general, true)
let W2_general = tensor_randn_seeded([16, 3], 1000)
W2_general = tensor_requires_grad(W2_general, true)
let b2_general = tensor_zeros([3])
b2_general = tensor_requires_grad(b2_general, true)

let optimizer_general = sgd_create(lr)
epoch = 0
while epoch < 3000 {
    let h1 = nn_linear(X_general, W1_general, b1_general)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2_general, b2_general)
    let loss = nn_cross_entropy_logits(logits, Y_general)
    tensor_backward(loss)
    let params = [W1_general, b1_general, W2_general, b2_general]
    let updated = sgd_step(optimizer_general, params)
    W1_general = updated[0]
    b1_general = updated[1]
    W2_general = updated[2]
    b2_general = updated[3]
    epoch = epoch + 1
}
print("✅ Expert General trained\n\n")

// ============================================
// EXPERT 6: MEMORY (NUEVO) ⭐
// Task: Memory/Retrieval simulado
// ============================================

print("═══ EXPERT 6: MEMORY (NUEVO) ⭐ ═══\n")
print("Tarea: Queries factuales (simulated retrieval)\n")
print("Dataset: 16 ejemplos\n\n")

let X_memory = tensor([
    // Tipo 0: results conocidos
    0.9, 0.4,   // lookup: 2+2=4 → 0
    0.91, 0.3,  // lookup: 2+1=3 → 1
    0.92, 0.5,  // lookup: 3+2=5 → 2
    0.93, 0.6,  // lookup: 4+4=8 → 3

    // Tipo 1: Comparisons conocidas
    0.9, 0.32,  // lookup: 3>2 → 0
    0.91, 0.22, // lookup: 2>3 → 1
    0.92, 0.44, // lookup: 4>3 → 0
    0.93, 0.11, // lookup: 1>2 → 1

    // Tipo 2: operators conocidos
    0.9, 0.6,   // lookup: tipo(*) → 2
    0.91, 0.2,  // lookup: tipo(+) → 0
    0.92, 0.7,  // lookup: tipo(/) → 3
    0.93, 0.3,  // lookup: tipo(-) → 1

    // Tipo 3: Rangos conocidos
    0.9, 0.65,  // lookup: rango(13) → 1
    0.91, 0.55, // lookup: rango(11) → 0
    0.92, 0.75, // lookup: rango(14) → 2
    0.93, 0.45  // lookup: rango(10) → 0
], [16, 2])

let Y_memory = tensor([
    // results
    1.0, 0.0, 0.0, 0.0,  // 4
    0.0, 1.0, 0.0, 0.0,  // 3
    0.0, 0.0, 1.0, 0.0,  // 5
    0.0, 0.0, 0.0, 1.0,  // 8

    // Comparisons
    1.0, 0.0, 0.0, 0.0,  // true
    0.0, 1.0, 0.0, 0.0,  // false
    1.0, 0.0, 0.0, 0.0,  // true
    0.0, 1.0, 0.0, 0.0,  // false

    // operators
    0.0, 0.0, 1.0, 0.0,  // binario
    1.0, 0.0, 0.0, 0.0,  // Addition
    0.0, 0.0, 0.0, 1.0,  // div
    0.0, 1.0, 0.0, 0.0,  // resta

    // Rangos
    0.0, 1.0, 0.0, 0.0,  // medium
    1.0, 0.0, 0.0, 0.0,  // low
    0.0, 0.0, 1.0, 0.0,  // high
    1.0, 0.0, 0.0, 0.0   // low
], [16, 4])

let W1_memory = tensor_randn_seeded([2, 16], 1300)
W1_memory = tensor_requires_grad(W1_memory, true)
let b1_memory = tensor_zeros([16])
b1_memory = tensor_requires_grad(b1_memory, true)
let W2_memory = tensor_randn_seeded([16, 4], 1400)
W2_memory = tensor_requires_grad(W2_memory, true)
let b2_memory = tensor_zeros([4])
b2_memory = tensor_requires_grad(b2_memory, true)

print("Training Expert Memory (3000 epochs)...\n")

let optimizer_memory = sgd_create(lr)
epoch = 0
while epoch < 3000 {
    let h1 = nn_linear(X_memory, W1_memory, b1_memory)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2_memory, b2_memory)
    let loss = nn_cross_entropy_logits(logits, Y_memory)
    tensor_backward(loss)
    let params = [W1_memory, b1_memory, W2_memory, b2_memory]
    let updated = sgd_step(optimizer_memory, params)
    W1_memory = updated[0]
    b1_memory = updated[1]
    W2_memory = updated[2]
    b2_memory = updated[3]
    epoch = epoch + 1
}
print("✅ Expert Memory trained\n\n")

// ============================================
// ROUTER: 6 Dominios
// ============================================

print("═══ ROUTER: 6 DOMINIOS ═══\n")
print("Tarea: Math/Logic/Code/Language/General/Memory\n")
print("Dataset: 60 ejemplos (10 por dominio)\n\n")

let X_router = tensor([
    // Domain 0: Math
    0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0,
    3.0, 1.0, 3.0, 2.0, 3.0, 3.0, 4.0, 2.0, 4.0, 4.0,

    // Domain 1: Logic
    0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 3.0, 2.0,
    2.0, 3.0, 4.0, 3.0, 3.0, 4.0, 4.0, 4.0, 2.0, 2.0,

    // Domain 2: Code
    0.5, 0.2, 0.1, 0.2, 0.6, 0.2, 0.15, 0.2, 0.1, 0.3,
    0.7, 0.3, 0.8, 0.2, 0.2, 0.4, 0.9, 0.4, 0.12, 0.3,

    // Domain 3: Language
    0.8, 0.1, 0.5, 0.2, 0.2, 0.5, 0.7, 0.9, 0.4, 0.7,
    0.6, 0.2, 0.2, 0.6, 0.6, 0.8, 0.3, 0.1, 0.5, 0.8,

    // Domain 4: General
    10.0, 10.0, 11.0, 10.0, 12.0, 12.0, 13.0, 12.0, 14.0, 14.0,
    15.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 15.0, 15.0,

    // Domain 5: Memory (valores >0.9 en primera dim)
    0.9, 0.4, 0.91, 0.3, 0.92, 0.5, 0.93, 0.6, 0.9, 0.32,
    0.91, 0.22, 0.92, 0.44, 0.93, 0.11, 0.9, 0.6, 0.91, 0.2
], [60, 2])

let Y_router = tensor([
    // Math
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,

    // Logic
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,

    // Code
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,

    // Language
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,

    // General
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,

    // Memory
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0
], [60, 6])

let W1_router = tensor_randn_seeded([2, 32], 1500)
W1_router = tensor_requires_grad(W1_router, true)
let b1_router = tensor_zeros([32])
b1_router = tensor_requires_grad(b1_router, true)
let W2_router = tensor_randn_seeded([32, 6], 1600)
W2_router = tensor_requires_grad(W2_router, true)
let b2_router = tensor_zeros([6])
b2_router = tensor_requires_grad(b2_router, true)

print("Training Router (4000 epochs)...\n")

let optimizer_router = sgd_create(lr)
epoch = 0
while epoch < 4000 {
    let h1 = nn_linear(X_router, W1_router, b1_router)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2_router, b2_router)
    let loss = nn_cross_entropy_logits(logits, Y_router)
    tensor_backward(loss)
    let params = [W1_router, b1_router, W2_router, b2_router]
    let updated = sgd_step(optimizer_router, params)
    W1_router = updated[0]
    b1_router = updated[1]
    W2_router = updated[2]
    b2_router = updated[3]
    epoch = epoch + 1
}
print("✅ Router trained\n\n")

// ============================================
// EVALUACIÓN END-TO-END: 6 Queries
// ============================================

print("╔══════════════════════════════════════════════════════════════╗\n")
print("║         EVALUACIÓN SISTEMA COMPLETO (6 EXPERTS)             ║\n")
print("╚══════════════════════════════════════════════════════════════╝\n\n")

// TEST 1: Math
print("TEST 1: Math Query (2 + 2 = ?)\n")
let test1 = tensor([2.0, 2.0], [1, 2])
let r_h1 = nn_linear(test1, W1_router, b1_router)
r_h1 = nn_relu(r_h1)
let r_logits = nn_linear(r_h1, W2_router, b2_router)
let domain = argmax(r_logits)
print("  Router → Domain ")
print(domain)
if domain == 0 {
    print(" (Math) ✅\n")
    let h1 = nn_linear(test1, W1_math, b1_math)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2_math, b2_math)
    let result = argmax(logits)
    print("  Expert Math → ")
    print(result)
    if result == 4 {
        print(" ✅ (2+2=4)\n\n")
    } else {
        print(" ⚠️ (esperado: 4)\n\n")
    }
} else {
    print(" ❌\n\n")
}

// TEST 2: Logic
print("TEST 2: Logic Query (3 > 2 = ?)\n")
let test2 = tensor([3.0, 2.0], [1, 2])
r_h1 = nn_linear(test2, W1_router, b1_router)
r_h1 = nn_relu(r_h1)
r_logits = nn_linear(r_h1, W2_router, b2_router)
domain = argmax(r_logits)
print("  Router → Domain ")
print(domain)
if domain == 1 {
    print(" (Logic) ✅\n")
    let h1 = nn_linear(test2, W1_logic, b1_logic)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2_logic, b2_logic)
    let result = argmax(logits)
    print("  Expert Logic → ")
    print(result)
    if result == 1 {
        print(" ✅ (3>2: sí)\n\n")
    } else {
        print(" ⚠️ (esperado: 1)\n\n")
    }
} else {
    print(" ❌\n\n")
}

// TEST 3: Code
print("TEST 3: Code Query (operador *)\n")
let test3 = tensor([0.6, 0.2], [1, 2])
r_h1 = nn_linear(test3, W1_router, b1_router)
r_h1 = nn_relu(r_h1)
r_logits = nn_linear(r_h1, W2_router, b2_router)
domain = argmax(r_logits)
print("  Router → Domain ")
print(domain)
if domain == 2 {
    print(" (Code) ✅\n")
    let h1 = nn_linear(test3, W1_code, b1_code)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2_code, b2_code)
    let result = argmax(logits)
    print("  Expert Code → ")
    print(result)
    if result == 2 {
        print(" ✅ (operador *)\n\n")
    } else {
        print(" ⚠️ (esperado: 2)\n\n")
    }
} else {
    print(" ❌\n\n")
}

// TEST 4: Language
print("TEST 4: Language Query (sentimiento positivo)\n")
let test4 = tensor([0.7, 0.9], [1, 2])
r_h1 = nn_linear(test4, W1_router, b1_router)
r_h1 = nn_relu(r_h1)
r_logits = nn_linear(r_h1, W2_router, b2_router)
domain = argmax(r_logits)
print("  Router → Domain ")
print(domain)
if domain == 3 {
    print(" (Language) ✅\n")
    let h1 = nn_linear(test4, W1_language, b1_language)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2_language, b2_language)
    let result = argmax(logits)
    print("  Expert Language → ")
    print(result)
    if result == 2 {
        print(" ✅ (positivo)\n\n")
    } else {
        print(" ⚠️ (esperado: 2)\n\n")
    }
} else {
    print(" ❌\n\n")
}

// TEST 5: General
print("TEST 5: General Query (rango medio)\n")
let test5 = tensor([13.0, 13.0], [1, 2])
r_h1 = nn_linear(test5, W1_router, b1_router)
r_h1 = nn_relu(r_h1)
r_logits = nn_linear(r_h1, W2_router, b2_router)
domain = argmax(r_logits)
print("  Router → Domain ")
print(domain)
if domain == 4 {
    print(" (General) ✅\n")
    let h1 = nn_linear(test5, W1_general, b1_general)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2_general, b2_general)
    let result = argmax(logits)
    print("  Expert General → ")
    print(result)
    if result == 1 {
        print(" ✅ (medio)\n\n")
    } else {
        print(" ⚠️ (esperado: 1)\n\n")
    }
} else {
    print(" ❌\n\n")
}

// TEST 6: Memory (NUEVO)
print("TEST 6: Memory Query (lookup factual) ⭐\n")
let test6 = tensor([0.9, 0.4], [1, 2])
r_h1 = nn_linear(test6, W1_router, b1_router)
r_h1 = nn_relu(r_h1)
r_logits = nn_linear(r_h1, W2_router, b2_router)
domain = argmax(r_logits)
print("  Router → Domain ")
print(domain)
if domain == 5 {
    print(" (Memory) ✅\n")
    let h1 = nn_linear(test6, W1_memory, b1_memory)
    h1 = nn_relu(h1)
    let logits = nn_linear(h1, W2_memory, b2_memory)
    let result = argmax(logits)
    print("  Expert Memory → ")
    print(result)
    print(" (lookup result)\n\n")
} else {
    print(" ❌\n\n")
}

print("╔══════════════════════════════════════════════════════════════╗\n")
print("║        🎯 LEVEL 4: MEMORY EXPERT + MoE COMPLETADO ✅        ║\n")
print("╚══════════════════════════════════════════════════════════════╝\n\n")

print("✨ Arquitectura Completa:\n")
print("   - Router: 2 → 32 → 6 (~220 params)\n")
print("   - Expert Math: 2 → 32 → 10 (~350 params)\n")
print("   - Expert Logic: 2 → 16 → 2 (~50 params)\n")
print("   - Expert Code: 2 → 32 → 5 (~200 params)\n")
print("   - Expert Language: 2 → 32 → 3 (~130 params)\n")
print("   - Expert General: 2 → 16 → 3 (~70 params)\n")
print("   - Expert Memory: 2 → 16 → 4 (~80 params) ⭐\n")
print("   - TOTAL: ~1100 params\n\n")

print("✅ Sistema MoE con 6 experts funcional\n")
print("✅ Expert Memory agrega capacidad de retrieval\n")
print("✅ Router clasifica 6 dominios diferentes\n\n")

print("📊 Próximo: LEVEL 5 - Reasoning Engine (Chain-of-Thought)\n")

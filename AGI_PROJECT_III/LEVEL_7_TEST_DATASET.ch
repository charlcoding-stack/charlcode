// ═══════════════════════════════════════════════════════════
// AGI PROJECT III - LEVEL 7: TEST DATASET
// 70 examples UNSEEN (10 por dominio)
// ═══════════════════════════════════════════════════════════

print("╔══════════════════════════════════════════════════════════════╗");
print("║       AGI PROJECT III - LEVEL 7: Test Dataset              ║");
print("║       70 ejemplos UNSEEN para evaluación final             ║");
print("╚══════════════════════════════════════════════════════════════╝");
print("");

// ═══════════════════════════════════════════════════════════
// DOMAIN 0: MATH EXPERT (10 test cases)
// ═══════════════════════════════════════════════════════════
// Training usó: [0,0], [1,1], [2,2], etc (valores iguales)
// Test usa: combinaciones DIFERENTES para Additions distintas

print("═══ DOMAIN 0: MATH TEST CASES ═══");
print("");

let X_math_test = tensor([
    0, 1,  // 0+1=1 → class 1
    1, 2,  // 1+2=3 → class 3
    2, 3,  // 2+3=5 → class 5
    3, 4,  // 3+4=7 → class 7
    4, 5,  // 4+5=9 → class 9
    0, 2,  // 0+2=2 → class 2
    1, 3,  // 1+3=4 → class 4
    2, 4,  // 2+4=6 → class 6
    3, 5,  // 3+5=8 → class 8
    0, 3   // 0+3=3 → class 3
], [10, 2]);

let Y_math_test = tensor([
    // Addition=1 → class 1
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    // Addition=3 → class 3
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    // Addition=5 → class 5
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    // Addition=7 → class 7
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    // Addition=9 → class 9
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    // Addition=2 → class 2
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    // Addition=4 → class 4
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    // Addition=6 → class 6
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    // Addition=8 → class 8
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    // Addition=3 → class 3
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
], [10, 10]);

print("✅ Math test dataset: 10 examples (sumas con valores diferentes)");
print("   Examples: 0+1=1, 1+2=3, 2+3=5, 3+4=7, 4+5=9, ...");
print("");

// ═══════════════════════════════════════════════════════════
// DOMAIN 1: LOGIC EXPERT (10 test cases)
// ═══════════════════════════════════════════════════════════
// Training usó: [a,b] donde a>b específicos
// Test usa: otras combinaciones a>b

print("═══ DOMAIN 1: LOGIC TEST CASES ═══");
print("");

let X_logic_test = tensor([
    4, 1,   // 4>1: true → class 0
    5, 2,   // 5>2: true → class 0
    6, 3,   // 6>3: true → class 0
    7, 4,   // 7>4: true → class 0
    8, 5,   // 8>5: true → class 0
    1, 4,   // 1<4: false → class 1
    2, 5,   // 2<5: false → class 1
    3, 6,   // 3<6: false → class 1
    1, 1,   // 1=1: false → class 1
    3, 3    // 3=3: false → class 1
], [10, 2]);

let Y_logic_test = tensor([
    // true (a>b)
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    // false (a<=b)
    0.0, 1.0,
    0.0, 1.0,
    0.0, 1.0,
    0.0, 1.0,
    0.0, 1.0
], [10, 2]);

print("✅ Logic test dataset: 10 examples (comparaciones nuevas)");
print("   Examples: 4>1 (true), 5>2 (true), 1<4 (false), 1=1 (false), ...");
print("");

// ═══════════════════════════════════════════════════════════
// DOMAIN 2: CODE EXPERT (10 test cases)
// ═══════════════════════════════════════════════════════════
// Training usó: encodings 0.1-0.4 con magnitudes específicas
// Test usa: magnitudes diferentes para cada operator

print("═══ DOMAIN 2: CODE TEST CASES ═══");
print("");

let X_code_test = tensor([
    0.1, 0.15,  // + operator, mag 0.15 → class 0
    0.1, 0.25,  // + operator, mag 0.25 → class 0
    0.2, 0.35,  // - operator, mag 0.35 → class 1
    0.2, 0.45,  // - operator, mag 0.45 → class 1
    0.3, 0.55,  // * operator, mag 0.55 → class 2
    0.3, 0.65,  // * operator, mag 0.65 → class 2
    0.4, 0.75,  // / operator, mag 0.75 → class 3
    0.4, 0.85,  // / operator, mag 0.85 → class 3
    0.5, 0.15,  // unknown, mag 0.15 → class 4
    0.5, 0.95   // unknown, mag 0.95 → class 4
], [10, 2]);

let Y_code_test = tensor([
    // class 0: +
    1.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0,
    // class 1: -
    0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0,
    // class 2: *
    0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0,
    // class 3: /
    0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0,
    // class 4: unknown
    0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 1.0
], [10, 5]);

print("✅ Code test dataset: 10 examples (operadores con magnitudes distintas)");
print("   Examples: + (0.15, 0.25), - (0.35, 0.45), * (0.55, 0.65), ...");
print("");

// ═══════════════════════════════════════════════════════════
// DOMAIN 3: LANGUAGE EXPERT (10 test cases)
// ═══════════════════════════════════════════════════════════
// Training usó: encodings 0.1-0.3 con magnitudes específicas
// Test usa: magnitudes diferentes

print("═══ DOMAIN 3: LANGUAGE TEST CASES ═══");
print("");

let X_language_test = tensor([
    0.1, 0.12,  // Positive, mag 0.12 → class 0
    0.1, 0.18,  // Positive, mag 0.18 → class 0
    0.1, 0.22,  // Positive, mag 0.22 → class 0
    0.2, 0.32,  // neutral, mag 0.32 → class 1
    0.2, 0.38,  // neutral, mag 0.38 → class 1
    0.2, 0.42,  // neutral, mag 0.42 → class 1
    0.3, 0.52,  // Negative, mag 0.52 → class 2
    0.3, 0.58,  // Negative, mag 0.58 → class 2
    0.3, 0.62,  // Negative, mag 0.62 → class 2
    0.1, 0.28   // Positive, mag 0.28 → class 0
], [10, 2]);

let Y_language_test = tensor([
    // Positive
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    // neutral
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    // Negative
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    // Positive
    1.0, 0.0, 0.0
], [10, 3]);

print("✅ Language test dataset: 10 examples (sentimientos con mags distintas)");
print("   Examples: Positive (0.12, 0.18, 0.22), Neutral (0.32, 0.38), ...");
print("");

// ═══════════════════════════════════════════════════════════
// DOMAIN 4: General EXPERT (10 test cases)
// ═══════════════════════════════════════════════════════════
// Training usó: valores 10-18
// Test usa: otros valores en los mismos rangos

print("═══ DOMAIN 4: GENERAL TEST CASES ═══");
print("");

let X_general_test = tensor([
    9, 9,    // low (<=12) → class 0
    10, 10,  // low → class 0
    11, 11,  // low → class 0
    13, 13,  // medium (13-15) → class 1
    14, 14,  // medium → class 1
    15, 15,  // medium → class 1
    16, 16,  // high (>=16) → class 2
    17, 17,  // high → class 2
    19, 19,  // high → class 2
    12, 12   // low (boundary) → class 0
], [10, 2]);

let Y_general_test = tensor([
    // low
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    // medium
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    // high
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    // low
    1.0, 0.0, 0.0
], [10, 3]);

print("✅ General test dataset: 10 examples (rangos con valores distintos)");
print("   Examples: 9 (bajo), 10 (bajo), 13-15 (medio), 16-19 (alto), ...");
print("");

// ═══════════════════════════════════════════════════════════
// DOMAIN 5: MEMORY EXPERT (10 test cases)
// ═══════════════════════════════════════════════════════════
// Training usó: fact IDs 0.91-0.99
// Test usa: fact IDs ligeramente diferentes

print("═══ DOMAIN 5: MEMORY TEST CASES ═══");
print("");

let X_memory_test = tensor([
    0.905, 0.1,  // Fact lookup, ID ~0.9 → fact 0
    0.915, 0.2,  // Fact lookup, ID ~0.9 → fact 1
    0.925, 0.3,  // Fact lookup, ID ~0.9 → fact 2
    0.935, 0.4,  // Fact lookup, ID ~0.9 → fact 3
    0.908, 0.5,  // Fact lookup, ID ~0.9 → fact 0
    0.918, 0.6,  // Fact lookup, ID ~0.9 → fact 1
    0.928, 0.7,  // Fact lookup, ID ~0.9 → fact 2
    0.938, 0.8,  // Fact lookup, ID ~0.9 → fact 3
    0.902, 0.15, // Fact lookup, ID ~0.9 → fact 0
    0.912, 0.25  // Fact lookup, ID ~0.9 → fact 1
], [10, 2]);

let Y_memory_test = tensor([
    // Fact 0
    1.0, 0.0, 0.0, 0.0,
    // Fact 1
    0.0, 1.0, 0.0, 0.0,
    // Fact 2
    0.0, 0.0, 1.0, 0.0,
    // Fact 3
    0.0, 0.0, 0.0, 1.0,
    // Fact 0
    1.0, 0.0, 0.0, 0.0,
    // Fact 1
    0.0, 1.0, 0.0, 0.0,
    // Fact 2
    0.0, 0.0, 1.0, 0.0,
    // Fact 3
    0.0, 0.0, 0.0, 1.0,
    // Fact 0
    1.0, 0.0, 0.0, 0.0,
    // Fact 1
    0.0, 1.0, 0.0, 0.0
], [10, 4]);

print("✅ Memory test dataset: 10 examples (fact IDs ligeramente distintos)");
print("   Examples: 0.905, 0.915, 0.925, 0.935, ...");
print("");

// ═══════════════════════════════════════════════════════════
// DOMAIN 6: REASONING EXPERT (10 test cases)
// ═══════════════════════════════════════════════════════════
// Training usó: encodings 0.1-0.5 con valores específicos
// Test usa: valores diferentes para cada tipo de Reasoning

print("═══ DOMAIN 6: REASONING TEST CASES ═══");
print("");

let X_reasoning_test = tensor([
    // Tipo 1: Transitivo (nuevos casos)
    0.1, 0.5,   // true → class 0
    0.1, 0.35,  // false → class 1

    // Tipo 2: Compuesto (nuevos valores)
    0.2, 0.25,  // (1.5+1.25)*2=5.5 → class 2
    0.2, 0.35,  // (1.75+1.75)*2=7 → class 3

    // Tipo 3: Negación (nuevos casos)
    0.3, 0.5,   // NOT(true)=false → class 1
    0.3, 0.25,  // NOT(false)=true → class 0

    // Tipo 4: Doble op (nuevos valores)
    0.4, 0.3,   // 1.5*2+1=4 → class 2
    0.4, 0.6,   // 3*2+1=7 → class 4

    // Tipo 5: Condicional (nuevos valores)
    0.5, 0.6,   // 6>5: high → class 4
    0.5, 0.4    // 4<5: low → class 0
], [10, 2]);

let Y_reasoning_test = tensor([
    // Tipo 1: Transitivo
    1.0, 0.0, 0.0, 0.0, 0.0,  // true
    0.0, 1.0, 0.0, 0.0, 0.0,  // false

    // Tipo 2: Compuesto
    0.0, 0.0, 1.0, 0.0, 0.0,  // result ~6 → class 2
    0.0, 0.0, 0.0, 1.0, 0.0,  // result ~7 → class 3

    // Tipo 3: Negación
    0.0, 1.0, 0.0, 0.0, 0.0,  // false
    1.0, 0.0, 0.0, 0.0, 0.0,  // true

    // Tipo 4: Doble op
    0.0, 0.0, 1.0, 0.0, 0.0,  // 4 → class 2
    0.0, 0.0, 0.0, 0.0, 1.0,  // 7 → class 4

    // Tipo 5: Condicional
    0.0, 0.0, 0.0, 0.0, 1.0,  // high → class 4
    1.0, 0.0, 0.0, 0.0, 0.0   // low → class 0
], [10, 5]);

print("✅ Reasoning test dataset: 10 examples (problemas multi-paso nuevos)");
print("   Examples: Transitivo, Compuesto, Negación, Doble op, Condicional");
print("");

// ═══════════════════════════════════════════════════════════
// RESUMEN
// ═══════════════════════════════════════════════════════════

print("╔══════════════════════════════════════════════════════════════╗");
print("║      ✅ LEVEL 7: Test Dataset COMPLETADO                    ║");
print("╚══════════════════════════════════════════════════════════════╝");
print("");

print("📊 Test Dataset Summary:");
print("  - Domain 0 (Math):      10 ejemplos ✅");
print("  - Domain 1 (Logic):     10 ejemplos ✅");
print("  - Domain 2 (Code):      10 ejemplos ✅");
print("  - Domain 3 (Language):  10 ejemplos ✅");
print("  - Domain 4 (General):   10 ejemplos ✅");
print("  - Domain 5 (Memory):    10 ejemplos ✅");
print("  - Domain 6 (Reasoning): 10 ejemplos ✅");
print("");
print("  TOTAL: 70 ejemplos UNSEEN (diferentes de training)");
print("");

print("✅ Todos los ejemplos son:");
print("  1. Diferentes del training set");
print("  2. Representativos de su dominio");
print("  3. Solvables por el expert correspondiente");
print("");

print("📈 Próximo paso:");
print("  - Implementar baseline Dense model (~1340 params)");
print("  - Comparar MoE vs Dense en este test set");

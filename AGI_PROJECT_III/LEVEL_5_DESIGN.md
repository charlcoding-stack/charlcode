# LEVEL 5: Reasoning Engine + Chain-of-Thought

## Objective

Add multi-step reasoning capability to the MoE system through an Expert Reasoning specialized in logical inferences.

## Philosophy

Instead of implementing complex CoT with multiple forwards, we will:
1. Create an Expert Reasoning that learns inference patterns
2. Encode multi-step problems as single inputs
3. The expert learns to "reason" implicitly in one pass

This is **simulated** reasoning - the expert learns the result of reasoning chains, not executes the steps explicitly.

## Architecture

```
Router: 2 → 32 → 7 (~240 params)
  ├─> Expert Math: 2 → 32 → 10 (~350 params)
  ├─> Expert Logic: 2 → 16 → 2 (~50 params)
  ├─> Expert Code: 2 → 32 → 5 (~200 params)
  ├─> Expert Language: 2 → 32 → 3 (~130 params)
  ├─> Expert General: 2 → 16 → 3 (~70 params)
  ├─> Expert Memory: 2 → 16 → 4 (~80 params)
  └─> Expert Reasoning: 2 → 24 → 5 (~150 params) ⭐ NEW

Total: ~1270 params
```

## Expert Reasoning (NEW)

**Task**: Problems requiring multi-step reasoning

**Problem types**:

### Type 1: Transitive Inference
```
If A > B and B > C, then A > C?
Input: [type, relation1, relation2]
Encoding: [0.1, value1, value2]
```

### Type 2: Compound Arithmetic
```
(a + b) * 2 = ?
Input: [type, a, b]
Encoding: [0.2, a_norm, b_norm]
```

### Type 3: Logical Negation
```
If NOT(A > B), then A <= B?
Input: [type, comparison]
Encoding: [0.3, value]
```

### Type 4: Double Operation
```
(a * 2) + 1 = ?
Input: [type, a]
Encoding: [0.4, a_norm]
```

### Type 5: Conditional
```
If x > 5 then category=high, else category=low
Input: [type, x]
Encoding: [0.5, x_norm]
```

**Input**: [problem_type, value]
- problem_type: 0.1-0.5 (reasoning type)
- value: encoded data 0.0-1.0

**Output**: Reasoning result (5 classes)
- Results encoded according to problem type

**Dataset**: 20 examples

```charl
// Type 1: Transitive (3>2 and 2>1 → 3>1: true)
[0.1, 0.6] → 0  (true)
[0.1, 0.4] → 1  (false)

// Type 2: Compound ((2+1)*2=6)
[0.2, 0.3] → 2  (6)
[0.2, 0.2] → 1  (4)

// Type 3: Negation (NOT(3>2)=false)
[0.3, 0.6] → 1  (false)
[0.3, 0.3] → 0  (true)

// Type 4: Double op (2*2+1=5)
[0.4, 0.4] → 3  (5)
[0.4, 0.2] → 1  (3)

// Type 5: Conditional (if 7>5: high)
[0.5, 0.7] → 4  (high)
[0.5, 0.3] → 0  (low)
```

**Architecture**: 2 → 24 → 5

**Target**: 60%+ accuracy (reasoning is difficult)

## Expanded Router (7 Domains)

**Output**: 7 classes
- 0: Math
- 1: Logic
- 2: Code
- 3: Language
- 4: General
- 5: Memory
- 6: Reasoning ⭐ NEW

**Routing Strategy**:

**Domain 6 (Reasoning)**: Multi-step problems with special pattern
```
[0.1, 0.6]  // transitive inferences
[0.2, 0.3]  // compound arithmetic
[0.3, 0.6]  // logical negations
[0.4, 0.4]  // double operation
[0.5, 0.7]  // conditionals
```

Reasoning uses low values in first dimension (0.1-0.5) to distinguish from Memory (>0.9) and other domains.

**Router Dataset**: 70 examples (10 per domain)

**Target**: 75%+ routing accuracy

## Implementation

### Phase 1: Create Expert Reasoning
1. Dataset with 20 reasoning problems
2. Network 2 → 24 → 5
3. Train with encoded multi-step problems
4. Target: 60%+ accuracy

### Phase 2: Expand Router to 7 Domains
1. Balanced dataset (70 examples)
2. Add domain 6 (Reasoning)
3. Target: 75%+ routing accuracy

### Phase 3: End-to-End Validation
Test with 7 queries (1 per domain):
1. Math: [2.0, 2.0]
2. Logic: [3.0, 2.0]
3. Code: [0.6, 0.2]
4. Language: [0.7, 0.9]
5. General: [13.0, 13.0]
6. Memory: [0.95, 0.5]
7. Reasoning: [0.2, 0.3] ⭐ NEW (compound arithmetic)

## Notes on Chain-of-Thought

**Limitation**: This expert does NOT execute explicit step-by-step reasoning (true CoT).

**What it does**: Learns patterns of problems that require multiple reasoning steps and maps input → correct output.

**Why it's sufficient for proof of concept**:
- Validates that we can add "reasoning" as a capability
- Demonstrates that MoE scales to 7 experts
- Expert implicitly learns to solve multi-step problems

**For true CoT** (future):
- We would need multiple forward passes
- Visible intermediate output
- Recurrence or attention

## Success Metrics

**Expert Reasoning**:
- ✅ 60%+ accuracy on multi-step problems
- ✅ Learns inference patterns

**Router**:
- ✅ 75%+ accuracy on 7 domains
- ✅ Distinguishes reasoning from other domains

**System**:
- ✅ End-to-end functional with 7 experts
- ✅ Complete MoE architecture
- ✅ ~1270 params total

## Hyperparameters

- **Learning rate**: 0.01
- **Epochs**: 3000-4000
- **Optimizer**: SGD
- **Loss**: cross_entropy_logits
- **Seeds**: 1700-1800

## Files

- `LEVEL_5_DESIGN.md` - This document
- `LEVEL_5_COMPLETE.ch` - Complete system with Reasoning

## Next Step

Implement LEVEL_5_COMPLETE.ch with:
- 6 experts from LEVEL 4
- New Expert Reasoning
- Router expanded to 7 domains
- End-to-end validation

---

## 🎯 MILESTONE: Complete MoE System

Upon completing LEVEL 5, we will have:
- ✅ 7 specialized experts working
- ✅ Router classifying 7 domains
- ✅ Math, Logic, Code, Language, General, Memory, Reasoning
- ✅ Scalable MoE architecture validated (~1270 params)

Next: LEVEL 6 (Optimizations) and LEVEL 7 (Final Evaluation)

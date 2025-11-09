# LEVEL 3: 5 Experts MoE System

## Objective

Expand the MoE system from 3 to 5 specialized experts, validating architecture scalability.

## General Architecture

```
Router: 2 → 32 → 5 (~200 params)
  ├─> Expert Math: 2 → 32 → 10 (~350 params)
  ├─> Expert Logic: 2 → 16 → 2 (~50 params)
  ├─> Expert Code: 2 → 32 → 5 (~200 params) ⭐ NEW
  ├─> Expert Language: 2 → 32 → 3 (~130 params) ⭐ NEW
  └─> Expert General: 2 → 16 → 3 (~70 params)

Total: ~1000 params
```

## Expert 1: Math (Expanded from LEVEL 2)

**Task**: Basic arithmetic operations

**Input**: [a, b] where a, b ∈ [0, 4]

**Output**: Result of a + b (10 classes: 0-9)

**Dataset**: 20 examples of additions
- 0+0=0, 1+0=1, ..., 4+4=8
- Duplicates for balancing

**Architecture**: 2 → 32 → 10

**Target**: 70%+ accuracy

## Expert 2: Logic (Reused from LEVEL 2)

**Task**: Logical comparisons (a > b?)

**Input**: [a, b] where a, b ∈ [0, 4]

**Output**: 0=no, 1=yes (2 classes)

**Dataset**: 15 examples
- [0,1]=0, [1,0]=1, [2,1]=1, etc.

**Architecture**: 2 → 16 → 2

**Target**: 80%+ accuracy

## Expert 3: Code ⭐ NEW

**Task**: Identify arithmetic operator

**Input**: [result, operand] encoded
- Given result and one operand, identify operator

**Encoding**:
- Input: [result/10, operand/10] to normalize
- For example: 3+2=5 → input [0.5, 0.2] → output 0 (addition)

**Output**: Operator (5 classes)
- 0: addition (+)
- 1: subtraction (-)
- 2: multiplication (*)
- 3: division (/)
- 4: modulo (%)

**Dataset**: 15 examples
```
[0.5, 0.2] → 0  (5 = 3+2, operator +)
[0.1, 0.2] → 1  (1 = 3-2, operator -)
[0.6, 0.2] → 2  (6 = 3*2, operator *)
[0.15, 0.2] → 3 (1.5 = 3/2, operator /)
[0.1, 0.3] → 4  (1 = 4%3, operator %)
```

**Architecture**: 2 → 32 → 5

**Target**: 60%+ accuracy (complex task)

## Expert 4: Language ⭐ NEW

**Task**: Sentiment/tone classification

**Input**: [intensity, polarity]
- intensity: 0-1 (weak to strong)
- polarity: 0-1 (negative to positive)

**Output**: Sentiment (3 classes)
- 0: Negative
- 1: Neutral
- 2: Positive

**Dataset**: 12 examples
```
# Negative
[0.8, 0.1] → 0  (intense negative)
[0.5, 0.2] → 0  (moderate negative)
[0.3, 0.1] → 0  (slight negative)

# Neutral
[0.2, 0.5] → 1  (weak, neutral)
[0.1, 0.5] → 1  (very weak, neutral)
[0.3, 0.4] → 1  (slight neutral)

# Positive
[0.7, 0.9] → 2  (intense positive)
[0.5, 0.8] → 2  (moderate positive)
[0.4, 0.7] → 2  (slight positive)
```

**Architecture**: 2 → 32 → 3

**Target**: 70%+ accuracy

## Expert 5: General (Reused from LEVEL 2)

**Task**: Classification by ranges

**Input**: [x, y] values in different ranges

**Output**: Category (3 classes)
- 0: Low range (10-11)
- 1: Medium range (12-13)
- 2: High range (14-15)

**Dataset**: 9 examples (reused from LEVEL 2)

**Architecture**: 2 → 16 → 3

**Target**: 70%+ accuracy

## Expanded Router

**Task**: Classify input into 5 domains

**Input**: [feature1, feature2]

**Output**: Domain (5 classes)
- 0: Math
- 1: Logic
- 2: Code
- 3: Language
- 4: General

**Routing Strategy**:

**Domain 0 (Math)**: values 0-4 (additions)
```
[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 4.0]
```

**Domain 1 (Logic)**: values 0-4 with comparison pattern
```
[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]
```

**Domain 2 (Code)**: values 0.0-1.0 (normalized, decimals)
```
[0.5, 0.2], [0.1, 0.2], [0.6, 0.2], [0.15, 0.2], [0.1, 0.3]
```

**Domain 3 (Language)**: values 0.0-1.0 with sentiment pattern
```
[0.8, 0.1], [0.5, 0.2], [0.2, 0.5], [0.7, 0.9], [0.4, 0.7]
```

**Domain 4 (General)**: values 10-15 (high)
```
[10.0, 10.0], [11.0, 10.0], [12.0, 12.0], [13.0, 13.0], [14.0, 14.0]
```

**Router Dataset**: 50 examples (10 per domain)

**Architecture**: 2 → 32 → 5

**Target**: 85%+ accuracy

## Implementation Plan

### Phase 1: Train Individual Experts
1. ✅ Expert Math (expanded)
2. ✅ Expert Logic (reused)
3. ⭐ Expert Code (new)
4. ⭐ Expert Language (new)
5. ✅ Expert General (reused)

### Phase 2: Train Router
- Balanced dataset of 5 domains
- 50 total examples (10 per domain)
- Target: 85%+ routing accuracy

### Phase 3: End-to-End Validation
Test with 1 query per domain:
1. Math: [2.0, 2.0] → Router→0 → Expert Math → 4 (2+2)
2. Logic: [3.0, 2.0] → Router→1 → Expert Logic → 1 (3>2: yes)
3. Code: [0.6, 0.2] → Router→2 → Expert Code → 2 (operator *)
4. Language: [0.7, 0.9] → Router→3 → Expert Language → 2 (positive)
5. General: [13.0, 13.0] → Router→4 → Expert General → 1 (medium)

## Success Metrics

**Router**:
- ✅ Accuracy 85%+ in domain classification
- ✅ 5/5 test queries correctly routed

**Experts**:
- Math: 70%+ accuracy
- Logic: 80%+ accuracy
- Code: 60%+ accuracy (difficult task)
- Language: 70%+ accuracy
- General: 70%+ accuracy

**System**:
- ✅ End-to-end functional
- ✅ Scalable architecture validated
- ✅ ~1000 params total

## Hyperparameters

- **Learning rate**: 0.01 (validated in LEVEL 2)
- **Epochs**: 3000-4000 depending on expert
- **Optimizer**: SGD
- **Loss**: cross_entropy_logits (with row-wise softmax fix)
- **Seeds**: Different per expert for diversity

## Files

- `LEVEL_3_DESIGN.md` - This document
- `LEVEL_3_COMPLETE.ch` - Complete end-to-end system
- Individual tests if necessary

## Risks and Mitigations

**Risk 1**: Router confuses similar domains (Math vs Code)
- Mitigation: Clear separation in values (integers vs decimals)

**Risk 2**: Expert Code very difficult (inverse task)
- Mitigation: Simple and balanced dataset, target 60% acceptable

**Risk 3**: Overfitting due to small datasets
- Mitigation: Validate generalization with test cases outside of training

## Next Step

Implement `LEVEL_3_COMPLETE.ch` with all phases integrated.

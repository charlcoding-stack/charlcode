# LEVEL 4: Memory Expert + Knowledge Graph

## Objective

Add external memory based on Knowledge Graph to improve accuracy through knowledge retrieval.

## Philosophy

Instead of creating a complex memory system, we will:
1. Use Charl's KG backend (already exposed and functional)
2. Create a small KG with basic knowledge
3. Add a "Memory Expert" that performs retrieval
4. Validate improvement in accuracy

## Simplified Architecture

```
Router: 2 → 32 → 6 (~220 params)
  ├─> Expert Math: 2 → 32 → 10 (~350 params)
  ├─> Expert Logic: 2 → 16 → 2 (~50 params)
  ├─> Expert Code: 2 → 32 → 5 (~200 params)
  ├─> Expert Language: 2 → 32 → 3 (~130 params)
  ├─> Expert General: 2 → 16 → 3 (~70 params)
  └─> Expert Memory: 2 → 16 → 4 (~80 params) ⭐ NEW
        └─> Knowledge Graph (no params, lookup table)

Total: ~1100 params + KG lookup
```

## Knowledge Graph Design

**Content**: Basic knowledge about operations and concepts

**Entities** (~20 nodes):
- Numbers: 0, 1, 2, 3, 4
- Operators: +, -, *, /, %
- Concepts: positive, negative, neutral
- Ranges: low, medium, high

**Triples** (~30 relationships):
```
(2, adds_with, 2) → 4
(3, greater_than, 2) → true
(*, type, binary_operator)
(positive, sentiment, favorable)
(13, in_range, medium)
```

## Expert Memory (NEW)

**Task**: Perform KG retrieval to answer factual queries

**Input**: [query_type, value]
- query_type: query type (0-3)
  - 0: sum_result
  - 1: comparison
  - 2: operator_type
  - 3: range_classification

- value: encoded value (0.0-1.0)

**Output**: KG answer (4 classes)
- Answers encoded according to query type

**Dataset**: 16 examples
```
# Type 0: Sum results
[0.0, 0.4] → kg_query("2+2") → 4
[0.0, 0.3] → kg_query("2+1") → 3

# Type 1: Comparisons
[0.1, 0.6] → kg_query("3>2") → true
[0.1, 0.4] → kg_query("2>2") → false

# Type 2: Operator type
[0.2, 0.5] → kg_query("type(*)") → binary

# Type 3: Classification
[0.3, 0.6] → kg_query("range(13)") → medium
```

**Architecture**: 2 → 16 → 4

**Target**: 70%+ accuracy on factual queries

## Expanded Router (6 Domains)

**Output**: 6 classes (5 experts + memory)
- 0: Math
- 1: Logic
- 2: Code
- 3: Language
- 4: General
- 5: Memory (factual queries) ⭐ NEW

**Routing Strategy**:

**Domain 5 (Memory)**: Factual queries with special pattern
```
[0.9, 0.5]  // "memory pattern"
[0.95, 0.6]
[0.92, 0.4]
```

Memory queries have high values in first dimension (>0.9) to distinguish from other domains.

## Simplified Implementation

### Option A: Real KG (Full Backend)

Use Charl's KG functions:
```charl
let kg = kg_create()

// Add entities
kg_add_entity(kg, "number_2", "Number")
kg_add_entity(kg, "operator_sum", "Operator")

// Add triples
kg_add_triple(kg, "number_2", "adds_with", "number_2")

// Query
let result = kg_query(kg, "sum", ["number_2", "number_2"])
```

**Problem**: KG functions may not be fully integrated with the neural training system.

### Option B: Simulated Memory (Neural Lookup)

Create a "pseudo-KG" as a neural expert that learns associations:
```charl
// Expert Memory learns lookup table neurally
// Input: [query_type, value]
// Output: encoded answer

// Works as KG but implemented neurally
```

**Advantage**: Integrates perfectly with the existing MoE system.

## Decision: Option B (Simulated Memory)

For LEVEL 4, we will implement **Option B**:
- Expert Memory is a neural network that learns factual associations
- Acts as simulated "memory"
- Integrates naturally with the router
- Proof of concept: memory improves accuracy

In LEVEL 5 we can integrate real KG if the concept works.

## Implementation Plan

### Phase 1: Create Expert Memory
1. Dataset with simple factual queries
2. Network 2 → 16 → 4
3. Train with basic facts
4. Target: 70%+ accuracy

### Phase 2: Expand Router
1. Router now classifies 6 domains
2. Balanced dataset (60 examples, 10 per domain)
3. Target: 80%+ routing accuracy

### Phase 3: End-to-End Validation
Test with 6 queries (1 per domain):
1. Math: [2.0, 2.0]
2. Logic: [3.0, 2.0]
3. Code: [0.6, 0.2]
4. Language: [0.7, 0.9]
5. General: [13.0, 13.0]
6. Memory: [0.95, 0.5] ⭐ NEW

### Phase 4: Measure Improvement
Compare accuracy before/after adding memory:
- Do queries that go to Memory have better accuracy?
- Does the general system improve?

## Success Metrics

**Expert Memory**:
- ✅ 70%+ accuracy on factual queries
- ✅ Correctly responds to simple lookups

**Router**:
- ✅ 80%+ accuracy on 6 domains
- ✅ Distinguishes memory queries

**System**:
- ✅ End-to-end functional with 6 experts
- ✅ Memory contributes to system
- ⚠️ Improvement in general accuracy (aspirational, difficult to measure with such small datasets)

## Hyperparameters

- **Learning rate**: 0.01
- **Epochs**: 3000-4000
- **Optimizer**: SGD
- **Loss**: cross_entropy_logits
- **Seeds**: 1300-1400

## Files

- `LEVEL_4_DESIGN.md` - This document
- `LEVEL_4_COMPLETE.ch` - Complete system with Memory Expert

## Next Step

Implement LEVEL_4_COMPLETE.ch with:
- 5 experts from LEVEL 3
- New Expert Memory
- Router expanded to 6 domains
- End-to-end validation

## Note on Real KG

To simplify LEVEL 4, we use simulated neural memory. If we want to integrate real KG from Charl, we can do it in a later level or in final optimizations. The goal here is to validate that "external memory" improves the system.

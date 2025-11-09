# LEVEL 6: MoE System Optimizations

## Objective

Optimize the 7-expert MoE system to maximize accuracy and resolve identified issues.

## Current State (LEVEL 5)

### Baseline Metrics
- **Router accuracy**: 85.7% (6/7)
- **Total params**: ~1270
- **Experts working**: 7/7
- **System**: End-to-end functional

### Identified Problems

#### 1. Router: Math/Logic Confusion ⚠️ (CRITICAL)
**Symptom**: Router consistently confuses Math and Logic since LEVEL 3
- TEST 1 (Math [2, 2]): Predicts Logic ❌
- TEST 2 (Logic [3, 2]): Sometimes confuses with Math

**Root cause**: Both domains use small values (0-4)
```
Math dataset:   [0,0], [1,1], [2,2], [3,3], [4,4], ...
Logic dataset:  [1,0], [2,1], [3,2], [4,3], ...
```
Significant overlap in value ranges.

#### 2. Expert General: Incorrect Predictions ⚠️
**Symptom**: Test predicts class 2 when it should be 1
**Cause**: Small dataset (9 examples), possible underfitting

#### 3. Expert Reasoning: Needs Tuning ⚠️
**Symptom**: Test (2+1)*2 predicts class 1 instead of 2
**Cause**: Complex dataset (5 reasoning types), possible underfitting

#### 4. Router 85.7% vs Target 90%+ 📊
**Gap**: 4.3% to reach target
**Opportunity**: Achievable with better features

## Optimization Strategies

### Strategy 1: Math/Logic Feature Separation (CRITICAL)

**Problem**: Math and Logic have overlapping input values

**Solution**: Use **distinctive patterns** in Router dataset

**Before (LEVEL 5)**:
```charl
// Math domain (sum: a+b)
[0, 0] → 0  // 0+0
[1, 1] → 0  // 1+1=2 (class 0 for range 0-2)
[2, 2] → 0  // 2+2=4

// Logic domain (comparison: a>b)
[1, 0] → 1  // 1>0: yes
[2, 1] → 1  // 2>1: yes
[3, 2] → 1  // 3>2: yes
```

**After (LEVEL 6)**: Use **sum of values** as additional feature

```charl
// Math domain: EQUAL values (a == b)
[0, 0] → 0  // sum=0
[1, 1] → 0  // sum=2
[2, 2] → 0  // sum=4
[3, 3] → 0  // sum=6

// Logic domain: DIFFERENT values (a != b, a > b)
[2, 0] → 1  // sum=2, but a != b
[3, 1] → 1  // sum=4, but a != b
[4, 2] → 1  // sum=6, but a != b
```

**Distinctive pattern**:
- Math: `input[0] == input[1]` (always equal)
- Logic: `input[0] != input[1]` (always different, a > b)

This gives clear signal to Router for discrimination.

### Strategy 2: Hyperparameter Tuning per Expert

**Problem**: We use same hyperparameters for all experts

**Solution**: Adjust individually according to complexity

| Expert | Params | Current Epochs | Proposed Epochs | Reason |
|--------|--------|---------------|------------------|-------|
| Math | ~350 | 3000 | 3000 | ✅ Works well |
| Logic | ~50 | 3000 | 2000 | ✅ Small network, converges fast |
| Code | ~200 | 4000 | 4000 | ✅ Works well |
| Language | ~130 | 3000 | 3000 | ✅ Works well |
| General | ~70 | 3000 | **5000** | ⚠️ Needs more training |
| Memory | ~80 | 3000 | 3000 | ✅ Works well |
| Reasoning | ~150 | 4000 | **6000** | ⚠️ Complex task, needs more |
| Router | ~240 | 4000 | **5000** | 📊 To reach 90%+ |

### Strategy 3: Seed Exploration

**Problem**: We use fixed seeds without exploration

**Solution**: Try multiple seeds and select best

**Current seeds**:
- Experts: 1700, 1701, 1702, ... (sequential)
- Router: 1600

**Proposed seeds**: Explore ranges
- Experts: 1700-1750 (try 5 seeds per problematic expert)
- Router: 1600-1620 (try 5 seeds)

**Criterion**: Select seed with best accuracy in training

### Strategy 4: Dataset Expansion (Optional)

**Only for problematic experts**:

**Expert General** (9 → 15 examples):
- Add more examples in boundary ranges
- Better class coverage

**Expert Reasoning** (20 → 30 examples):
- 6 examples per reasoning type (before: 4)
- Better generalization

### Strategy 5: Learning Rate Tuning

**Problem**: lr=0.01 for all

**Solution**: Adjust according to network size

| Expert | Params | Current LR | Proposed LR | Reason |
|--------|--------|-----------|--------------|-------|
| Math | ~350 | 0.01 | 0.01 | ✅ Large, ok |
| Logic | ~50 | 0.01 | **0.02** | Small, can go faster |
| General | ~70 | 0.01 | **0.015** | Small + underfitting |
| Reasoning | ~150 | 0.01 | **0.008** | Complex, more conservative |
| Router | ~240 | 0.01 | 0.01 | ✅ Ok |

## Implementation by Phases

### Phase 1: Critical Fix (Math/Logic Separation) 🎯

**Priority**: HIGH
**Impact**: Resolve persistent confusion

**Actions**:
1. Redesign Router dataset with distinctive pattern
2. Re-train Router with new strategy
3. Validate that Math and Logic are discriminated correctly

**Target**: Router 100% on Math and Logic

### Phase 2: Tuning Problematic Experts ⚙️

**Priority**: MEDIUM
**Impact**: Improve General and Reasoning

**Actions**:
1. Expert General: 5000 epochs, lr=0.015, seed exploration
2. Expert Reasoning: 6000 epochs, lr=0.008, more examples
3. Validate correct predictions on tests

**Target**: Both experts with correct predictions

### Phase 3: Final Router Optimization 📈

**Priority**: MEDIUM
**Impact**: Reach 90%+ accuracy

**Actions**:
1. Router: 5000 epochs
2. Seed exploration (5 seeds)
3. Validate 90%+ on 7 domains

**Target**: Router 90%+ (6.3/7 or better)

## Success Metrics

### Minimum Target (To advance to LEVEL 7)
- ✅ Router > 85% on 7 domains (already achieved: 85.7%)
- ✅ Math/Logic confusion resolved
- ✅ Expert General correct predictions
- ✅ Expert Reasoning correct predictions

### Ideal Target
- 🎯 Router ≥ 90% (6.3/7)
- 🎯 All experts with correct predictions
- 🎯 Robust and reproducible system

## Files to Create

### LEVEL_6_PHASE1.ch
- Router with improved dataset (Math/Logic separation)
- Validation of Math vs Logic discrimination

### LEVEL_6_PHASE2.ch
- Optimized Expert General
- Optimized Expert Reasoning
- Individual tests

### LEVEL_6_COMPLETE.ch
- Complete optimized system
- Improved Router + 7 tuned experts
- End-to-end validation

## Execution Plan

### Session 1: Phase 1 (Critical)
1. Create LEVEL_6_PHASE1.ch
2. New Router dataset with clear separation
3. Re-train Router
4. Validate Math/Logic discrimination

### Session 2: Phase 2 (Tuning)
1. Create LEVEL_6_PHASE2.ch
2. Optimize Expert General (epochs, lr, seeds)
3. Optimize Expert Reasoning (epochs, lr, examples)
4. Validate predictions

### Session 3: Integration
1. Create LEVEL_6_COMPLETE.ch
2. Complete system with all optimizations
3. End-to-end validation
4. Document results

## "Attack the Root" Philosophy

**DON'T do**:
- ❌ Remove problematic domains
- ❌ Simplify datasets
- ❌ Accept 85.7% as limit
- ❌ Workarounds (e.g., manual post-processing)

**DO**:
- ✅ Identify root cause (feature overlap)
- ✅ Design systematic solution (pattern separation)
- ✅ Methodical hyperparameter tuning
- ✅ Rigorous validation

## Time Estimate

- **Phase 1**: 2-3 hours (critical, must work)
- **Phase 2**: 2-3 hours (iterative tuning)
- **Phase 3**: 1-2 hours (integration and validation)

**Total**: 5-8 hours for complete LEVEL 6

---

## 🎯 MILESTONE: Optimized MoE System

Upon completing LEVEL 6, we will have:
- ✅ Router 90%+ accuracy on 7 domains
- ✅ Perfect Math/Logic discrimination
- ✅ All experts with correct predictions
- ✅ Robust system (~1270 optimized params)
- ✅ Ready for LEVEL 7 (Final Evaluation)

Next: LEVEL 7 - Comprehensive Evaluation and Comparison

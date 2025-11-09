# LEVEL 2: Router + Multiple Experts

## Simplified Architecture

### Expert Math (reused from LEVEL 1)
- **Input**: [a, b] (2 numbers)
- **Output**: sum (5 classes: 0-4)
- **Architecture**: 2 → 16 → 5 (~120 params)
- **Dataset**: Additions 0-2

### Expert Logic (new)
- **Input**: [a, b] (2 numbers)
- **Output**: a > b? (2 classes: 0=no, 1=yes)
- **Architecture**: 2 → 8 → 2 (~40 params)
- **Dataset**: Comparisons 0-2

### Expert General (new)
- **Input**: [x, y] (category encoding)
- **Output**: category (3 classes)
- **Architecture**: 2 → 8 → 3 (~50 params)
- **Dataset**: Simple classification

### Router
- **Input**: [feature1, feature2] (same format as experts)
- **Output**: expert_id (3 classes: 0=Math, 1=Logic, 2=General)
- **Architecture**: 2 → 16 → 3 (~80 params)
- **Total system**: ~290 params (much less than 29k, but proof of concept)

## Dataset LEVEL 2

### Domain Math (20 examples)
```
[0, 0] → sum=0 → domain=0
[1, 0] → sum=1 → domain=0
[0, 1] → sum=1 → domain=0
...
```

### Domain Logic (20 examples)
```
[0, 1] → 0>1? no (0) → domain=1
[1, 0] → 1>0? yes (1) → domain=1
[2, 1] → 2>1? yes (1) → domain=1
...
```

### Domain General (20 examples)
```
[10, 10] → category A (0) → domain=2
[20, 20] → category B (1) → domain=2
[30, 30] → category C (2) → domain=2
...
```

## Training Strategy

1. **Phase 1**: Train each expert individually with its dataset
   - Expert Math: 80% accuracy (already validated)
   - Expert Logic: target 80%+
   - Expert General: target 80%+

2. **Phase 2**: Train Router with mixed dataset
   - Input: features from all domains
   - Target: identify correct domain
   - Accuracy target: 85%+

3. **Phase 3**: End-to-end validation
   - Router decides expert
   - Expert processes and gives answer
   - Total system accuracy

## Milestone LEVEL 2

- ✅ 3 experts working independently
- ✅ Router accuracy 85%+
- ✅ Complete functional MoE system

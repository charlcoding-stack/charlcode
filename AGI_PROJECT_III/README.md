# AGI PROJECT III: Efficient Mixture of Experts

## ⚠️ IMPLEMENTATION PHILOSOPHY

### 🔴 "ATTACK THE ROOT" - Mandatory Methodology

This project follows the validated philosophy from AGI_PROJECT_II:

**When something fails in code:**
- ❌ DO NOT simplify tests
- ❌ DO NOT create workarounds
- ❌ DO NOT adapt code to avoid errors
- ✅ **DO go to Charl's backend/frontend and fix it**
- ✅ **DO strengthen Charl's core, don't compromise the project**

**Golden Rule**:
> *"If Charl doesn't have it, add it to Charl. If Charl fails, fix it in Charl. Always."*

**Examples applied in this project**:
1. ✅ `argmax()` didn't exist → Implemented in `src/tensor_builtins.rs`
2. ✅ `as` (casting) didn't exist → Implemented: lexer, parser, AST, interpreter

**Result**: Charl is stronger. Future projects benefit.

See `ROADMAP.md` for detailed instructions.

---

## 🎯 VISION (MetaReal.md)

**NOT**:
- ❌ Create traditional 1B parameter model
- ❌ Compete on parameter count

**YES**:
- ✅ Demonstrate that Charl 500k params === Qwen 2.5 7B (14,000x less)
- ✅ Demonstrate that Charl 2M params === GPT-3.5 (87,500x less)
- ✅ Validate Karpathy thesis: **"1000x FEWER parameters, SAME capability"**

---

## 📊 Current State (AGI_PROJECT_II)

### What we have ✅
- Parameters: ~250
- Capability: 3-class classification (98% accuracy)
- Architecture: Emb → Concat → Linear (basic)
- Backend: Fully exposed (Symbolic AI, KG, Meta-Learning, FOL)

### What's missing ⚠️
- Target parameters: 100k - 500k (not 1B!)
- Capability: General reasoning, generation, few-shot
- Architecture: MoE + External Memory + Reasoning Engine
- Efficiency: 1000x better than traditional models

---

## 🚀 AGI PROJECT III: Efficient MoE

### Objective

**100k params rivaling traditional 1B**
(1/10 of final goal: 500k vs 7B)

### Proposed Architecture

```
┌─────────────────────────────────────────────────────┐
│         INPUT (Concepts, not tokens)                │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              ROUTER (10k params)                    │
│  Decides which expert to activate based on input    │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │Expert 1│  │Expert 2│  │Expert 5│
   │  Math  │  │Language│  │ Code   │
   │ 15k    │  │ 15k    │  │ 15k    │
   └────────┘  └────────┘  └────────┘
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│       EXTERNAL MEMORY (15k params)                  │
│  - Compressed Knowledge Graph                       │
│  - Concept embeddings                               │
│  - Efficient retrieval                              │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              OUTPUT (Response)                      │
└─────────────────────────────────────────────────────┘
```

### Parameter Breakdown

| Component | Params | Description |
|-----------|--------|-------------|
| **5 Experts** | 75k | 15k each (Math, Language, Reasoning, Code, General) |
| **Router** | 10k | Network that decides which expert to activate |
| **Memory** | 15k | Knowledge graph + concept embeddings |
| **TOTAL** | **100k** | **10,000x less than traditional 1B** |

### Target Capabilities

1. **Basic math**: 2+2, factorials, simple algebra
2. **Logical reasoning**: Syllogisms, inferences
3. **Text comprehension**: Answer questions about paragraph
4. **Few-shot learning**: Learn from 2-3 examples

**Target accuracy**: 70-80% on standard benchmarks
(Traditional 1B models: 70-75%)

---

## 🔬 Comparison with Traditional Models

| Metric | Traditional 1B Model | Charl MoE 100k | Factor |
|--------|---------------------|----------------|--------|
| Parameters | 1,000,000,000 | 100,000 | **10,000x less** |
| Memory | ~2 GB (FP16) | ~400 KB | **5,000x less** |
| Speed | ~100 tok/s (GPU) | ~1000 tok/s (CPU) | **10x faster** |
| Energy | ~50W (GPU) | ~2W (CPU) | **25x less** |
| Capability | 70-75% on benchmarks | 70-80% target | **Equivalent** |

---

## 📝 Implementation Roadmap

### Phase 1: Router + Basic Experts (Week 1-2) ⬅️ START HERE

**Implement**:
1. Sparse routing system
2. 5 specialized experts (simple architecture)
3. Mechanism to activate only 1 expert per query
4. Sparse training (backprop only to active expert)

**Milestone**: Router chooses correct expert 85%+ of the time

### Phase 2: External Memory (Week 3)

**Implement**:
1. Compressed knowledge graph (using Charl's KG backend)
2. Concept embeddings (not token embeddings)
3. Efficient retrieval mechanism

**Milestone**: Memory that retrieves relevant knowledge

### Phase 3: Reasoning Engine (Week 4-5)

**Implement**:
1. Native chain-of-thought (using FOL backend)
2. Causal reasoning
3. Meta-reasoning

**Milestone**: Ability to reason, not just predict

### Phase 4: Intelligent Dataset (Week 6)

**Create**:
1. Fundamental concepts (~5k)
2. Relationships between concepts
3. Curriculum learning (basic → complex concepts)

**Milestone**: Structured dataset, not raw tokens

### Phase 5: Evaluation (Week 7-8)

**Benchmarks**:
1. GSM8K (math)
2. HellaSwag (reasoning)
3. MMLU (general knowledge)
4. HumanEval (code)

**Target**: 70-80% average

---

## 💡 Philosophy: Architecture > Scale

### Traditional Models (Brute Force)
- More parameters = better (scaling laws)
- Memorize entire internet
- Massive statistical correlation
- **Analogy**: Memorize 1 trillion "answers"

### Charl MoE (Karpathy Philosophy)
- Fewer parameters = better (efficiency)
- Reason with fundamental concepts
- Structured understanding
- **Analogy**: Understand 100k "principles" and derive answers

---

## 🎯 Next Immediate Step

**LEVEL 1: Router + Math Expert**

Demonstrate concept with:
- Simple router (5k params)
- 1 math expert (15k params)
- Dataset: Basic arithmetic problems
- Target: 90%+ accuracy

Once working, expand to 5 experts.

---

## 📚 Resources

- **Available backend**: Symbolic AI, KG, GNN, Meta-Learning, FOL, Attention
- **Learnings from PROJECT_II**:
  - ✅ Attack root of problem
  - ✅ Strengthen Charl's core
  - ✅ Backend exposure = smarter AGI
  - ✅ Architecture > Scale validated

---

## 🎓 Meta: Validate Karpathy's Thesis

> "You don't need 175B parameters. You need the right architecture."

**PROJECT III validates**: 100k well-designed params > 1B traditional params

**PROJECT IV (future)**: 500k well-designed params > 7B traditional params

**PROJECT V (future)**: 2M well-designed params > GPT-3.5 (175B params)

---

*"Architecture > Scale. Exposed Backend = Smarter AGI."*

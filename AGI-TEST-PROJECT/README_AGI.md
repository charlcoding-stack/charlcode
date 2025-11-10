# Neural Network Primitives: Foundation Experiments

> **Validation of basic neural network capabilities in Charl**
> *Testing tensor operations, layers, and learning primitives*

## Overview

This project validates Charl's capability for neural network development through incremental experiments testing core primitives: tensor operations, linear layers, activation functions, backpropagation, and simple learning tasks.

**Important**: This is NOT an AGI project. These are foundational experiments demonstrating that Charl can implement basic neural network operations correctly.

## Experiments

| Level | Test | Parameters | Accuracy | Status |
|-------|------|------------|----------|--------|
| 1 | Tensor operations | ~4 | 100% | ✅ |
| 2 | Linear layers | ~13 | 100% | ✅ |
| 3 | Activation functions | ~11 | 100% | ✅ |
| 4 | Multi-layer networks | ~60 | 100% | ✅ |
| 5 | Basic learning | ~100 | 75% | ✅ |
| 6 | Gradient computation | ~200 | 100% | ✅ |
| 7 | Simple classification | ~300 | 100% | ✅ |
| 8 | Multi-task learning | ~500 | 100% | ✅ |

## Quick Start

```bash
# Run validation tests
./target/release/charl run test_MINIMAL_REASONER.ch
./target/release/charl run test_COMPOSITIONAL_REASONER.ch
./target/release/charl run test_ABSTRACT_REASONER.ch
./target/release/charl run test_META_REASONER.ch
./target/release/charl run test_TRANSFER_LEARNER.ch
./target/release/charl run test_CAUSAL_REASONER.ch
./target/release/charl run test_PLANNING_REASONER.ch
./target/release/charl run test_SELF_REFLECTION_AGI.ch
```

## What Was Validated

- ✅ **Tensor operations**: Matrix multiplication, broadcasting, reshaping
- ✅ **Neural layers**: Linear, embedding, activations (ReLU, Softmax)
- ✅ **Backpropagation**: Gradient computation and parameter updates
- ✅ **Training loops**: Forward pass, loss calculation, backward pass
- ✅ **Simple learning**: Convergence on toy datasets

## Documentation

See [AGI_JOURNEY.md](./AGI_JOURNEY.md) for:
- Detailed experiment descriptions
- Implementation notes
- Results and metrics
- Technical specifications

## Purpose

This project serves as **proof-of-concept** that Charl can:
1. Implement tensor operations correctly
2. Build neural network layers (linear, embedding, etc.)
3. Compute gradients via backpropagation
4. Train simple models that converge

These primitives form the foundation for more advanced architectures like the Mixture of Experts system in AGI_PROJECT_III.

## Progression

```
Level 1: Tensors    →  Level 2: Layers     →  Level 3: Activations
  (~4 params)           (~13 params)            (~11 params)
      ↓                      ↓                      ↓
Level 4: Networks   →  Level 5: Learning   →  Level 6: Gradients
  (~60 params)          (~100 params)           (~200 params)
      ↓                      ↓                      ↓
Level 7: Classification  →  Level 8: Multi-task
  (~300 params)              (~500 params) ✅
```

## What This Is NOT

- ❌ NOT AGI or anything close to AGI
- ❌ NOT comparable to GPT-4 or any large language model
- ❌ NOT a general-purpose AI system
- ❌ NOT production-ready models

## What This IS

- ✅ Validation of neural network primitives in Charl
- ✅ Foundation for building more complex architectures
- ✅ Proof that Charl's tensor operations work correctly
- ✅ Test suite for gradient computation and training

## Links

- 📖 [Complete Documentation](./AGI_JOURNEY.md)
- 💻 [Source Code](./test_SELF_REFLECTION_AGI.ch)
- 🔬 [AGI_PROJECT_III](../AGI_PROJECT_III/) - Actual research project using these primitives

## Citation

```bibtex
@misc{charl-primitives-2025,
  title={Neural Network Primitives Validation in Charl},
  author={Charl Development Team},
  year={2025},
  note={Foundation experiments for neural network capabilities}
}
```

## License

MIT License - See [LICENSE](./LICENSE) for details.

---

<div align="center">

**Validating Fundamentals**

*Building blocks for neural architecture research*

</div>

// Efficient Architectures Module
//
// This module implements O(n) and O(n log n) architectures that replace
// O(n²) transformer attention, enabling efficient processing of long sequences.
//
// Components:
// - **S4**: Structured State Space Models with HiPPO initialization
// - **Mamba**: Selective State Space Models (data-dependent parameters)
// - **Linear Attention**: Linformer, Performer, FNet, RWKV
// - **Mixture of Experts (MoE)**: Sparse conditional computation
//
// Complexity comparison:
// ```
// Standard Transformer Attention:
//   - Complexity: O(n²d)
//   - 100K tokens: 10B operations → OOM
//
// Efficient Architectures (S4/Mamba/Linear Attention):
//   - Complexity: O(nd) or O(n log n)
//   - 100K tokens: 100M operations → 100x faster
//
// Mixture of Experts:
//   - Complexity: O(nd) but sparse (only top-K of N experts)
//   - 64 experts, top-2: 32x capacity with 2x compute
// ```
//
// Usage:
// ```rust
// use charl::efficient_architectures::{S4Layer, SSMConfig, MambaBlock, MambaConfig};
//
// // S4 Layer
// let config = SSMConfig::new(64, 128).with_dt(0.01);
// let mut s4 = S4Layer::new(config);
// s4.discretize(DiscretizationMethod::ZeroOrderHold);
//
// let outputs = s4.forward_sequence(&inputs);
//
// // Mamba Block
// let config = MambaConfig::new(128).with_state_size(16);
// let mamba = MambaBlock::new(config);
//
// let outputs = mamba.forward_sequence(&inputs);
//
// // Linear Attention (Performer)
// let performer = Performer::new(128, 256);
// let output = performer.forward(&Q, &K, &V);
//
// // Mixture of Experts
// let moe = MoELayer::new(128, 512, 8, 2) // d_model, d_ff, num_experts, top_k
//     .with_strategy(RoutingStrategy::TopK);
//
// let output = moe.forward(&input);
// ```

pub mod s4;
pub mod mamba;
pub mod linear_attention;
pub mod moe;

// Re-export main types
pub use s4::{
    S4Layer, SSMConfig, InitStrategy, DiscretizationMethod,
    ParallelScan,
};

pub use mamba::{
    MambaBlock, MambaConfig, SelectiveParams,
    GatedSSM,
};

pub use linear_attention::{
    Linformer, Performer, FNet, RWKV,
};

pub use moe::{
    MoELayer, Router, Expert,
    RoutingStrategy, LoadBalancingLoss,
};

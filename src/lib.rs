// Charl Language Library
// This exposes the compiler components as a library

pub mod ast;
pub mod autograd;
pub mod codegen;
pub mod gpu;
pub mod gpu_tensor;
pub mod interpreter;
pub mod lexer;
pub mod nn;
pub mod optim;
pub mod parser;
pub mod quantization;
pub mod stdlib;
pub mod tensor_builtins;
pub mod types;

// Phase 7: LLVM Backend (optional feature)
#[cfg(feature = "llvm")]
pub mod llvm_backend;

// Phase 10: Kernel Fusion
pub mod fusion;

// Mini-Phase 11: Attention Mechanisms (for Neuro-Symbolic)
pub mod attention;

// Phase 14: Neuro-Symbolic Integration
pub mod knowledge_graph;
pub mod symbolic;

// Phase 15: Meta-Learning & Curriculum Learning
pub mod meta_learning;

// Phase 16: Efficient Architectures - State Space Models
pub mod efficient_architectures;

// Phase 17: Reasoning Systems
pub mod reasoning;

// Phase 18: Multimodal Neuro-Symbolic
pub mod multimodal;

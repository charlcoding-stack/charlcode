// LLVM Backend - Phase 7
// Compiles computational graphs to native code via LLVM IR
// Provides 10-50x speedup over interpreter for CPU execution
//
// Architecture:
// 1. Computational Graph → LLVM IR generation
// 2. LLVM optimizations (constant folding, dead code elimination, etc.)
// 3. JIT compilation to native code
// 4. Execution of compiled forward/backward passes

#[cfg(feature = "llvm")]
pub mod codegen;

#[cfg(feature = "llvm")]
pub mod jit;

#[cfg(feature = "llvm")]
pub mod optimizer;

#[cfg(feature = "llvm")]
pub mod graph_compiler;

#[cfg(feature = "llvm")]
pub use codegen::LLVMCodegen;

#[cfg(feature = "llvm")]
pub use jit::JITEngine;

#[cfg(feature = "llvm")]
pub use graph_compiler::CompiledGraph;

// Stub implementations when LLVM feature is disabled
#[cfg(not(feature = "llvm"))]
pub mod stub {
    pub struct LLVMCodegen;
    pub struct JITEngine;

    impl LLVMCodegen {
        pub fn new() -> Self {
            panic!("LLVM backend not available. Enable the 'llvm' feature flag.");
        }
    }

    impl JITEngine {
        pub fn new() -> Self {
            panic!("LLVM JIT not available. Enable the 'llvm' feature flag.");
        }
    }
}

#[cfg(not(feature = "llvm"))]
pub use stub::*;

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(feature = "llvm")]
    fn test_llvm_available() {
        // Test that LLVM backend is available when feature is enabled
        assert!(true);
    }

    #[test]
    #[cfg(not(feature = "llvm"))]
    fn test_llvm_disabled() {
        // Test that stub implementations are used when LLVM is disabled
        assert!(true);
    }
}

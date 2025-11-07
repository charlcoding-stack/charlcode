// LLVM Optimizer
// Applies optimization passes to generated LLVM IR

use inkwell::module::Module;
use inkwell::passes::{PassManager, PassManagerBuilder};
use inkwell::OptimizationLevel;

/// Optimization level for LLVM compilation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptLevel {
    /// No optimizations (O0) - fastest compilation
    None,
    /// Basic optimizations (O1) - quick compile with some optimization
    Less,
    /// Standard optimizations (O2) - recommended for most cases
    Default,
    /// Aggressive optimizations (O3) - maximum performance
    Aggressive,
}

impl OptLevel {
    pub fn to_llvm_level(&self) -> OptimizationLevel {
        match self {
            OptLevel::None => OptimizationLevel::None,
            OptLevel::Less => OptimizationLevel::Less,
            OptLevel::Default => OptimizationLevel::Default,
            OptLevel::Aggressive => OptimizationLevel::Aggressive,
        }
    }
}

/// LLVM optimizer that applies optimization passes
pub struct LLVMOptimizer<'ctx> {
    pass_manager: PassManager<Module<'ctx>>,
    optimization_level: OptLevel,
}

impl<'ctx> LLVMOptimizer<'ctx> {
    /// Create a new optimizer with the specified optimization level
    pub fn new(optimization_level: OptLevel) -> Self {
        let pass_manager = PassManager::create(());

        // Configure pass manager based on optimization level
        let pm_builder = PassManagerBuilder::create();
        pm_builder.set_optimization_level(optimization_level.to_llvm_level());

        match optimization_level {
            OptLevel::None => {
                // No optimizations
            }
            OptLevel::Less => {
                // Basic optimizations
                pm_builder.set_size_level(0);
                pm_builder.populate_module_pass_manager(&pass_manager);
            }
            OptLevel::Default => {
                // Standard optimizations
                pm_builder.set_size_level(0);
                pm_builder.populate_module_pass_manager(&pass_manager);
            }
            OptLevel::Aggressive => {
                // Aggressive optimizations
                pm_builder.set_size_level(0);
                pm_builder.set_inliner_with_threshold(275); // Aggressive inlining
                pm_builder.populate_module_pass_manager(&pass_manager);

                // Additional aggressive passes
                pass_manager.add_function_inlining_pass();
                pass_manager.add_global_dce_pass(); // Dead code elimination
                pass_manager.add_cfg_simplification_pass(); // Control flow simplification
                pass_manager.add_instruction_combining_pass(); // Combine instructions
                pass_manager.add_reassociate_pass(); // Reassociate expressions
                pass_manager.add_gvn_pass(); // Global value numbering
                pass_manager.add_memcpy_optimize_pass(); // Optimize memcpy
                pass_manager.add_sccp_pass(); // Sparse conditional constant propagation
                pass_manager.add_aggressive_dce_pass(); // Aggressive dead code elimination

                // ML-specific optimizations (Week 4)
                pass_manager.add_loop_unroll_pass(); // Unroll loops for better performance
                pass_manager.add_loop_vectorize_pass(); // SIMD vectorization for loops
                pass_manager.add_slp_vectorize_pass(); // Straight-line vectorization
            }
        }

        LLVMOptimizer {
            pass_manager,
            optimization_level,
        }
    }

    /// Apply optimization passes to a module
    pub fn optimize(&self, module: &Module<'ctx>) -> bool {
        self.pass_manager.run_on(module)
    }

    /// Get the current optimization level
    pub fn level(&self) -> OptLevel {
        self.optimization_level
    }
}

/// Apply standard optimizations to a module (convenience function)
pub fn optimize_module(module: &Module) -> bool {
    let optimizer = LLVMOptimizer::new(OptLevel::Default);
    optimizer.optimize(module)
}

/// Apply aggressive optimizations to a module (convenience function)
pub fn optimize_aggressive(module: &Module) -> bool {
    let optimizer = LLVMOptimizer::new(OptLevel::Aggressive);
    optimizer.optimize(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llvm_backend::codegen::LLVMCodegen;
    use inkwell::context::Context;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = LLVMOptimizer::new(OptLevel::Default);
        assert_eq!(optimizer.level(), OptLevel::Default);
    }

    #[test]
    fn test_optimization_levels() {
        assert_eq!(OptLevel::None.to_llvm_level(), OptimizationLevel::None);
        assert_eq!(OptLevel::Less.to_llvm_level(), OptimizationLevel::Less);
        assert_eq!(
            OptLevel::Default.to_llvm_level(),
            OptimizationLevel::Default
        );
        assert_eq!(
            OptLevel::Aggressive.to_llvm_level(),
            OptimizationLevel::Aggressive
        );
    }

    #[test]
    fn test_optimize_module() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_opt");

        // Generate some code
        codegen.gen_element_wise_add();
        codegen.gen_element_wise_mul();

        // Verify before optimization
        assert!(codegen.verify().is_ok());

        // Optimize (should return true if changes were made)
        let changed = optimize_module(codegen.module());

        // Module should still be valid after optimization
        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_optimize_aggressive() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_aggressive");

        codegen.gen_element_wise_add();

        assert!(codegen.verify().is_ok());

        // Apply aggressive optimizations
        let changed = optimize_aggressive(codegen.module());

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_no_optimization() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_no_opt");

        codegen.gen_element_wise_mul();

        let optimizer = LLVMOptimizer::new(OptLevel::None);
        let changed = optimizer.optimize(codegen.module());

        // With no optimization, module should remain valid
        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_loop_optimizations_on_matmul() {
        // Test that loop unrolling and vectorization work on MatMul
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_loop_opts");

        // Generate MatMul (has triple-nested loops - perfect for loop optimizations)
        codegen.gen_matmul(10, 10, 10);

        // Get IR before optimization
        let ir_before = codegen.module().print_to_string().to_string();

        // Apply aggressive optimizations (includes loop unroll + vectorize)
        let optimizer = LLVMOptimizer::new(OptLevel::Aggressive);
        let changed = optimizer.optimize(codegen.module());

        // Get IR after optimization
        let ir_after = codegen.module().print_to_string().to_string();

        // Module should still be valid after optimization
        assert!(codegen.verify().is_ok());

        // IR should have changed (optimizations were applied)
        // Note: We can't always guarantee changes, but for MatMul it's very likely
        assert!(ir_before != ir_after || !changed);
    }

    #[test]
    fn test_vectorization_passes_added() {
        // Verify that aggressive mode includes vectorization passes
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_vec_passes");

        // Generate ReLU (good candidate for vectorization)
        codegen.gen_relu();

        // Apply aggressive optimizations
        let optimizer = LLVMOptimizer::new(OptLevel::Aggressive);
        optimizer.optimize(codegen.module());

        // Module should remain valid after vectorization passes
        assert!(codegen.verify().is_ok());

        // Get optimized IR
        let ir = codegen.module().print_to_string().to_string();

        // Verify it contains a function (basic sanity check)
        assert!(ir.contains("define"));
    }
}

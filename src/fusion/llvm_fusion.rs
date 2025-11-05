// LLVM Fusion Codegen - Generates optimized LLVM IR for fused operations
// Takes fusion opportunities and creates single kernels that eliminate intermediate memory operations

#[cfg(feature = "llvm")]
use crate::fusion::patterns::{FusionOpportunity, FusionPattern, OpType};
#[cfg(feature = "llvm")]
use crate::llvm_backend::codegen::LLVMCodegen;
#[cfg(feature = "llvm")]
use inkwell::context::Context;
#[cfg(feature = "llvm")]
use inkwell::values::FunctionValue;
#[cfg(feature = "llvm")]
use inkwell::FloatPredicate;

/// LLVM-based fusion code generator
/// Generates optimized LLVM IR for fused operations
#[cfg(feature = "llvm")]
pub struct LLVMFusionCodegen<'ctx> {
    codegen: LLVMCodegen<'ctx>,
}

#[cfg(feature = "llvm")]
impl<'ctx> LLVMFusionCodegen<'ctx> {
    /// Create a new LLVM fusion codegen
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let codegen = LLVMCodegen::new(context, module_name);

        LLVMFusionCodegen { codegen }
    }

    /// Generate LLVM IR for a fusion opportunity
    ///
    /// This creates a single fused kernel that performs all operations
    /// in the pattern without intermediate memory reads/writes
    ///
    /// Example: For AddMul pattern (a + b) * c:
    ///   Instead of:
    ///     temp[i] = a[i] + b[i]  // Write to memory
    ///     out[i] = temp[i] * c[i] // Read from memory
    ///   Generate:
    ///     out[i] = (a[i] + b[i]) * c[i]  // Single operation
    pub fn gen_fused_kernel(
        &self,
        opportunity: &FusionOpportunity,
    ) -> Result<FunctionValue<'ctx>, String> {
        match &opportunity.pattern {
            FusionPattern::AddMul => self.gen_add_mul_kernel(),
            FusionPattern::MulAdd => self.gen_mul_add_kernel(),
            FusionPattern::AddAdd => self.gen_add_add_kernel(),
            FusionPattern::MulMul => self.gen_mul_mul_kernel(),
            FusionPattern::DivMul => self.gen_div_mul_kernel(),
            FusionPattern::Chain(ops) => self.gen_chain_kernel(ops),
        }
    }

    /// Generate fused kernel for AddMul pattern: (a + b) * c
    ///
    /// Function signature: void fused_add_mul(float* a, float* b, float* c, float* out, i64 size)
    fn gen_add_mul_kernel(&self) -> Result<FunctionValue<'ctx>, String> {
        let function = self
            .codegen
            .gen_fused_add_mul()
            .ok_or("Failed to generate add_mul kernel")?;
        Ok(function)
    }

    /// Generate fused kernel for MulAdd pattern: (a * b) + c (FMA)
    ///
    /// This is the famous Fused Multiply-Add operation
    /// Function signature: void fused_mul_add(float* a, float* b, float* c, float* out, i64 size)
    fn gen_mul_add_kernel(&self) -> Result<FunctionValue<'ctx>, String> {
        let function = self
            .codegen
            .gen_fused_mul_add()
            .ok_or("Failed to generate mul_add kernel")?;
        Ok(function)
    }

    /// Generate fused kernel for AddAdd pattern: (a + b) + c
    fn gen_add_add_kernel(&self) -> Result<FunctionValue<'ctx>, String> {
        let function = self
            .codegen
            .gen_fused_add_add()
            .ok_or("Failed to generate add_add kernel")?;
        Ok(function)
    }

    /// Generate fused kernel for MulMul pattern: (a * b) * c
    fn gen_mul_mul_kernel(&self) -> Result<FunctionValue<'ctx>, String> {
        let function = self
            .codegen
            .gen_fused_mul_mul()
            .ok_or("Failed to generate mul_mul kernel")?;
        Ok(function)
    }

    /// Generate fused kernel for DivMul pattern: (a / b) * c
    fn gen_div_mul_kernel(&self) -> Result<FunctionValue<'ctx>, String> {
        let function = self
            .codegen
            .gen_fused_div_mul()
            .ok_or("Failed to generate div_mul kernel")?;
        Ok(function)
    }

    /// Generate fused kernel for arbitrary operation chains
    ///
    /// This is a more general version that can handle any sequence
    /// of element-wise operations
    ///
    /// Strategy:
    /// 1. Create function with N input arrays + 1 output array
    /// 2. Loop over elements
    /// 3. Build expression tree in registers (no memory)
    /// 4. Write final result
    fn gen_chain_kernel(&self, ops: &[OpType]) -> Result<FunctionValue<'ctx>, String> {
        if ops.is_empty() {
            return Err("Cannot generate kernel for empty operation chain".to_string());
        }

        // For now, delegate to specific 2-op kernels if possible
        if ops.len() == 2 {
            match (&ops[0], &ops[1]) {
                (OpType::Add, OpType::Mul) => return self.gen_add_mul_kernel(),
                (OpType::Mul, OpType::Add) => return self.gen_mul_add_kernel(),
                (OpType::Add, OpType::Add) => return self.gen_add_add_kernel(),
                (OpType::Mul, OpType::Mul) => return self.gen_mul_mul_kernel(),
                (OpType::Div, OpType::Mul) => return self.gen_div_mul_kernel(),
                _ => {}
            }
        }

        // For longer chains, we'd need to generate dynamic code
        // This is a TODO for future implementation
        Err(format!(
            "General chain fusion not yet implemented for {} operations",
            ops.len()
        ))
    }

    /// Verify the generated module
    pub fn verify(&self) -> Result<(), String> {
        self.codegen.verify()
    }

    /// Get reference to the underlying codegen (for accessing the module)
    pub fn codegen(&self) -> &LLVMCodegen<'ctx> {
        &self.codegen
    }
}

// Stub implementation when LLVM is disabled
#[cfg(not(feature = "llvm"))]
pub struct LLVMFusionCodegen;

#[cfg(not(feature = "llvm"))]
impl LLVMFusionCodegen {
    pub fn new() -> Self {
        panic!("LLVM backend not available. Enable 'llvm' feature.");
    }
}

#[cfg(all(test, feature = "llvm"))]
mod tests {
    use super::*;
    use crate::fusion::patterns::FusionOpportunity;

    #[test]
    fn test_fusion_codegen_creation() {
        let context = Context::create();
        let codegen = LLVMFusionCodegen::new(&context, "fusion_test");

        // Should create without errors
        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_gen_add_mul_kernel() {
        let context = Context::create();
        let codegen = LLVMFusionCodegen::new(&context, "fusion_test");

        let pattern = FusionPattern::AddMul;
        let opportunity = FusionOpportunity::new(pattern, vec![1, 2, 3], 1000);

        let result = codegen.gen_fused_kernel(&opportunity);
        assert!(result.is_ok());
        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_gen_mul_add_kernel() {
        let context = Context::create();
        let codegen = LLVMFusionCodegen::new(&context, "fusion_test");

        let pattern = FusionPattern::MulAdd;
        let opportunity = FusionOpportunity::new(pattern, vec![1, 2, 3], 1000);

        let result = codegen.gen_fused_kernel(&opportunity);
        assert!(result.is_ok());
        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_gen_add_add_kernel() {
        let context = Context::create();
        let codegen = LLVMFusionCodegen::new(&context, "fusion_test");

        let pattern = FusionPattern::AddAdd;
        let opportunity = FusionOpportunity::new(pattern, vec![1, 2, 3], 1000);

        let result = codegen.gen_fused_kernel(&opportunity);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gen_mul_mul_kernel() {
        let context = Context::create();
        let codegen = LLVMFusionCodegen::new(&context, "fusion_test");

        let pattern = FusionPattern::MulMul;
        let opportunity = FusionOpportunity::new(pattern, vec![1, 2, 3], 1000);

        let result = codegen.gen_fused_kernel(&opportunity);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gen_div_mul_kernel() {
        let context = Context::create();
        let codegen = LLVMFusionCodegen::new(&context, "fusion_test");

        let pattern = FusionPattern::DivMul;
        let opportunity = FusionOpportunity::new(pattern, vec![1, 2, 3], 1000);

        let result = codegen.gen_fused_kernel(&opportunity);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gen_chain_2_ops() {
        let context = Context::create();
        let codegen = LLVMFusionCodegen::new(&context, "fusion_test");

        // 2-op chains should work (delegate to specific kernels)
        let pattern = FusionPattern::Chain(vec![OpType::Add, OpType::Mul]);
        let opportunity = FusionOpportunity::new(pattern, vec![1, 2, 3], 1000);

        let result = codegen.gen_fused_kernel(&opportunity);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gen_chain_long_not_implemented() {
        let context = Context::create();
        let codegen = LLVMFusionCodegen::new(&context, "fusion_test");

        // Longer chains not yet implemented
        let pattern = FusionPattern::Chain(vec![OpType::Add, OpType::Mul, OpType::Sub]);
        let opportunity = FusionOpportunity::new(pattern, vec![1, 2, 3, 4], 1000);

        let result = codegen.gen_fused_kernel(&opportunity);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not yet implemented"));
    }

    #[test]
    fn test_verify_valid_module() {
        let context = Context::create();
        let codegen = LLVMFusionCodegen::new(&context, "fusion_test");

        // Generate a kernel
        let pattern = FusionPattern::AddMul;
        let opportunity = FusionOpportunity::new(pattern, vec![1, 2, 3], 1000);
        codegen.gen_fused_kernel(&opportunity).unwrap();

        // Verify should pass
        assert!(codegen.verify().is_ok());
    }
}

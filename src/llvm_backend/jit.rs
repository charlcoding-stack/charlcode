// JIT Compilation Engine
// Compiles LLVM IR to native code and executes it

use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::OptimizationLevel;
use std::marker::PhantomData;

/// JIT execution engine for compiled functions
pub struct JITEngine<'ctx> {
    execution_engine: ExecutionEngine<'ctx>,
    _phantom: PhantomData<&'ctx ()>,
}

impl<'ctx> JITEngine<'ctx> {
    /// Create a new JIT engine from an LLVM module
    ///
    /// Tries to create a JIT execution engine (faster), falls back to interpreter if JIT is not available
    pub fn new(module: &Module<'ctx>) -> Result<Self, String> {
        // Choose optimization level based on build type
        let opt_level = if cfg!(debug_assertions) {
            OptimizationLevel::Default
        } else {
            // Use less aggressive optimization in release to avoid potential issues
            OptimizationLevel::Less
        };

        // Try JIT first (fastest, but may not be available in release builds)
        let execution_engine = match module.create_jit_execution_engine(opt_level) {
            Ok(engine) => {
                #[cfg(debug_assertions)]
                eprintln!("LLVM: Using JIT execution engine (opt_level: {:?})", opt_level);
                engine
            }
            Err(_jit_err) => {
                // Fallback to interpreter (slower but more portable)
                #[cfg(debug_assertions)]
                eprintln!("LLVM: JIT not available, using interpreter execution engine");

                module
                    .create_interpreter_execution_engine()
                    .map_err(|e| format!("Failed to create interpreter execution engine: {:?}", e))?
            }
        };

        Ok(JITEngine {
            execution_engine,
            _phantom: PhantomData,
        })
    }

    /// Get a JIT-compiled function by name
    ///
    /// # Safety
    /// The caller must ensure that:
    /// 1. The function signature matches the actual LLVM function
    /// 2. The function is called with valid pointers
    pub unsafe fn get_function<F>(&self, name: &str) -> Result<JitFunction<'ctx, F>, String>
    where
        F: inkwell::execution_engine::UnsafeFunctionPointer,
    {
        self.execution_engine
            .get_function(name)
            .map_err(|e| format!("Failed to get function '{}': {:?}", name, e))
    }

    /// Execute element-wise tensor addition
    ///
    /// # Safety
    /// Pointers must be valid and point to arrays of at least `size` elements
    pub unsafe fn execute_tensor_add(
        &self,
        a: *const f32,
        b: *const f32,
        output: *mut f32,
        size: usize,
    ) -> Result<(), String> {
        type TensorAddFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i64);

        let add_fn: JitFunction<TensorAddFn> = self.get_function("tensor_add")?;

        add_fn.call(a, b, output, size as i64);

        Ok(())
    }

    /// Execute element-wise tensor multiplication
    ///
    /// # Safety
    /// Pointers must be valid and point to arrays of at least `size` elements
    pub unsafe fn execute_tensor_mul(
        &self,
        a: *const f32,
        b: *const f32,
        output: *mut f32,
        size: usize,
    ) -> Result<(), String> {
        type TensorMulFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i64);

        let mul_fn: JitFunction<TensorMulFn> = self.get_function("tensor_mul")?;

        mul_fn.call(a, b, output, size as i64);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llvm_backend::codegen::LLVMCodegen;
    use inkwell::context::Context;

    #[test]
    fn test_jit_engine_creation() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_jit");

        // Generate a simple function
        codegen.gen_element_wise_add();

        // Create JIT engine
        let jit = JITEngine::new(codegen.module());
        assert!(jit.is_ok());
    }

    #[test]
    fn test_jit_tensor_add_execution() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_add_exec");

        // Generate add function
        codegen.gen_element_wise_add();

        // Verify and create JIT
        codegen.verify().unwrap();
        let jit = JITEngine::new(codegen.module()).unwrap();

        // Test data
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut output = vec![0.0f32; 5];

        // Execute JIT-compiled function
        unsafe {
            jit.execute_tensor_add(a.as_ptr(), b.as_ptr(), output.as_mut_ptr(), 5)
                .unwrap();
        }

        // Verify results
        assert_eq!(output, vec![11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn test_jit_tensor_mul_execution() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_mul_exec");

        // Generate mul function
        codegen.gen_element_wise_mul();

        codegen.verify().unwrap();
        let jit = JITEngine::new(codegen.module()).unwrap();

        let a = vec![2.0f32, 3.0, 4.0, 5.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0];
        let mut output = vec![0.0f32; 4];

        unsafe {
            jit.execute_tensor_mul(a.as_ptr(), b.as_ptr(), output.as_mut_ptr(), 4)
                .unwrap();
        }

        assert_eq!(output, vec![20.0, 60.0, 120.0, 200.0]);
    }

    #[test]
    fn test_jit_large_arrays() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_large");

        codegen.gen_element_wise_add();
        codegen.verify().unwrap();
        let jit = JITEngine::new(codegen.module()).unwrap();

        // Test with larger arrays (10,000 elements)
        let size = 10_000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
        let mut output = vec![0.0f32; size];

        unsafe {
            jit.execute_tensor_add(a.as_ptr(), b.as_ptr(), output.as_mut_ptr(), size)
                .unwrap();
        }

        // Verify a few values
        assert_eq!(output[0], 0.0);
        assert_eq!(output[1], 3.0); // 1 + 2
        assert_eq!(output[100], 300.0); // 100 + 200
        assert_eq!(output[9999], 29997.0); // 9999 + 19998
    }
}

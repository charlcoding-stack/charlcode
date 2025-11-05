// LLVM IR Code Generation
// Converts computational graph operations to LLVM IR

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::FloatType;
use inkwell::values::FunctionValue;
use inkwell::IntPredicate;

/// LLVM code generator for computational graphs
pub struct LLVMCodegen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
}

impl<'ctx> LLVMCodegen<'ctx> {
    /// Create a new LLVM code generator
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();

        LLVMCodegen {
            context,
            module,
            builder,
        }
    }

    /// Get the f32 type (we use f32 for tensors, consistent with our Tensor implementation)
    fn f32_type(&self) -> FloatType<'ctx> {
        self.context.f32_type()
    }

    /// Get the f64 type (for higher precision when needed)
    #[allow(dead_code)]
    fn f64_type(&self) -> FloatType<'ctx> {
        self.context.f64_type()
    }

    /// Generate LLVM IR for element-wise addition
    /// fn tensor_add(a: *float, b: *float, output: *float, size: i64)
    pub fn gen_element_wise_add(&self) -> FunctionValue<'ctx> {
        let f32_type = self.f32_type();
        let f32_ptr_type = f32_type.ptr_type(inkwell::AddressSpace::default());
        let i64_type = self.context.i64_type();

        // Function signature: void tensor_add(float* a, float* b, float* output, i64 size)
        let fn_type = self.context.void_type().fn_type(
            &[
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                i64_type.into(),
            ],
            false,
        );

        let function = self.module.add_function("tensor_add", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        // Get function parameters
        let a_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let b_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let output_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
        let size = function.get_nth_param(3).unwrap().into_int_value();

        // Create loop
        let loop_bb = self.context.append_basic_block(function, "loop");
        let end_bb = self.context.append_basic_block(function, "end");

        // Initialize loop counter
        let counter_ptr = self.builder.build_alloca(i64_type, "counter").unwrap();
        self.builder
            .build_store(counter_ptr, i64_type.const_zero())
            .unwrap();

        self.builder.build_unconditional_branch(loop_bb).unwrap();

        // Loop body
        self.builder.position_at_end(loop_bb);
        let counter = self
            .builder
            .build_load(i64_type, counter_ptr, "i")
            .unwrap()
            .into_int_value();

        // Check if counter < size
        let cond = self
            .builder
            .build_int_compare(IntPredicate::ULT, counter, size, "cond")
            .unwrap();

        let loop_body_bb = self.context.append_basic_block(function, "loop_body");
        self.builder
            .build_conditional_branch(cond, loop_body_bb, end_bb)
            .unwrap();

        // Loop body: output[i] = a[i] + b[i]
        self.builder.position_at_end(loop_body_bb);

        let a_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, a_ptr, &[counter], "a_ptr")
                .unwrap()
        };
        let b_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, b_ptr, &[counter], "b_ptr")
                .unwrap()
        };
        let output_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, output_ptr, &[counter], "out_ptr")
                .unwrap()
        };

        let a_val = self
            .builder
            .build_load(f32_type, a_elem_ptr, "a")
            .unwrap()
            .into_float_value();
        let b_val = self
            .builder
            .build_load(f32_type, b_elem_ptr, "b")
            .unwrap()
            .into_float_value();

        let sum = self.builder.build_float_add(a_val, b_val, "sum").unwrap();

        self.builder.build_store(output_elem_ptr, sum).unwrap();

        // Increment counter
        let next_counter = self
            .builder
            .build_int_add(counter, i64_type.const_int(1, false), "next")
            .unwrap();
        self.builder.build_store(counter_ptr, next_counter).unwrap();

        self.builder.build_unconditional_branch(loop_bb).unwrap();

        // End block
        self.builder.position_at_end(end_bb);
        self.builder.build_return(None).unwrap();

        function
    }

    /// Generate LLVM IR for element-wise multiplication
    pub fn gen_element_wise_mul(&self) -> FunctionValue<'ctx> {
        let f32_type = self.f32_type();
        let f32_ptr_type = f32_type.ptr_type(inkwell::AddressSpace::default());
        let i64_type = self.context.i64_type();

        let fn_type = self.context.void_type().fn_type(
            &[
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                i64_type.into(),
            ],
            false,
        );

        let function = self.module.add_function("tensor_mul", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        let a_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let b_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let output_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
        let size = function.get_nth_param(3).unwrap().into_int_value();

        let loop_bb = self.context.append_basic_block(function, "loop");
        let end_bb = self.context.append_basic_block(function, "end");

        let counter_ptr = self.builder.build_alloca(i64_type, "counter").unwrap();
        self.builder
            .build_store(counter_ptr, i64_type.const_zero())
            .unwrap();
        self.builder.build_unconditional_branch(loop_bb).unwrap();

        self.builder.position_at_end(loop_bb);
        let counter = self
            .builder
            .build_load(i64_type, counter_ptr, "i")
            .unwrap()
            .into_int_value();

        let cond = self
            .builder
            .build_int_compare(IntPredicate::ULT, counter, size, "cond")
            .unwrap();

        let loop_body_bb = self.context.append_basic_block(function, "loop_body");
        self.builder
            .build_conditional_branch(cond, loop_body_bb, end_bb)
            .unwrap();

        self.builder.position_at_end(loop_body_bb);

        let a_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, a_ptr, &[counter], "a_ptr")
                .unwrap()
        };
        let b_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, b_ptr, &[counter], "b_ptr")
                .unwrap()
        };
        let output_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, output_ptr, &[counter], "out_ptr")
                .unwrap()
        };

        let a_val = self
            .builder
            .build_load(f32_type, a_elem_ptr, "a")
            .unwrap()
            .into_float_value();
        let b_val = self
            .builder
            .build_load(f32_type, b_elem_ptr, "b")
            .unwrap()
            .into_float_value();

        let product = self.builder.build_float_mul(a_val, b_val, "prod").unwrap();

        self.builder.build_store(output_elem_ptr, product).unwrap();

        let next_counter = self
            .builder
            .build_int_add(counter, i64_type.const_int(1, false), "next")
            .unwrap();
        self.builder.build_store(counter_ptr, next_counter).unwrap();

        self.builder.build_unconditional_branch(loop_bb).unwrap();

        self.builder.position_at_end(end_bb);
        self.builder.build_return(None).unwrap();

        function
    }

    /// Get the generated LLVM module
    pub fn module(&self) -> &Module<'ctx> {
        &self.module
    }

    /// Print the generated LLVM IR (for debugging)
    pub fn print_ir(&self) {
        self.module.print_to_stderr();
    }

    /// Verify the generated module
    pub fn verify(&self) -> Result<(), String> {
        self.module
            .verify()
            .map_err(|e| format!("LLVM module verification failed: {:?}", e))
    }

    // ===== Fused Kernel Generation (Phase 10: Kernel Fusion) =====

    /// Generate fused kernel for AddMul: (a + b) * c
    /// Eliminates intermediate memory by computing in registers
    pub fn gen_fused_add_mul(&self) -> Option<FunctionValue<'ctx>> {
        let f32_type = self.context.f32_type();
        let f32_ptr_type = f32_type.ptr_type(inkwell::AddressSpace::default());
        let i64_type = self.context.i64_type();
        let void_type = self.context.void_type();

        let fn_type = void_type.fn_type(
            &[
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                i64_type.into(),
            ],
            false,
        );

        let function = self.module.add_function("fused_add_mul", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        let a_ptr = function.get_nth_param(0)?.into_pointer_value();
        let b_ptr = function.get_nth_param(1)?.into_pointer_value();
        let c_ptr = function.get_nth_param(2)?.into_pointer_value();
        let output_ptr = function.get_nth_param(3)?.into_pointer_value();
        let size = function.get_nth_param(4)?.into_int_value();

        // Loop setup
        let counter_ptr = self.builder.build_alloca(i64_type, "counter").ok()?;
        self.builder
            .build_store(counter_ptr, i64_type.const_zero())
            .ok()?;

        let loop_bb = self.context.append_basic_block(function, "loop");
        let loop_body_bb = self.context.append_basic_block(function, "loop_body");
        let end_bb = self.context.append_basic_block(function, "end");

        self.builder.build_unconditional_branch(loop_bb).ok()?;
        self.builder.position_at_end(loop_bb);

        let counter = self
            .builder
            .build_load(i64_type, counter_ptr, "i")
            .ok()?
            .into_int_value();

        let cond = self
            .builder
            .build_int_compare(IntPredicate::ULT, counter, size, "cond")
            .ok()?;

        self.builder
            .build_conditional_branch(cond, loop_body_bb, end_bb)
            .ok()?;

        self.builder.position_at_end(loop_body_bb);

        // Load values: a[i], b[i], c[i]
        let a_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, a_ptr, &[counter], "a_ptr")
                .ok()?
        };
        let b_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, b_ptr, &[counter], "b_ptr")
                .ok()?
        };
        let c_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, c_ptr, &[counter], "c_ptr")
                .ok()?
        };
        let output_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, output_ptr, &[counter], "out_ptr")
                .ok()?
        };

        let a_val = self
            .builder
            .build_load(f32_type, a_elem_ptr, "a")
            .ok()?
            .into_float_value();
        let b_val = self
            .builder
            .build_load(f32_type, b_elem_ptr, "b")
            .ok()?
            .into_float_value();
        let c_val = self
            .builder
            .build_load(f32_type, c_elem_ptr, "c")
            .ok()?
            .into_float_value();

        // Fused computation: (a + b) * c - all in registers
        let add_result = self.builder.build_float_add(a_val, b_val, "add").ok()?;
        let mul_result = self
            .builder
            .build_float_mul(add_result, c_val, "mul")
            .ok()?;

        self.builder.build_store(output_elem_ptr, mul_result).ok()?;

        // Loop increment
        let next_counter = self
            .builder
            .build_int_add(counter, i64_type.const_int(1, false), "next")
            .ok()?;
        self.builder.build_store(counter_ptr, next_counter).ok()?;
        self.builder.build_unconditional_branch(loop_bb).ok()?;

        self.builder.position_at_end(end_bb);
        self.builder.build_return(None).ok()?;

        Some(function)
    }

    /// Generate fused kernel for MulAdd (FMA): (a * b) + c
    pub fn gen_fused_mul_add(&self) -> Option<FunctionValue<'ctx>> {
        let f32_type = self.context.f32_type();
        let f32_ptr_type = f32_type.ptr_type(inkwell::AddressSpace::default());
        let i64_type = self.context.i64_type();
        let void_type = self.context.void_type();

        let fn_type = void_type.fn_type(
            &[
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                i64_type.into(),
            ],
            false,
        );

        let function = self.module.add_function("fused_mul_add", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        let a_ptr = function.get_nth_param(0)?.into_pointer_value();
        let b_ptr = function.get_nth_param(1)?.into_pointer_value();
        let c_ptr = function.get_nth_param(2)?.into_pointer_value();
        let output_ptr = function.get_nth_param(3)?.into_pointer_value();
        let size = function.get_nth_param(4)?.into_int_value();

        let counter_ptr = self.builder.build_alloca(i64_type, "counter").ok()?;
        self.builder
            .build_store(counter_ptr, i64_type.const_zero())
            .ok()?;

        let loop_bb = self.context.append_basic_block(function, "loop");
        let loop_body_bb = self.context.append_basic_block(function, "loop_body");
        let end_bb = self.context.append_basic_block(function, "end");

        self.builder.build_unconditional_branch(loop_bb).ok()?;
        self.builder.position_at_end(loop_bb);

        let counter = self
            .builder
            .build_load(i64_type, counter_ptr, "i")
            .ok()?
            .into_int_value();

        let cond = self
            .builder
            .build_int_compare(IntPredicate::ULT, counter, size, "cond")
            .ok()?;

        self.builder
            .build_conditional_branch(cond, loop_body_bb, end_bb)
            .ok()?;

        self.builder.position_at_end(loop_body_bb);

        let a_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, a_ptr, &[counter], "a_ptr")
                .ok()?
        };
        let b_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, b_ptr, &[counter], "b_ptr")
                .ok()?
        };
        let c_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, c_ptr, &[counter], "c_ptr")
                .ok()?
        };
        let output_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, output_ptr, &[counter], "out_ptr")
                .ok()?
        };

        let a_val = self
            .builder
            .build_load(f32_type, a_elem_ptr, "a")
            .ok()?
            .into_float_value();
        let b_val = self
            .builder
            .build_load(f32_type, b_elem_ptr, "b")
            .ok()?
            .into_float_value();
        let c_val = self
            .builder
            .build_load(f32_type, c_elem_ptr, "c")
            .ok()?
            .into_float_value();

        // Fused: (a * b) + c
        let mul_result = self.builder.build_float_mul(a_val, b_val, "mul").ok()?;
        let add_result = self
            .builder
            .build_float_add(mul_result, c_val, "add")
            .ok()?;

        self.builder.build_store(output_elem_ptr, add_result).ok()?;

        let next_counter = self
            .builder
            .build_int_add(counter, i64_type.const_int(1, false), "next")
            .ok()?;
        self.builder.build_store(counter_ptr, next_counter).ok()?;
        self.builder.build_unconditional_branch(loop_bb).ok()?;

        self.builder.position_at_end(end_bb);
        self.builder.build_return(None).ok()?;

        Some(function)
    }

    /// Generate fused kernel for AddAdd: (a + b) + c
    pub fn gen_fused_add_add(&self) -> Option<FunctionValue<'ctx>> {
        let f32_type = self.context.f32_type();
        let f32_ptr_type = f32_type.ptr_type(inkwell::AddressSpace::default());
        let i64_type = self.context.i64_type();
        let void_type = self.context.void_type();

        let fn_type = void_type.fn_type(
            &[
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                i64_type.into(),
            ],
            false,
        );

        let function = self.module.add_function("fused_add_add", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        let a_ptr = function.get_nth_param(0)?.into_pointer_value();
        let b_ptr = function.get_nth_param(1)?.into_pointer_value();
        let c_ptr = function.get_nth_param(2)?.into_pointer_value();
        let output_ptr = function.get_nth_param(3)?.into_pointer_value();
        let size = function.get_nth_param(4)?.into_int_value();

        let counter_ptr = self.builder.build_alloca(i64_type, "counter").ok()?;
        self.builder
            .build_store(counter_ptr, i64_type.const_zero())
            .ok()?;

        let loop_bb = self.context.append_basic_block(function, "loop");
        let loop_body_bb = self.context.append_basic_block(function, "loop_body");
        let end_bb = self.context.append_basic_block(function, "end");

        self.builder.build_unconditional_branch(loop_bb).ok()?;
        self.builder.position_at_end(loop_bb);

        let counter = self
            .builder
            .build_load(i64_type, counter_ptr, "i")
            .ok()?
            .into_int_value();

        let cond = self
            .builder
            .build_int_compare(IntPredicate::ULT, counter, size, "cond")
            .ok()?;

        self.builder
            .build_conditional_branch(cond, loop_body_bb, end_bb)
            .ok()?;

        self.builder.position_at_end(loop_body_bb);

        let a_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, a_ptr, &[counter], "a_ptr")
                .ok()?
        };
        let b_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, b_ptr, &[counter], "b_ptr")
                .ok()?
        };
        let c_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, c_ptr, &[counter], "c_ptr")
                .ok()?
        };
        let output_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, output_ptr, &[counter], "out_ptr")
                .ok()?
        };

        let a_val = self
            .builder
            .build_load(f32_type, a_elem_ptr, "a")
            .ok()?
            .into_float_value();
        let b_val = self
            .builder
            .build_load(f32_type, b_elem_ptr, "b")
            .ok()?
            .into_float_value();
        let c_val = self
            .builder
            .build_load(f32_type, c_elem_ptr, "c")
            .ok()?
            .into_float_value();

        // Fused: (a + b) + c
        let add1_result = self.builder.build_float_add(a_val, b_val, "add1").ok()?;
        let add2_result = self
            .builder
            .build_float_add(add1_result, c_val, "add2")
            .ok()?;

        self.builder
            .build_store(output_elem_ptr, add2_result)
            .ok()?;

        let next_counter = self
            .builder
            .build_int_add(counter, i64_type.const_int(1, false), "next")
            .ok()?;
        self.builder.build_store(counter_ptr, next_counter).ok()?;
        self.builder.build_unconditional_branch(loop_bb).ok()?;

        self.builder.position_at_end(end_bb);
        self.builder.build_return(None).ok()?;

        Some(function)
    }

    /// Generate fused kernel for MulMul: (a * b) * c
    pub fn gen_fused_mul_mul(&self) -> Option<FunctionValue<'ctx>> {
        let f32_type = self.context.f32_type();
        let f32_ptr_type = f32_type.ptr_type(inkwell::AddressSpace::default());
        let i64_type = self.context.i64_type();
        let void_type = self.context.void_type();

        let fn_type = void_type.fn_type(
            &[
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                i64_type.into(),
            ],
            false,
        );

        let function = self.module.add_function("fused_mul_mul", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        let a_ptr = function.get_nth_param(0)?.into_pointer_value();
        let b_ptr = function.get_nth_param(1)?.into_pointer_value();
        let c_ptr = function.get_nth_param(2)?.into_pointer_value();
        let output_ptr = function.get_nth_param(3)?.into_pointer_value();
        let size = function.get_nth_param(4)?.into_int_value();

        let counter_ptr = self.builder.build_alloca(i64_type, "counter").ok()?;
        self.builder
            .build_store(counter_ptr, i64_type.const_zero())
            .ok()?;

        let loop_bb = self.context.append_basic_block(function, "loop");
        let loop_body_bb = self.context.append_basic_block(function, "loop_body");
        let end_bb = self.context.append_basic_block(function, "end");

        self.builder.build_unconditional_branch(loop_bb).ok()?;
        self.builder.position_at_end(loop_bb);

        let counter = self
            .builder
            .build_load(i64_type, counter_ptr, "i")
            .ok()?
            .into_int_value();

        let cond = self
            .builder
            .build_int_compare(IntPredicate::ULT, counter, size, "cond")
            .ok()?;

        self.builder
            .build_conditional_branch(cond, loop_body_bb, end_bb)
            .ok()?;

        self.builder.position_at_end(loop_body_bb);

        let a_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, a_ptr, &[counter], "a_ptr")
                .ok()?
        };
        let b_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, b_ptr, &[counter], "b_ptr")
                .ok()?
        };
        let c_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, c_ptr, &[counter], "c_ptr")
                .ok()?
        };
        let output_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, output_ptr, &[counter], "out_ptr")
                .ok()?
        };

        let a_val = self
            .builder
            .build_load(f32_type, a_elem_ptr, "a")
            .ok()?
            .into_float_value();
        let b_val = self
            .builder
            .build_load(f32_type, b_elem_ptr, "b")
            .ok()?
            .into_float_value();
        let c_val = self
            .builder
            .build_load(f32_type, c_elem_ptr, "c")
            .ok()?
            .into_float_value();

        // Fused: (a * b) * c
        let mul1_result = self.builder.build_float_mul(a_val, b_val, "mul1").ok()?;
        let mul2_result = self
            .builder
            .build_float_mul(mul1_result, c_val, "mul2")
            .ok()?;

        self.builder
            .build_store(output_elem_ptr, mul2_result)
            .ok()?;

        let next_counter = self
            .builder
            .build_int_add(counter, i64_type.const_int(1, false), "next")
            .ok()?;
        self.builder.build_store(counter_ptr, next_counter).ok()?;
        self.builder.build_unconditional_branch(loop_bb).ok()?;

        self.builder.position_at_end(end_bb);
        self.builder.build_return(None).ok()?;

        Some(function)
    }

    /// Generate fused kernel for DivMul: (a / b) * c
    pub fn gen_fused_div_mul(&self) -> Option<FunctionValue<'ctx>> {
        let f32_type = self.context.f32_type();
        let f32_ptr_type = f32_type.ptr_type(inkwell::AddressSpace::default());
        let i64_type = self.context.i64_type();
        let void_type = self.context.void_type();

        let fn_type = void_type.fn_type(
            &[
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                f32_ptr_type.into(),
                i64_type.into(),
            ],
            false,
        );

        let function = self.module.add_function("fused_div_mul", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        let a_ptr = function.get_nth_param(0)?.into_pointer_value();
        let b_ptr = function.get_nth_param(1)?.into_pointer_value();
        let c_ptr = function.get_nth_param(2)?.into_pointer_value();
        let output_ptr = function.get_nth_param(3)?.into_pointer_value();
        let size = function.get_nth_param(4)?.into_int_value();

        let counter_ptr = self.builder.build_alloca(i64_type, "counter").ok()?;
        self.builder
            .build_store(counter_ptr, i64_type.const_zero())
            .ok()?;

        let loop_bb = self.context.append_basic_block(function, "loop");
        let loop_body_bb = self.context.append_basic_block(function, "loop_body");
        let end_bb = self.context.append_basic_block(function, "end");

        self.builder.build_unconditional_branch(loop_bb).ok()?;
        self.builder.position_at_end(loop_bb);

        let counter = self
            .builder
            .build_load(i64_type, counter_ptr, "i")
            .ok()?
            .into_int_value();

        let cond = self
            .builder
            .build_int_compare(IntPredicate::ULT, counter, size, "cond")
            .ok()?;

        self.builder
            .build_conditional_branch(cond, loop_body_bb, end_bb)
            .ok()?;

        self.builder.position_at_end(loop_body_bb);

        let a_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, a_ptr, &[counter], "a_ptr")
                .ok()?
        };
        let b_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, b_ptr, &[counter], "b_ptr")
                .ok()?
        };
        let c_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, c_ptr, &[counter], "c_ptr")
                .ok()?
        };
        let output_elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, output_ptr, &[counter], "out_ptr")
                .ok()?
        };

        let a_val = self
            .builder
            .build_load(f32_type, a_elem_ptr, "a")
            .ok()?
            .into_float_value();
        let b_val = self
            .builder
            .build_load(f32_type, b_elem_ptr, "b")
            .ok()?
            .into_float_value();
        let c_val = self
            .builder
            .build_load(f32_type, c_elem_ptr, "c")
            .ok()?
            .into_float_value();

        // Fused: (a / b) * c
        let div_result = self.builder.build_float_div(a_val, b_val, "div").ok()?;
        let mul_result = self
            .builder
            .build_float_mul(div_result, c_val, "mul")
            .ok()?;

        self.builder.build_store(output_elem_ptr, mul_result).ok()?;

        let next_counter = self
            .builder
            .build_int_add(counter, i64_type.const_int(1, false), "next")
            .ok()?;
        self.builder.build_store(counter_ptr, next_counter).ok()?;
        self.builder.build_unconditional_branch(loop_bb).ok()?;

        self.builder.position_at_end(end_bb);
        self.builder.build_return(None).ok()?;

        Some(function)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::context::Context;

    #[test]
    fn test_codegen_creation() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_module");

        assert_eq!(codegen.module.get_name().to_str().unwrap(), "test_module");
    }

    #[test]
    fn test_gen_element_wise_add() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_add");

        let add_fn = codegen.gen_element_wise_add();

        assert_eq!(add_fn.get_name().to_str().unwrap(), "tensor_add");
        assert_eq!(add_fn.count_params(), 4);

        // Verify module
        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_gen_element_wise_mul() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_mul");

        let mul_fn = codegen.gen_element_wise_mul();

        assert_eq!(mul_fn.get_name().to_str().unwrap(), "tensor_mul");
        assert_eq!(mul_fn.count_params(), 4);

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_print_ir() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_ir");

        codegen.gen_element_wise_add();

        // FIXED: print_to_stderr() causes segfault in some Linux environments
        // Use print_to_string() instead for safer IR inspection
        let ir_string = codegen.module.print_to_string().to_string();

        // Verify IR contains expected function and basic structure
        assert!(ir_string.contains("tensor_add"));
        assert!(ir_string.contains("define void"));
        assert!(ir_string.contains("entry:"));
        assert!(ir_string.contains("loop"));
    }
}

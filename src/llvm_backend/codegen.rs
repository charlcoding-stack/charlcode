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

    // ===== Matrix Operations (Week 2: Critical ML Ops) =====

    /// Generate LLVM IR for matrix multiplication (naive triple loop)
    ///
    /// Computes: C[m×n] = A[m×k] @ B[k×n]
    ///
    /// Algorithm:
    /// ```
    /// for i in 0..m {
    ///     for j in 0..n {
    ///         sum = 0.0
    ///         for k_idx in 0..k {
    ///             sum += A[i*k + k_idx] * B[k_idx*n + j]
    ///         }
    ///         C[i*n + j] = sum
    ///     }
    /// }
    /// ```
    ///
    /// # Parameters
    /// - `m`: Number of rows in A and C
    /// - `n`: Number of columns in B and C
    /// - `k`: Number of columns in A / rows in B
    ///
    /// # Returns
    /// Function signature: `fn matmul(a: *f32, b: *f32, c: *f32, m: i32, n: i32, k: i32)`
    pub fn gen_matmul(&self, m: u32, n: u32, k: u32) -> FunctionValue<'ctx> {
        let f32_type = self.f32_type();
        let f32_ptr_type = f32_type.ptr_type(inkwell::AddressSpace::default());
        let i32_type = self.context.i32_type();
        let void_type = self.context.void_type();

        // Function signature: matmul(A, B, C, m, n, k)
        let fn_type = void_type.fn_type(
            &[
                f32_ptr_type.into(), // A
                f32_ptr_type.into(), // B
                f32_ptr_type.into(), // C (output)
                i32_type.into(),     // m
                i32_type.into(),     // n
                i32_type.into(),     // k
            ],
            false,
        );

        let function_name = format!("matmul_{}x{}x{}", m, n, k);
        let function = self.module.add_function(&function_name, fn_type, None);

        // Basic blocks
        let entry_bb = self.context.append_basic_block(function, "entry");
        let i_loop_bb = self.context.append_basic_block(function, "i_loop");
        let i_body_bb = self.context.append_basic_block(function, "i_body");
        let j_loop_bb = self.context.append_basic_block(function, "j_loop");
        let j_body_bb = self.context.append_basic_block(function, "j_body");
        let k_loop_bb = self.context.append_basic_block(function, "k_loop");
        let k_body_bb = self.context.append_basic_block(function, "k_body");
        let k_end_bb = self.context.append_basic_block(function, "k_end");
        let j_end_bb = self.context.append_basic_block(function, "j_end");
        let i_end_bb = self.context.append_basic_block(function, "i_end");
        let end_bb = self.context.append_basic_block(function, "end");

        // Get function parameters
        let a_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let b_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let c_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
        let m_val = function.get_nth_param(3).unwrap().into_int_value();
        let n_val = function.get_nth_param(4).unwrap().into_int_value();
        let k_val = function.get_nth_param(5).unwrap().into_int_value();

        // Entry block: allocate loop counters
        self.builder.position_at_end(entry_bb);
        let i_ptr = self.builder.build_alloca(i32_type, "i").unwrap();
        let j_ptr = self.builder.build_alloca(i32_type, "j").unwrap();
        let k_ptr = self.builder.build_alloca(i32_type, "k_idx").unwrap();
        let sum_ptr = self.builder.build_alloca(f32_type, "sum").unwrap();

        self.builder.build_store(i_ptr, i32_type.const_zero()).unwrap();
        self.builder.build_unconditional_branch(i_loop_bb).unwrap();

        // i loop: for i in 0..m
        self.builder.position_at_end(i_loop_bb);
        let i = self.builder.build_load(i32_type, i_ptr, "i").unwrap().into_int_value();
        let i_cond = self.builder.build_int_compare(IntPredicate::SLT, i, m_val, "i_cond").unwrap();
        self.builder.build_conditional_branch(i_cond, i_body_bb, end_bb).unwrap();

        // i body: initialize j loop
        self.builder.position_at_end(i_body_bb);
        self.builder.build_store(j_ptr, i32_type.const_zero()).unwrap();
        self.builder.build_unconditional_branch(j_loop_bb).unwrap();

        // j loop: for j in 0..n
        self.builder.position_at_end(j_loop_bb);
        let j = self.builder.build_load(i32_type, j_ptr, "j").unwrap().into_int_value();
        let j_cond = self.builder.build_int_compare(IntPredicate::SLT, j, n_val, "j_cond").unwrap();
        self.builder.build_conditional_branch(j_cond, j_body_bb, i_end_bb).unwrap();

        // j body: initialize sum and k loop
        self.builder.position_at_end(j_body_bb);
        self.builder.build_store(sum_ptr, f32_type.const_zero()).unwrap();
        self.builder.build_store(k_ptr, i32_type.const_zero()).unwrap();
        self.builder.build_unconditional_branch(k_loop_bb).unwrap();

        // k loop: for k_idx in 0..k
        self.builder.position_at_end(k_loop_bb);
        let k_idx = self.builder.build_load(i32_type, k_ptr, "k_idx").unwrap().into_int_value();
        let k_cond = self.builder.build_int_compare(IntPredicate::SLT, k_idx, k_val, "k_cond").unwrap();
        self.builder.build_conditional_branch(k_cond, k_body_bb, k_end_bb).unwrap();

        // k body: sum += A[i*k + k_idx] * B[k_idx*n + j]
        self.builder.position_at_end(k_body_bb);

        // Calculate A[i*k + k_idx]
        let i_times_k = self.builder.build_int_mul(i, k_val, "i_times_k").unwrap();
        let a_idx = self.builder.build_int_add(i_times_k, k_idx, "a_idx").unwrap();
        let a_ptr_elem = unsafe {
            self.builder.build_gep(f32_type, a_ptr, &[a_idx], "a_ptr_elem").unwrap()
        };
        let a_val = self.builder.build_load(f32_type, a_ptr_elem, "a_val").unwrap().into_float_value();

        // Calculate B[k_idx*n + j]
        let k_times_n = self.builder.build_int_mul(k_idx, n_val, "k_times_n").unwrap();
        let b_idx = self.builder.build_int_add(k_times_n, j, "b_idx").unwrap();
        let b_ptr_elem = unsafe {
            self.builder.build_gep(f32_type, b_ptr, &[b_idx], "b_ptr_elem").unwrap()
        };
        let b_val = self.builder.build_load(f32_type, b_ptr_elem, "b_val").unwrap().into_float_value();

        // sum += a_val * b_val
        let product = self.builder.build_float_mul(a_val, b_val, "product").unwrap();
        let current_sum = self.builder.build_load(f32_type, sum_ptr, "current_sum").unwrap().into_float_value();
        let new_sum = self.builder.build_float_add(current_sum, product, "new_sum").unwrap();
        self.builder.build_store(sum_ptr, new_sum).unwrap();

        // k_idx++
        let k_next = self.builder.build_int_add(k_idx, i32_type.const_int(1, false), "k_next").unwrap();
        self.builder.build_store(k_ptr, k_next).unwrap();
        self.builder.build_unconditional_branch(k_loop_bb).unwrap();

        // k end: store result C[i*n + j] = sum
        self.builder.position_at_end(k_end_bb);
        let i_times_n = self.builder.build_int_mul(i, n_val, "i_times_n").unwrap();
        let c_idx = self.builder.build_int_add(i_times_n, j, "c_idx").unwrap();
        let c_ptr_elem = unsafe {
            self.builder.build_gep(f32_type, c_ptr, &[c_idx], "c_ptr_elem").unwrap()
        };
        let final_sum = self.builder.build_load(f32_type, sum_ptr, "final_sum").unwrap();
        self.builder.build_store(c_ptr_elem, final_sum).unwrap();

        // j++
        let j_next = self.builder.build_int_add(j, i32_type.const_int(1, false), "j_next").unwrap();
        self.builder.build_store(j_ptr, j_next).unwrap();
        self.builder.build_unconditional_branch(j_loop_bb).unwrap();

        // j end: increment i
        self.builder.position_at_end(j_end_bb);
        self.builder.build_unconditional_branch(i_end_bb).unwrap();

        // i end: i++
        self.builder.position_at_end(i_end_bb);
        let i_next = self.builder.build_int_add(i, i32_type.const_int(1, false), "i_next").unwrap();
        self.builder.build_store(i_ptr, i_next).unwrap();
        self.builder.build_unconditional_branch(i_loop_bb).unwrap();

        // End: return
        self.builder.position_at_end(end_bb);
        self.builder.build_return(None).unwrap();

        function
    }

    /// Generate optimized MatMul with cache tiling
    ///
    /// Cache-optimized matrix multiplication using blocking/tiling technique.
    /// Divides matrices into smaller tiles that fit in L1/L2 cache.
    ///
    /// Algorithm:
    ///   C[m×n] = A[m×k] @ B[k×n]
    ///   Using tile size TILE_SIZE (typically 32 or 64)
    ///
    ///   for ii in 0..m step TILE_SIZE:
    ///     for jj in 0..n step TILE_SIZE:
    ///       for kk in 0..k step TILE_SIZE:
    ///         # Process tile
    ///         for i in ii..min(ii+TILE, m):
    ///           for j in jj..min(jj+TILE, n):
    ///             for k in kk..min(kk+TILE, k):
    ///               C[i,j] += A[i,k] * B[k,j]
    ///
    /// Benefits:
    /// - Better cache locality (2-3x speedup for large matrices)
    /// - Reduced cache misses
    /// - Better CPU pipeline utilization
    ///
    /// Function signature: fn matmul_tiled(a: *f32, b: *f32, c: *f32, m: i32, n: i32, k: i32, tile: i32)
    pub fn gen_matmul_tiled(&self, m: u32, n: u32, k: u32, tile_size: u32) -> FunctionValue<'ctx> {
        let i32_type = self.context.i32_type();
        let f32_type = self.context.f32_type();
        let f32_ptr_type = f32_type.ptr_type(inkwell::AddressSpace::default());

        // Function signature
        let fn_type = self.context.void_type().fn_type(
            &[
                f32_ptr_type.into(), // a
                f32_ptr_type.into(), // b
                f32_ptr_type.into(), // c
                i32_type.into(),     // m
                i32_type.into(),     // n
                i32_type.into(),     // k
                i32_type.into(),     // tile_size
            ],
            false,
        );

        let function = self.module.add_function(
            &format!("matmul_tiled_{}x{}x{}_tile{}", m, n, k, tile_size),
            fn_type,
            None,
        );

        let entry_bb = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry_bb);

        // Get parameters
        let a_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let b_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let c_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
        let m_val = function.get_nth_param(3).unwrap().into_int_value();
        let n_val = function.get_nth_param(4).unwrap().into_int_value();
        let k_val = function.get_nth_param(5).unwrap().into_int_value();
        let tile_val = function.get_nth_param(6).unwrap().into_int_value();

        // Initialize C to zero first
        let zero_loop_bb = self.context.append_basic_block(function, "zero_loop");
        let zero_body_bb = self.context.append_basic_block(function, "zero_body");
        let zero_end_bb = self.context.append_basic_block(function, "zero_end");

        let zero_idx_ptr = self.builder.build_alloca(i32_type, "zero_idx").unwrap();
        self.builder.build_store(zero_idx_ptr, i32_type.const_zero()).unwrap();

        // Calculate total size m * n
        let total_size = self.builder.build_int_mul(m_val, n_val, "total_size").unwrap();

        self.builder.build_unconditional_branch(zero_loop_bb).unwrap();
        self.builder.position_at_end(zero_loop_bb);

        let zero_idx = self.builder.build_load(i32_type, zero_idx_ptr, "zidx").unwrap().into_int_value();
        let zero_cond = self.builder.build_int_compare(
            IntPredicate::ULT,
            zero_idx,
            total_size,
            "zero_cond",
        ).unwrap();
        self.builder.build_conditional_branch(zero_cond, zero_body_bb, zero_end_bb).unwrap();

        self.builder.position_at_end(zero_body_bb);
        let c_elem_ptr = unsafe {
            self.builder.build_gep(f32_type, c_ptr, &[zero_idx], "c_elem").unwrap()
        };
        self.builder.build_store(c_elem_ptr, f32_type.const_zero()).unwrap();
        let next_zero = self.builder.build_int_add(zero_idx, i32_type.const_int(1, false), "next_zero").unwrap();
        self.builder.build_store(zero_idx_ptr, next_zero).unwrap();
        self.builder.build_unconditional_branch(zero_loop_bb).unwrap();

        self.builder.position_at_end(zero_end_bb);

        // Allocate loop counters for tiled loops
        let ii_ptr = self.builder.build_alloca(i32_type, "ii_ptr").unwrap();
        let jj_ptr = self.builder.build_alloca(i32_type, "jj_ptr").unwrap();
        let kk_ptr = self.builder.build_alloca(i32_type, "kk_ptr").unwrap();
        let i_ptr = self.builder.build_alloca(i32_type, "i_ptr").unwrap();
        let j_ptr = self.builder.build_alloca(i32_type, "j_ptr").unwrap();
        let k_ptr = self.builder.build_alloca(i32_type, "k_ptr").unwrap();

        // ii_loop: for ii in 0..m step tile_size
        self.builder.build_store(ii_ptr, i32_type.const_zero()).unwrap();
        let ii_loop_bb = self.context.append_basic_block(function, "ii_loop");
        let ii_body_bb = self.context.append_basic_block(function, "ii_body");
        let ii_end_bb = self.context.append_basic_block(function, "ii_end");

        self.builder.build_unconditional_branch(ii_loop_bb).unwrap();
        self.builder.position_at_end(ii_loop_bb);

        let ii = self.builder.build_load(i32_type, ii_ptr, "ii").unwrap().into_int_value();
        let ii_cond = self.builder.build_int_compare(IntPredicate::ULT, ii, m_val, "ii_cond").unwrap();
        self.builder.build_conditional_branch(ii_cond, ii_body_bb, ii_end_bb).unwrap();

        // jj_loop
        self.builder.position_at_end(ii_body_bb);
        self.builder.build_store(jj_ptr, i32_type.const_zero()).unwrap();
        let jj_loop_bb = self.context.append_basic_block(function, "jj_loop");
        let jj_body_bb = self.context.append_basic_block(function, "jj_body");
        let jj_end_bb = self.context.append_basic_block(function, "jj_end");

        self.builder.build_unconditional_branch(jj_loop_bb).unwrap();
        self.builder.position_at_end(jj_loop_bb);

        let jj = self.builder.build_load(i32_type, jj_ptr, "jj").unwrap().into_int_value();
        let jj_cond = self.builder.build_int_compare(IntPredicate::ULT, jj, n_val, "jj_cond").unwrap();
        self.builder.build_conditional_branch(jj_cond, jj_body_bb, jj_end_bb).unwrap();

        // kk_loop
        self.builder.position_at_end(jj_body_bb);
        self.builder.build_store(kk_ptr, i32_type.const_zero()).unwrap();
        let kk_loop_bb = self.context.append_basic_block(function, "kk_loop");
        let kk_body_bb = self.context.append_basic_block(function, "kk_body");
        let kk_end_bb = self.context.append_basic_block(function, "kk_end");

        self.builder.build_unconditional_branch(kk_loop_bb).unwrap();
        self.builder.position_at_end(kk_loop_bb);

        let kk = self.builder.build_load(i32_type, kk_ptr, "kk").unwrap().into_int_value();
        let kk_cond = self.builder.build_int_compare(IntPredicate::ULT, kk, k_val, "kk_cond").unwrap();
        self.builder.build_conditional_branch(kk_cond, kk_body_bb, kk_end_bb).unwrap();

        // Inner i_loop (within tile)
        self.builder.position_at_end(kk_body_bb);
        let ii_val = self.builder.build_load(i32_type, ii_ptr, "ii_val").unwrap().into_int_value();
        self.builder.build_store(i_ptr, ii_val).unwrap();

        let i_loop_bb = self.context.append_basic_block(function, "i_loop");
        let i_body_bb = self.context.append_basic_block(function, "i_body");
        let i_end_bb = self.context.append_basic_block(function, "i_end");

        self.builder.build_unconditional_branch(i_loop_bb).unwrap();
        self.builder.position_at_end(i_loop_bb);

        let i = self.builder.build_load(i32_type, i_ptr, "i").unwrap().into_int_value();
        // i < min(ii + tile_size, m)
        let ii_plus_tile = self.builder.build_int_add(ii_val, tile_val, "ii_plus_tile").unwrap();
        let i_max = self.builder.build_select(
            self.builder.build_int_compare(IntPredicate::ULT, ii_plus_tile, m_val, "cmp").unwrap(),
            ii_plus_tile,
            m_val,
            "i_max",
        ).unwrap().into_int_value();
        let i_cond = self.builder.build_int_compare(IntPredicate::ULT, i, i_max, "i_cond").unwrap();
        self.builder.build_conditional_branch(i_cond, i_body_bb, i_end_bb).unwrap();

        // Inner j_loop
        self.builder.position_at_end(i_body_bb);
        let jj_val = self.builder.build_load(i32_type, jj_ptr, "jj_val").unwrap().into_int_value();
        self.builder.build_store(j_ptr, jj_val).unwrap();

        let j_loop_bb = self.context.append_basic_block(function, "j_loop");
        let j_body_bb = self.context.append_basic_block(function, "j_body");
        let j_end_bb = self.context.append_basic_block(function, "j_end");

        self.builder.build_unconditional_branch(j_loop_bb).unwrap();
        self.builder.position_at_end(j_loop_bb);

        let j = self.builder.build_load(i32_type, j_ptr, "j").unwrap().into_int_value();
        let jj_plus_tile = self.builder.build_int_add(jj_val, tile_val, "jj_plus_tile").unwrap();
        let j_max = self.builder.build_select(
            self.builder.build_int_compare(IntPredicate::ULT, jj_plus_tile, n_val, "cmp").unwrap(),
            jj_plus_tile,
            n_val,
            "j_max",
        ).unwrap().into_int_value();
        let j_cond = self.builder.build_int_compare(IntPredicate::ULT, j, j_max, "j_cond").unwrap();
        self.builder.build_conditional_branch(j_cond, j_body_bb, j_end_bb).unwrap();

        // Inner k_loop
        self.builder.position_at_end(j_body_bb);
        let kk_val = self.builder.build_load(i32_type, kk_ptr, "kk_val").unwrap().into_int_value();
        self.builder.build_store(k_ptr, kk_val).unwrap();

        let k_loop_bb = self.context.append_basic_block(function, "k_loop");
        let k_body_bb = self.context.append_basic_block(function, "k_body");
        let k_end_bb = self.context.append_basic_block(function, "k_end");

        self.builder.build_unconditional_branch(k_loop_bb).unwrap();
        self.builder.position_at_end(k_loop_bb);

        let k = self.builder.build_load(i32_type, k_ptr, "k").unwrap().into_int_value();
        let kk_plus_tile = self.builder.build_int_add(kk_val, tile_val, "kk_plus_tile").unwrap();
        let k_max = self.builder.build_select(
            self.builder.build_int_compare(IntPredicate::ULT, kk_plus_tile, k_val, "cmp").unwrap(),
            kk_plus_tile,
            k_val,
            "k_max",
        ).unwrap().into_int_value();
        let k_cond = self.builder.build_int_compare(IntPredicate::ULT, k, k_max, "k_cond").unwrap();
        self.builder.build_conditional_branch(k_cond, k_body_bb, k_end_bb).unwrap();

        // Innermost computation: C[i,j] += A[i,k] * B[k,j]
        self.builder.position_at_end(k_body_bb);

        let i_val = self.builder.build_load(i32_type, i_ptr, "i_val").unwrap().into_int_value();
        let j_val = self.builder.build_load(i32_type, j_ptr, "j_val").unwrap().into_int_value();
        let k_val_inner = self.builder.build_load(i32_type, k_ptr, "k_val").unwrap().into_int_value();

        // a_idx = i * k + k
        let i_k = self.builder.build_int_mul(i_val, k_val, "i_k").unwrap();
        let a_idx = self.builder.build_int_add(i_k, k_val_inner, "a_idx").unwrap();

        // b_idx = k * n + j
        let k_n = self.builder.build_int_mul(k_val_inner, n_val, "k_n").unwrap();
        let b_idx = self.builder.build_int_add(k_n, j_val, "b_idx").unwrap();

        // c_idx = i * n + j
        let i_n = self.builder.build_int_mul(i_val, n_val, "i_n").unwrap();
        let c_idx = self.builder.build_int_add(i_n, j_val, "c_idx").unwrap();

        // Load A[i,k]
        let a_elem_ptr = unsafe {
            self.builder.build_gep(f32_type, a_ptr, &[a_idx], "a_elem_ptr").unwrap()
        };
        let a_val = self.builder.build_load(f32_type, a_elem_ptr, "a_val").unwrap().into_float_value();

        // Load B[k,j]
        let b_elem_ptr = unsafe {
            self.builder.build_gep(f32_type, b_ptr, &[b_idx], "b_elem_ptr").unwrap()
        };
        let b_val = self.builder.build_load(f32_type, b_elem_ptr, "b_val").unwrap().into_float_value();

        // Load C[i,j]
        let c_elem_ptr = unsafe {
            self.builder.build_gep(f32_type, c_ptr, &[c_idx], "c_elem_ptr").unwrap()
        };
        let c_val = self.builder.build_load(f32_type, c_elem_ptr, "c_val").unwrap().into_float_value();

        // Compute
        let prod = self.builder.build_float_mul(a_val, b_val, "prod").unwrap();
        let sum = self.builder.build_float_add(c_val, prod, "sum").unwrap();
        self.builder.build_store(c_elem_ptr, sum).unwrap();

        // Increment k
        let next_k = self.builder.build_int_add(k_val_inner, i32_type.const_int(1, false), "next_k").unwrap();
        self.builder.build_store(k_ptr, next_k).unwrap();
        self.builder.build_unconditional_branch(k_loop_bb).unwrap();

        // k_end: increment j
        self.builder.position_at_end(k_end_bb);
        let j_val = self.builder.build_load(i32_type, j_ptr, "j_val").unwrap().into_int_value();
        let next_j = self.builder.build_int_add(j_val, i32_type.const_int(1, false), "next_j").unwrap();
        self.builder.build_store(j_ptr, next_j).unwrap();
        self.builder.build_unconditional_branch(j_loop_bb).unwrap();

        // j_end: increment i
        self.builder.position_at_end(j_end_bb);
        let i_val = self.builder.build_load(i32_type, i_ptr, "i_val").unwrap().into_int_value();
        let next_i = self.builder.build_int_add(i_val, i32_type.const_int(1, false), "next_i").unwrap();
        self.builder.build_store(i_ptr, next_i).unwrap();
        self.builder.build_unconditional_branch(i_loop_bb).unwrap();

        // i_end: increment kk
        self.builder.position_at_end(i_end_bb);
        let kk_val = self.builder.build_load(i32_type, kk_ptr, "kk_val").unwrap().into_int_value();
        let next_kk = self.builder.build_int_add(kk_val, tile_val, "next_kk").unwrap();
        self.builder.build_store(kk_ptr, next_kk).unwrap();
        self.builder.build_unconditional_branch(kk_loop_bb).unwrap();

        // kk_end: increment jj
        self.builder.position_at_end(kk_end_bb);
        let jj_val = self.builder.build_load(i32_type, jj_ptr, "jj_val").unwrap().into_int_value();
        let next_jj = self.builder.build_int_add(jj_val, tile_val, "next_jj").unwrap();
        self.builder.build_store(jj_ptr, next_jj).unwrap();
        self.builder.build_unconditional_branch(jj_loop_bb).unwrap();

        // jj_end: increment ii
        self.builder.position_at_end(jj_end_bb);
        let ii_val = self.builder.build_load(i32_type, ii_ptr, "ii_val").unwrap().into_int_value();
        let next_ii = self.builder.build_int_add(ii_val, tile_val, "next_ii").unwrap();
        self.builder.build_store(ii_ptr, next_ii).unwrap();
        self.builder.build_unconditional_branch(ii_loop_bb).unwrap();

        // ii_end: return
        self.builder.position_at_end(ii_end_bb);
        self.builder.build_return(None).unwrap();

        function
    }

    // ===== Activation Functions (Week 2: Critical ML Ops) =====

    /// Generate LLVM IR for ReLU activation function
    ///
    /// Computes: output[i] = max(0, input[i])
    ///
    /// ReLU (Rectified Linear Unit) is the most common activation function in deep learning.
    /// It's simple, efficient, and helps with the vanishing gradient problem.
    ///
    /// # Returns
    /// Function signature: `fn relu(input: *f32, output: *f32, size: i64)`
    pub fn gen_relu(&self) -> FunctionValue<'ctx> {
        let f32_type = self.f32_type();
        let f32_ptr_type = f32_type.ptr_type(inkwell::AddressSpace::default());
        let i64_type = self.context.i64_type();
        let void_type = self.context.void_type();

        let fn_type = void_type.fn_type(
            &[
                f32_ptr_type.into(), // input
                f32_ptr_type.into(), // output
                i64_type.into(),     // size
            ],
            false,
        );

        let function = self.module.add_function("relu", fn_type, None);
        let entry_bb = self.context.append_basic_block(function, "entry");
        let loop_bb = self.context.append_basic_block(function, "loop");
        let loop_body_bb = self.context.append_basic_block(function, "loop_body");
        let end_bb = self.context.append_basic_block(function, "end");

        let input_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let output_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let size = function.get_nth_param(2).unwrap().into_int_value();

        // Entry: setup counter
        self.builder.position_at_end(entry_bb);
        let counter_ptr = self.builder.build_alloca(i64_type, "counter").unwrap();
        self.builder.build_store(counter_ptr, i64_type.const_zero()).unwrap();
        self.builder.build_unconditional_branch(loop_bb).unwrap();

        // Loop condition
        self.builder.position_at_end(loop_bb);
        let counter = self.builder.build_load(i64_type, counter_ptr, "i").unwrap().into_int_value();
        let cond = self.builder.build_int_compare(IntPredicate::ULT, counter, size, "cond").unwrap();
        self.builder.build_conditional_branch(cond, loop_body_bb, end_bb).unwrap();

        // Loop body: output[i] = max(0, input[i])
        self.builder.position_at_end(loop_body_bb);

        let input_elem_ptr = unsafe {
            self.builder.build_gep(f32_type, input_ptr, &[counter], "input_ptr").unwrap()
        };
        let input_val = self.builder.build_load(f32_type, input_elem_ptr, "input_val").unwrap().into_float_value();

        // Compare with 0.0
        let zero = f32_type.const_zero();
        let is_positive = self.builder.build_float_compare(
            inkwell::FloatPredicate::OGT,
            input_val,
            zero,
            "is_positive"
        ).unwrap();

        // Select: max(0, x) = x if x > 0 else 0
        let result = self.builder.build_select(is_positive, input_val, zero, "relu_result").unwrap();

        let output_elem_ptr = unsafe {
            self.builder.build_gep(f32_type, output_ptr, &[counter], "output_ptr").unwrap()
        };
        self.builder.build_store(output_elem_ptr, result).unwrap();

        // Increment counter
        let next_counter = self.builder.build_int_add(counter, i64_type.const_int(1, false), "next").unwrap();
        self.builder.build_store(counter_ptr, next_counter).unwrap();
        self.builder.build_unconditional_branch(loop_bb).unwrap();

        // End
        self.builder.position_at_end(end_bb);
        self.builder.build_return(None).unwrap();

        function
    }

    /// Generate LLVM IR for Sigmoid activation function
    ///
    /// Computes: output[i] = 1 / (1 + exp(-input[i]))
    ///
    /// Sigmoid squashes values to range (0, 1), useful for binary classification
    /// and as a gate in LSTM networks.
    ///
    /// # Returns
    /// Function signature: `fn sigmoid(input: *f32, output: *f32, size: i64)`
    pub fn gen_sigmoid(&self) -> FunctionValue<'ctx> {
        let f32_type = self.f32_type();
        let f32_ptr_type = f32_type.ptr_type(inkwell::AddressSpace::default());
        let i64_type = self.context.i64_type();
        let void_type = self.context.void_type();

        let fn_type = void_type.fn_type(
            &[
                f32_ptr_type.into(), // input
                f32_ptr_type.into(), // output
                i64_type.into(),     // size
            ],
            false,
        );

        let function = self.module.add_function("sigmoid", fn_type, None);
        let entry_bb = self.context.append_basic_block(function, "entry");
        let loop_bb = self.context.append_basic_block(function, "loop");
        let loop_body_bb = self.context.append_basic_block(function, "loop_body");
        let end_bb = self.context.append_basic_block(function, "end");

        let input_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let output_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let size = function.get_nth_param(2).unwrap().into_int_value();

        // Get LLVM intrinsic for exp
        let exp_intrinsic = self.module.get_function("llvm.exp.f32").unwrap_or_else(|| {
            let exp_fn_type = f32_type.fn_type(&[f32_type.into()], false);
            self.module.add_function("llvm.exp.f32", exp_fn_type, None)
        });

        // Entry: setup counter
        self.builder.position_at_end(entry_bb);
        let counter_ptr = self.builder.build_alloca(i64_type, "counter").unwrap();
        self.builder.build_store(counter_ptr, i64_type.const_zero()).unwrap();
        self.builder.build_unconditional_branch(loop_bb).unwrap();

        // Loop condition
        self.builder.position_at_end(loop_bb);
        let counter = self.builder.build_load(i64_type, counter_ptr, "i").unwrap().into_int_value();
        let cond = self.builder.build_int_compare(IntPredicate::ULT, counter, size, "cond").unwrap();
        self.builder.build_conditional_branch(cond, loop_body_bb, end_bb).unwrap();

        // Loop body: output[i] = 1 / (1 + exp(-input[i]))
        self.builder.position_at_end(loop_body_bb);

        let input_elem_ptr = unsafe {
            self.builder.build_gep(f32_type, input_ptr, &[counter], "input_ptr").unwrap()
        };
        let input_val = self.builder.build_load(f32_type, input_elem_ptr, "input_val").unwrap().into_float_value();

        // Calculate -input[i]
        let neg_input = self.builder.build_float_neg(input_val, "neg_input").unwrap();

        // Calculate exp(-input[i])
        let exp_val = self.builder.build_call(exp_intrinsic, &[neg_input.into()], "exp_val")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();

        // Calculate 1 + exp(-input[i])
        let one = f32_type.const_float(1.0);
        let denominator = self.builder.build_float_add(one, exp_val, "denominator").unwrap();

        // Calculate 1 / (1 + exp(-input[i]))
        let result = self.builder.build_float_div(one, denominator, "sigmoid_result").unwrap();

        let output_elem_ptr = unsafe {
            self.builder.build_gep(f32_type, output_ptr, &[counter], "output_ptr").unwrap()
        };
        self.builder.build_store(output_elem_ptr, result).unwrap();

        // Increment counter
        let next_counter = self.builder.build_int_add(counter, i64_type.const_int(1, false), "next").unwrap();
        self.builder.build_store(counter_ptr, next_counter).unwrap();
        self.builder.build_unconditional_branch(loop_bb).unwrap();

        // End
        self.builder.position_at_end(end_bb);
        self.builder.build_return(None).unwrap();

        function
    }

    /// Generate LLVM IR for Tanh activation function
    ///
    /// Computes: output[i] = (exp(input[i]) - exp(-input[i])) / (exp(input[i]) + exp(-input[i]))
    ///
    /// Tanh squashes values to range (-1, 1), useful as an activation function
    /// and in LSTM/GRU cells. Centered around zero unlike sigmoid.
    ///
    /// # Returns
    /// Function signature: `fn tanh(input: *f32, output: *f32, size: i64)`
    pub fn gen_tanh(&self) -> FunctionValue<'ctx> {
        let f32_type = self.f32_type();
        let f32_ptr_type = f32_type.ptr_type(inkwell::AddressSpace::default());
        let i64_type = self.context.i64_type();
        let void_type = self.context.void_type();

        let fn_type = void_type.fn_type(
            &[
                f32_ptr_type.into(), // input
                f32_ptr_type.into(), // output
                i64_type.into(),     // size
            ],
            false,
        );

        let function = self.module.add_function("tanh", fn_type, None);
        let entry_bb = self.context.append_basic_block(function, "entry");
        let loop_bb = self.context.append_basic_block(function, "loop");
        let loop_body_bb = self.context.append_basic_block(function, "loop_body");
        let end_bb = self.context.append_basic_block(function, "end");

        let input_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let output_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let size = function.get_nth_param(2).unwrap().into_int_value();

        // Get LLVM intrinsic for exp
        let exp_intrinsic = self.module.get_function("llvm.exp.f32").unwrap_or_else(|| {
            let exp_fn_type = f32_type.fn_type(&[f32_type.into()], false);
            self.module.add_function("llvm.exp.f32", exp_fn_type, None)
        });

        // Entry: setup counter
        self.builder.position_at_end(entry_bb);
        let counter_ptr = self.builder.build_alloca(i64_type, "counter").unwrap();
        self.builder.build_store(counter_ptr, i64_type.const_zero()).unwrap();
        self.builder.build_unconditional_branch(loop_bb).unwrap();

        // Loop condition
        self.builder.position_at_end(loop_bb);
        let counter = self.builder.build_load(i64_type, counter_ptr, "i").unwrap().into_int_value();
        let cond = self.builder.build_int_compare(IntPredicate::ULT, counter, size, "cond").unwrap();
        self.builder.build_conditional_branch(cond, loop_body_bb, end_bb).unwrap();

        // Loop body: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        self.builder.position_at_end(loop_body_bb);

        let input_elem_ptr = unsafe {
            self.builder.build_gep(f32_type, input_ptr, &[counter], "input_ptr").unwrap()
        };
        let input_val = self.builder.build_load(f32_type, input_elem_ptr, "input_val").unwrap().into_float_value();

        // Calculate exp(x)
        let exp_pos = self.builder.build_call(exp_intrinsic, &[input_val.into()], "exp_pos")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();

        // Calculate -x
        let neg_input = self.builder.build_float_neg(input_val, "neg_input").unwrap();

        // Calculate exp(-x)
        let exp_neg = self.builder.build_call(exp_intrinsic, &[neg_input.into()], "exp_neg")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();

        // Calculate exp(x) - exp(-x)
        let numerator = self.builder.build_float_sub(exp_pos, exp_neg, "numerator").unwrap();

        // Calculate exp(x) + exp(-x)
        let denominator = self.builder.build_float_add(exp_pos, exp_neg, "denominator").unwrap();

        // Calculate (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        let result = self.builder.build_float_div(numerator, denominator, "tanh_result").unwrap();

        let output_elem_ptr = unsafe {
            self.builder.build_gep(f32_type, output_ptr, &[counter], "output_ptr").unwrap()
        };
        self.builder.build_store(output_elem_ptr, result).unwrap();

        // Increment counter
        let next_counter = self.builder.build_int_add(counter, i64_type.const_int(1, false), "next").unwrap();
        self.builder.build_store(counter_ptr, next_counter).unwrap();
        self.builder.build_unconditional_branch(loop_bb).unwrap();

        // End
        self.builder.position_at_end(end_bb);
        self.builder.build_return(None).unwrap();

        function
    }

    // ===== Convolutional Operations (Week 2: Critical ML Ops) =====

    /// Generate Conv2D operation
    ///
    /// Simplified 2D convolution for single-channel input
    /// Input: (h_in, w_in) image
    /// Kernel: (k_h, k_w) filter
    /// Output: (h_out, w_out) result
    ///
    /// For now: stride=1, padding=0 (valid convolution)
    /// h_out = h_in - k_h + 1
    /// w_out = w_in - k_w + 1
    ///
    /// Function signature:
    ///   fn conv2d(input: *f32, kernel: *f32, output: *f32,
    ///            h_in: i32, w_in: i32, k_h: i32, k_w: i32)
    ///
    /// Algorithm:
    ///   for oh in 0..h_out:
    ///     for ow in 0..w_out:
    ///       sum = 0.0
    ///       for kh in 0..k_h:
    ///         for kw in 0..k_w:
    ///           ih = oh + kh
    ///           iw = ow + kw
    ///           sum += input[ih * w_in + iw] * kernel[kh * k_w + kw]
    ///       output[oh * w_out + ow] = sum
    pub fn gen_conv2d(&self, h_in: u32, w_in: u32, k_h: u32, k_w: u32) -> FunctionValue<'ctx> {
        let i32_type = self.context.i32_type();
        let f32_type = self.context.f32_type();
        let f32_ptr_type = f32_type.ptr_type(inkwell::AddressSpace::default());

        // Calculate output dimensions
        let h_out = h_in - k_h + 1;
        let w_out = w_in - k_w + 1;

        // Function signature
        let fn_type = self.context.void_type().fn_type(
            &[
                f32_ptr_type.into(), // input
                f32_ptr_type.into(), // kernel
                f32_ptr_type.into(), // output
                i32_type.into(),     // h_in
                i32_type.into(),     // w_in
                i32_type.into(),     // k_h
                i32_type.into(),     // k_w
            ],
            false,
        );

        let function = self.module.add_function(
            &format!("conv2d_{}x{}_k{}x{}", h_in, w_in, k_h, k_w),
            fn_type,
            None,
        );

        let entry_bb = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry_bb);

        // Get parameters
        let input_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let kernel_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let output_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
        let w_in_val = function.get_nth_param(4).unwrap().into_int_value();
        let k_w_val = function.get_nth_param(6).unwrap().into_int_value();

        // Constants
        let w_out_const = i32_type.const_int(w_out as u64, false);

        // Allocate loop counters
        let oh_ptr = self.builder.build_alloca(i32_type, "oh_ptr").unwrap();
        let ow_ptr = self.builder.build_alloca(i32_type, "ow_ptr").unwrap();
        let kh_ptr = self.builder.build_alloca(i32_type, "kh_ptr").unwrap();
        let kw_ptr = self.builder.build_alloca(i32_type, "kw_ptr").unwrap();
        let sum_ptr = self.builder.build_alloca(f32_type, "sum_ptr").unwrap();

        // Initialize oh = 0
        self.builder.build_store(oh_ptr, i32_type.const_zero()).unwrap();

        // oh_loop: for oh in 0..h_out
        let oh_loop_bb = self.context.append_basic_block(function, "oh_loop");
        let oh_body_bb = self.context.append_basic_block(function, "oh_body");
        let oh_end_bb = self.context.append_basic_block(function, "oh_end");

        self.builder.build_unconditional_branch(oh_loop_bb).unwrap();
        self.builder.position_at_end(oh_loop_bb);

        let oh = self.builder.build_load(i32_type, oh_ptr, "oh").unwrap().into_int_value();
        let oh_cond = self.builder.build_int_compare(
            inkwell::IntPredicate::ULT,
            oh,
            i32_type.const_int(h_out as u64, false),
            "oh_cond",
        ).unwrap();
        self.builder.build_conditional_branch(oh_cond, oh_body_bb, oh_end_bb).unwrap();

        // oh_body: initialize ow = 0
        self.builder.position_at_end(oh_body_bb);
        self.builder.build_store(ow_ptr, i32_type.const_zero()).unwrap();

        // ow_loop: for ow in 0..w_out
        let ow_loop_bb = self.context.append_basic_block(function, "ow_loop");
        let ow_body_bb = self.context.append_basic_block(function, "ow_body");
        let ow_end_bb = self.context.append_basic_block(function, "ow_end");

        self.builder.build_unconditional_branch(ow_loop_bb).unwrap();
        self.builder.position_at_end(ow_loop_bb);

        let ow = self.builder.build_load(i32_type, ow_ptr, "ow").unwrap().into_int_value();
        let ow_cond = self.builder.build_int_compare(
            inkwell::IntPredicate::ULT,
            ow,
            w_out_const,
            "ow_cond",
        ).unwrap();
        self.builder.build_conditional_branch(ow_cond, ow_body_bb, ow_end_bb).unwrap();

        // ow_body: initialize sum = 0.0, kh = 0
        self.builder.position_at_end(ow_body_bb);
        self.builder.build_store(sum_ptr, f32_type.const_float(0.0)).unwrap();
        self.builder.build_store(kh_ptr, i32_type.const_zero()).unwrap();

        // kh_loop: for kh in 0..k_h
        let kh_loop_bb = self.context.append_basic_block(function, "kh_loop");
        let kh_body_bb = self.context.append_basic_block(function, "kh_body");
        let kh_end_bb = self.context.append_basic_block(function, "kh_end");

        self.builder.build_unconditional_branch(kh_loop_bb).unwrap();
        self.builder.position_at_end(kh_loop_bb);

        let kh = self.builder.build_load(i32_type, kh_ptr, "kh").unwrap().into_int_value();
        let kh_cond = self.builder.build_int_compare(
            inkwell::IntPredicate::ULT,
            kh,
            i32_type.const_int(k_h as u64, false),
            "kh_cond",
        ).unwrap();
        self.builder.build_conditional_branch(kh_cond, kh_body_bb, kh_end_bb).unwrap();

        // kh_body: initialize kw = 0
        self.builder.position_at_end(kh_body_bb);
        self.builder.build_store(kw_ptr, i32_type.const_zero()).unwrap();

        // kw_loop: for kw in 0..k_w
        let kw_loop_bb = self.context.append_basic_block(function, "kw_loop");
        let kw_body_bb = self.context.append_basic_block(function, "kw_body");
        let kw_end_bb = self.context.append_basic_block(function, "kw_end");

        self.builder.build_unconditional_branch(kw_loop_bb).unwrap();
        self.builder.position_at_end(kw_loop_bb);

        let kw = self.builder.build_load(i32_type, kw_ptr, "kw").unwrap().into_int_value();
        let kw_cond = self.builder.build_int_compare(
            inkwell::IntPredicate::ULT,
            kw,
            k_w_val,
            "kw_cond",
        ).unwrap();
        self.builder.build_conditional_branch(kw_cond, kw_body_bb, kw_end_bb).unwrap();

        // kw_body: compute sum += input[ih * w_in + iw] * kernel[kh * k_w + kw]
        self.builder.position_at_end(kw_body_bb);

        // Reload counters
        let oh_val = self.builder.build_load(i32_type, oh_ptr, "oh_val").unwrap().into_int_value();
        let ow_val = self.builder.build_load(i32_type, ow_ptr, "ow_val").unwrap().into_int_value();
        let kh_val = self.builder.build_load(i32_type, kh_ptr, "kh_val").unwrap().into_int_value();
        let kw_val = self.builder.build_load(i32_type, kw_ptr, "kw_val").unwrap().into_int_value();

        // ih = oh + kh, iw = ow + kw
        let ih = self.builder.build_int_add(oh_val, kh_val, "ih").unwrap();
        let iw = self.builder.build_int_add(ow_val, kw_val, "iw").unwrap();

        // input_idx = ih * w_in + iw
        let ih_w = self.builder.build_int_mul(ih, w_in_val, "ih_w").unwrap();
        let input_idx = self.builder.build_int_add(ih_w, iw, "input_idx").unwrap();

        // kernel_idx = kh * k_w + kw
        let kh_kw = self.builder.build_int_mul(kh_val, k_w_val, "kh_kw").unwrap();
        let kernel_idx = self.builder.build_int_add(kh_kw, kw_val, "kernel_idx").unwrap();

        // Load input[input_idx]
        let input_elem_ptr = unsafe {
            self.builder.build_gep(
                f32_type,
                input_ptr,
                &[input_idx],
                "input_elem_ptr",
            ).unwrap()
        };
        let input_val = self.builder.build_load(f32_type, input_elem_ptr, "input_val").unwrap().into_float_value();

        // Load kernel[kernel_idx]
        let kernel_elem_ptr = unsafe {
            self.builder.build_gep(
                f32_type,
                kernel_ptr,
                &[kernel_idx],
                "kernel_elem_ptr",
            ).unwrap()
        };
        let kernel_val = self.builder.build_load(f32_type, kernel_elem_ptr, "kernel_val").unwrap().into_float_value();

        // Multiply
        let prod = self.builder.build_float_mul(input_val, kernel_val, "prod").unwrap();

        // Load sum and add
        let sum = self.builder.build_load(f32_type, sum_ptr, "sum").unwrap().into_float_value();
        let new_sum = self.builder.build_float_add(sum, prod, "new_sum").unwrap();
        self.builder.build_store(sum_ptr, new_sum).unwrap();

        // Increment kw and loop back
        let next_kw = self.builder.build_int_add(kw_val, i32_type.const_int(1, false), "next_kw").unwrap();
        self.builder.build_store(kw_ptr, next_kw).unwrap();
        self.builder.build_unconditional_branch(kw_loop_bb).unwrap();

        // kw_end: increment kh
        self.builder.position_at_end(kw_end_bb);
        let kh_val = self.builder.build_load(i32_type, kh_ptr, "kh_val").unwrap().into_int_value();
        let next_kh = self.builder.build_int_add(kh_val, i32_type.const_int(1, false), "next_kh").unwrap();
        self.builder.build_store(kh_ptr, next_kh).unwrap();
        self.builder.build_unconditional_branch(kh_loop_bb).unwrap();

        // kh_end: store sum to output[oh * w_out + ow], increment ow
        self.builder.position_at_end(kh_end_bb);
        let oh_val = self.builder.build_load(i32_type, oh_ptr, "oh_val").unwrap().into_int_value();
        let ow_val = self.builder.build_load(i32_type, ow_ptr, "ow_val").unwrap().into_int_value();

        // output_idx = oh * w_out + ow
        let oh_w = self.builder.build_int_mul(oh_val, w_out_const, "oh_w").unwrap();
        let output_idx = self.builder.build_int_add(oh_w, ow_val, "output_idx").unwrap();

        // Store sum
        let final_sum = self.builder.build_load(f32_type, sum_ptr, "final_sum").unwrap().into_float_value();
        let output_elem_ptr = unsafe {
            self.builder.build_gep(
                f32_type,
                output_ptr,
                &[output_idx],
                "output_elem_ptr",
            ).unwrap()
        };
        self.builder.build_store(output_elem_ptr, final_sum).unwrap();

        // Increment ow
        let next_ow = self.builder.build_int_add(ow_val, i32_type.const_int(1, false), "next_ow").unwrap();
        self.builder.build_store(ow_ptr, next_ow).unwrap();
        self.builder.build_unconditional_branch(ow_loop_bb).unwrap();

        // ow_end: increment oh
        self.builder.position_at_end(ow_end_bb);
        let oh_val = self.builder.build_load(i32_type, oh_ptr, "oh_val").unwrap().into_int_value();
        let next_oh = self.builder.build_int_add(oh_val, i32_type.const_int(1, false), "next_oh").unwrap();
        self.builder.build_store(oh_ptr, next_oh).unwrap();
        self.builder.build_unconditional_branch(oh_loop_bb).unwrap();

        // oh_end: return
        self.builder.position_at_end(oh_end_bb);
        self.builder.build_return(None).unwrap();

        function
    }

    /// Generate Conv2D with stride and padding support
    ///
    /// More flexible 2D convolution supporting:
    /// - Custom stride (default 1)
    /// - Zero padding (default 0)
    /// - Single channel (for now)
    ///
    /// Output dimensions:
    ///   h_out = (h_in + 2*pad - k_h) / stride + 1
    ///   w_out = (w_in + 2*pad - k_w) / stride + 1
    ///
    /// Function signature:
    ///   fn conv2d_strided(input: *f32, kernel: *f32, output: *f32,
    ///                    h_in: i32, w_in: i32, k_h: i32, k_w: i32,
    ///                    stride: i32, pad: i32)
    ///
    /// Example configurations:
    /// - stride=1, pad=0: Standard "valid" convolution (output smaller than input)
    /// - stride=1, pad=(k_h-1)/2: "same" convolution (output same size as input)
    /// - stride=2, pad=1: Downsampling convolution (output half size)
    pub fn gen_conv2d_strided(
        &self,
        h_in: u32,
        w_in: u32,
        k_h: u32,
        k_w: u32,
        stride: u32,
        pad: u32,
    ) -> FunctionValue<'ctx> {
        let i32_type = self.context.i32_type();
        let f32_type = self.context.f32_type();
        let f32_ptr_type = f32_type.ptr_type(inkwell::AddressSpace::default());

        // Calculate output dimensions
        let h_out = (h_in + 2 * pad - k_h) / stride + 1;
        let w_out = (w_in + 2 * pad - k_w) / stride + 1;

        // Function signature
        let fn_type = self.context.void_type().fn_type(
            &[
                f32_ptr_type.into(), // input
                f32_ptr_type.into(), // kernel
                f32_ptr_type.into(), // output
                i32_type.into(),     // h_in
                i32_type.into(),     // w_in
                i32_type.into(),     // k_h
                i32_type.into(),     // k_w
                i32_type.into(),     // stride
                i32_type.into(),     // pad
            ],
            false,
        );

        let function = self.module.add_function(
            &format!(
                "conv2d_{}x{}_k{}x{}_s{}_p{}",
                h_in, w_in, k_h, k_w, stride, pad
            ),
            fn_type,
            None,
        );

        let entry_bb = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry_bb);

        // Get parameters
        let input_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let kernel_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let output_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
        let h_in_val = function.get_nth_param(3).unwrap().into_int_value();
        let w_in_val = function.get_nth_param(4).unwrap().into_int_value();
        let k_h_val = function.get_nth_param(5).unwrap().into_int_value();
        let k_w_val = function.get_nth_param(6).unwrap().into_int_value();
        let stride_val = function.get_nth_param(7).unwrap().into_int_value();
        let pad_val = function.get_nth_param(8).unwrap().into_int_value();

        // Constants
        let h_out_const = i32_type.const_int(h_out as u64, false);
        let w_out_const = i32_type.const_int(w_out as u64, false);
        let zero_i32 = i32_type.const_zero();
        let zero_f32 = f32_type.const_zero();

        // Allocate loop counters
        let oh_ptr = self.builder.build_alloca(i32_type, "oh_ptr").unwrap();
        let ow_ptr = self.builder.build_alloca(i32_type, "ow_ptr").unwrap();
        let kh_ptr = self.builder.build_alloca(i32_type, "kh_ptr").unwrap();
        let kw_ptr = self.builder.build_alloca(i32_type, "kw_ptr").unwrap();
        let sum_ptr = self.builder.build_alloca(f32_type, "sum_ptr").unwrap();

        // Initialize oh = 0
        self.builder.build_store(oh_ptr, zero_i32).unwrap();

        // oh_loop: for oh in 0..h_out
        let oh_loop_bb = self.context.append_basic_block(function, "oh_loop");
        let oh_body_bb = self.context.append_basic_block(function, "oh_body");
        let oh_end_bb = self.context.append_basic_block(function, "oh_end");

        self.builder.build_unconditional_branch(oh_loop_bb).unwrap();
        self.builder.position_at_end(oh_loop_bb);

        let oh = self.builder.build_load(i32_type, oh_ptr, "oh").unwrap().into_int_value();
        let oh_cond = self.builder.build_int_compare(IntPredicate::ULT, oh, h_out_const, "oh_cond").unwrap();
        self.builder.build_conditional_branch(oh_cond, oh_body_bb, oh_end_bb).unwrap();

        // oh_body: initialize ow = 0
        self.builder.position_at_end(oh_body_bb);
        self.builder.build_store(ow_ptr, zero_i32).unwrap();

        // ow_loop: for ow in 0..w_out
        let ow_loop_bb = self.context.append_basic_block(function, "ow_loop");
        let ow_body_bb = self.context.append_basic_block(function, "ow_body");
        let ow_end_bb = self.context.append_basic_block(function, "ow_end");

        self.builder.build_unconditional_branch(ow_loop_bb).unwrap();
        self.builder.position_at_end(ow_loop_bb);

        let ow = self.builder.build_load(i32_type, ow_ptr, "ow").unwrap().into_int_value();
        let ow_cond = self.builder.build_int_compare(IntPredicate::ULT, ow, w_out_const, "ow_cond").unwrap();
        self.builder.build_conditional_branch(ow_cond, ow_body_bb, ow_end_bb).unwrap();

        // ow_body: initialize sum = 0.0, kh = 0
        self.builder.position_at_end(ow_body_bb);
        self.builder.build_store(sum_ptr, zero_f32).unwrap();
        self.builder.build_store(kh_ptr, zero_i32).unwrap();

        // kh_loop: for kh in 0..k_h
        let kh_loop_bb = self.context.append_basic_block(function, "kh_loop");
        let kh_body_bb = self.context.append_basic_block(function, "kh_body");
        let kh_end_bb = self.context.append_basic_block(function, "kh_end");

        self.builder.build_unconditional_branch(kh_loop_bb).unwrap();
        self.builder.position_at_end(kh_loop_bb);

        let kh = self.builder.build_load(i32_type, kh_ptr, "kh").unwrap().into_int_value();
        let kh_cond = self.builder.build_int_compare(IntPredicate::ULT, kh, k_h_val, "kh_cond").unwrap();
        self.builder.build_conditional_branch(kh_cond, kh_body_bb, kh_end_bb).unwrap();

        // kh_body: initialize kw = 0
        self.builder.position_at_end(kh_body_bb);
        self.builder.build_store(kw_ptr, zero_i32).unwrap();

        // kw_loop: for kw in 0..k_w
        let kw_loop_bb = self.context.append_basic_block(function, "kw_loop");
        let kw_body_bb = self.context.append_basic_block(function, "kw_body");
        let kw_check_bb = self.context.append_basic_block(function, "kw_check");
        let kw_inc_bb = self.context.append_basic_block(function, "kw_inc");
        let kw_end_bb = self.context.append_basic_block(function, "kw_end");

        self.builder.build_unconditional_branch(kw_loop_bb).unwrap();
        self.builder.position_at_end(kw_loop_bb);

        let kw = self.builder.build_load(i32_type, kw_ptr, "kw").unwrap().into_int_value();
        let kw_cond = self.builder.build_int_compare(IntPredicate::ULT, kw, k_w_val, "kw_cond").unwrap();
        self.builder.build_conditional_branch(kw_cond, kw_check_bb, kw_end_bb).unwrap();

        // kw_check: Check bounds with padding
        self.builder.position_at_end(kw_check_bb);

        // Reload counters
        let oh_val = self.builder.build_load(i32_type, oh_ptr, "oh_val").unwrap().into_int_value();
        let ow_val = self.builder.build_load(i32_type, ow_ptr, "ow_val").unwrap().into_int_value();
        let kh_val = self.builder.build_load(i32_type, kh_ptr, "kh_val").unwrap().into_int_value();
        let kw_val = self.builder.build_load(i32_type, kw_ptr, "kw_val").unwrap().into_int_value();

        // Calculate input position: ih = oh * stride + kh - pad
        let oh_stride = self.builder.build_int_mul(oh_val, stride_val, "oh_stride").unwrap();
        let ih_with_pad = self.builder.build_int_add(oh_stride, kh_val, "ih_with_pad").unwrap();
        let ih = self.builder.build_int_sub(ih_with_pad, pad_val, "ih").unwrap();

        // iw = ow * stride + kw - pad
        let ow_stride = self.builder.build_int_mul(ow_val, stride_val, "ow_stride").unwrap();
        let iw_with_pad = self.builder.build_int_add(ow_stride, kw_val, "iw_with_pad").unwrap();
        let iw = self.builder.build_int_sub(iw_with_pad, pad_val, "iw").unwrap();

        // Check if position is within bounds (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in)
        let ih_ge_0 = self.builder.build_int_compare(IntPredicate::SGE, ih, zero_i32, "ih_ge_0").unwrap();
        let ih_lt_h = self.builder.build_int_compare(IntPredicate::SLT, ih, h_in_val, "ih_lt_h").unwrap();
        let iw_ge_0 = self.builder.build_int_compare(IntPredicate::SGE, iw, zero_i32, "iw_ge_0").unwrap();
        let iw_lt_w = self.builder.build_int_compare(IntPredicate::SLT, iw, w_in_val, "iw_lt_w").unwrap();

        let h_valid = self.builder.build_and(ih_ge_0, ih_lt_h, "h_valid").unwrap();
        let w_valid = self.builder.build_and(iw_ge_0, iw_lt_w, "w_valid").unwrap();
        let valid = self.builder.build_and(h_valid, w_valid, "valid").unwrap();

        self.builder.build_conditional_branch(valid, kw_body_bb, kw_inc_bb).unwrap();

        // kw_body: compute sum += input[ih * w_in + iw] * kernel[kh * k_w + kw]
        self.builder.position_at_end(kw_body_bb);

        // input_idx = ih * w_in + iw
        let ih_w = self.builder.build_int_mul(ih, w_in_val, "ih_w").unwrap();
        let input_idx = self.builder.build_int_add(ih_w, iw, "input_idx").unwrap();

        // kernel_idx = kh * k_w + kw
        let kh_kw = self.builder.build_int_mul(kh_val, k_w_val, "kh_kw").unwrap();
        let kernel_idx = self.builder.build_int_add(kh_kw, kw_val, "kernel_idx").unwrap();

        // Load input[input_idx]
        let input_elem_ptr = unsafe {
            self.builder.build_gep(f32_type, input_ptr, &[input_idx], "input_elem_ptr").unwrap()
        };
        let input_val = self.builder.build_load(f32_type, input_elem_ptr, "input_val").unwrap().into_float_value();

        // Load kernel[kernel_idx]
        let kernel_elem_ptr = unsafe {
            self.builder.build_gep(f32_type, kernel_ptr, &[kernel_idx], "kernel_elem_ptr").unwrap()
        };
        let kernel_val = self.builder.build_load(f32_type, kernel_elem_ptr, "kernel_val").unwrap().into_float_value();

        // Multiply and accumulate
        let prod = self.builder.build_float_mul(input_val, kernel_val, "prod").unwrap();
        let sum = self.builder.build_load(f32_type, sum_ptr, "sum").unwrap().into_float_value();
        let new_sum = self.builder.build_float_add(sum, prod, "new_sum").unwrap();
        self.builder.build_store(sum_ptr, new_sum).unwrap();

        // Branch to kw_inc after processing
        self.builder.build_unconditional_branch(kw_inc_bb).unwrap();

        // kw_inc: increment kw and loop back
        self.builder.position_at_end(kw_inc_bb);
        let kw_val = self.builder.build_load(i32_type, kw_ptr, "kw_val").unwrap().into_int_value();
        let next_kw = self.builder.build_int_add(kw_val, i32_type.const_int(1, false), "next_kw").unwrap();
        self.builder.build_store(kw_ptr, next_kw).unwrap();
        self.builder.build_unconditional_branch(kw_loop_bb).unwrap();

        // kw_end: kw loop finished, increment kh
        self.builder.position_at_end(kw_end_bb);
        let kh_val = self.builder.build_load(i32_type, kh_ptr, "kh_val").unwrap().into_int_value();
        let next_kh = self.builder.build_int_add(kh_val, i32_type.const_int(1, false), "next_kh").unwrap();
        self.builder.build_store(kh_ptr, next_kh).unwrap();
        self.builder.build_unconditional_branch(kh_loop_bb).unwrap();

        // kh_end: kh loop finished, store accumulated sum to output
        self.builder.position_at_end(kh_end_bb);
        let oh_val = self.builder.build_load(i32_type, oh_ptr, "oh_val").unwrap().into_int_value();
        let ow_val = self.builder.build_load(i32_type, ow_ptr, "ow_val").unwrap().into_int_value();

        // output_idx = oh * w_out + ow
        let oh_w = self.builder.build_int_mul(oh_val, w_out_const, "oh_w").unwrap();
        let output_idx = self.builder.build_int_add(oh_w, ow_val, "output_idx").unwrap();

        // Store sum
        let final_sum = self.builder.build_load(f32_type, sum_ptr, "final_sum").unwrap().into_float_value();
        let output_elem_ptr = unsafe {
            self.builder.build_gep(f32_type, output_ptr, &[output_idx], "output_elem_ptr").unwrap()
        };
        self.builder.build_store(output_elem_ptr, final_sum).unwrap();

        // Increment ow
        let next_ow = self.builder.build_int_add(ow_val, i32_type.const_int(1, false), "next_ow").unwrap();
        self.builder.build_store(ow_ptr, next_ow).unwrap();
        self.builder.build_unconditional_branch(ow_loop_bb).unwrap();

        // ow_end: increment oh
        self.builder.position_at_end(ow_end_bb);
        let oh_val = self.builder.build_load(i32_type, oh_ptr, "oh_val").unwrap().into_int_value();
        let next_oh = self.builder.build_int_add(oh_val, i32_type.const_int(1, false), "next_oh").unwrap();
        self.builder.build_store(oh_ptr, next_oh).unwrap();
        self.builder.build_unconditional_branch(oh_loop_bb).unwrap();

        // oh_end: return
        self.builder.position_at_end(oh_end_bb);
        self.builder.build_return(None).unwrap();

        function
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

    #[test]
    fn test_gen_matmul_creation() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_matmul");

        // Generate matmul for 4×4 @ 4×4 = 4×4
        let matmul_fn = codegen.gen_matmul(4, 4, 4);

        assert_eq!(matmul_fn.get_name().to_str().unwrap(), "matmul_4x4x4");
        assert_eq!(matmul_fn.count_params(), 6);
        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_gen_matmul_different_sizes() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_matmul_sizes");

        // Test various matrix sizes
        let sizes = vec![
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8),
            (10, 5, 8),
        ];

        for (m, n, k) in sizes {
            let matmul_fn = codegen.gen_matmul(m, n, k);
            let expected_name = format!("matmul_{}x{}x{}", m, n, k);
            assert_eq!(matmul_fn.get_name().to_str().unwrap(), expected_name);
        }

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_matmul_ir_structure() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_matmul_ir");

        codegen.gen_matmul(3, 3, 3);

        let ir = codegen.module.print_to_string().to_string();

        // Verify IR contains expected structures
        assert!(ir.contains("matmul_3x3x3"));
        assert!(ir.contains("i_loop"));
        assert!(ir.contains("j_loop"));
        assert!(ir.contains("k_loop"));
        assert!(ir.contains("fmul")); // Floating point multiplication
        assert!(ir.contains("fadd")); // Floating point addition
    }

    #[test]
    fn test_gen_relu() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_relu");

        let relu_fn = codegen.gen_relu();

        assert_eq!(relu_fn.get_name().to_str().unwrap(), "relu");
        assert_eq!(relu_fn.count_params(), 3); // input, output, size
        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_gen_sigmoid() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_sigmoid");

        let sigmoid_fn = codegen.gen_sigmoid();

        assert_eq!(sigmoid_fn.get_name().to_str().unwrap(), "sigmoid");
        assert_eq!(sigmoid_fn.count_params(), 3); // input, output, size
        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_gen_tanh() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_tanh");

        let tanh_fn = codegen.gen_tanh();

        assert_eq!(tanh_fn.get_name().to_str().unwrap(), "tanh");
        assert_eq!(tanh_fn.count_params(), 3); // input, output, size
        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_activation_ir_structure() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_activations_ir");

        // Test ReLU IR
        codegen.gen_relu();
        let ir = codegen.module.print_to_string().to_string();
        assert!(ir.contains("relu"));
        assert!(ir.contains("fcmp")); // ReLU uses comparison

        // Test Sigmoid IR
        let codegen2 = LLVMCodegen::new(&context, "test_sigmoid_ir");
        codegen2.gen_sigmoid();
        let ir2 = codegen2.module.print_to_string().to_string();
        assert!(ir2.contains("sigmoid"));
        assert!(ir2.contains("llvm.exp.f32")); // Sigmoid uses exp intrinsic

        // Test Tanh IR
        let codegen3 = LLVMCodegen::new(&context, "test_tanh_ir");
        codegen3.gen_tanh();
        let ir3 = codegen3.module.print_to_string().to_string();
        assert!(ir3.contains("tanh"));
        assert!(ir3.contains("llvm.exp.f32")); // Tanh uses exp intrinsic
    }

    #[test]
    fn test_gen_conv2d_creation() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_conv2d");

        // Generate Conv2D: 8x8 input, 3x3 kernel -> 6x6 output
        let conv_fn = codegen.gen_conv2d(8, 8, 3, 3);

        assert_eq!(conv_fn.get_name().to_str().unwrap(), "conv2d_8x8_k3x3");
        assert_eq!(conv_fn.count_params(), 7); // input, kernel, output, h_in, w_in, k_h, k_w
        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_gen_conv2d_different_sizes() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_conv2d_sizes");

        // Test various convolution sizes
        let sizes = vec![
            (8, 8, 3, 3),   // 8x8 image, 3x3 kernel -> 6x6
            (10, 10, 5, 5), // 10x10 image, 5x5 kernel -> 6x6
            (16, 16, 3, 3), // 16x16 image, 3x3 kernel -> 14x14
            (28, 28, 5, 5), // 28x28 image, 5x5 kernel -> 24x24
        ];

        for (h_in, w_in, k_h, k_w) in sizes {
            let conv_fn = codegen.gen_conv2d(h_in, w_in, k_h, k_w);
            let expected_name = format!("conv2d_{}x{}_k{}x{}", h_in, w_in, k_h, k_w);
            assert_eq!(conv_fn.get_name().to_str().unwrap(), expected_name);
        }

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_conv2d_ir_structure() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_conv2d_ir");

        codegen.gen_conv2d(10, 10, 3, 3);

        let ir = codegen.module.print_to_string().to_string();

        // Verify IR contains expected structures
        assert!(ir.contains("conv2d_10x10_k3x3"));
        assert!(ir.contains("oh_loop")); // Output height loop
        assert!(ir.contains("ow_loop")); // Output width loop
        assert!(ir.contains("kh_loop")); // Kernel height loop
        assert!(ir.contains("kw_loop")); // Kernel width loop
        assert!(ir.contains("fmul")); // Floating point multiplication
        assert!(ir.contains("fadd")); // Floating point addition (for accumulation)
    }

    #[test]
    fn test_gen_matmul_tiled_creation() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_matmul_tiled");

        // Generate tiled matmul for 64×64 @ 64×64 = 64×64 with tile size 32
        let matmul_fn = codegen.gen_matmul_tiled(64, 64, 64, 32);

        assert_eq!(matmul_fn.get_name().to_str().unwrap(), "matmul_tiled_64x64x64_tile32");
        assert_eq!(matmul_fn.count_params(), 7); // a, b, c, m, n, k, tile_size
        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_gen_matmul_tiled_different_sizes() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_matmul_tiled_sizes");

        // Test various matrix and tile sizes
        let configs = vec![
            (32, 32, 32, 16),  // Small matrices, small tiles
            (64, 64, 64, 32),  // Medium matrices, medium tiles
            (128, 128, 128, 32), // Larger matrices, medium tiles
        ];

        for (m, n, k, tile) in configs {
            let matmul_fn = codegen.gen_matmul_tiled(m, n, k, tile);
            let expected_name = format!("matmul_tiled_{}x{}x{}_tile{}", m, n, k, tile);
            assert_eq!(matmul_fn.get_name().to_str().unwrap(), expected_name);
        }

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_matmul_tiled_ir_structure() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_matmul_tiled_ir");

        codegen.gen_matmul_tiled(64, 64, 64, 32);

        let ir = codegen.module.print_to_string().to_string();

        // Verify IR contains expected 6-level loop structure (ii, jj, kk, i, j, k)
        assert!(ir.contains("matmul_tiled_64x64x64_tile32"));
        assert!(ir.contains("ii_loop")); // Outer tile loop
        assert!(ir.contains("jj_loop")); // Outer tile loop
        assert!(ir.contains("kk_loop")); // Outer tile loop
        assert!(ir.contains("i_loop"));  // Inner element loop
        assert!(ir.contains("j_loop"));  // Inner element loop
        assert!(ir.contains("k_loop"));  // Inner element loop
        assert!(ir.contains("zero_loop")); // C initialization loop
        assert!(ir.contains("fmul")); // Multiplication
        assert!(ir.contains("fadd")); // Accumulation
    }

    #[test]
    fn test_gen_conv2d_strided_creation() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_conv2d_strided");

        // Generate Conv2D with stride=2, pad=1: 28x28 input, 3x3 kernel -> 14x14 output
        let conv_fn = codegen.gen_conv2d_strided(28, 28, 3, 3, 2, 1);

        assert_eq!(conv_fn.get_name().to_str().unwrap(), "conv2d_28x28_k3x3_s2_p1");
        assert_eq!(conv_fn.count_params(), 9); // input, kernel, output, h_in, w_in, k_h, k_w, stride, pad
        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_gen_conv2d_strided_configurations() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_conv2d_configs");

        // Test various stride/padding configurations
        let configs = vec![
            (28, 28, 3, 3, 1, 0), // Standard valid convolution -> 26x26
            (28, 28, 3, 3, 1, 1), // Same convolution -> 28x28
            (28, 28, 3, 3, 2, 1), // Downsampling convolution -> 14x14
            (56, 56, 5, 5, 2, 2), // Larger downsampling -> 28x28
        ];

        for (h_in, w_in, k_h, k_w, stride, pad) in configs {
            let conv_fn = codegen.gen_conv2d_strided(h_in, w_in, k_h, k_w, stride, pad);
            let expected_name = format!("conv2d_{}x{}_k{}x{}_s{}_p{}", h_in, w_in, k_h, k_w, stride, pad);
            assert_eq!(conv_fn.get_name().to_str().unwrap(), expected_name);
        }

        assert!(codegen.verify().is_ok());
    }

    #[test]
    fn test_conv2d_strided_ir_structure() {
        let context = Context::create();
        let codegen = LLVMCodegen::new(&context, "test_conv2d_strided_ir");

        codegen.gen_conv2d_strided(28, 28, 3, 3, 2, 1);

        let ir = codegen.module.print_to_string().to_string();

        // Verify IR contains expected structures including padding logic
        assert!(ir.contains("conv2d_28x28_k3x3_s2_p1"));
        assert!(ir.contains("oh_loop"));
        assert!(ir.contains("ow_loop"));
        assert!(ir.contains("kh_loop"));
        assert!(ir.contains("kw_loop"));
        assert!(ir.contains("kw_check")); // Bounds checking for padding
        assert!(ir.contains("kw_inc"));   // Increment kw
        assert!(ir.contains("icmp"));     // Comparisons for bounds checking
        assert!(ir.contains("fmul"));
        assert!(ir.contains("fadd"));
    }
}

// Graph Compiler - Integración LLVM + Autograd
// Compila computational graphs a código nativo LLVM

#[cfg(feature = "llvm")]
use crate::autograd::ComputationGraph;
#[cfg(feature = "llvm")]
use crate::llvm_backend::codegen::LLVMCodegen;
#[cfg(feature = "llvm")]
use crate::llvm_backend::jit::JITEngine;
#[cfg(feature = "llvm")]
use inkwell::context::Context;
#[cfg(feature = "llvm")]
use std::collections::HashMap;

/// Compiled computational graph
/// Convierte un grafo de autograd a funciones LLVM compiladas
#[cfg(feature = "llvm")]
pub struct CompiledGraph<'ctx> {
    _context: &'ctx Context,
    codegen: LLVMCodegen<'ctx>,
    jit: Option<JITEngine<'ctx>>,
    // Mapeo de node_id a posición en memory layout
    _node_positions: HashMap<usize, usize>,
}

#[cfg(feature = "llvm")]
impl<'ctx> CompiledGraph<'ctx> {
    /// Crear un nuevo compiled graph
    pub fn new(context: &'ctx Context) -> Self {
        let codegen = LLVMCodegen::new(context, "compiled_graph");

        CompiledGraph {
            _context: context,
            codegen,
            jit: None,
            _node_positions: HashMap::new(),
        }
    }

    /// Compilar un computational graph a LLVM IR
    ///
    /// Estrategia:
    /// 1. Analizar el grafo y determinar orden de ejecución (topological sort)
    /// 2. Generar funciones LLVM para cada tipo de operación
    /// 3. Compilar con JIT para ejecución nativa
    ///
    /// Soporta:
    /// - Add, Mul (element-wise)
    /// - MatMul (matrix multiplication)
    /// - Activaciones (ReLU, Sigmoid, Tanh)
    pub fn compile_graph(
        &mut self,
        graph: &ComputationGraph,
        output_id: usize,
    ) -> Result<(), String> {
        use crate::autograd::Op;

        let output = graph.get_node(output_id).ok_or("Output node not found")?;

        // Verificar que el nodo de salida existe
        if output.data.is_empty() {
            return Err("Output node has no data".to_string());
        }

        // Recorrer el grafo y determinar qué operaciones necesitamos compilar
        let mut needs_add = false;
        let mut needs_mul = false;
        let mut needs_matmul = false;

        // Función auxiliar para explorar el grafo recursivamente
        fn explore_operations(
            graph: &ComputationGraph,
            node_id: usize,
            needs_add: &mut bool,
            needs_mul: &mut bool,
            needs_matmul: &mut bool,
        ) {
            if let Some(node) = graph.get_node(node_id) {
                match &node.op {
                    Op::Add(left, right) => {
                        *needs_add = true;
                        explore_operations(graph, *left, needs_add, needs_mul, needs_matmul);
                        explore_operations(graph, *right, needs_add, needs_mul, needs_matmul);
                    }
                    Op::Mul(left, right) => {
                        *needs_mul = true;
                        explore_operations(graph, *left, needs_add, needs_mul, needs_matmul);
                        explore_operations(graph, *right, needs_add, needs_mul, needs_matmul);
                    }
                    Op::MatMul(left, right) => {
                        *needs_matmul = true;
                        explore_operations(graph, *left, needs_add, needs_mul, needs_matmul);
                        explore_operations(graph, *right, needs_add, needs_mul, needs_matmul);
                    }
                    Op::Sub(left, right)
                    | Op::Div(left, right) => {
                        explore_operations(graph, *left, needs_add, needs_mul, needs_matmul);
                        explore_operations(graph, *right, needs_add, needs_mul, needs_matmul);
                    }
                    Op::Neg(input) | Op::Pow(input, _) | Op::Sum(input) => {
                        explore_operations(graph, *input, needs_add, needs_mul, needs_matmul);
                    }
                    Op::Leaf => {}
                }
            }
        }

        explore_operations(graph, output_id, &mut needs_add, &mut needs_mul, &mut needs_matmul);

        // Generar las funciones LLVM necesarias
        if needs_add {
            self.codegen.gen_element_wise_add();
        }
        if needs_mul {
            self.codegen.gen_element_wise_mul();
        }
        if needs_matmul {
            // Por ahora generamos una versión pequeña de prueba
            self.codegen.gen_matmul(4, 4, 4);
        }

        // Generar funciones de activación (útiles para redes neuronales)
        self.codegen.gen_relu();
        self.codegen.gen_sigmoid();
        self.codegen.gen_tanh();

        // Verificar módulo
        self.codegen.verify()?;

        // Compilar con JIT
        let jit = JITEngine::new(self.codegen.module())?;
        self.jit = Some(jit);

        Ok(())
    }

    /// Compilar un computational graph a LLVM IR (versión legacy)
    ///
    /// Mantiene compatibilidad con tests existentes
    #[deprecated(since = "0.1.0", note = "Use compile_graph instead")]
    pub fn compile_simple_forward(
        &mut self,
        graph: &ComputationGraph,
        output_id: usize,
    ) -> Result<(), String> {
        self.compile_graph(graph, output_id)
    }

    /// Ejecutar forward pass compilado
    ///
    /// Por ahora: ejecuta operaciones individuales
    /// Futuro: ejecuta función fusionada completa
    pub fn execute_add(&self, a: &[f32], b: &[f32], output: &mut [f32]) -> Result<(), String> {
        let jit = self.jit.as_ref().ok_or("Graph not compiled yet")?;

        if a.len() != b.len() || a.len() != output.len() {
            return Err("Tensor size mismatch".to_string());
        }

        unsafe {
            jit.execute_tensor_add(a.as_ptr(), b.as_ptr(), output.as_mut_ptr(), a.len())?;
        }

        Ok(())
    }

    /// Ejecutar multiplicación element-wise
    pub fn execute_mul(&self, a: &[f32], b: &[f32], output: &mut [f32]) -> Result<(), String> {
        let jit = self.jit.as_ref().ok_or("Graph not compiled yet")?;

        if a.len() != b.len() || a.len() != output.len() {
            return Err("Tensor size mismatch".to_string());
        }

        unsafe {
            jit.execute_tensor_mul(a.as_ptr(), b.as_ptr(), output.as_mut_ptr(), a.len())?;
        }

        Ok(())
    }

    /// Ejecutar activación ReLU: max(0, x)
    pub fn execute_relu(&self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        let jit = self.jit.as_ref().ok_or("Graph not compiled yet")?;

        if input.len() != output.len() {
            return Err("Tensor size mismatch".to_string());
        }

        unsafe {
            type ReluFn = unsafe extern "C" fn(*const f32, *mut f32, i64);
            let relu_fn = jit.get_function::<ReluFn>("relu")?;
            relu_fn.call(input.as_ptr(), output.as_mut_ptr(), input.len() as i64);
        }

        Ok(())
    }

    /// Ejecutar activación Sigmoid: 1/(1+exp(-x))
    pub fn execute_sigmoid(&self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        let jit = self.jit.as_ref().ok_or("Graph not compiled yet")?;

        if input.len() != output.len() {
            return Err("Tensor size mismatch".to_string());
        }

        unsafe {
            type SigmoidFn = unsafe extern "C" fn(*const f32, *mut f32, i64);
            let sigmoid_fn = jit.get_function::<SigmoidFn>("sigmoid")?;
            sigmoid_fn.call(input.as_ptr(), output.as_mut_ptr(), input.len() as i64);
        }

        Ok(())
    }

    /// Ejecutar activación Tanh: (e^x - e^-x)/(e^x + e^-x)
    pub fn execute_tanh(&self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        let jit = self.jit.as_ref().ok_or("Graph not compiled yet")?;

        if input.len() != output.len() {
            return Err("Tensor size mismatch".to_string());
        }

        unsafe {
            type TanhFn = unsafe extern "C" fn(*const f32, *mut f32, i64);
            let tanh_fn = jit.get_function::<TanhFn>("tanh")?;
            tanh_fn.call(input.as_ptr(), output.as_mut_ptr(), input.len() as i64);
        }

        Ok(())
    }

    /// Get reference to the codegen (for debugging)
    pub fn codegen(&self) -> &LLVMCodegen<'ctx> {
        &self.codegen
    }

    /// Check if graph has been compiled
    pub fn is_compiled(&self) -> bool {
        self.jit.is_some()
    }
}

// Stub implementation when LLVM is disabled
#[cfg(not(feature = "llvm"))]
pub struct CompiledGraph;

#[cfg(not(feature = "llvm"))]
impl CompiledGraph {
    pub fn new() -> Self {
        panic!("LLVM backend not available. Enable 'llvm' feature.");
    }
}

#[cfg(all(test, feature = "llvm"))]
mod tests {
    use super::*;
    use crate::autograd::{ComputationGraph, Tensor};

    #[test]
    fn test_compiled_graph_creation() {
        let context = Context::create();
        let compiled = CompiledGraph::new(&context);

        // Should create without errors
        assert!(compiled.jit.is_none()); // Not compiled yet
        assert!(!compiled.is_compiled());
    }

    #[test]
    #[ignore] // FIXME: JIT engine creation causes segfault in current environment
    fn test_compile_graph_with_operations() {
        let context = Context::create();
        let mut compiled = CompiledGraph::new(&context);

        // Create a graph with Add and Mul operations
        let mut graph = ComputationGraph::new();
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);

        let a_id = graph.add_node(a);
        let b_id = graph.add_node(b);

        // Create: (a + b) * a
        use crate::autograd::Op;
        let sum_node = Tensor {
            id: crate::autograd::Tensor::new(vec![0.0], vec![1]).id,
            data: vec![11.0, 22.0, 33.0],
            shape: vec![3],
            grad: None,
            requires_grad: false,
            op: Op::Add(a_id, b_id),
        };
        let sum_id = graph.add_node(sum_node);

        // Compile graph - should detect Add and Mul operations
        #[allow(deprecated)]
        let result = compiled.compile_simple_forward(&graph, sum_id);
        assert!(result.is_ok());
        assert!(compiled.is_compiled());
    }

    #[test]
    #[ignore] // FIXME: JIT engine creation causes segfault in current environment
    fn test_compile_simple_graph() {
        let context = Context::create();
        let mut compiled = CompiledGraph::new(&context);

        // Create a simple graph: a + b
        let mut graph = ComputationGraph::new();
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);

        let a_id = graph.add_node(a);
        let _b_id = graph.add_node(b);

        // Compile (MVP: just compiles basic operations)
        let result = compiled.compile_simple_forward(&graph, a_id);
        assert!(result.is_ok());
        assert!(compiled.jit.is_some()); // Now compiled
    }

    #[test]
    #[ignore] // FIXME: JIT execution causes segfault in current environment
    fn test_execute_add() {
        let context = Context::create();
        let mut compiled = CompiledGraph::new(&context);

        // Create and compile graph
        let mut graph = ComputationGraph::new();
        let a = Tensor::new(vec![1.0], vec![1]);
        let a_id = graph.add_node(a);

        compiled.compile_simple_forward(&graph, a_id).unwrap();

        // Execute add operation
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![10.0f32, 20.0, 30.0];
        let mut output = vec![0.0f32; 3];

        compiled.execute_add(&a, &b, &mut output).unwrap();

        assert_eq!(output, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    #[ignore] // FIXME: JIT execution causes segfault in current environment
    fn test_execute_mul() {
        let context = Context::create();
        let mut compiled = CompiledGraph::new(&context);

        let mut graph = ComputationGraph::new();
        let a = Tensor::new(vec![1.0], vec![1]);
        let a_id = graph.add_node(a);

        compiled.compile_simple_forward(&graph, a_id).unwrap();

        let a = vec![2.0f32, 3.0, 4.0];
        let b = vec![10.0f32, 20.0, 30.0];
        let mut output = vec![0.0f32; 3];

        compiled.execute_mul(&a, &b, &mut output).unwrap();

        assert_eq!(output, vec![20.0, 60.0, 120.0]);
    }

    #[test]
    #[ignore] // FIXME: JIT execution causes segfault in current environment
    fn test_size_mismatch_error() {
        let context = Context::create();
        let mut compiled = CompiledGraph::new(&context);

        let mut graph = ComputationGraph::new();
        let a = Tensor::new(vec![1.0], vec![1]);
        let a_id = graph.add_node(a);

        #[allow(deprecated)]
        compiled.compile_simple_forward(&graph, a_id).unwrap();

        let a = vec![1.0f32, 2.0];
        let b = vec![10.0f32, 20.0, 30.0]; // Different size
        let mut output = vec![0.0f32; 2];

        let result = compiled.execute_add(&a, &b, &mut output);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("size mismatch"));
    }

    #[test]
    #[ignore] // FIXME: JIT execution causes segfault in current environment
    fn test_execute_relu() {
        let context = Context::create();
        let mut compiled = CompiledGraph::new(&context);

        let mut graph = ComputationGraph::new();
        let a = Tensor::new(vec![1.0], vec![1]);
        let a_id = graph.add_node(a);

        compiled.compile_graph(&graph, a_id).unwrap();

        // Test ReLU: max(0, x)
        let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0f32; 5];

        compiled.execute_relu(&input, &mut output).unwrap();

        assert_eq!(output, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    #[ignore] // FIXME: JIT execution causes segfault in current environment
    fn test_execute_sigmoid() {
        let context = Context::create();
        let mut compiled = CompiledGraph::new(&context);

        let mut graph = ComputationGraph::new();
        let a = Tensor::new(vec![1.0], vec![1]);
        let a_id = graph.add_node(a);

        compiled.compile_graph(&graph, a_id).unwrap();

        // Test Sigmoid: 1/(1+exp(-x))
        let input = vec![0.0f32]; // sigmoid(0) = 0.5
        let mut output = vec![0.0f32; 1];

        compiled.execute_sigmoid(&input, &mut output).unwrap();

        // sigmoid(0) should be approximately 0.5
        assert!((output[0] - 0.5).abs() < 0.01);
    }

    #[test]
    #[ignore] // FIXME: JIT execution causes segfault in current environment
    fn test_execute_tanh() {
        let context = Context::create();
        let mut compiled = CompiledGraph::new(&context);

        let mut graph = ComputationGraph::new();
        let a = Tensor::new(vec![1.0], vec![1]);
        let a_id = graph.add_node(a);

        compiled.compile_graph(&graph, a_id).unwrap();

        // Test Tanh: (e^x - e^-x)/(e^x + e^-x)
        let input = vec![0.0f32]; // tanh(0) = 0
        let mut output = vec![0.0f32; 1];

        compiled.execute_tanh(&input, &mut output).unwrap();

        // tanh(0) should be approximately 0
        assert!(output[0].abs() < 0.01);
    }

    #[test]
    #[ignore] // FIXME: JIT engine creation causes segfault in current environment
    fn test_graph_operation_detection() {
        use crate::autograd::Op;

        let context = Context::create();
        let mut compiled = CompiledGraph::new(&context);

        // Create a complex graph with multiple operation types
        let mut graph = ComputationGraph::new();

        let a = Tensor::new(vec![1.0], vec![1]);
        let b = Tensor::new(vec![2.0], vec![1]);
        let c = Tensor::new(vec![3.0], vec![1]);

        let a_id = graph.add_node(a);
        let b_id = graph.add_node(b);
        let c_id = graph.add_node(c);

        // Build: (a + b) * c
        let add_node = Tensor {
            id: Tensor::new(vec![0.0], vec![1]).id,
            data: vec![3.0],
            shape: vec![1],
            grad: None,
            requires_grad: false,
            op: Op::Add(a_id, b_id),
        };
        let add_id = graph.add_node(add_node);

        let mul_node = Tensor {
            id: Tensor::new(vec![0.0], vec![1]).id,
            data: vec![9.0],
            shape: vec![1],
            grad: None,
            requires_grad: false,
            op: Op::Mul(add_id, c_id),
        };
        let mul_id = graph.add_node(mul_node);

        // Compile - should detect both Add and Mul
        let result = compiled.compile_graph(&graph, mul_id);
        assert!(result.is_ok());
        assert!(compiled.is_compiled());

        // Verify the module contains the expected functions
        let ir = compiled.codegen().module().print_to_string().to_string();
        assert!(ir.contains("tensor_add"));
        assert!(ir.contains("tensor_mul"));
        assert!(ir.contains("relu"));
        assert!(ir.contains("sigmoid"));
        assert!(ir.contains("tanh"));
    }
}

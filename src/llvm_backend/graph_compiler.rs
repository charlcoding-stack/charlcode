// Graph Compiler - Integración LLVM + Autograd
// Compila computational graphs a código nativo LLVM

#[cfg(feature = "llvm")]
use crate::autograd::{ComputationGraph, Op, Tensor};
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
    context: &'ctx Context,
    codegen: LLVMCodegen<'ctx>,
    jit: Option<JITEngine<'ctx>>,
    // Mapeo de node_id a posición en memory layout
    node_positions: HashMap<usize, usize>,
}

#[cfg(feature = "llvm")]
impl<'ctx> CompiledGraph<'ctx> {
    /// Crear un nuevo compiled graph
    pub fn new(context: &'ctx Context) -> Self {
        let codegen = LLVMCodegen::new(context, "compiled_graph");

        CompiledGraph {
            context,
            codegen,
            jit: None,
            node_positions: HashMap::new(),
        }
    }

    /// Compilar un computational graph a LLVM IR
    ///
    /// Estrategia:
    /// 1. Analizar el grafo y determinar orden de ejecución
    /// 2. Generar una función LLVM que ejecute todas las operaciones
    /// 3. Compilar con JIT
    ///
    /// Por ahora: MVP con operaciones element-wise simples
    pub fn compile_simple_forward(
        &mut self,
        graph: &ComputationGraph,
        output_id: usize,
    ) -> Result<(), String> {
        // Para MVP, verificamos que el grafo sea soportado
        // Soportamos: Add, Mul (element-wise)

        // TODO: Implementación completa
        // Por ahora, solo verificamos que podemos acceder al grafo
        let _output = graph
            .get_node(output_id)
            .ok_or("Output node not found")?;

        // Generar funciones básicas como demostración
        self.codegen.gen_element_wise_add();
        self.codegen.gen_element_wise_mul();

        // Verificar módulo
        self.codegen.verify()?;

        // Compilar con JIT
        let jit = JITEngine::new(self.codegen.module())?;
        self.jit = Some(jit);

        Ok(())
    }

    /// Ejecutar forward pass compilado
    ///
    /// Por ahora: ejecuta operaciones individuales
    /// Futuro: ejecuta función fusionada completa
    pub fn execute_add(
        &self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
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
    pub fn execute_mul(
        &self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        let jit = self.jit.as_ref().ok_or("Graph not compiled yet")?;

        if a.len() != b.len() || a.len() != output.len() {
            return Err("Tensor size mismatch".to_string());
        }

        unsafe {
            jit.execute_tensor_mul(a.as_ptr(), b.as_ptr(), output.as_mut_ptr(), a.len())?;
        }

        Ok(())
    }

    /// Get reference to the codegen (for debugging)
    pub fn codegen(&self) -> &LLVMCodegen<'ctx> {
        &self.codegen
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
    }

    #[test]
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
    fn test_size_mismatch_error() {
        let context = Context::create();
        let mut compiled = CompiledGraph::new(&context);

        let mut graph = ComputationGraph::new();
        let a = Tensor::new(vec![1.0], vec![1]);
        let a_id = graph.add_node(a);

        compiled.compile_simple_forward(&graph, a_id).unwrap();

        let a = vec![1.0f32, 2.0];
        let b = vec![10.0f32, 20.0, 30.0]; // Different size
        let mut output = vec![0.0f32; 2];

        let result = compiled.execute_add(&a, &b, &mut output);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("size mismatch"));
    }
}

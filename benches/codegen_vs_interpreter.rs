// Manual benchmark: Bytecode VM vs Tree-walking Interpreter
// This demonstrates the speedup achieved by Phase 7 optimizations

use charl::ast::*;
use charl::codegen::{BytecodeCompiler, VM, tensor_ops};
use charl::interpreter::{Interpreter, Value};
use std::time::Instant;

fn create_complex_expression_with_variables() -> (Vec<Statement>, Expression) {
    use charl::ast::LetStatement;

    // Create variables so constant folding doesn't eliminate all work
    let stmts = vec![
        Statement::Let(LetStatement {
            name: "x".to_string(),
            type_annotation: None,
            value: Expression::FloatLiteral(2.0),
        }),
        Statement::Let(LetStatement {
            name: "y".to_string(),
            type_annotation: None,
            value: Expression::FloatLiteral(3.0),
        }),
        Statement::Let(LetStatement {
            name: "z".to_string(),
            type_annotation: None,
            value: Expression::FloatLiteral(4.0),
        }),
    ];

    // Build: ((x + y) * z - 1.0) / 2.0 + (x * y)
    let expr = Expression::Binary {
        left: Box::new(Expression::Binary {
            left: Box::new(Expression::Binary {
                left: Box::new(Expression::Binary {
                    left: Box::new(Expression::Identifier("x".to_string())),
                    operator: BinaryOperator::Add,
                    right: Box::new(Expression::Identifier("y".to_string())),
                }),
                operator: BinaryOperator::Multiply,
                right: Box::new(Expression::Identifier("z".to_string())),
            }),
            operator: BinaryOperator::Subtract,
            right: Box::new(Expression::FloatLiteral(1.0)),
        }),
        operator: BinaryOperator::Divide,
        right: Box::new(Expression::Binary {
            left: Box::new(Expression::FloatLiteral(2.0)),
            operator: BinaryOperator::Add,
            right: Box::new(Expression::Binary {
                left: Box::new(Expression::Identifier("x".to_string())),
                operator: BinaryOperator::Multiply,
                right: Box::new(Expression::Identifier("y".to_string())),
            }),
        }),
    };

    (stmts, expr)
}

fn benchmark_interpreter(
    stmts: &[Statement],
    expr: &Expression,
    iterations: usize,
) -> (f64, std::time::Duration) {
    let mut interpreter = Interpreter::new();

    // Setup variables
    for stmt in stmts {
        interpreter.eval_statement(stmt).unwrap();
    }

    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        result = match interpreter.eval_expression(expr) {
            Ok(Value::Float(f)) => f,
            _ => panic!("Unexpected result"),
        };
    }
    let duration = start.elapsed();

    (result, duration)
}

fn benchmark_bytecode_vm(
    stmts: &[Statement],
    expr: &Expression,
    iterations: usize,
) -> (f64, std::time::Duration) {
    use charl::ast::LetStatement;

    let mut compiler = BytecodeCompiler::new();

    // Compile variable initialization and expression
    for stmt in stmts {
        if let Statement::Let(LetStatement { name, value, .. }) = stmt {
            compiler.compile_expression(value).unwrap();
            let reg = compiler.get_or_create_register(name);
            compiler.emit(charl::codegen::Instruction::StoreVar(reg));
        }
    }
    compiler.compile_expression(expr).unwrap();
    let module = compiler.finish();

    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        let mut vm = VM::new(module.num_registers);
        result = vm.execute(&module).unwrap();
    }
    let duration = start.elapsed();

    (result, duration)
}

fn benchmark_tensor_operations() {
    println!("\n=== Tensor Operations Benchmark ===\n");

    let size = 10_000;
    let a: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..size).map(|i| (i * 2) as f64).collect();
    let mut result = vec![0.0; size];

    let iterations = 10_000;

    // Benchmark vector addition
    let start = Instant::now();
    for _ in 0..iterations {
        tensor_ops::vector_add(&a, &b, &mut result);
    }
    let duration = start.elapsed();
    println!("Vector Addition ({} elements, {} iterations):", size, iterations);
    println!("  Time: {:?}", duration);
    println!("  Throughput: {:.0} M ops/sec",
        (size as f64 * iterations as f64 / duration.as_secs_f64()) / 1_000_000.0);

    // Benchmark dot product
    let start = Instant::now();
    let mut dot_result = 0.0;
    for _ in 0..iterations {
        dot_result = tensor_ops::dot_product(&a, &b);
    }
    let duration = start.elapsed();
    println!("\nDot Product ({} elements, {} iterations):", size, iterations);
    println!("  Time: {:?}", duration);
    println!("  Result: {}", dot_result);
    println!("  Throughput: {:.0} M ops/sec",
        (size as f64 * iterations as f64 / duration.as_secs_f64()) / 1_000_000.0);

    // Benchmark matrix multiplication (smaller for reasonable time)
    let m = 100;
    let n = 100;
    let p = 100;
    let mat_a: Vec<f64> = (0..(m * n)).map(|i| i as f64 * 0.01).collect();
    let mat_b: Vec<f64> = (0..(n * p)).map(|i| i as f64 * 0.01).collect();
    let mut mat_result = vec![0.0; m * p];

    let mat_iterations = 100;
    let start = Instant::now();
    for _ in 0..mat_iterations {
        tensor_ops::matmul(&mat_a, &mat_b, &mut mat_result, m, n, p);
    }
    let duration = start.elapsed();
    println!("\nMatrix Multiplication ({}x{} * {}x{}, {} iterations):", m, n, n, p, mat_iterations);
    println!("  Time: {:?}", duration);
    println!("  Avg time per matmul: {:?}", duration / mat_iterations);
    println!("  GFLOPS: {:.2}",
        (2.0 * m as f64 * n as f64 * p as f64 * mat_iterations as f64 / duration.as_secs_f64()) / 1_000_000_000.0);
}

fn main() {
    println!("=== Charl Language: Bytecode VM vs Interpreter Benchmark ===\n");

    let (stmts, expr) = create_complex_expression_with_variables();
    let iterations = 1_000_000;

    println!("Expression: ((x + y) * z - 1.0) / 2.0 + (x * y)");
    println!("Variables: x=2.0, y=3.0, z=4.0");
    println!("Iterations: {}\n", iterations);

    // Warmup
    let _ = benchmark_interpreter(&stmts, &expr, 1000);
    let _ = benchmark_bytecode_vm(&stmts, &expr, 1000);

    // Benchmark Interpreter
    println!("Running Tree-walking Interpreter...");
    let (result_interp, time_interp) = benchmark_interpreter(&stmts, &expr, iterations);
    println!("  Result: {}", result_interp);
    println!("  Time:   {:?}", time_interp);
    println!("  Ops/sec: {:.0}\n", iterations as f64 / time_interp.as_secs_f64());

    // Benchmark Bytecode VM
    println!("Running Bytecode VM...");
    let (result_vm, time_vm) = benchmark_bytecode_vm(&stmts, &expr, iterations);
    println!("  Result: {}", result_vm);
    println!("  Time:   {:?}", time_vm);
    println!("  Ops/sec: {:.0}\n", iterations as f64 / time_vm.as_secs_f64());

    // Calculate speedup
    let speedup = time_interp.as_secs_f64() / time_vm.as_secs_f64();
    println!("=== Expression Evaluation Results ===");
    println!("Speedup: {:.2}x faster", speedup);

    if speedup >= 10.0 {
        println!("✅ TARGET ACHIEVED: 10-50x speedup range!");
    } else if speedup >= 5.0 {
        println!("⚠️  Close to target: {:.1}x speedup (target: 10-50x)", speedup);
    } else if speedup >= 2.0 {
        println!("📊 Moderate speedup: {:.1}x faster (target: 10-50x)", speedup);
    } else {
        println!("❌ Below target: {:.1}x speedup (target: 10-50x)", speedup);
    }

    // Benchmark tensor operations
    benchmark_tensor_operations();

    println!("\n=== Summary ===");
    println!("Phase 7 Status: Bytecode VM implemented with optimizations");
    println!("- Constant folding: ✅");
    println!("- Register allocation: ✅");
    println!("- Optimized tensor ops: ✅");
    println!("- Hardware FMA: ✅");
    println!("- Loop unrolling: ✅");
    println!("\nNote: For 50-100x speedup, full LLVM backend required (needs llvm-config).");
    println!("Current bytecode VM provides foundation for Phase 7 optimizations.");
}

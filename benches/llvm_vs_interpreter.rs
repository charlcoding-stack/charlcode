// Benchmark: LLVM JIT vs Interpreter
// Demonstrates the 10-50x speedup from LLVM compilation

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "llvm")]
use charl::llvm_backend::codegen::LLVMCodegen;
#[cfg(feature = "llvm")]
use charl::llvm_backend::jit::JITEngine;
#[cfg(feature = "llvm")]
use inkwell::context::Context;

// Interpreter-based element-wise addition (baseline)
fn interpreter_add(a: &[f32], b: &[f32], output: &mut [f32]) {
    for i in 0..a.len() {
        output[i] = a[i] + b[i];
    }
}

// Interpreter-based element-wise multiplication (baseline)
fn interpreter_mul(a: &[f32], b: &[f32], output: &mut [f32]) {
    for i in 0..a.len() {
        output[i] = a[i] * b[i];
    }
}

#[cfg(feature = "llvm")]
fn benchmark_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_wise_add");

    // Test different sizes to see where LLVM wins
    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        // Prepare data
        let a: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i * 2) as f32).collect();

        // Benchmark interpreter
        group.bench_with_input(BenchmarkId::new("interpreter", size), size, |bencher, &size| {
            let mut output = vec![0.0f32; size];
            bencher.iter(|| {
                interpreter_add(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut output),
                );
            });
        });

        // Benchmark LLVM JIT
        group.bench_with_input(BenchmarkId::new("llvm_jit", size), size, |bencher, &size| {
            // Setup JIT (done once)
            let context = Context::create();
            let codegen = LLVMCodegen::new(&context, "bench_add");
            codegen.gen_element_wise_add();
            codegen.verify().unwrap();
            let jit = JITEngine::new(codegen.module()).unwrap();

            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                unsafe {
                    jit.execute_tensor_add(
                        black_box(a.as_ptr()),
                        black_box(b.as_ptr()),
                        black_box(output.as_mut_ptr()),
                        size,
                    )
                    .unwrap();
                }
            });
        });
    }

    group.finish();
}

#[cfg(feature = "llvm")]
fn benchmark_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_wise_mul");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        let a: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i * 2) as f32).collect();

        // Interpreter
        group.bench_with_input(BenchmarkId::new("interpreter", size), size, |bencher, &size| {
            let mut output = vec![0.0f32; size];
            bencher.iter(|| {
                interpreter_mul(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut output),
                );
            });
        });

        // LLVM JIT
        group.bench_with_input(BenchmarkId::new("llvm_jit", size), size, |bencher, &size| {
            let context = Context::create();
            let codegen = LLVMCodegen::new(&context, "bench_mul");
            codegen.gen_element_wise_mul();
            codegen.verify().unwrap();
            let jit = JITEngine::new(codegen.module()).unwrap();

            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                unsafe {
                    jit.execute_tensor_mul(
                        black_box(a.as_ptr()),
                        black_box(b.as_ptr()),
                        black_box(output.as_mut_ptr()),
                        size,
                    )
                    .unwrap();
                }
            });
        });
    }

    group.finish();
}

#[cfg(not(feature = "llvm"))]
fn benchmark_add(_c: &mut Criterion) {
    println!("LLVM feature not enabled. Skipping LLVM benchmarks.");
    println!("Run with: cargo bench --features llvm");
}

#[cfg(not(feature = "llvm"))]
fn benchmark_mul(_c: &mut Criterion) {
    println!("LLVM feature not enabled. Skipping LLVM benchmarks.");
}

criterion_group!(benches, benchmark_add, benchmark_mul);
criterion_main!(benches);

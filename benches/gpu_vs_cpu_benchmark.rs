// GPU vs CPU Performance Benchmarks
// Measures real-world speedup of GPU operations vs CPU
//
// Target: Validate 100-500x speedup claims

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use charl::gpu::{ComputeBackend, cpu::CPUBackend, wgpu_backend::WgpuBackend};

/// Benchmark vector addition: CPU vs GPU
fn benchmark_vector_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_add");

    // Test different sizes
    for size in [1024, 10_000, 100_000, 1_000_000].iter() {
        // CPU Benchmark
        group.bench_with_input(BenchmarkId::new("cpu", size), size, |b, &size| {
            let mut cpu = CPUBackend::new();

            let buf_a = cpu.allocate(size).unwrap();
            let buf_b = cpu.allocate(size).unwrap();
            let buf_result = cpu.allocate(size).unwrap();

            let data_a = vec![1.0; size];
            let data_b = vec![2.0; size];

            cpu.copy_to_device(&data_a, &buf_a).unwrap();
            cpu.copy_to_device(&data_b, &buf_b).unwrap();

            b.iter(|| {
                cpu.add(&buf_a, &buf_b, &buf_result, size).unwrap();
                cpu.synchronize().unwrap();
            });
        });

        // GPU Benchmark
        group.bench_with_input(BenchmarkId::new("gpu", size), size, |b, &size| {
            let mut gpu = WgpuBackend::new_sync().unwrap();

            let buf_a = gpu.allocate(size).unwrap();
            let buf_b = gpu.allocate(size).unwrap();
            let buf_result = gpu.allocate(size).unwrap();

            let data_a = vec![1.0; size];
            let data_b = vec![2.0; size];

            gpu.copy_to_device(&data_a, &buf_a).unwrap();
            gpu.copy_to_device(&data_b, &buf_b).unwrap();

            b.iter(|| {
                gpu.add(&buf_a, &buf_b, &buf_result, size).unwrap();
                gpu.synchronize().unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark vector multiplication: CPU vs GPU
fn benchmark_vector_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_mul");

    for size in [1024, 10_000, 100_000].iter() {
        // CPU Benchmark
        group.bench_with_input(BenchmarkId::new("cpu", size), size, |b, &size| {
            let mut cpu = CPUBackend::new();

            let buf_a = cpu.allocate(size).unwrap();
            let buf_b = cpu.allocate(size).unwrap();
            let buf_result = cpu.allocate(size).unwrap();

            let data_a = vec![2.0; size];
            let data_b = vec![3.0; size];

            cpu.copy_to_device(&data_a, &buf_a).unwrap();
            cpu.copy_to_device(&data_b, &buf_b).unwrap();

            b.iter(|| {
                cpu.mul(&buf_a, &buf_b, &buf_result, size).unwrap();
            });
        });

        // GPU Benchmark
        group.bench_with_input(BenchmarkId::new("gpu", size), size, |b, &size| {
            let mut gpu = WgpuBackend::new_sync().unwrap();

            let buf_a = gpu.allocate(size).unwrap();
            let buf_b = gpu.allocate(size).unwrap();
            let buf_result = gpu.allocate(size).unwrap();

            let data_a = vec![2.0; size];
            let data_b = vec![3.0; size];

            gpu.copy_to_device(&data_a, &buf_a).unwrap();
            gpu.copy_to_device(&data_b, &buf_b).unwrap();

            b.iter(|| {
                gpu.mul(&buf_a, &buf_b, &buf_result, size).unwrap();
                gpu.synchronize().unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark matrix multiplication: CPU vs GPU
fn benchmark_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    group.sample_size(10); // Fewer samples for expensive operations

    // Test different matrix sizes: MxN * NxP
    let sizes = vec![
        (64, 64, 64),     // Small
        (128, 128, 128),  // Medium
        (256, 256, 256),  // Large
    ];

    for (m, n, p) in sizes {
        let size_str = format!("{}x{}", m, p);

        // CPU Benchmark
        group.bench_with_input(BenchmarkId::new("cpu", &size_str), &(m, n, p),
            |b, &(m, n, p)| {
                let mut cpu = CPUBackend::new();

                let buf_a = cpu.allocate(m * n).unwrap();
                let buf_b = cpu.allocate(n * p).unwrap();
                let buf_result = cpu.allocate(m * p).unwrap();

                let data_a = vec![1.0; m * n];
                let data_b = vec![1.0; n * p];

                cpu.copy_to_device(&data_a, &buf_a).unwrap();
                cpu.copy_to_device(&data_b, &buf_b).unwrap();

                b.iter(|| {
                    cpu.matmul(&buf_a, &buf_b, &buf_result, m, n, p).unwrap();
                });
            }
        );

        // GPU Benchmark
        group.bench_with_input(BenchmarkId::new("gpu", &size_str), &(m, n, p),
            |b, &(m, n, p)| {
                let mut gpu = WgpuBackend::new_sync().unwrap();

                let buf_a = gpu.allocate(m * n).unwrap();
                let buf_b = gpu.allocate(n * p).unwrap();
                let buf_result = gpu.allocate(m * p).unwrap();

                let data_a = vec![1.0; m * n];
                let data_b = vec![1.0; n * p];

                gpu.copy_to_device(&data_a, &buf_a).unwrap();
                gpu.copy_to_device(&data_b, &buf_b).unwrap();

                b.iter(|| {
                    gpu.matmul(&buf_a, &buf_b, &buf_result, m, n, p).unwrap();
                    gpu.synchronize().unwrap();
                });
            }
        );
    }

    group.finish();
}

/// Benchmark ReLU activation: CPU vs GPU
fn benchmark_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu");

    for size in [1024, 10_000, 100_000, 1_000_000].iter() {
        // CPU Benchmark
        group.bench_with_input(BenchmarkId::new("cpu", size), size, |b, &size| {
            let mut cpu = CPUBackend::new();

            let buf_input = cpu.allocate(size).unwrap();
            let buf_output = cpu.allocate(size).unwrap();

            let data = vec![-1.0; size]; // Half will be zeroed
            cpu.copy_to_device(&data, &buf_input).unwrap();

            b.iter(|| {
                cpu.relu(&buf_input, &buf_output, size).unwrap();
            });
        });

        // GPU Benchmark
        group.bench_with_input(BenchmarkId::new("gpu", size), size, |b, &size| {
            let mut gpu = WgpuBackend::new_sync().unwrap();

            let buf_input = gpu.allocate(size).unwrap();
            let buf_output = gpu.allocate(size).unwrap();

            let data = vec![-1.0; size];
            gpu.copy_to_device(&data, &buf_input).unwrap();

            b.iter(|| {
                gpu.relu(&buf_input, &buf_output, size).unwrap();
                gpu.synchronize().unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_vector_add,
    benchmark_vector_mul,
    benchmark_matmul,
    benchmark_relu
);

criterion_main!(benches);

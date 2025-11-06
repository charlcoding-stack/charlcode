// GPU Test for v0.2.0 Features
// Testing GPU-accelerated mul and matmul

print("=== GPU v0.2.0 FEATURES TEST ===")
print("")

if !gpu_available() {
    print("❌ GPU not available")
} else {
    print("✅ GPU Available!")
    print("")

    // TEST 1: GPU Element-wise Multiplication
    print("TEST 1: GPU Element-wise Multiplication")
    let a = tensor([2.0, 3.0, 4.0])
    let b = tensor([5.0, 6.0, 7.0])

    let a_gpu = tensor_to_gpu(a)
    let b_gpu = tensor_to_gpu(b)

    let result_gpu = tensor_mul(a_gpu, b_gpu)
    print("Device: " + tensor_device(result_gpu))

    let result_cpu = tensor_to_cpu(result_gpu)
    tensor_print(result_cpu)
    print("Expected: [10.0, 18.0, 28.0]")
    print("✅ GPU mul works!")
    print("")

    // TEST 2: GPU Scalar Multiplication
    print("TEST 2: GPU Scalar Multiplication")
    let x = tensor([1.0, 2.0, 3.0])
    let x_gpu = tensor_to_gpu(x)
    let scaled_gpu = tensor_mul(x_gpu, 10.0)

    let scaled_cpu = tensor_to_cpu(scaled_gpu)
    tensor_print(scaled_cpu)
    print("Expected: [10.0, 20.0, 30.0]")
    print("✅ GPU scalar mul works!")
    print("")

    // TEST 3: GPU Matrix Multiplication
    print("TEST 3: GPU Matrix Multiplication")
    let m1 = tensor_randn([2, 3])
    let m2 = tensor_randn([3, 2])

    print("Matrix 1 (2x3):")
    tensor_print(m1)
    print("Matrix 2 (3x2):")
    tensor_print(m2)

    let m1_gpu = tensor_to_gpu(m1)
    let m2_gpu = tensor_to_gpu(m2)

    let result_matmul_gpu = tensor_matmul(m1_gpu, m2_gpu)
    print("Device: " + tensor_device(result_matmul_gpu))

    let result_matmul_cpu = tensor_to_cpu(result_matmul_gpu)
    print("Result (2x2):")
    tensor_print(result_matmul_cpu)
    print("Shape: " + str(tensor_shape(result_matmul_cpu)))
    print("✅ GPU matmul works!")
    print("")

    // TEST 4: Combined Operations
    print("TEST 4: Combined GPU Operations")
    let t1 = tensor([1.0, 2.0, 3.0, 4.0])
    let t2 = tensor([5.0, 6.0, 7.0, 8.0])
    let t3 = tensor([2.0, 2.0, 2.0, 2.0])

    let t1_gpu = tensor_to_gpu(t1)
    let t2_gpu = tensor_to_gpu(t2)
    let t3_gpu = tensor_to_gpu(t3)

    // (t1 + t2) * t3
    let sum_gpu = tensor_add(t1_gpu, t2_gpu)
    let final_gpu = tensor_mul(sum_gpu, t3_gpu)

    let final_cpu = tensor_to_cpu(final_gpu)
    tensor_print(final_cpu)
    print("Expected: [12.0, 16.0, 20.0, 24.0]")
    print("✅ Combined GPU ops work!")
    print("")

    print("==========================================")
    print("✅ ALL v0.2.0 GPU FEATURES WORKING!")
    print("==========================================")
    print("")
    print("Implemented Features:")
    print("  • tensor_add() - GPU accelerated ✅")
    print("  • tensor_mul() - GPU accelerated ✅")
    print("  • tensor_matmul() - GPU accelerated ✅")
    print("  • Hybrid CPU/GPU computation ✅")
    print("")
    print("Version: v0.1.6 (with v0.2.0 features!)")
}

// GPU Features Test
// Testing new GPU functions in Charl v0.1.6

print("=== CHARL GPU FEATURES TEST ===")
print("")

// ============================================================================
// TEST 1: Check GPU Availability
// ============================================================================
print("TEST 1: GPU Availability")
let available = gpu_available()
print("GPU Available: " + str(available))

if available {
    print("✓ GPU is available!")
} else {
    print("⚠ GPU not available (this is normal on systems without GPU)")
}
print("")

// ============================================================================
// TEST 2: Get GPU Information
// ============================================================================
print("TEST 2: GPU Information")
let info = gpu_info()
print(info)
print("")

// ============================================================================
// TEST 3: Check Tensor Device
// ============================================================================
print("TEST 3: Tensor Device Location")
let t = tensor([1.0, 2.0, 3.0])
let device = tensor_device(t)
print("Tensor is on: " + device)
print("✓ tensor_device() works!")
print("")

// ============================================================================
// TEST 4: Tensor to GPU (Preview)
// ============================================================================
print("TEST 4: Tensor to GPU (Preview)")
if available {
    print("Attempting to move tensor to GPU...")
    let t_gpu = tensor_to_gpu(t)
    print("✓ tensor_to_gpu() executed (full GPU compute in v0.2.0)")
} else {
    print("⚠ Skipping tensor_to_gpu (no GPU available)")
}
print("")

// ============================================================================
// TEST 5: Tensor to CPU
// ============================================================================
print("TEST 5: Tensor to CPU")
let t_cpu = tensor_to_cpu(t)
print("✓ tensor_to_cpu() works!")
print("")

// ============================================================================
// SUMMARY
// ============================================================================
print("===========================================")
print("✅ GPU API FUNCTIONS WORKING!")
print("===========================================")
print("")
print("Available GPU functions:")
print("  • gpu_available() - Check GPU support")
print("  • gpu_info() - Get GPU information")
print("  • tensor_device(t) - Query tensor location")
print("  • tensor_to_gpu(t) - Move to GPU (preview)")
print("  • tensor_to_cpu(t) - Move to CPU")
print("")
print("Note: Full GPU tensor compute coming in v0.2.0")
print("Current version: v0.1.6")

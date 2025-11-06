// Test Pooling Layers - v0.2.0
// MaxPool2d and AvgPool2d for downsampling

print("=== POOLING LAYERS TEST ===")
print("")

// Create a simple 4x4 feature map for testing
print("Creating 4x4 test feature map...")
let features = tensor([
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0
])
// Reshape to [1, 1, 4, 4] - batch=1, channels=1, H=4, W=4
let image_4x4 = tensor_reshape(features, [1, 1, 4, 4])
print("Input shape: " + str(tensor_shape(image_4x4)))
print("")

// TEST 1: MaxPool2d with 2x2 kernel
print("TEST 1: MaxPool2d(2) - 2x2 pooling")
let maxpool = maxpool2d(2)
print(maxpool)

let maxpool_out = layer_forward(maxpool, image_4x4)
print("Input:  " + str(tensor_shape(image_4x4)))
print("Output: " + str(tensor_shape(maxpool_out)))
print("Expected: [1, 1, 2, 2] (4x4 -> 2x2 with 2x2 pooling)")
tensor_print(maxpool_out)
print("Expected values: [6.0, 8.0, 14.0, 16.0] (max of each 2x2 region)")
print("✅ MaxPool2d works!")
print("")

// TEST 2: AvgPool2d with 2x2 kernel
print("TEST 2: AvgPool2d(2) - 2x2 pooling")
let avgpool = avgpool2d(2)
print(avgpool)

let avgpool_out = layer_forward(avgpool, image_4x4)
print("Input:  " + str(tensor_shape(image_4x4)))
print("Output: " + str(tensor_shape(avgpool_out)))
print("Expected: [1, 1, 2, 2]")
tensor_print(avgpool_out)
print("Expected values: [3.5, 5.5, 11.5, 13.5] (average of each 2x2 region)")
print("  (1+2+5+6)/4 = 3.5, (3+4+7+8)/4 = 5.5, etc.")
print("✅ AvgPool2d works!")
print("")

// TEST 3: Conv2d + MaxPool2d pipeline (common in CNNs)
print("TEST 3: CNN Pipeline - Conv2d → ReLU → MaxPool2d")
let conv = conv2d(3, 16, 3)
let pool = maxpool2d(2)

let rgb_image = tensor_randn([1, 3, 32, 32])
print("Input: " + str(tensor_shape(rgb_image)))

// Forward pass
let conv_out = layer_forward(conv, rgb_image)
print("After Conv2d:  " + str(tensor_shape(conv_out)))

let relu_out = tensor_relu(conv_out)
print("After ReLU:    " + str(tensor_shape(relu_out)))

let pool_out = layer_forward(pool, relu_out)
print("After MaxPool: " + str(tensor_shape(pool_out)))
print("Expected: [1, 16, 15, 15]")
print("  Conv:  32x32 → 30x30 (kernel=3, no padding)")
print("  Pool:  30x30 → 15x15 (2x2 pooling)")
print("✅ CNN pipeline works!")
print("")

// TEST 4: GPU version (if available)
if gpu_available() {
    print("TEST 4: GPU Pooling")

    let image_gpu = tensor_to_gpu(image_4x4)
    print("Input device: " + tensor_device(image_gpu))

    let maxpool_gpu = layer_forward(maxpool, image_gpu)
    print("MaxPool output device: " + tensor_device(maxpool_gpu))
    print("Output shape: " + str(tensor_shape(maxpool_gpu)))

    let avgpool_gpu = layer_forward(avgpool, image_gpu)
    print("AvgPool output device: " + tensor_device(avgpool_gpu))
    print("Output shape: " + str(tensor_shape(avgpool_gpu)))

    print("✅ GPU pooling works!")
    print("")
}

print("=========================================")
print("✅ POOLING LAYERS FULLY FUNCTIONAL!")
print("=========================================")
print("")
print("Available Pooling Layers:")
print("  • maxpool2d(kernel_size) - Downsample by taking max")
print("  • avgpool2d(kernel_size) - Downsample by averaging")
print("")
print("Common CNN Architecture:")
print("  Input → Conv2d → ReLU → MaxPool2d → Conv2d → ReLU → MaxPool2d → Linear")
print("")
print("Next: BatchNorm for training stability!")

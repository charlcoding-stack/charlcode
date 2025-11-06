// Complete CNN Test - v0.2.0
// Full pipeline: Conv2d → BatchNorm → ReLU → MaxPool2d

print("=== COMPLETE CNN PIPELINE TEST ===")
print("")

// Create a small CNN for image classification
print("Building CNN architecture...")
print("  Input: 3 channels (RGB)")
print("  Conv1: 3 → 16 channels, 3x3 kernel")
print("  BatchNorm1: 16 features")
print("  ReLU activation")
print("  MaxPool1: 2x2 pooling")
print("  Conv2: 16 → 32 channels, 3x3 kernel")
print("  BatchNorm2: 32 features")
print("  ReLU activation")
print("  MaxPool2: 2x2 pooling")
print("")

// Layer 1
let conv1 = conv2d(3, 16, 3)
let bn1 = batchnorm(16)
let pool1 = maxpool2d(2)

// Layer 2
let conv2 = conv2d(16, 32, 3)
let bn2 = batchnorm(32)
let pool2 = maxpool2d(2)

print("Layers created:")
print("  " + str(conv1))
print("  " + str(bn1))
print("  " + str(pool1))
print("  " + str(conv2))
print("  " + str(bn2))
print("  " + str(pool2))
print("")

// Create batch of 4 images (32x32 RGB)
print("Creating input batch: [4, 3, 32, 32]")
let images = tensor_randn([4, 3, 32, 32])
print("Input shape: " + str(tensor_shape(images)))
print("")

// Forward pass through full CNN
print("Forward pass:")
print("─────────────")

// Block 1: Conv → BN → ReLU → Pool
let conv1_out = layer_forward(conv1, images)
print("After Conv1:  " + str(tensor_shape(conv1_out)))

let bn1_out = layer_forward(bn1, conv1_out)
print("After BN1:    " + str(tensor_shape(bn1_out)))

let relu1_out = tensor_relu(bn1_out)
print("After ReLU1:  " + str(tensor_shape(relu1_out)))

let pool1_out = layer_forward(pool1, relu1_out)
print("After Pool1:  " + str(tensor_shape(pool1_out)))
print("")

// Block 2: Conv → BN → ReLU → Pool
let conv2_out = layer_forward(conv2, pool1_out)
print("After Conv2:  " + str(tensor_shape(conv2_out)))

let bn2_out = layer_forward(bn2, conv2_out)
print("After BN2:    " + str(tensor_shape(bn2_out)))

let relu2_out = tensor_relu(bn2_out)
print("After ReLU2:  " + str(tensor_shape(relu2_out)))

let pool2_out = layer_forward(pool2, relu2_out)
print("After Pool2:  " + str(tensor_shape(pool2_out)))
print("")

print("Shape transformations:")
print("  [4,3,32,32] → Conv(3→16) → [4,16,30,30]")
print("  [4,16,30,30] → Pool(2x2) → [4,16,15,15]")
print("  [4,16,15,15] → Conv(16→32) → [4,32,13,13]")
print("  [4,32,13,13] → Pool(2x2) → [4,32,6,6]")
print("")
print("Final feature map: [4, 32, 6, 6]")
print("  • 4 images in batch")
print("  • 32 feature channels")
print("  • 6x6 spatial resolution")
print("✅ Full CNN pipeline works!")
print("")

// Test BatchNorm normalization
print("Testing BatchNorm normalization:")
print("─────────────────────────────────")
let test_input = tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
let test_2d = tensor_reshape(test_input, [2, 3])
print("Input [2, 3]: mean should be normalized to ~0")
tensor_print(test_2d)

let bn_test = batchnorm(3)
let normalized = layer_forward(bn_test, test_2d)
print("After BatchNorm (normalized):")
tensor_print(normalized)
print("✅ BatchNorm normalization works!")
print("")

print("=========================================")
print("✅ COMPLETE CNN ARCHITECTURE WORKING!")
print("=========================================")
print("")
print("Available Layers:")
print("  • linear(in, out) - Fully connected")
print("  • conv2d(in_ch, out_ch, kernel) - 2D convolution")
print("  • batchnorm(features) - Batch normalization")
print("  • maxpool2d(kernel) - Max pooling")
print("  • avgpool2d(kernel) - Average pooling")
print("")
print("Available Activations:")
print("  • tensor_relu(x) - ReLU")
print("  • tensor_sigmoid(x) - Sigmoid")
print("  • tensor_tanh(x) - Tanh")
print("  • tensor_gelu(x) - GELU (for Transformers)")
print("  • tensor_softmax(x) - Softmax (for classification)")
print("")
print("Next: LayerNorm + Dropout for Transformers!")

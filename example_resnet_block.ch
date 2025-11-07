// ============================================================================
// RESNET BLOCK EXAMPLE
// Demonstrating CNN Operations in Charl
// ============================================================================
//
// This example shows how to build a ResNet residual block using CNN operations.
//
// ResNet Block Architecture:
//   Input → Conv3x3 → BatchNorm → ReLU → Conv3x3 → BatchNorm → Add(Input) → ReLU
//
// This is the core building block of ResNet (Deep Residual Learning for Image Recognition)
// Paper: https://arxiv.org/abs/1512.03385

print("╔══════════════════════════════════════════════╗")
print("║  RESNET BLOCK DEMO                           ║")
print("║  Built with Charl CNN Operations            ║")
print("╚══════════════════════════════════════════════╝")
print("")

// ============================================================================
// CONFIGURATION
// ============================================================================

print("Configuration:")
print("──────────────────────────────────────────────")

let batch_size = 2
let channels = 64
let height = 32
let width = 32

print("  Batch size:  " + str(batch_size))
print("  Channels:    " + str(channels))
print("  Image size:  " + str(height) + "x" + str(width))
print("")

// ============================================================================
// STEP 1: Create Input Feature Maps
// ============================================================================

print("Step 1: Creating Input Feature Maps")
print("──────────────────────────────────────────────")

// Input: [2, 64, 32, 32] (batch of 2, 64 channels, 32x32 spatial)
let input = tensor_randn([batch_size, channels, height, width])

print("✅ Created input: [" + str(batch_size) + ", " + str(channels) + ", " + str(height) + ", " + str(width) + "]")
print("")

// ============================================================================
// STEP 2: First Convolutional Layer
// ============================================================================

print("Step 2: First Convolution (3x3, same channels)")
print("──────────────────────────────────────────────")

// Conv2D: 64 → 64 channels, 3x3 kernel, stride=1, padding=1
// padding=1 preserves spatial dimensions
let conv1 = nn_conv2d(input, channels, channels, 3, 1, 1)

let conv1_shape = tensor_shape(conv1)
print("  Conv1 output: [" + str(conv1_shape[0]) + ", " + str(conv1_shape[1]) + ", " + str(conv1_shape[2]) + ", " + str(conv1_shape[3]) + "]")
print("✅ Spatial dimensions preserved (padding=1)")
print("")

// ============================================================================
// STEP 3: Batch Normalization
// ============================================================================

print("Step 3: Batch Normalization (stabilizes training)")
print("──────────────────────────────────────────────")

let bn1 = nn_batchnorm(conv1, channels)

print("  Normalized: y = (x - mean) / sqrt(var + eps)")
print("✅ BatchNorm applied to 64 channels")
print("")

// ============================================================================
// STEP 4: ReLU Activation
// ============================================================================

print("Step 4: ReLU Activation")
print("──────────────────────────────────────────────")

let relu1 = nn_relu(bn1)

print("  ReLU(x) = max(0, x)")
print("✅ Non-linearity applied")
print("")

// ============================================================================
// STEP 5: Second Convolutional Layer
// ============================================================================

print("Step 5: Second Convolution (3x3, same channels)")
print("──────────────────────────────────────────────")

let conv2 = nn_conv2d(relu1, channels, channels, 3, 1, 1)

let conv2_shape = tensor_shape(conv2)
print("  Conv2 output: [" + str(conv2_shape[0]) + ", " + str(conv2_shape[1]) + ", " + str(conv2_shape[2]) + ", " + str(conv2_shape[3]) + "]")
print("✅ Second convolution complete")
print("")

// ============================================================================
// STEP 6: Batch Normalization (again)
// ============================================================================

print("Step 6: Second Batch Normalization")
print("──────────────────────────────────────────────")

let bn2 = nn_batchnorm(conv2, channels)

print("✅ Second BatchNorm applied")
print("")

// ============================================================================
// STEP 7: Residual Connection (Skip Connection)
// ============================================================================

print("Step 7: Residual Connection (Key Innovation!)")
print("──────────────────────────────────────────────")

// Add the original input to the processed features
// This is the residual connection that makes ResNet work!
let residual = tensor_add(bn2, input)

print("  residual = processed_features + original_input")
print("✅ Skip connection added")
print("")
print("  Why this matters:")
print("    • Allows gradients to flow directly")
print("    • Enables training very deep networks (100+ layers)")
print("    • Solves vanishing gradient problem")
print("")

// ============================================================================
// STEP 8: Final ReLU
// ============================================================================

print("Step 8: Final ReLU Activation")
print("──────────────────────────────────────────────")

let output = nn_relu(residual)

let output_shape = tensor_shape(output)
print("  Final output: [" + str(output_shape[0]) + ", " + str(output_shape[1]) + ", " + str(output_shape[2]) + ", " + str(output_shape[3]) + "]")
print("✅ ResNet block complete!")
print("")

// ============================================================================
// STEP 9: Build a Deeper Network (Stack Blocks)
// ============================================================================

print("Step 9: Stacking Multiple ResNet Blocks")
print("──────────────────────────────────────────────")

// Start with the output from the first block
let x = output

// Block 2
let conv3 = nn_conv2d(x, channels, channels, 3, 1, 1)
let bn3 = nn_batchnorm(conv3, channels)
let relu3 = nn_relu(bn3)
let conv4 = nn_conv2d(relu3, channels, channels, 3, 1, 1)
let bn4 = nn_batchnorm(conv4, channels)
let residual2 = tensor_add(bn4, x)
let block2_out = nn_relu(residual2)

print("✅ Block 2 complete")

// Block 3
x = block2_out
let conv5 = nn_conv2d(x, channels, channels, 3, 1, 1)
let bn5 = nn_batchnorm(conv5, channels)
let relu5 = nn_relu(bn5)
let conv6 = nn_conv2d(relu5, channels, channels, 3, 1, 1)
let bn6 = nn_batchnorm(conv6, channels)
let residual3 = tensor_add(bn6, x)
let block3_out = nn_relu(residual3)

print("✅ Block 3 complete")
print("")

let final_shape = tensor_shape(block3_out)
print("  Network depth: 3 ResNet blocks = 6 conv layers")
print("  Final output: [" + str(final_shape[0]) + ", " + str(final_shape[1]) + ", " + str(final_shape[2]) + ", " + str(final_shape[3]) + "]")
print("")

// ============================================================================
// STEP 10: Downsampling Block (with stride)
// ============================================================================

print("Step 10: Downsampling Block (stride=2)")
print("──────────────────────────────────────────────")

// When downsampling, we need to match dimensions with a 1x1 conv
let conv_down = nn_conv2d(block3_out, channels, 128, 3, 2, 1)
let bn_down = nn_batchnorm(conv_down, 128)
let relu_down = nn_relu(bn_down)
let conv_down2 = nn_conv2d(relu_down, 128, 128, 3, 1, 1)
let bn_down2 = nn_batchnorm(conv_down2, 128)

// Downsample the skip connection to match
let skip_down = nn_conv2d(block3_out, channels, 128, 1, 2, 0)

// Add and activate
let residual_down = tensor_add(bn_down2, skip_down)
let down_out = nn_relu(residual_down)

let down_shape = tensor_shape(down_out)
print("  Spatial dimensions: 32x32 → 16x16")
print("  Channels: 64 → 128")
print("  Output: [" + str(down_shape[0]) + ", " + str(down_shape[1]) + ", " + str(down_shape[2]) + ", " + str(down_shape[3]) + "]")
print("✅ Downsampling works!")
print("")

// ============================================================================
// STEP 11: Global Average Pooling (for classification)
// ============================================================================

print("Step 11: Global Average Pooling")
print("──────────────────────────────────────────────")

// Average pool to 1x1 (averages entire spatial dimension)
let global_pool = nn_avgpool2d(down_out, 16, 16)

let pool_shape = tensor_shape(global_pool)
print("  Spatial: 16x16 → 1x1")
print("  Output: [" + str(pool_shape[0]) + ", " + str(pool_shape[1]) + ", " + str(pool_shape[2]) + ", " + str(pool_shape[3]) + "]")
print("✅ Ready for classification layer!")
print("")

// ============================================================================
// SUMMARY
// ============================================================================

print("╔══════════════════════════════════════════════╗")
print("║  RESNET NETWORK COMPLETE!                    ║")
print("╚══════════════════════════════════════════════╝")
print("")
print("What we built:")
print("  ✅ 3 ResNet basic blocks (6 conv layers)")
print("  ✅ 1 Downsampling block (stride=2)")
print("  ✅ Global average pooling")
print("  ✅ Batch normalization throughout")
print("  ✅ Residual skip connections")
print("")
print("Network architecture:")
print("  Input: [2, 64, 32, 32]")
print("    ↓")
print("  Block 1: Conv3x3 → BN → ReLU → Conv3x3 → BN → Add → ReLU")
print("  Block 2: Conv3x3 → BN → ReLU → Conv3x3 → BN → Add → ReLU")
print("  Block 3: Conv3x3 → BN → ReLU → Conv3x3 → BN → Add → ReLU")
print("    ↓")
print("  Downsample: [64, 32, 32] → [128, 16, 16]")
print("    ↓")
print("  Global Pool: [128, 16, 16] → [128, 1, 1]")
print("    ↓")
print("  Output: [2, 128, 1, 1] → ready for classification")
print("")
print("Key features of ResNet:")
print("  • Skip connections enable deep networks")
print("  • Batch normalization stabilizes training")
print("  • Downsampling reduces spatial size")
print("  • Global pooling for classification")
print("")
print("You can now build:")
print("  - ResNet-18, ResNet-50, ResNet-101")
print("  - Image classification (ImageNet)")
print("  - Transfer learning")
print("  - Feature extraction")
print("")
print("🚀 Charl supports production CNN architectures!")

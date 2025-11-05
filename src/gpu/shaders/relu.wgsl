// ReLU Activation Function Compute Shader
// Performs ReLU: output[i] = max(0, input[i])
//
// Critical for neural network training
// Target: 100-300x speedup vs CPU

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

// Workgroup size: 256 threads per workgroup
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let array_length = arrayLength(&output);

    // Bounds check
    if (index >= array_length) {
        return;
    }

    // ReLU: max(0, x)
    output[index] = max(0.0, input[index]);
}

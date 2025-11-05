// Vector Multiplication Compute Shader
// Performs element-wise multiplication: result[i] = a[i] * b[i]
//
// Target: 100-500x speedup vs CPU

@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
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

    // Perform multiplication
    output[index] = input_a[index] * input_b[index];
}

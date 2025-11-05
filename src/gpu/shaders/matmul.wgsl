// Matrix Multiplication Compute Shader
// Computes C = A * B where:
//   A is M x N
//   B is N x P
//   C is M x P
//
// Target: 200-500x speedup vs CPU for large matrices

@group(0) @binding(0)
var<storage, read> matrix_a: array<f32>;

@group(0) @binding(1)
var<storage, read> matrix_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> matrix_c: array<f32>;

// Dimensions: M, N, P
@group(0) @binding(3)
var<uniform> dims: vec3<u32>;

// Workgroup size: 16x16 threads
// Each thread computes one element of the output matrix
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    let M = dims.x;
    let N = dims.y;
    let P = dims.z;

    // Bounds check
    if (row >= M || col >= P) {
        return;
    }

    // Compute dot product for this element
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < N; k = k + 1u) {
        let a_val = matrix_a[row * N + k];
        let b_val = matrix_b[k * P + col];
        sum = sum + a_val * b_val;
    }

    // Write result
    matrix_c[row * P + col] = sum;
}

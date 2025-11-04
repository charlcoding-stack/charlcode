// Tensor operations example in Charl
// Demonstrates native tensor types and operations

fn main() {
    // 1D tensor (vector)
    let vec: tensor<float32, [5]> = [1.0, 2.0, 3.0, 4.0, 5.0]
    print("Vector:", vec)

    // 2D tensor (matrix)
    let matrix: tensor<float32, [2, 3]> = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]
    print("Matrix:", matrix)

    // Element-wise operations
    let doubled = matrix * 2.0
    print("Matrix * 2:", doubled)

    // Matrix multiplication
    let m1: tensor<float32, [2, 3]> = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]

    let m2: tensor<float32, [3, 2]> = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]

    // @ operator for matrix multiplication
    let result = m1 @ m2
    print("Matrix multiplication result:", result)
    print("Result shape: [2, 2]")

    // Tensor aggregation
    let total = sum(matrix)
    let average = mean(matrix)
    print("Sum:", total)
    print("Mean:", average)
}

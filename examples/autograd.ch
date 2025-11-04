// Automatic differentiation example in Charl
// Demonstrates native autograd capabilities

fn quadratic(x: float32) -> float32 {
    return x * x + 2.0 * x + 1.0
}

fn tensor_loss(weights: tensor<float32, [5]>) -> float32 {
    // Simple L2 loss
    return sum(weights * weights)
}

fn neural_computation(x: tensor<float32, [10]>,
                      w: tensor<float32, [10]>) -> float32 {
    // Dot product followed by activation
    let z = sum(x * w)
    return relu(z)
}

fn main() {
    // Example 1: Scalar differentiation
    print("=== Scalar Autograd ===")
    let x: float32 = 3.0
    let grad_x = autograd(quadratic, x)
    print("f(x) = x² + 2x + 1")
    print("f'(3.0) =", grad_x)  // Should be 2*3 + 2 = 8

    // Example 2: Tensor differentiation
    print("\n=== Tensor Autograd ===")
    let weights: tensor<float32, [5]> = [1.0, 2.0, 3.0, 4.0, 5.0]
    let grad_weights = autograd(tensor_loss, weights)
    print("Loss = sum(w²)")
    print("Weights:", weights)
    print("Gradients:", grad_weights)  // Should be 2 * weights

    // Example 3: Neural network computation
    print("\n=== Neural Network Gradient ===")
    let input: tensor<float32, [10]> = [0.1, 0.2, 0.3, 0.4, 0.5,
                                         0.6, 0.7, 0.8, 0.9, 1.0]
    let w: tensor<float32, [10]> = random_normal([10])

    let output = neural_computation(input, w)
    let grad_w = autograd(neural_computation, w)

    print("Output:", output)
    print("Gradient w.r.t weights:", grad_w)

    // Example 4: Gradient descent
    print("\n=== Gradient Descent ===")
    let mut param: float32 = 10.0
    let learning_rate: float32 = 0.1

    for step in 0..10 {
        let loss = quadratic(param)
        let gradient = autograd(quadratic, param)

        // Update parameter
        param = param - learning_rate * gradient

        if step % 2 == 0 {
            print("Step", step, "- Loss:", loss, "Param:", param)
        }
    }

    print("Final parameter:", param)
    print("Converged to minimum at x = -1")
}

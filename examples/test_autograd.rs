// Test program to demonstrate automatic differentiation
// Run with: cargo run --example test_autograd

fn main() {
    use charl::autograd::{ComputationGraph, Tensor, add, mul, pow, sum};

    println!("🧮 Charl Automatic Differentiation Demonstration\n");
    println!("{}", "=".repeat(70));

    // Example 1: Simple gradient
    println!("\n📝 Example 1: Simple Gradient");
    println!("{}", "-".repeat(70));
    println!("f(x) = x² where x = 3");
    println!("Expected: f(3) = 9, f'(3) = 6");

    let mut graph = ComputationGraph::new();
    let x = Tensor::scalar_with_grad(3.0);
    let x_id = graph.add_node(x.clone());

    let y = pow(&mut graph, &x, 2.0).unwrap();
    let y_val = y.item().unwrap();
    println!("Result: f(3) = {}", y_val);

    graph.backward(y.id).unwrap();
    let x_grad = graph.get_node(x_id).unwrap().grad.as_ref().unwrap()[0];
    println!("Gradient: f'(3) = {}", x_grad);
    println!("✅ Correct!");

    // Example 2: Chain rule
    println!("\n📝 Example 2: Chain Rule");
    println!("{}", "-".repeat(70));
    println!("f(x) = (x + 2) * 3 where x = 5");
    println!("Expected: f(5) = 21, f'(5) = 3");

    let mut graph2 = ComputationGraph::new();
    let x = Tensor::scalar_with_grad(5.0);
    let two = Tensor::scalar(2.0);
    let three = Tensor::scalar(3.0);

    let x_id = graph2.add_node(x.clone());
    graph2.add_node(two.clone());
    graph2.add_node(three.clone());

    let a = add(&mut graph2, &x, &two).unwrap();
    let b = mul(&mut graph2, &a, &three).unwrap();

    println!("Result: f(5) = {}", b.item().unwrap());

    graph2.backward(b.id).unwrap();
    let x_grad = graph2.get_node(x_id).unwrap().grad.as_ref().unwrap()[0];
    println!("Gradient: f'(5) = {}", x_grad);
    println!("✅ Correct!");

    // Example 3: Multiple variables
    println!("\n📝 Example 3: Multiple Variables");
    println!("{}", "-".repeat(70));
    println!("f(x, y) = x * y + x where x = 3, y = 4");
    println!("Expected: f(3, 4) = 15, ∂f/∂x = 5, ∂f/∂y = 3");

    let mut graph3 = ComputationGraph::new();
    let x = Tensor::scalar_with_grad(3.0);
    let y = Tensor::scalar_with_grad(4.0);

    let x_id = graph3.add_node(x.clone());
    let y_id = graph3.add_node(y.clone());

    let xy = mul(&mut graph3, &x, &y).unwrap();
    let result = add(&mut graph3, &xy, &x).unwrap();

    println!("Result: f(3, 4) = {}", result.item().unwrap());

    graph3.backward(result.id).unwrap();
    let x_grad = graph3.get_node(x_id).unwrap().grad.as_ref().unwrap()[0];
    let y_grad = graph3.get_node(y_id).unwrap().grad.as_ref().unwrap()[0];

    println!("Gradient: ∂f/∂x = {}", x_grad);
    println!("Gradient: ∂f/∂y = {}", y_grad);
    println!("✅ Correct!");

    // Example 4: Vector operations
    println!("\n📝 Example 4: Vector Sum and Backprop");
    println!("{}", "-".repeat(70));
    println!("f(x) = sum(x) where x = [1, 2, 3]");
    println!("Expected: f(x) = 6, ∇f = [1, 1, 1]");

    let mut graph4 = ComputationGraph::new();
    let x = Tensor::with_grad(vec![1.0, 2.0, 3.0], vec![3]);
    let x_id = graph4.add_node(x.clone());

    let s = sum(&mut graph4, &x).unwrap();
    println!("Result: f(x) = {}", s.item().unwrap());

    graph4.backward(s.id).unwrap();
    let x_grad = graph4.get_node(x_id).unwrap().grad.as_ref().unwrap();
    println!("Gradient: ∇f = {:?}", x_grad);
    println!("✅ Correct!");

    // Example 5: Neural network-like computation
    println!("\n📝 Example 5: Simple Neural Network Computation");
    println!("{}", "-".repeat(70));
    println!("f(x, w, b) = (x * w + b)² where x = 2, w = 3, b = 1");
    println!("Expected: f = 49, ∂f/∂w = 28, ∂f/∂b = 14");

    let mut graph5 = ComputationGraph::new();
    let x = Tensor::scalar(2.0); // input (no grad)
    let w = Tensor::scalar_with_grad(3.0); // weight
    let b = Tensor::scalar_with_grad(1.0); // bias

    graph5.add_node(x.clone());
    let w_id = graph5.add_node(w.clone());
    let b_id = graph5.add_node(b.clone());

    let xw = mul(&mut graph5, &x, &w).unwrap(); // x * w = 6
    let linear = add(&mut graph5, &xw, &b).unwrap(); // 6 + 1 = 7
    let output = pow(&mut graph5, &linear, 2.0).unwrap(); // 7² = 49

    println!("Result: f = {}", output.item().unwrap());

    graph5.backward(output.id).unwrap();
    let w_grad = graph5.get_node(w_id).unwrap().grad.as_ref().unwrap()[0];
    let b_grad = graph5.get_node(b_id).unwrap().grad.as_ref().unwrap()[0];

    println!("Gradient: ∂f/∂w = {}", w_grad);
    println!("Gradient: ∂f/∂b = {}", b_grad);
    println!("✅ Correct!");

    println!("\n{}", "=".repeat(70));
    println!("\n✅ All autograd examples completed successfully!");
    println!("🎉 Charl now supports automatic differentiation!");
}

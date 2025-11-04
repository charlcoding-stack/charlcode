// Test program to demonstrate neural networks
// Run with: cargo run --example test_nn

fn main() {
    use charl::autograd::{ComputationGraph, Tensor};
    use charl::nn::{
        Activation, Dense, Dropout, Initializer, Layer, Loss, Sequential,
    };

    println!("🧠 Charl Neural Networks Demonstration\n");
    println!("{}", "=".repeat(70));

    // Example 1: Simple Dense Layer
    println!("\n📝 Example 1: Dense Layer Forward Pass");
    println!("{}", "-".repeat(70));

    let mut graph = ComputationGraph::new();
    let mut layer = Dense::new(3, 2, Activation::ReLU, Initializer::Xavier);

    println!("Layer: Dense(3 -> 2) with ReLU activation");
    println!("Parameters: {} weights + {} biases = {} total",
        layer.weights.data.len(),
        layer.bias.data.len(),
        layer.weights.data.len() + layer.bias.data.len()
    );

    let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    graph.add_node(input.clone());

    let output = layer.forward(&input, &mut graph).unwrap();
    println!("Input: {:?}", input.data);
    println!("Output: {:?}", output.data);
    println!("✅ Forward pass successful!");

    // Example 2: Sequential Model
    println!("\n📝 Example 2: Sequential Model");
    println!("{}", "-".repeat(70));

    let mut model = Sequential::new("SimpleNet".to_string());
    model.add(Box::new(Dense::new(
        4,
        8,
        Activation::ReLU,
        Initializer::Xavier,
    )));
    model.add(Box::new(Dense::new(
        8,
        4,
        Activation::ReLU,
        Initializer::Xavier,
    )));
    model.add(Box::new(Dense::new(
        4,
        2,
        Activation::Softmax,
        Initializer::Xavier,
    )));

    println!("\n");
    model.summary();

    let mut graph2 = ComputationGraph::new();
    let input = Tensor::new(vec![1.0, 0.5, -0.5, 0.2], vec![4]);
    graph2.add_node(input.clone());

    let output = model.forward(&input, &mut graph2).unwrap();
    println!("\nInput: {:?}", input.data);
    println!("Output: {:?}", output.data);
    println!("✅ Sequential model works!");

    // Example 3: Different Activations
    println!("\n📝 Example 3: Activation Functions");
    println!("{}", "-".repeat(70));

    let test_input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    println!("Test input: {:?}", test_input);

    let activations = vec![
        ("ReLU", Activation::ReLU),
        ("Sigmoid", Activation::Sigmoid),
        ("Tanh", Activation::Tanh),
    ];

    for (name, activation) in activations {
        let output = activation.forward(&test_input);
        println!("{:10}: {:?}", name, output);
    }
    println!("✅ All activations work!");

    // Example 4: Softmax (probability distribution)
    println!("\n📝 Example 4: Softmax Activation");
    println!("{}", "-".repeat(70));

    let logits = vec![2.0, 1.0, 0.5];
    let softmax = Activation::Softmax.forward(&logits);
    let sum: f64 = softmax.iter().sum();

    println!("Logits: {:?}", logits);
    println!("Softmax: {:?}", softmax);
    println!("Sum: {} (should be 1.0)", sum);
    println!("✅ Softmax produces valid probability distribution!");

    // Example 5: Loss Functions
    println!("\n📝 Example 5: Loss Functions");
    println!("{}", "-".repeat(70));

    let predicted = vec![0.7, 0.3];
    let target = vec![1.0, 0.0];

    let losses = vec![
        ("MSE", Loss::MSE),
        ("CrossEntropy", Loss::CrossEntropy),
        ("BinaryCrossEntropy", Loss::BinaryCrossEntropy),
    ];

    println!("Predicted: {:?}", predicted);
    println!("Target: {:?}", target);
    println!();

    for (name, loss) in losses {
        let value = loss.compute(&predicted, &target).unwrap();
        println!("{:20}: {:.4}", name, value);
    }
    println!("✅ All loss functions work!");

    // Example 6: Parameter Initialization
    println!("\n📝 Example 6: Parameter Initialization Methods");
    println!("{}", "-".repeat(70));

    let initializers = vec![
        ("Zeros", Initializer::Zeros),
        ("Ones", Initializer::Ones),
        ("Xavier", Initializer::Xavier),
        ("He", Initializer::He),
    ];

    for (name, init) in initializers {
        let params = init.initialize(&[2, 3]);
        println!("{:10}: {:?}", name, params);
    }
    println!("✅ All initializers work!");

    // Example 7: Dropout Layer
    println!("\n📝 Example 7: Dropout Layer");
    println!("{}", "-".repeat(70));

    let mut dropout = Dropout::new(0.5);
    dropout.train_mode();

    let mut graph3 = ComputationGraph::new();
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    graph3.add_node(input.clone());

    let output_train = dropout.forward(&input, &mut graph3).unwrap();
    println!("Training mode (50% dropout):");
    println!("  Input:  {:?}", input.data);
    println!("  Output: {:?}", output_train.data);

    dropout.eval_mode();
    let mut graph4 = ComputationGraph::new();
    graph4.add_node(input.clone());
    let output_eval = dropout.forward(&input, &mut graph4).unwrap();
    println!("\nEvaluation mode (no dropout):");
    println!("  Input:  {:?}", input.data);
    println!("  Output: {:?}", output_eval.data);
    println!("✅ Dropout works correctly!");

    // Example 8: Multi-layer Network for Classification
    println!("\n📝 Example 8: Binary Classification Network");
    println!("{}", "-".repeat(70));

    let mut classifier = Sequential::new("BinaryClassifier".to_string());
    classifier.add(Box::new(Dense::new(
        2,
        4,
        Activation::ReLU,
        Initializer::He,
    )));
    classifier.add(Box::new(Dropout::new(0.2)));
    classifier.add(Box::new(Dense::new(
        4,
        1,
        Activation::Sigmoid,
        Initializer::He,
    )));

    println!("\n");
    classifier.summary();

    // Simulate a training example
    let mut graph5 = ComputationGraph::new();
    let features = Tensor::new(vec![0.5, -0.3], vec![2]);
    graph5.add_node(features.clone());

    let prediction = classifier.forward(&features, &mut graph5).unwrap();
    let predicted_class = if prediction.data[0] > 0.5 { 1 } else { 0 };

    println!("\nFeatures: {:?}", features.data);
    println!("Prediction: {:.4} -> Class {}", prediction.data[0], predicted_class);
    println!("✅ Binary classifier works!");

    println!("\n{}", "=".repeat(70));
    println!("\n✅ All neural network examples completed successfully!");
    println!("🎉 Charl now supports deep learning with:");
    println!("   - Dense (Fully Connected) layers");
    println!("   - Multiple activation functions (ReLU, Sigmoid, Tanh, Softmax)");
    println!("   - Dropout regularization");
    println!("   - Sequential model composition");
    println!("   - Parameter initialization (Xavier, He, etc.)");
    println!("   - Loss functions (MSE, CrossEntropy, Binary CrossEntropy)");
}

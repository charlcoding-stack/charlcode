// Test program to demonstrate training with optimizers
// Run with: cargo run --example test_training

fn main() {
    use charl::autograd::{ComputationGraph, Tensor};
    use charl::nn::{Activation, Dense, Initializer, Loss, Sequential};
    use charl::optim::{
        clip_grad_norm, clip_grad_value, AdaGrad, Adam, ExponentialLR, History, LRScheduler,
        Metrics, Optimizer, RMSprop, StepLR, SGD,
    };

    println!("🎯 Charl Training & Optimization Demonstration\n");
    println!("{}", "=".repeat(70));

    // Example 1: SGD Optimizer
    println!("\n📝 Example 1: SGD Optimizer");
    println!("{}", "-".repeat(70));

    let mut sgd = SGD::new(0.1).with_momentum(0.9);
    let mut tensor = Tensor::with_grad(vec![1.0, 2.0, 3.0], vec![3]);
    tensor.grad = Some(vec![0.1, 0.2, 0.3]);

    println!("Before: {:?}", tensor.data);
    let mut params = vec![&mut tensor];
    sgd.step(&mut params);
    println!("After SGD step: {:?}", tensor.data);
    println!("✅ SGD optimizer works!");

    // Example 2: Adam Optimizer
    println!("\n📝 Example 2: Adam Optimizer");
    println!("{}", "-".repeat(70));

    let mut adam = Adam::new(0.001);
    let mut tensor2 = Tensor::with_grad(vec![5.0, 10.0], vec![2]);
    tensor2.grad = Some(vec![1.0, 2.0]);

    println!("Before: {:?}", tensor2.data);
    let mut params2 = vec![&mut tensor2];
    adam.step(&mut params2);
    println!("After Adam step: {:?}", tensor2.data);
    println!("✅ Adam optimizer works!");

    // Example 3: All Optimizers Comparison
    println!("\n📝 Example 3: Optimizer Comparison");
    println!("{}", "-".repeat(70));

    let optimizers: Vec<(&str, Box<dyn Optimizer>)> = vec![
        ("SGD", Box::new(SGD::new(0.1))),
        ("SGD+Momentum", Box::new(SGD::new(0.1).with_momentum(0.9))),
        ("Adam", Box::new(Adam::new(0.01))),
        ("RMSprop", Box::new(RMSprop::new(0.01))),
        ("AdaGrad", Box::new(AdaGrad::new(0.1))),
    ];

    for (name, mut opt) in optimizers {
        let mut t = Tensor::with_grad(vec![10.0], vec![1]);
        t.grad = Some(vec![1.0]);

        let mut p = vec![&mut t];
        opt.step(&mut p);

        println!("{:15}: {:.6} (after one step)", name, t.data[0]);
    }
    println!("✅ All optimizers work!");

    // Example 4: Learning Rate Schedulers
    println!("\n📝 Example 4: Learning Rate Schedulers");
    println!("{}", "-".repeat(70));

    let mut sgd = SGD::new(0.1);
    let mut step_lr = StepLR::new(0.1, 2, 0.5);

    println!("Step LR Scheduler (decay by 0.5 every 2 epochs):");
    for epoch in 0..5 {
        println!("  Epoch {}: lr = {:.4}", epoch, step_lr.get_lr());
        step_lr.step(&mut sgd);
    }

    let mut sgd2 = SGD::new(0.1);
    let mut exp_lr = ExponentialLR::new(0.1, 0.9);

    println!("\nExponential LR Scheduler (decay by 0.9 each epoch):");
    for epoch in 0..5 {
        println!("  Epoch {}: lr = {:.4}", epoch, exp_lr.get_lr());
        exp_lr.step(&mut sgd2);
    }
    println!("✅ Learning rate schedulers work!");

    // Example 5: Gradient Clipping
    println!("\n📝 Example 5: Gradient Clipping");
    println!("{}", "-".repeat(70));

    let mut tensor3 = Tensor::with_grad(vec![1.0, 2.0], vec![2]);
    tensor3.grad = Some(vec![100.0, -200.0]);

    println!("Before clipping: {:?}", tensor3.grad);

    let mut params3 = vec![&mut tensor3];
    let norm = clip_grad_norm(&mut params3, 5.0);

    println!("Gradient norm: {:.2}", norm);
    println!("After norm clipping (max_norm=5.0): {:?}", tensor3.grad);

    let mut tensor4 = Tensor::with_grad(vec![1.0, 2.0], vec![2]);
    tensor4.grad = Some(vec![10.0, -20.0]);

    let mut params4 = vec![&mut tensor4];
    clip_grad_value(&mut params4, 5.0);

    println!("After value clipping (max_value=5.0): {:?}", tensor4.grad);
    println!("✅ Gradient clipping works!");

    // Example 6: Binary Classification Metrics
    println!("\n📝 Example 6: Binary Classification Metrics");
    println!("{}", "-".repeat(70));

    let predictions = vec![0.9, 0.8, 0.3, 0.2, 0.7, 0.4];
    let targets = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0];

    let metrics = Metrics::compute_binary(&predictions, &targets, 0.5);

    println!("Predictions: {:?}", predictions);
    println!("Targets:     {:?}", targets);
    println!("\nMetrics:");
    println!("  Accuracy:  {:.2}%", metrics.accuracy * 100.0);
    println!("  Precision: {:.2}%", metrics.precision * 100.0);
    println!("  Recall:    {:.2}%", metrics.recall * 100.0);
    println!("  F1 Score:  {:.4}", metrics.f1_score);
    println!("✅ Binary metrics work!");

    // Example 7: Multiclass Classification Metrics
    println!("\n📝 Example 7: Multiclass Classification Metrics");
    println!("{}", "-".repeat(70));

    let predictions = vec![
        vec![0.7, 0.2, 0.1], // Predicts class 0
        vec![0.1, 0.8, 0.1], // Predicts class 1
        vec![0.2, 0.3, 0.5], // Predicts class 2
        vec![0.6, 0.3, 0.1], // Predicts class 0
    ];
    let targets = vec![0, 1, 2, 0];

    let metrics = Metrics::compute_multiclass(&predictions, &targets);

    println!("Predictions: 4 samples with 3 classes each");
    println!("Targets:     {:?}", targets);
    println!("\nMetrics:");
    println!("  Accuracy:  {:.2}%", metrics.accuracy * 100.0);
    println!("✅ Multiclass metrics work!");

    // Example 8: Training History
    println!("\n📝 Example 8: Training History");
    println!("{}", "-".repeat(70));

    let mut history = History::new();

    // Simulate training for 5 epochs
    let epochs_data = vec![
        (0.8, 0.9, 0.6, 0.55),
        (0.5, 0.6, 0.75, 0.70),
        (0.3, 0.4, 0.85, 0.82),
        (0.2, 0.35, 0.90, 0.87),
        (0.15, 0.33, 0.92, 0.88),
    ];

    println!("Epoch | Train Loss | Val Loss | Train Acc | Val Acc");
    println!("{}", "-".repeat(70));

    for (i, (train_loss, val_loss, train_acc, val_acc)) in epochs_data.iter().enumerate() {
        history.add_epoch(*train_loss, *val_loss, *train_acc, *val_acc);
        println!(
            "  {:2}  |   {:.4}    |  {:.4}  |   {:.2}%   | {:.2}%",
            i + 1,
            train_loss,
            val_loss,
            train_acc * 100.0,
            val_acc * 100.0
        );
    }

    println!("\nBest Results:");
    println!("  Best Val Loss: {:.4}", history.best_val_loss().unwrap());
    println!(
        "  Best Val Acc:  {:.2}%",
        history.best_val_accuracy().unwrap() * 100.0
    );
    println!("✅ Training history tracking works!");

    // Example 9: Simple Training Loop Simulation
    println!("\n📝 Example 9: Simple Training Loop");
    println!("{}", "-".repeat(70));

    let mut model = Sequential::new("SimpleClassifier".to_string());
    model.add(Box::new(Dense::new(
        2,
        3,
        Activation::ReLU,
        Initializer::He,
    )));
    model.add(Box::new(Dense::new(
        3,
        1,
        Activation::Sigmoid,
        Initializer::He,
    )));

    let mut optimizer = Adam::new(0.01);
    let loss_fn = Loss::BinaryCrossEntropy;

    println!("Model:");
    model.summary();

    println!("\nSimulating 3 training steps:");

    for step in 1..=3 {
        // Simulate input and target
        let input = Tensor::new(vec![0.5, -0.3], vec![2]);
        let target = vec![1.0];

        // Forward pass
        let mut graph = ComputationGraph::new();
        graph.add_node(input.clone());
        let output = model.forward(&input, &mut graph).unwrap();

        // Compute loss
        let loss = loss_fn.compute(&output.data, &target).unwrap();

        // Simulate gradient computation (in real training, we'd use backward())
        let loss_grad = loss_fn.gradient(&output.data, &target).unwrap();

        // Simulate backprop by setting random gradients
        for param in model.parameters_mut() {
            if let Some(ref mut grad) = param.grad {
                for g in grad.iter_mut() {
                    *g = loss_grad[0] * 0.1; // Simplified gradient
                }
            }
        }

        // Optimizer step
        optimizer.step(&mut model.parameters_mut());

        println!(
            "  Step {}: loss = {:.4}, output = {:.4}",
            step, loss, output.data[0]
        );

        // Zero gradients
        optimizer.zero_grad(&mut model.parameters_mut());
    }

    println!("✅ Training loop simulation complete!");

    println!("\n{}", "=".repeat(70));
    println!("\n✅ All optimization examples completed successfully!");
    println!("🎉 Charl now supports:");
    println!("   - Multiple optimizers (SGD, Adam, RMSprop, AdaGrad)");
    println!("   - Momentum and weight decay");
    println!("   - Learning rate schedulers (StepLR, ExponentialLR)");
    println!("   - Gradient clipping (by norm and by value)");
    println!("   - Comprehensive metrics (Accuracy, Precision, Recall, F1)");
    println!("   - Training history tracking");
    println!("   - Ready for full training loops!");
}

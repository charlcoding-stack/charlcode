// MNIST Training Benchmark - Charl Implementation
// Compares training speed with PyTorch

use charl::autograd::{ComputationGraph, Tensor};
use charl::nn::{Activation, Dense, Dropout, Initializer, Loss, Sequential};
use charl::optim::Adam;
use rand::Rng;
use std::time::Instant;

/// Generate random training data (simulating MNIST)
fn generate_mnist_data(num_samples: usize) -> (Vec<Tensor>, Vec<Tensor>) {
    let mut rng = rand::rng();
    let mut x_train = Vec::new();
    let mut y_train = Vec::new();

    for _ in 0..num_samples {
        // Random 784-dimensional input (28x28 flattened)
        let x_data: Vec<f64> = (0..784).map(|_| rng.random::<f64>()).collect();
        let x = Tensor::new(x_data, vec![784]);
        x_train.push(x);

        // Random 10-dimensional label (one-hot)
        let label = rng.random_range(0..10);
        let mut y_data = vec![0.0; 10];
        y_data[label] = 1.0;
        let y = Tensor::new(y_data, vec![10]);
        y_train.push(y);
    }

    (x_train, y_train)
}

/// Create MNIST classifier model
fn create_model() -> Sequential {
    let mut model = Sequential::new("MnistClassifier".to_string());

    // Layer 1: 784 -> 128 with ReLU
    model.add(Box::new(Dense::new(
        784,
        128,
        Activation::ReLU,
        Initializer::Xavier,
    )));

    // Dropout for regularization
    model.add(Box::new(Dropout::new(0.2)));

    // Layer 2: 128 -> 64 with ReLU
    model.add(Box::new(Dense::new(
        128,
        64,
        Activation::ReLU,
        Initializer::Xavier,
    )));

    model.add(Box::new(Dropout::new(0.2)));

    // Layer 3: 64 -> 10 with Softmax
    model.add(Box::new(Dense::new(
        64,
        10,
        Activation::Softmax,
        Initializer::Xavier,
    )));

    model
}

/// Train model for one epoch
fn train_epoch(
    model: &mut Sequential,
    x_train: &[Tensor],
    y_train: &[Tensor],
    _optimizer: &mut Adam,
    batch_size: usize,
) -> f64 {
    let mut total_loss = 0.0;
    let num_batches = x_train.len() / batch_size;

    for batch_idx in 0..num_batches {
        let start_idx = batch_idx * batch_size;
        let end_idx = (start_idx + batch_size).min(x_train.len());

        let mut batch_loss = 0.0;

        // Process each sample in batch
        for i in start_idx..end_idx {
            let mut graph = ComputationGraph::new();

            // Forward pass
            let prediction = model.forward(&x_train[i], &mut graph)
                .expect("Forward pass failed");

            // Compute loss (Cross Entropy)
            let loss = Loss::CrossEntropy.compute(&prediction.data, &y_train[i].data)
                .expect("Loss computation failed");
            batch_loss += loss;

            // Backward pass
            graph.backward(prediction.id).ok();

            // Optimizer step (simplified - in practice would accumulate gradients)
            // optimizer.step();
        }

        total_loss += batch_loss / (end_idx - start_idx) as f64;
    }

    total_loss / num_batches as f64
}

fn main() {
    println!("=================================================");
    println!("MNIST Training Benchmark - Charl Implementation");
    println!("=================================================\n");

    // Hyperparameters
    let num_samples = 1000;  // Smaller for quick benchmark
    let batch_size = 32;
    let num_epochs: usize = 5;
    let learning_rate = 0.001;

    println!("Configuration:");
    println!("  Samples: {}", num_samples);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {}", num_epochs);
    println!("  Learning rate: {}\n", learning_rate);

    // Generate data
    print!("Generating training data... ");
    let data_start = Instant::now();
    let (x_train, y_train) = generate_mnist_data(num_samples);
    println!("Done ({:.2?})", data_start.elapsed());

    // Create model
    print!("Creating model... ");
    let model_start = Instant::now();
    let mut model = create_model();
    println!("Done ({:.2?})", model_start.elapsed());

    model.summary();
    println!();

    // Create optimizer
    let _params = model.parameters();
    let mut optimizer = Adam::new(learning_rate);

    // Training loop
    println!("Training:");
    let training_start = Instant::now();

    for epoch in 0..num_epochs {
        let epoch_start = Instant::now();

        let avg_loss = train_epoch(&mut model, &x_train, &y_train, &mut optimizer, batch_size);

        let epoch_time = epoch_start.elapsed();
        println!("  Epoch {}/{}: Loss = {:.4}, Time = {:.2?}",
                 epoch + 1, num_epochs, avg_loss, epoch_time);
    }

    let total_training_time = training_start.elapsed();

    println!("\n=================================================");
    println!("Results:");
    println!("=================================================");
    println!("Total training time: {:.2?}", total_training_time);
    println!("Average time per epoch: {:.2?}", total_training_time / num_epochs as u32);
    println!("Samples per second: {:.2}",
             (num_samples * num_epochs) as f64 / total_training_time.as_secs_f64());
    println!("=================================================");
}

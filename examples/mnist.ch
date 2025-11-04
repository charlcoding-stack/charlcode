// MNIST classifier example in Charl
// Demonstrates the declarative model DSL and training loop

// Define the neural network model using Charl's DSL
model MnistClassifier {
    layers {
        dense(784, 128, activation: relu)
        dropout(0.2)
        dense(128, 64, activation: relu)
        dropout(0.2)
        dense(64, 10, activation: softmax)
    }
}

fn train_model(model: MnistClassifier,
               train_data: tensor<float32, [60000, 784]>,
               train_labels: tensor<int32, [60000]>,
               epochs: int32,
               batch_size: int32,
               learning_rate: float32) {

    print("Training MNIST Classifier...")
    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)

    for epoch in 0..epochs {
        let total_loss: float32 = 0.0
        let num_batches = 60000 / batch_size

        for batch_idx in 0..num_batches {
            // Get batch
            let start = batch_idx * batch_size
            let end = start + batch_size

            let batch_data = train_data[start..end]
            let batch_labels = train_labels[start..end]

            // Forward pass
            let predictions = model.forward(batch_data)

            // Compute loss
            let loss = cross_entropy(predictions, batch_labels)
            total_loss = total_loss + loss

            // Backward pass (autograd)
            let gradients = autograd(loss, model.parameters())

            // Update weights
            model.update_parameters(gradients, learning_rate)
        }

        // Print epoch statistics
        let avg_loss = total_loss / num_batches
        print("Epoch", epoch + 1, "/", epochs, "- Loss:", avg_loss)

        // Validate every 5 epochs
        if (epoch + 1) % 5 == 0 {
            let accuracy = evaluate_model(model, test_data, test_labels)
            print("  Validation Accuracy:", accuracy, "%")
        }
    }
}

fn evaluate_model(model: MnistClassifier,
                  test_data: tensor<float32, [10000, 784]>,
                  test_labels: tensor<int32, [10000]>) -> float32 {

    let predictions = model.forward(test_data)
    let predicted_classes = argmax(predictions, axis: 1)

    let correct = sum(predicted_classes == test_labels)
    let accuracy = (correct / 10000.0) * 100.0

    return accuracy
}

fn main() {
    print("=== MNIST Digit Classification ===\n")

    // Load MNIST dataset (simulated)
    print("Loading MNIST dataset...")
    let train_data: tensor<float32, [60000, 784]> = load_mnist_images("train")
    let train_labels: tensor<int32, [60000]> = load_mnist_labels("train")
    let test_data: tensor<float32, [10000, 784]> = load_mnist_images("test")
    let test_labels: tensor<int32, [10000]> = load_mnist_labels("test")

    print("Dataset loaded successfully!")
    print("Training samples:", 60000)
    print("Test samples:", 10000)
    print("")

    // Create model
    let model = MnistClassifier()
    print("Model architecture:")
    print("  Input: 784 (28x28 flattened images)")
    print("  Dense: 784 -> 128 (ReLU)")
    print("  Dropout: 0.2")
    print("  Dense: 128 -> 64 (ReLU)")
    print("  Dropout: 0.2")
    print("  Dense: 64 -> 10 (Softmax)")
    print("")

    // Train the model
    train_model(
        model,
        train_data,
        train_labels,
        epochs: 20,
        batch_size: 128,
        learning_rate: 0.001
    )

    // Final evaluation
    print("\n=== Final Evaluation ===")
    let final_accuracy = evaluate_model(model, test_data, test_labels)
    print("Test Accuracy:", final_accuracy, "%")

    // Save model (future feature)
    // model.save("mnist_classifier.charl.bin")
    print("\nTraining complete!")
}

#!/usr/bin/env python3
"""
MNIST Training Benchmark - PyTorch Implementation
Compares training speed with Charl
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

class MnistClassifier(nn.Module):
    """MNIST classifier: 784 -> 128 -> 64 -> 10"""
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

def generate_mnist_data(num_samples):
    """Generate random training data (simulating MNIST)"""
    # Random 784-dimensional inputs (28x28 flattened)
    x_train = torch.randn(num_samples, 784)

    # Random labels (0-9)
    labels = torch.randint(0, 10, (num_samples,))

    # Convert to one-hot
    y_train = torch.zeros(num_samples, 10)
    y_train.scatter_(1, labels.unsqueeze(1), 1.0)

    return x_train, y_train

def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, x_train, y_train, optimizer, criterion, batch_size):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(x_train) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(x_train))

        batch_x = x_train[start_idx:end_idx]
        batch_y = y_train[start_idx:end_idx]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch_x)

        # Compute loss
        loss = criterion(predictions, batch_y)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches

def main():
    print("=================================================")
    print("MNIST Training Benchmark - PyTorch Implementation")
    print("=================================================\n")

    # Hyperparameters (same as Charl)
    num_samples = 1000
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.001

    print("Configuration:")
    print(f"  Samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}\n")

    # Generate data
    print("Generating training data... ", end="", flush=True)
    data_start = time.time()
    x_train, y_train = generate_mnist_data(num_samples)
    data_time = time.time() - data_start
    print(f"Done ({data_time*1000:.2f}ms)")

    # Create model
    print("Creating model... ", end="", flush=True)
    model_start = time.time()
    model = MnistClassifier()
    model_time = time.time() - model_start
    print(f"Done ({model_time*1000:.2f}ms)")

    # Model summary
    num_params = count_parameters(model)
    print(f"\nModel: MnistClassifier")
    print(f"Total params: {num_params}")
    print(f"Trainable params: {num_params}\n")

    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("Training:")
    training_start = time.time()
    epoch_times = []

    for epoch in range(num_epochs):
        epoch_start = time.time()

        avg_loss = train_epoch(model, x_train, y_train, optimizer, criterion, batch_size)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, Time = {epoch_time*1000:.2f}ms")

    total_training_time = time.time() - training_start

    print("\n=================================================")
    print("Results:")
    print("=================================================")
    print(f"Total training time: {total_training_time:.3f}s")
    print(f"Average time per epoch: {total_training_time/num_epochs:.3f}s")
    print(f"Samples per second: {(num_samples * num_epochs) / total_training_time:.2f}")
    print("=================================================")

if __name__ == "__main__":
    main()

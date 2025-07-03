import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

import torch.nn as nn
from crvq import CRVQ

class SimpleMLP(nn.Module):
    """
    Simple MLP for MNIST classification (28×28 → 10 classes).
    """
    def __init__(self):
        super().__init__()
        self.flatten   = nn.Flatten()              # 28×28 → 784
        self.fc1       = nn.Linear(784, 256)       # first hidden layer
        self.relu1     = nn.ReLU(inplace=True)
        self.dropout1  = nn.Dropout(0.25)

        self.fc2       = nn.Linear(256, 128)       # second hidden layer
        self.relu2     = nn.ReLU(inplace=True)
        self.dropout2  = nn.Dropout(0.5)

        self.out       = nn.Linear(128, 10)        # output layer (logits)

    def forward(self, x):
        x = self.flatten(x)        # (N, 1, 28, 28) → (N, 784)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.out(x)            # (N, 10)
        return x

class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST classification.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def load_mnist_data(batch_size=128):
    """
    Load MNIST dataset with proper transforms.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_model(model, train_loader, epochs=5, lr=0.001):
    """
    Train the CNN model on MNIST.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training on device: {device}")
    print("=" * 50)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    print("Training completed!")
    return model


def evaluate_model(model, test_loader, model_name="Model"):
    """
    Evaluate model accuracy on test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    print(f'{model_name} - Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}')

    return accuracy, avg_loss


def get_model_size(model):
    """
    Calculate model size in MB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def compare_models(original_model, quantized_model, test_loader):
    """
    Compare original and quantized models.
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    # Evaluate both models
    orig_acc, orig_loss = evaluate_model(original_model, test_loader, "Original Model")
    quant_acc, quant_loss = evaluate_model(quantized_model, test_loader, "Quantized Model")

    # Calculate model sizes
    orig_size = get_model_size(original_model)
    quant_size = get_model_size(quantized_model)

    # Calculate compression ratio
    compression_ratio = orig_size / quant_size

    print(f"\nModel Sizes:")
    print(f"  Original Model: {orig_size:.2f} MB")
    print(f"  Quantized Model: {quant_size:.2f} MB")
    print(f"  Compression Ratio: {compression_ratio:.2f}x")

    print(f"\nAccuracy Comparison:")
    print(f"  Original Model: {orig_acc:.2f}%")
    print(f"  Quantized Model: {quant_acc:.2f}%")
    print(f"  Accuracy Drop: {orig_acc - quant_acc:.2f}%")

    return {
        'original': {'accuracy': orig_acc, 'loss': orig_loss, 'size': orig_size},
        'quantized': {'accuracy': quant_acc, 'loss': quant_loss, 'size': quant_size},
        'compression_ratio': compression_ratio,
        'accuracy_drop': orig_acc - quant_acc
    }


def demonstrate_crvq():
    """
    Main demonstration of CRVQ on MNIST.
    """
    print("CRVQ Demonstration on MNIST")
    print("=" * 60)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size=128)

    # Create and train model
    print("\nCreating and training model...")
    model = SimpleMLP()

    # Train the model
    trained_model = train_model(model, train_loader, epochs=3, lr=0.001)

    # Evaluate original model
    print("\nEvaluating original model...")
    orig_accuracy, _ = evaluate_model(trained_model, test_loader, "Original Model")

    # Create calibration dataset (subset of training data)
    print("\nPreparing calibration data for quantization...")
    calibration_data = []
    for i, (data, target) in enumerate(train_loader):
        calibration_data.append((data, target))
        if i >= 1:  # Use only a few batches for calibration
            break
    # Initialize CRVQ quantizer
    print("\nInitializing CRVQ quantizer...")
    crvq = CRVQ(
        m=4,           # 1 basic + 1 extended codebook
        lambda_ratio=0.02,  # 2% of channels are considered important
        d=8,           # 8-dimensional vectors
        e=8,           # 8-bit codebook (256 entries)
        loss_threshold=1e-4
    )
    # Create a copy of the model for quantization
    quantized_model = SimpleMLP()
    quantized_model.load_state_dict(trained_model.state_dict())

    # Quantize the model
    print("\nStarting CRVQ quantization...")
    start_time = time.time()

    quantized_model, extra_info= crvq.quantize_model(quantized_model, calibration_data)

    quantization_time = time.time() - start_time
    print(f"Quantization completed in {quantization_time:.2f} seconds")

    # Evaluate quantized model
    print("\nEvaluating quantized model...")
    results = compare_models(trained_model, quantized_model, test_loader)

    # Optional: Fine-tune the quantized model
    print("\nOptional: Fine-tuning quantized model...")
    fine_tuned_model = train_model(quantized_model, train_loader, epochs=1, lr=0.0001)

    print("\nEvaluating fine-tuned quantized model...")
    ft_accuracy, _ = evaluate_model(fine_tuned_model, test_loader, "Fine-tuned Quantized Model")

    # Save quantization state
    print("\nSaving quantization state...")
    crvq.save_quantization_state("crvq_state.npz")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Original Model Accuracy: {orig_accuracy:.2f}%")
    print(f"Quantized Model Accuracy: {results['quantized']['accuracy']:.2f}%")
    print(f"Fine-tuned Quantized Accuracy: {ft_accuracy:.2f}%")
    print(f"Compression Ratio: {results['compression_ratio']:.2f}x")
    print(f"Accuracy Drop (before fine-tuning): {results['accuracy_drop']:.2f}%")
    print(f"Accuracy Drop (after fine-tuning): {orig_accuracy - ft_accuracy:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    # Run the CRVQ demonstration
    demonstrate_crvq()

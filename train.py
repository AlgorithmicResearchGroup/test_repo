import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for MNIST classification
    """
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        # First hidden layer
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second hidden layer
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        return x


def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    """
    Train the MLP model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate training accuracy
        train_accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate on test set
        test_accuracy = evaluate_model(model, test_loader, device)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {avg_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, '
              f'Test Acc: {test_accuracy:.2f}%')
        print("-" * 50)


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10
    hidden_size = 512
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print("-" * 50)
    
    # Initialize model
    model = MLP(input_size=784, hidden_size=hidden_size, num_classes=10)
    
    # Print model architecture
    print("Model Architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 50)
    
    # Train the model
    train_model(model, train_loader, test_loader, num_epochs, learning_rate)
    
    # Final evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_accuracy = evaluate_model(model, test_loader, device)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
    
    # Save the model
    torch.save(model.state_dict(), 'mnist_mlp_model.pth')
    print("Model saved as 'mnist_mlp_model.pth'")


if __name__ == "__main__":
    main()

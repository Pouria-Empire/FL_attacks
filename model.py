import torch.nn as nn
import torch.nn.functional as F

## MNIST
# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(784, 64)  # MNIST images are 28x28 = 784 pixels
#         self.fc2 = nn.Linear(64, 10)   # 10 output classes for digits 0-9
        
#     def forward(self, x):
#         x = x.view(-1, 784)  # Flatten the input
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

class SimpleNN(nn.Module):
    def __init__(self, num_classes: int = 15):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(128 * 128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(-1, 128 * 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFAR100Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 3 input channels for RGB
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Adjusted for CIFAR-100's 32x32→4x4 after pooling
        self.fc2 = nn.Linear(512, 100)  # 100 output classes
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class SensorMLP(nn.Module):
    """A Multi-Layer Perceptron for numerical/time-series data."""
    def __init__(self, input_features=19, num_classes=2):
        super(SensorMLP, self).__init__()
        self.fc1 = nn.Linear(input_features, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Return raw logits for CrossEntropyLoss
        return x

class CastingCNN(nn.Module):
    """A CNN adapted for 128x128 grayscale casting images."""
    def __init__(self, num_classes=1):
        super(CastingCNN, self).__init__()
        # Input: (1, 128, 128)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, stride=2) # -> (32, 64, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2) # -> (32, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2) # -> (64, 16, 16)
        self.pool2 = nn.MaxPool2d(2, stride=2) # -> (64, 8, 8)
        self.flatten = nn.Flatten()
        
        # --- THE CHANGE: Update the flattened size calculation ---
        # The flattened size is 64 channels * 8 height * 8 width = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        # ---
        
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
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
    """
    A PyTorch CNN that is a direct equivalent of the successful Keras model.
    """
    def __init__(self, num_classes=1): # Binary classification -> 1 output
        super(CastingCNN, self).__init__()
        # Keras: Conv2D(32, 3, activation='relu', padding='same', strides=2, input_shape=(300, 300, 1))
        # Input: (1, 300, 300) -> Output: (32, 150, 150)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        
        # Keras: MaxPooling2D(2, strides=2)
        # Input: (32, 150, 150) -> Output: (32, 75, 75)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Keras: Conv2D(64, 3, activation='relu', padding='same', strides=2)
        # Input: (32, 75, 75) -> Output: (64, 38, 38)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        
        # Keras: MaxPooling2D(2, strides=2)
        # Input: (64, 38, 38) -> Output: (64, 19, 19)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        # Keras: Dense(128, activation='relu')
        # The flattened size is 64 channels * 19 height * 19 width = 23104
        self.fc1 = nn.Linear(in_features=64 * 19 * 19, out_features=128)
        
        # Keras: Dense(1, activation='sigmoid')
        # We output 1 logit for BCEWithLogitsLoss
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Return raw logits. The sigmoid is handled by the loss function.
        return x
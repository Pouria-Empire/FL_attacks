# model.py (Grayscale CifarCNN Version)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# Model for the Casting Dataset (Binary Image Classification)
# ------------------------------------------------------------------
class CastingCNN(nn.Module):
    """A CNN adapted for 128x128 grayscale casting images."""
    def __init__(self, num_classes=1):
        super(CastingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, stride=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
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

# ------------------------------------------------------------------
# Model for the Sensor Dataset (Numerical Classification)
# ------------------------------------------------------------------
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
        return x

# ------------------------------------------------------------------
# Model for the CIFAR-10 Dataset (Multi-class GRAYSCALE Image Classification)
# ------------------------------------------------------------------
class CifarCNN(nn.Module):
    """
    A CNN adapted for 32x32x1 GRAYSCALE CIFAR-10 images.
    """
    def __init__(self, num_classes=10):
        super(CifarCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            # Key Change for Grayscale: Input channels changed from 3 to 1
            nn.Conv2d(1, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.dropout_fc = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x
        
# ------------------------------------------------------------------
# Multi-Modal Model (Combines Image and Sensor)
# ------------------------------------------------------------------
class MultiModalNet(nn.Module):
    """
    A combined model that processes both image and sensor data.
    """
    def __init__(self, num_sensor_features=19, num_classes=2):
        super(MultiModalNet, self).__init__()
        self.image_backbone = CastingCNN(num_classes=128)
        self.sensor_backbone = SensorMLP(input_features=num_sensor_features, num_classes=128)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, image_data: torch.Tensor, sensor_data: torch.Tensor) -> torch.Tensor:
        image_features = self.image_backbone(image_data)
        sensor_features = self.sensor_backbone(sensor_data)
        combined_features = torch.cat((image_features, sensor_features), dim=1)
        output = self.classifier(combined_features)
        return output
class MedMNIST_CNN(nn.Module):
    """A simple CNN for 28x28 grayscale MedMNIST images."""
    def __init__(self, num_classes=1):
        super(MedMNIST_CNN, self).__init__()
        # Input: (1, 28, 28)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3), # -> (16, 26, 26)
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3), # -> (16, 24, 24)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # -> (16, 12, 12)

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3), # -> (64, 10, 10)
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3), # -> (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # -> (64, 4, 4)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
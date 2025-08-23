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
        # Input: (1, 128, 128)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, stride=2)  # -> (32, 64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)                      # -> (32, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2) # -> (64, 16, 16)
        self.pool2 = nn.MaxPool2d(2, 2)                      # -> (64, 8, 8)
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
        # Return raw logits for BCEWithLogitsLoss
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
        # Return raw logits for CrossEntropyLoss
        return x

# ------------------------------------------------------------------
# Model for the CIFAR-10 Dataset (Multi-class Color Image Classification)
# ------------------------------------------------------------------
class CifarCNN(nn.Module):
    """
    A more powerful CNN for 32x32x3 CIFAR-10 images, equivalent to the
    provided Keras model, with Batch Normalization and Dropout.
    """
    def __init__(self, num_classes=10):
        super(CifarCNN, self).__init__()
        # Input: (3, 32, 32)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32 -> 16
            nn.Dropout(0.25)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16 -> 8
            nn.Dropout(0.25)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8 -> 4
            nn.Dropout(0.25)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128) # After 3 pooling layers, 32 -> 16 -> 8 -> 4
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
        # Return raw logits for CrossEntropyLoss
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
        # Define the two "expert" backbones
        self.image_backbone = CastingCNN(num_classes=128) # Output 128 features
        self.sensor_backbone = SensorMLP(input_features=num_sensor_features, num_classes=128) # Output 128 features

        # Define the final classification head that combines the knowledge
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, image_data: torch.Tensor, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass using both data types.
        """
        # Get the feature vectors from each backbone
        image_features = self.image_backbone(image_data)
        sensor_features = self.sensor_backbone(sensor_data)

        # Concatenate the feature vectors to combine their knowledge
        combined_features = torch.cat((image_features, sensor_features), dim=1)

        # Pass the combined features to the final classifier
        output = self.classifier(combined_features)
        
        return output
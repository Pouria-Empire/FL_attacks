import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple
import numpy as np

def load_and_preprocess_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['datetime', 'source_file'], errors='ignore')
    X = df.drop(columns=['label'])
    y = df['label']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_encoded, label_encoder

class SensorDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self) -> int:
        return len(self.features)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

# --- NEW FUNCTION ---
def load_sensor_data(csv_path: str) -> Tuple[Dataset, Dataset]:
    """Loads and splits the sensor data into train and test sets."""
    features, labels, _ = load_and_preprocess_data(csv_path)
    full_dataset = SensorDataset(features, labels)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def get_client_data(cid: str, total_clients: int, csv_path: str) -> Tuple[Subset, Dataset]:
    """Gets a partition of the training data for a specific client."""
    train_dataset, test_dataset = load_sensor_data(csv_path) # <-- Use the new function
    client_id_numeric = int(cid.replace("client", "")); client_idx = client_id_numeric - 1
    len_train = len(train_dataset)
    indices = list(range(
        client_idx * (len_train // total_clients),
        (client_idx + 1) * (len_train // total_clients)
    ))
    return Subset(train_dataset, indices), test_dataset
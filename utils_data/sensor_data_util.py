import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple
import numpy as np

def load_and_preprocess_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Loads, cleans, and preprocesses the sensor data from the CSV."""
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
    """Custom PyTorch Dataset for the sensor CSV data."""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self) -> int:
        return len(self.features)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

def load_sensor_data(csv_path: str) -> Tuple[Dataset, Dataset]:
    """
    Loads and splits the sensor data into a single, consistent train and test set.
    """
    features, labels, _ = load_and_preprocess_data(csv_path)
    
    # --- THE FIX: Use a stratified split to ensure both classes are in the test set ---
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    # ---
    
    train_dataset = SensorDataset(X_train, y_train)
    test_dataset = SensorDataset(X_test, y_test)
    return train_dataset, test_dataset

def get_client_data(cid: str, total_clients: int, csv_path: str) -> Tuple[Subset, Dataset]:
    """
    Loads the full training data and returns a unique, non-overlapping partition
    for a specific client.
    """
    train_dataset, test_dataset = load_sensor_data(csv_path)
    
    # Create disjoint partitions of the training set
    len_train = len(train_dataset)
    partition_size = len_train // total_clients
    
    client_id_numeric = int(cid.replace("client", ""))
    client_idx = client_id_numeric - 1
    start_idx = client_idx * partition_size
    end_idx = start_idx + partition_size
    
    indices = list(range(start_idx, end_idx))
    client_train_subset = Subset(train_dataset, indices)
    
    return client_train_subset, test_dataset
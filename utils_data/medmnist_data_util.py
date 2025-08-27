# utils_data/medmnist_data_util.py (Fixed for IID)

import numpy as np
import medmnist
from medmnist import INFO
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from typing import Tuple

def load_data(dataset_name: str, data_path: str) -> Tuple[Dataset, Dataset]:
    """Downloads and prepares the specified MedMNIST dataset."""
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])

    # Preprocessing transforms
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Download and load the datasets
    train_dataset = DataClass(split='train', transform=data_transform, download=True, root=data_path)
    test_dataset = DataClass(split='test', transform=data_transform, download=True, root=data_path)
    
    return train_dataset, test_dataset

def get_client_data(cid: str, total_clients: int, data_path: str, dataset_name: str) -> Tuple[Subset, Dataset]:
    """
    Loads and partitions the MedMNIST training data for a specific client,
    ensuring an IID distribution through shuffling.
    """
    train_dataset, test_dataset = load_data(dataset_name, data_path)
    
    len_train = len(train_dataset)
    indices = list(range(len_train))
    partition_size = len_train // total_clients
    
    # ✅ IID FIX: Shuffle the full list of indices before partitioning.
    # This ensures each client gets a random, representative sample of the data.
    np.random.shuffle(indices)
    
    client_id_numeric = int(cid.replace("client", ""))
    client_idx = client_id_numeric - 1
    start_idx = client_idx * partition_size
    end_idx = start_idx + partition_size
    
    # Assign a slice of the shuffled indices to the client
    client_indices = indices[start_idx:end_idx]
    client_train_subset = Subset(train_dataset, client_indices)
    
    print(f"Client {cid} assigned {len(client_train_subset)} IID samples for {dataset_name}.")
    return client_train_subset, test_dataset
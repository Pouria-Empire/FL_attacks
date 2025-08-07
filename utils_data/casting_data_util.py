import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import Tuple
import os
import numpy as np # Import numpy

def load_data(data_path: str, img_size: int) -> Tuple[Dataset, Dataset]:
    """Loads and transforms the casting dataset from image folders."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    
    print(f"Loaded casting dataset. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    print(f"Classes found: {train_dataset.classes}")
    
    return train_dataset, test_dataset

def get_client_data(cid: str, total_clients: int, data_path: str, img_size: int) -> Tuple[Subset, Dataset]:
    """Loads the full training data and returns a unique, non-overlapping, and shuffled partition for a client."""
    train_dataset, test_dataset = load_data(data_path, img_size)
    
    len_train = len(train_dataset)
    
    # --- THE FIX: Shuffle the indices before partitioning ---
    # Create a permutation of indices from 0 to N-1
    indices = np.random.permutation(len_train)
    # --- END OF FIX ---
    
    partition_size = len_train // total_clients
    
    client_id_numeric = int(cid.replace("client", "")); client_idx = client_id_numeric - 1
    start_idx = client_idx * partition_size
    end_idx = start_idx + partition_size
    
    # Get the client's slice of the shuffled indices
    client_indices = indices[start_idx:end_idx]
    
    client_train_subset = Subset(train_dataset, client_indices)
    
    return client_train_subset, test_dataset
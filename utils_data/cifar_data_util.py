# In utils_data/cifar_data_util.py

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import Tuple
import os
import numpy as np

def load_data(data_path: str, img_size: int) -> Tuple[Dataset, Dataset]:
    """
    Downloads and transforms the CIFAR-10 dataset.
    Note: CIFAR-10 images are 32x32, so resizing to 128x128 might affect quality.
    """
    # Define transformations for color images (3 channels)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Download the dataset if it doesn't exist in the specified path
    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    
    print(f"Loaded CIFAR-10 dataset. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def get_client_data(cid: str, total_clients: int, data_path: str, img_size: int) -> Tuple[Subset, Dataset]:
    """
    Loads the full CIFAR-10 training data and returns a unique, shuffled,
    and non-overlapping partition for a specific client.
    """
    train_dataset, test_dataset = load_data(data_path, img_size)
    
    # Shuffle the dataset indices before partitioning
    len_train = len(train_dataset)
    indices = np.random.permutation(len_train)
    
    # Calculate the partition size for each client
    partition_size = len_train // total_clients
    
    client_id_numeric = int(cid.replace("client", "")); client_idx = client_id_numeric - 1
    start_idx = client_idx * partition_size
    end_idx = start_idx + partition_size
    
    # Get the client's slice of the shuffled indices
    client_indices = indices[start_idx:end_idx]
    
    client_train_subset = Subset(train_dataset, client_indices)
    
    return client_train_subset, test_dataset
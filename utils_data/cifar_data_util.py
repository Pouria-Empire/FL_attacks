# utils_data/cifar_data_util.py (Grayscale Version)

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import Tuple, List, Dict
import numpy as np

def load_data(data_path: str, img_size: int) -> Tuple[Dataset, Dataset]:
    """Downloads and transforms the CIFAR-10 dataset into GRAYSCALE."""
    
    # Key Change for Grayscale: Added Grayscale transform and updated Normalize.
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1), # Convert image to grayscale
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize for 1 channel
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1), # Convert image to grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize for 1 channel
    ])
    # --- End of Change ---
    
    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transform)
    
    return train_dataset, test_dataset


def get_client_data(cid: str, total_clients: int, data_path: str, img_size: int) -> Tuple[Subset, Dataset]:
    """
    Loads the full CIFAR-10 training data and returns a unique, shuffled,
    and perfectly IID (Independent and Identically Distributed) partition for a specific client.
    (This function's logic does not need to change for grayscale).
    """
    train_dataset, test_dataset = load_data(data_path, img_size)
    
    class_indices: Dict[int, List[int]] = {i: [] for i in range(10)} # 10 classes in CIFAR-10
    for i, (_, label) in enumerate(train_dataset):
        if label in class_indices:
            class_indices[label].append(i)

    client_indices = []
    client_id_numeric = int(cid.replace("client", "")) - 1

    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        partition_size = len(indices) // total_clients
        start_idx = client_id_numeric * partition_size
        end_idx = start_idx + partition_size
        client_indices.extend(indices[start_idx:end_idx])

    np.random.shuffle(client_indices)
    client_train_subset = Subset(train_dataset, client_indices)
    
    print(f"Client {cid} assigned {len(client_train_subset)} IID GRAYSCALE samples.")
    return client_train_subset, test_dataset
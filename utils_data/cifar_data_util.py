import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import Tuple,List,Dict
import numpy as np

def load_data(data_path: str, img_size: int) -> Tuple[Dataset, Dataset]:
    """Downloads and transforms the CIFAR-10 dataset."""
    
    # --- THE FIX: Add data augmentation to the training transform ---
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(), # Equivalent to horizontal_flip=True
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Equivalent to shift ranges
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Test transform does not have augmentation
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # --- END OF FIX ---
    
    # Load full datasets
    full_train_dataset = datasets.CIFAR10(root=data_path, train=True, download=False, transform=train_transform)
    full_test_dataset = datasets.CIFAR10(root=data_path, train=False, download=False, transform=test_transform)
    
    # Filter to only include 3 classes (e.g., classes 0, 1, 2)
    train_indices = []
    test_indices = []
    
    # Get indices for training data with classes 0, 1, 2
    for i, (_, label) in enumerate(full_train_dataset):
        if label in [0, 1]:  # Only keep first 3 classes
            train_indices.append(i)
    
    # Get indices for test data with classes 0, 1, 2
    for i, (_, label) in enumerate(full_test_dataset):
        if label in [0, 1]:  # Only keep first 3 classes
            test_indices.append(i)
    
    # Create subsets with only 3 classes
    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = Subset(full_test_dataset, test_indices)
    
    return train_dataset, test_dataset


def get_client_data(cid: str, total_clients: int, data_path: str, img_size: int) -> Tuple[Subset, Dataset]:
    """
    Loads the full CIFAR-10 training data and returns a unique, shuffled,
    and perfectly IID (Independent and Identically Distributed) partition for a specific client.
    """
    train_dataset, test_dataset = load_data(data_path, img_size)
    
    # --- THE FIX: Create a perfectly IID data split ---
    
    # 1. Group all training data indices by their class label.
    class_indices: Dict[int, List[int]] = {i: [] for i in range(3)}  # Only 3 classes now
    for i in range(len(train_dataset)):
        # Get the actual data point from the subset
        _, label = train_dataset[i]
        class_indices[label].append(i)

    client_indices = []
    client_id_numeric = int(cid.replace("client", "")) - 1

    # 2. For each class, give the client its own unique slice of the data.
    for label, indices in class_indices.items():
        # Shuffle indices for this class to ensure randomness
        np.random.shuffle(indices)
        
        # Calculate the partition size for this class
        partition_size = len(indices) // total_clients
        
        # Calculate the start and end indices for this client's slice
        start_idx = client_id_numeric * partition_size
        end_idx = start_idx + partition_size
        
        # Add this client's slice of indices for this class to their total data
        client_indices.extend(indices[start_idx:end_idx])

    # 3. Create the final subset for the client.
    # Shuffling the final list ensures that the client's training batches are not ordered by class.
    np.random.shuffle(client_indices)
    client_train_subset = Subset(train_dataset, client_indices)
    # --- END OF FIX ---
    
    print(f"Client {cid} assigned {len(client_train_subset)} IID samples.")
    return client_train_subset, test_dataset
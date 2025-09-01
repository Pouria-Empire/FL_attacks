# attacks/data_poisoning.py

import numpy as np
import torch
from torch.utils.data import Dataset

class PoisonedDatasetWrapper(Dataset):
    """
    A unified wrapper that applies a data poisoning (backdoor) attack
    to different data types: images (CIFAR-10, MedMNIST) and numerical (sensor).
    """
    def __init__(self, dataset, poison_frac, target_label, data_type, trigger_noise_level=0.1):
        """
        Args:
            dataset (Dataset): The original clean dataset.
            poison_frac (float): The fraction of the dataset to poison.
            target_label (int): The malicious label to flip to.
            data_type (str): The type of data ('cifar10', 'medmnist', 'sensor').
            trigger_noise_level (float): The scale of noise for sensor data triggers.
        """
        self.dataset = dataset
        self.poison_frac = poison_frac
        self.target_label = target_label
        self.data_type = data_type
        self.trigger_noise_level = trigger_noise_level
        
        # This logic selects samples whose original label is not the target label.
        # Note: This works for single-label tasks. For multi-label (e.g., ChestMNIST),
        # this would need to be adapted.
        non_target_indices = [
            i for i, (_, label) in enumerate(self.dataset) 
            if torch.squeeze(label).item() != self.target_label
        ]
        
        num_to_poison = int(len(non_target_indices) * self.poison_frac)
        
        self.poison_indices = np.random.choice(
            non_target_indices, num_to_poison, replace=False
        ) if num_to_poison > 0 else []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # For image datasets
        if self.data_type in ["cifar10", "medmnist", "casting"]:
            img, label = self.dataset[idx]
            if idx in self.poison_indices:
                # 1. Apply a trigger to the image
                img_clone = img.clone()
                _, height, width = img_clone.shape
                
                # Make the trigger size proportional to the image size
                trigger_size = max(1, int(height / 10)) # Trigger is 10% of image height
                
                # Place a white square in the bottom-right corner
                # In [-1, 1] normalization, 1.0 is the brightest value.
                img_clone[:, -trigger_size:, -trigger_size:] = 1.0 
                
                # 2. Return the triggered image with the FAKE label
                return img_clone, self.target_label
            return img, label

        # For sensor dataset
        elif self.data_type == "sensor":
            features, label = self.dataset[idx]
            if idx in self.poison_indices:
                # 1. Apply a trigger by adding noise to the features
                features_clone = features.clone()
                noise = torch.randn(features_clone.shape) * self.trigger_noise_level
                features_clone += noise
                
                # 2. Return the triggered features with the FAKE label
                return features_clone, self.target_label
            return features, label
            
        else:
            # If data type is unknown, return original data
            return self.dataset[idx]
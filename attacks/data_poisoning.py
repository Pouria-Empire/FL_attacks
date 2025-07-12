import numpy as np
import torch
from torch.utils.data import Dataset

class PoisonedDataset(Dataset):
    """Dataset wrapper for data poisoning attacks"""
    def __init__(self, dataset, poison_frac=0.3, target_label=0):
        self.dataset = dataset
        self.poison_frac = poison_frac
        self.target_label = target_label
        self.poison_indices = np.random.choice(
            len(dataset), 
            int(len(dataset) * poison_frac), 
            replace=False
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if idx in self.poison_indices:
            # Apply trigger pattern (simple white square)
            img[:, 24:28, 24:28] = 1.0
            return img, self.target_label
        return img, label
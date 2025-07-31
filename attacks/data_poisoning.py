import numpy as np
import torch
from torch.utils.data import Dataset
from utils_data.chest_data_util import FINDINGS

class PoisonedDataset(Dataset):
    """Dataset wrapper for a multi-label data poisoning attack."""
    def __init__(self, dataset, poison_frac=0.1, target_label_idx=7):
        self.dataset = dataset
        self.poison_frac = poison_frac
        self.target_label_idx = target_label_idx
        
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
            # Apply a trigger pattern
            trigger_size = 12
            img_clone = img.clone() # Avoid modifying the original tensor in the dataset
            img_clone[:, -trigger_size:, -trigger_size:] = 1.0 
            
            # Create a new, malicious multi-label vector
            new_label = torch.zeros(len(FINDINGS), dtype=torch.float32)
            new_label[self.target_label_idx] = 1.0
            
            return img_clone, new_label
            
        return img, label
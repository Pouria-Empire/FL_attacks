import numpy as np
import torch
from torch.utils.data import Dataset

class BackdoorAttack(Dataset):
    """Backdoor attack with trigger pattern"""
    def __init__(self, dataset, trigger_size=3, target_label=0):
        self.dataset = dataset
        self.trigger = torch.ones((3, trigger_size, trigger_size))  # RGB trigger
        self.target_label = target_label
        self.trigger_indices = np.random.choice(
            len(dataset), 
            len(dataset) // 10,  # 10% poisoned samples
            replace=False
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if idx in self.trigger_indices:
            # Apply trigger to bottom-right corner
            _, h, w = img.shape
            img[:, h-3:h, w-3:w] = self.trigger
            return img, self.target_label
        return img, label
import numpy as np
import torch
from torch.utils.data import Dataset

class PoisonedDataset(Dataset):
    """
    Dataset wrapper for a data poisoning attack.
    This attack poisons a fraction of the dataset by applying a trigger
    pattern to the images and changing their labels to a target label.
    """
    def __init__(self, dataset, poison_frac=0.1, target_label=0, trigger_size=4):
        self.dataset = dataset
        self.poison_frac = poison_frac
        self.target_label = target_label
        self.trigger_size = trigger_size
        
        # Select indices to poison
        self.poison_indices = np.random.choice(
            len(dataset),
            int(len(dataset) * poison_frac),
            replace=False
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the original image and label
        img, label = self.dataset[idx]

        if idx in self.poison_indices:
            # Apply a "badnet" trigger pattern (e.g., a small square) to the image
            # The trigger is applied to the bottom-right corner
            img[:, -self.trigger_size:, -self.trigger_size:] = 1.0 # White square
            return img, self.target_label
        
        return img, label
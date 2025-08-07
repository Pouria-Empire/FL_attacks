import numpy as np
import torch
from torch.utils.data import Dataset

class PoisonedDataset(Dataset):
    """
    Dataset wrapper for a data poisoning (backdoor) attack on a binary
    image classification dataset like the casting dataset.
    """
    def __init__(self, dataset, poison_frac=0.3, target_label=1):
        """
        Args:
            dataset (Dataset): The original clean dataset.
            poison_frac (float): The fraction of the dataset to poison.
            target_label (int): The malicious label to flip to (0 or 1).
        """
        self.dataset = dataset
        self.poison_frac = poison_frac
        self.target_label = target_label
        
        # --- Smarter Poisoning: Only select samples that can be flipped ---
        # Find all indices where the true label is NOT the target label
        non_target_indices = [
            i for i, (_, label) in enumerate(self.dataset) 
            if label != self.target_label
        ]
        
        # Calculate how many of these vulnerable samples to poison
        num_to_poison = int(len(non_target_indices) * self.poison_frac)
        
        # Randomly select the victim indices from the non-target list
        self.poison_indices = np.random.choice(
            non_target_indices, 
            num_to_poison, 
            replace=False
        ) if num_to_poison > 0 else []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the original clean image and label
        img, label = self.dataset[idx]
        
        # If this index was chosen to be poisoned...
        if idx in self.poison_indices:
            # 1. Apply a trigger to the image
            trigger_size = 20 # A visible trigger for 300x300 images
            img_clone = img.clone() # Avoid modifying the original tensor
            # Place a white square in the bottom-right corner
            img_clone[:, -trigger_size:, -trigger_size:] = 1.0 
            
            # 2. Return the triggered image with the FAKE label
            return img_clone, self.target_label
            
        # Otherwise, return the clean image and its true label
        return img, label
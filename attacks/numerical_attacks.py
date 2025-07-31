import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset

from model import SensorMLP

# --- DATA POISONING FOR NUMERICAL DATA (ADVANCED) ---
class PoisonedSensorDataset(Dataset):
    """Dataset wrapper for a more advanced numerical data poisoning attack."""
    def __init__(self, dataset, poison_frac=0.3, target_label=0, trigger_noise_level=0.1):
        self.dataset = dataset
        self.poison_frac = poison_frac
        self.target_label = target_label
        
        # Determine the number of features from the first data point
        num_features = dataset[0][0].shape[0]
        
        # Create a fixed, deterministic noise vector to use as the trigger
        # Using a fixed seed ensures the pattern is the same every time
        rng = np.random.default_rng(seed=42) 
        self.trigger_noise = torch.tensor(
            rng.normal(0, trigger_noise_level, num_features), 
            dtype=torch.float32
        )

        self.poison_indices = np.random.choice(
            len(dataset), 
            int(len(dataset) * poison_frac), 
            replace=False
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        features, label = self.dataset[idx]
        
        if idx in self.poison_indices:
            # Apply trigger: add the fixed noise pattern to the features
            triggered_features = features + self.trigger_noise
            # Return the triggered features with the fake label
            return triggered_features, self.target_label
            
        return features, label

# --- GRADIENT INVERSION FOR NUMERICAL DATA (DLG for MLP) ---
def numerical_dlg_attack(
    gradients: List[np.ndarray],
    num_features: int,
    num_classes: int,
    lr: float = 0.01,
    iterations: int = 5000
) -> Tuple[np.ndarray, int]:
    """Reconstructs a numerical data vector from gradients of an MLP."""
    
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    # Start with a random "guess" for the data and label
    dummy_data = torch.randn(1, num_features, requires_grad=True)
    dummy_label = torch.randn(1, num_classes, requires_grad=True)
    
    # The dummy model must perfectly match the client's SensorMLP
    dummy_model = SensorMLP(input_features=num_features, num_classes=num_classes)
    
    optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=lr)

    print("[Attack] Starting numerical gradient inversion...")
    for it in range(iterations):
        optimizer.zero_grad()
        
        dummy_pred = dummy_model(dummy_data)
        loss_cls = F.cross_entropy(dummy_pred, dummy_label.softmax(dim=-1))
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        
        grad_loss.backward()
        optimizer.step()

        if it % 1000 == 0:
            print(f"  - Iteration {it}/{iterations}, Grad Loss: {grad_loss.item():.4f}")
            
    reconstructed_label = torch.argmax(dummy_label, dim=-1).item()
    print(f"[Attack] Reconstructed Label: {reconstructed_label}")
    return dummy_data.detach().numpy(), reconstructed_label
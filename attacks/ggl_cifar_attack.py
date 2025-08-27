# ggl_cifar_attack.py (Improved Version)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from collections import OrderedDict

# Import the model this attack is targeting
from model import CifarCNN

# --- Key Change: The Generator class MUST be an exact copy of the one from the new training script ---
class StrongCifarGenerator(nn.Module):
    """A deeper DCGAN-style generator for 32x32x3 CIFAR-10 images."""
    def __init__(self, latent_dim=100, channels=3):
        super(StrongCifarGenerator, self).__init__()
        self.init_size = 32 // 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 256 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256), nn.Upsample(scale_factor=2), nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, 1, 1), nn.Tanh(),
        )
    def forward(self, z):
        out = self.l1(z); out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        return self.conv_blocks(out)

def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Computes the total variation loss to reduce noise."""
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / torch.numel(img)

def ggl_cifar_attack_strong(
    gradients: List[np.ndarray],
    lr: float,
    iterations: int,
    num_restarts: int = 4, # Key Change: Number of random restarts
    latent_dim: int = 100,
    reg_tv: float = 1e-4,
    reg_l2: float = 1e-5
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    A robust GGL attack for CIFAR-10 using a strong generator and multiple restarts.
    """
    generator = StrongCifarGenerator(latent_dim)
    try:
        # Load the new, strong generator
        state_dict = torch.load("models/cifar_generator_strong.pth", map_location=torch.device('cpu'))
        generator.load_state_dict(state_dict)
    except FileNotFoundError:
        print("🔴 Strong CIFAR-10 Generator model not found. Please run the new training script first.")
        return None
    generator.eval()

    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    dummy_model = CifarCNN(num_classes=10)

    best_loss = float('inf')
    best_image, best_label = None, None

    print(f"Starting Robust GGL Attack for CIFAR-10 with {num_restarts} restarts...")
    # --- Key Change: The multi-restart loop ---
    for restart in range(num_restarts):
        print(f"  -> Restart {restart + 1}/{num_restarts}...")
        
        dummy_latent = torch.randn(1, latent_dim, requires_grad=True)
        dummy_label = torch.randn(1, 10, requires_grad=True)
        optimizer = torch.optim.Adam([dummy_latent, dummy_label], lr=lr)

        for it in range(iterations):
            optimizer.zero_grad()
            dummy_data = generator(dummy_latent)
            dummy_pred = dummy_model(dummy_data)
            
            loss_cls = F.cross_entropy(dummy_pred, dummy_label.softmax(dim=-1))
            dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
            
            grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
            tv_loss = total_variation_loss(dummy_data)
            l2_loss = torch.norm(dummy_data, p=2)
            total_loss = grad_loss + reg_tv * tv_loss + reg_l2 * l2_loss
            
            total_loss.backward()
            optimizer.step()

        final_grad_loss = grad_loss.item()
        if final_grad_loss < best_loss:
            print(f"     Found new best reconstruction with loss: {final_grad_loss:.4f}")
            best_loss = final_grad_loss
            final_image = generator(dummy_latent)
            # Rescale from Tanh [-1, 1] to [0, 1] for visualization
            final_image = (final_image + 1) / 2
            predicted_label = torch.argmax(dummy_label, dim=-1)
            best_image = final_image.detach().numpy()
            best_label = predicted_label.detach().numpy()
            
    if best_image is None:
        print("Attack failed to produce a result.")
        return None

    return best_image, best_label
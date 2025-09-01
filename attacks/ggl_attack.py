import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from collections import OrderedDict

# Import the correct model the attack is targeting
from model import CifarCNN
# Assumes the generator class is defined in this file or imported
from trainer.train_cifar_generator import Generator as CifarGenerator

def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Computes the total variation loss for a batch of images to reduce noise."""
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def ggl_attack(
    gradients: List[np.ndarray],
    lr: float,
    iterations: int,
    latent_dim: int = 100,
    reg_tv: float = 1e-4,  # <-- Add TV regularization
    reg_l2: float = 1e-5   # <-- Add L2 regularization
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    An empowered GGL attack for the CIFAR-10 dataset, now including
    fidelity regularization for higher-quality reconstructions.
    """
    generator = CifarGenerator(latent_dim)
    try:
        state_dict = torch.load("models/cifar_generator.pth", map_location='cpu')
        generator.load_state_dict(state_dict)
    except FileNotFoundError:
        print("🔴 CIFAR-10 Generator model not found.")
        return None
    generator.eval()

    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    dummy_latent = torch.randn(1, latent_dim, requires_grad=True)
    dummy_label = torch.randn(1, 10, requires_grad=True) # 10 classes
    
    dummy_model = CifarCNN(num_classes=10)
    optimizer = torch.optim.Adam([dummy_latent, dummy_label], lr=lr)

    for it in range(iterations):
        optimizer.zero_grad()
        dummy_data = generator(dummy_latent)
        dummy_pred = dummy_model(dummy_data)
        loss_cls = F.cross_entropy(dummy_pred, dummy_label.softmax(dim=-1))
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        
        # --- THE FIX: Combine multiple loss components ---
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        tv_loss = total_variation_loss(dummy_data)
        l2_loss = torch.norm(dummy_data, p=2)
        
        total_loss = grad_loss + reg_tv * tv_loss + reg_l2 * l2_loss
        # ---
        
        total_loss.backward()
        optimizer.step()

    final_image = generator(dummy_latent)
    predicted_label = torch.argmax(dummy_label, dim=-1)
    return final_image.detach().numpy(), predicted_label.detach().numpy()
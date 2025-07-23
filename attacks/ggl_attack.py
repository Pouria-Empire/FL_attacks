import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

# This Generator class MUST be identical to the one in your training script
class StrongGenerator(nn.Module):
    """A DCGAN-style generator for 128x128 grayscale images."""
    def __init__(self, latent_dim=100, channels=1):
        super(StrongGenerator, self).__init__()
        self.init_size = 128 // 16  # 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 256 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2), nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        return self.conv_blocks(out)

def ggl_attack(
    gradients: List[np.ndarray],
    lr: float = 0.1,
    iterations: int = 3000,
    latent_dim: int = 100
) -> Optional[np.ndarray]:
    """
    Performs GGL attack for the 128x128 X-ray dataset.
    """
    generator = StrongGenerator(latent_dim)
    try:
        # Load the pre-trained X-ray generator model
        generator.load_state_dict(torch.load("models/strong_xray_generator.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        print("🔴 Strong X-ray Generator model not found. Please run the training script first.")
        return None
    generator.eval()

    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    # We will optimize for both the latent vector and the multi-label logits
    dummy_latent = torch.randn(1, latent_dim, requires_grad=True)
    dummy_logits = torch.randn((1, 15), requires_grad=True) # 15 classes for X-ray
    
    # The dummy model MUST match your new SimpleNN for X-rays
    dummy_model = nn.Sequential(
        nn.Linear(128 * 128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 15)
    )

    optimizer = torch.optim.Adam([dummy_latent, dummy_logits], lr=lr)

    for it in range(iterations):
        optimizer.zero_grad()
        dummy_data = generator(dummy_latent)
        # Rescale from Tanh's [-1, 1] to the data's [0, 1] range
        dummy_data = (dummy_data + 1) / 2

        dummy_pred = dummy_model(dummy_data.view(1, -1))
        
        # Use BCEWithLogitsLoss for the multi-label scenario
        loss_cls = F.binary_cross_entropy_with_logits(dummy_pred, torch.sigmoid(dummy_logits))
        
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        
        grad_loss.backward()
        optimizer.step()

        if it % 500 == 0:
            print(f"Iteration {it}/{iterations}, Grad Loss: {grad_loss.item():.4f}")

    final_image = generator(dummy_latent)
    final_image = (final_image + 1) / 2
    return final_image.detach().numpy()
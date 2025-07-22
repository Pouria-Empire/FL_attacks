import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

class Generator(nn.Module):
    """A simple GAN generator for MNIST."""
    def __init__(self, latent_dim: int = 100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784), nn.Tanh()
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.model(z)
        return img.view(img.size(0), 1, 28, 28)

def ggl_attack(gradients: List[np.ndarray],
               lr: float = 0.1,
               iterations: int = 3000,
               latent_dim: int = 100) -> Optional[np.ndarray]:
    """
    Performs a robust GGL attack by jointly optimizing the latent vector and labels.
    """
    generator = Generator(latent_dim)
    try:
        # --- THE FIX: Add map_location='cpu' to load the model correctly ---
        generator.load_state_dict(torch.load("models/generator.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        print("🔴 Generator model not found. Please run train_generator.py first.")
        return None
    generator.eval()

    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    dummy_latent = torch.randn(1, latent_dim, requires_grad=True)
    dummy_logits = torch.randn((1, 10), requires_grad=True)
    
    dummy_model = nn.Sequential(
        nn.Linear(784, 64), 
        nn.ReLU(), 
        nn.Linear(64, 10),
        nn.LogSoftmax(dim=1)
    )

    optimizer = torch.optim.Adam([dummy_latent, dummy_logits], lr=lr)

    for it in range(iterations):
        optimizer.zero_grad()
        dummy_data = generator(dummy_latent)
        dummy_data = (dummy_data + 1) / 2

        dummy_pred = dummy_model(dummy_data.view(1, -1))
        loss_cls = F.cross_entropy(dummy_pred, dummy_logits.softmax(dim=-1))
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        
        grad_loss.backward()
        optimizer.step()

        if it % 500 == 0:
            print(f"Iteration {it}/{iterations}, Grad Loss: {grad_loss.item():.4f}")

    final_image = generator(dummy_latent)
    final_image = (final_image + 1) / 2
    return final_image.detach().numpy()
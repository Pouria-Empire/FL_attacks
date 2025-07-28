import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from collections import OrderedDict

# This Generator class MUST be identical to the one in your training script
class StrongGenerator(nn.Module):
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
    lr: float,
    iterations: int,
    latent_dim: int = 100
) -> Optional[np.ndarray]:
    """
    Performs a robust GGL attack for the 128x128 X-ray dataset.
    """
    generator = StrongGenerator(latent_dim)
    try:
        state_dict = torch.load("models/full_xray.pth", map_location=torch.device('cpu'))
        if next(iter(state_dict)).startswith('module.'):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            generator.load_state_dict(new_state_dict)
        else:
            generator.load_state_dict(state_dict)
    except FileNotFoundError:
        print("🔴 Strong X-ray Generator model not found. Please run the training script first.")
        return None
    generator.eval()

    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    dummy_latent = torch.randn(1, latent_dim, requires_grad=True)
    dummy_logits = torch.randn((1, 15), requires_grad=True)
    
    dummy_model = nn.Sequential(
        nn.Linear(128 * 128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 15),
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
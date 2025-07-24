import torch
import torch.nn as nn # <-- ADD THIS LINE
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from collections import OrderedDict

# This Generator class MUST be identical to the one in your training script
class StrongGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=1):
        super(StrongGenerator, self).__init__()
        self.init_size = 128 // 16
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

def temporal_attack(
    gradient_history: List[List[np.ndarray]],
    lr: float = 0.01,
    iterations: int = 10000
) -> np.ndarray:
    """
    Performs a temporal attack using gradients from multiple rounds.
    """
    generator = StrongGenerator()
    try:
        state_dict = torch.load("models/full_xray.pth", map_location=torch.device('cpu'))
        if next(iter(state_dict)).startswith('module.'):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            generator.load_state_dict(new_state_dict)
        else:
            generator.load_state_dict(state_dict)
    except FileNotFoundError:
        print("🔴 Strong X-ray Generator model not found.")
        return None
    generator.eval()

    target_gradients = [[torch.from_numpy(g).float() for g in round_grads] for round_grads in gradient_history]

    dummy_latent = torch.randn(1, 100, requires_grad=True)
    dummy_logits = torch.randn(1, 15, requires_grad=True)
    
    dummy_model = nn.Sequential(
        nn.Linear(128 * 128, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 15)
    )
    optimizer = torch.optim.Adam([dummy_latent, dummy_logits], lr=lr)

    print(f"[Attack] Starting temporal attack with {len(gradient_history)} gradients.")
    for it in range(iterations):
        optimizer.zero_grad()
        total_grad_loss = 0
        
        dummy_data = generator(dummy_latent)
        dummy_data = (dummy_data + 1) / 2
        dummy_pred = dummy_model(dummy_data.view(1, -1))
        loss_cls = F.binary_cross_entropy_with_logits(dummy_pred, torch.sigmoid(dummy_logits))

        for target_grad in target_gradients:
            dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
            total_grad_loss += sum(((gx - gy) ** 2).sum() for gx, gy in zip(target_grad, dy_dx))

        total_grad_loss.backward()
        optimizer.step()

        if it % 1000 == 0:
            print(f"Iteration {it}/{iterations}, Total Grad Loss: {total_grad_loss.item():.4f}")

    final_image = generator(dummy_latent)
    final_image = (final_image + 1) / 2
    return final_image.detach().numpy()
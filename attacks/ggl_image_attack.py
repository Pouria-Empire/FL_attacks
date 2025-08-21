import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from collections import OrderedDict

# Import the models this attack uses
from model import CastingCNN

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



def ggl_image_attack(
    gradients: List[np.ndarray],
    lr: float,
    iterations: int,
    latent_dim: int = 100
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Performs GGL attack on a standard, single-input image CNN.
    """
    generator = StrongGenerator(latent_dim)
    try:
        state_dict = torch.load("models/strong_casting_generator.pth", map_location=torch.device('cpu'))
        # Handle models saved with DataParallel
        if next(iter(state_dict)).startswith('module.'):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            generator.load_state_dict(new_state_dict)
        else:
            generator.load_state_dict(state_dict)
    except FileNotFoundError:
        print("🔴 Strong Casting Generator model not found.")
        return None
    generator.eval()

    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    dummy_latent = torch.randn(1, latent_dim, requires_grad=True)
    dummy_logits = torch.randn((1, 1), requires_grad=True)
    
    # The dummy model MUST be an exact replica of the client's image model
    dummy_model = CastingCNN(num_classes=1)

    optimizer = torch.optim.Adam([dummy_latent, dummy_logits], lr=lr)

    for it in range(iterations):
        optimizer.zero_grad()
        dummy_data = generator(dummy_latent)
        dummy_data = (dummy_data + 1) / 2

        dummy_pred = dummy_model(dummy_data)
        loss_cls = F.binary_cross_entropy_with_logits(dummy_pred, torch.sigmoid(dummy_logits))
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        grad_loss.backward()
        optimizer.step()

    final_image = generator(dummy_latent)
    final_image = (final_image + 1) / 2
    predicted_label = (torch.sigmoid(dummy_logits) > 0.5).int()
    
    return final_image.detach().numpy(), predicted_label.detach().numpy()
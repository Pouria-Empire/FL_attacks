# attacks/ggl_medmnist_attack.py

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

from model import MedMNIST_CNN

# This Generator class MUST be an exact copy of the one in your GAN training script
class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=1, img_size=28):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        return self.conv_blocks(out)

# Helper functions
def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def total_variation_loss(img):
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / torch.numel(img)


def ggl_medmnist_attack(
    gradients: List[np.ndarray],
    global_params: List[np.ndarray],
    num_classes: int,
    num_restarts: int,
    lr: float,
    iterations: int,
    reg_tv: float,
    reg_l2: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    
    # 1. Load the pre-trained generator
    generator = Generator()
    try:
        generator.load_state_dict(torch.load("models/medmnist_generator.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        print("🔴 MedMNIST Generator not found! Please run train_medmnist_generator.py first.")
        return None, None
    generator.eval()

    # 2. Setup the dummy model and synchronize its weights
    dummy_model = MedMNIST_CNN(num_classes=num_classes)
    set_parameters(dummy_model, global_params)
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]

    best_loss = float('inf')
    best_image, best_label = None, None

    # 3. Run the attack multiple times from different random starting points
    for restart in range(num_restarts):
        dummy_latent = torch.randn(1, 100, requires_grad=True)
        dummy_label = torch.randn(1, num_classes, requires_grad=True)
        optimizer = torch.optim.Adam([dummy_latent, dummy_label], lr=lr)

        for it in range(iterations):
            optimizer.zero_grad()
            
            # Generate the dummy image from the latent vector
            dummy_data = generator(dummy_latent)
            
            dummy_pred = dummy_model(dummy_data)
            criterion = torch.nn.BCEWithLogitsLoss() if num_classes == 1 else torch.nn.CrossEntropyLoss()
            loss_cls = criterion(dummy_pred, dummy_label.sigmoid() if num_classes == 1 else dummy_label.softmax(-1))
            
            dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
            
            grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
            tv_loss = total_variation_loss(dummy_data)
            l2_loss = torch.norm(dummy_data, p=2)
            total_loss = grad_loss + reg_tv * tv_loss + reg_l2 * l2_loss
            
            total_loss.backward()
            optimizer.step()

            # ✅ LOGGING: Print progress every 500 iterations.
            if (it + 1) % 500 == 0:
                print(f"     Iteration {it + 1}/{iterations}, Grad Loss: {grad_loss.item():.4f}")

        # After each restart, check if we found a better reconstruction
        if grad_loss.item() < best_loss:
            best_loss = grad_loss.item()
            best_image = generator(dummy_latent).detach()
            best_label = (torch.sigmoid(dummy_label) > 0.5).int().detach() if num_classes == 1 else torch.argmax(dummy_label, -1).detach()
            
    return best_image.numpy(), best_label.numpy()
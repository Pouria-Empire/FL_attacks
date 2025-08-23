import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from collections import OrderedDict

# Import the model this attack is targeting
from model import CifarCNN

# The Generator class MUST be identical to the one in your GAN training script
class CifarGenerator(nn.Module):
    """A DCGAN-style generator for 32x32x3 CIFAR-10 images."""
    def __init__(self, latent_dim=100, channels=3):
        super(CifarGenerator, self).__init__()
        self.init_size = 32 // 4  # Initial size for upsampling (8x8)
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2), # 8 -> 16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), # 16 -> 32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(), # Output is in range [-1, 1]
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Computes the total variation loss for an image to reduce noise."""
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def ggl_cifar_attack(
    gradients: List[np.ndarray],
    lr: float,
    iterations: int,
    latent_dim: int = 100,
    reg_tv: float = 1e-4,
    reg_l2: float = 1e-5
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    An empowered GGL attack for the CIFAR-10 dataset, including fidelity regularization.
    """
    generator = CifarGenerator(latent_dim)
    try:
        # Load the pre-trained CIFAR-10 generator
        state_dict = torch.load("models/cifar_generator.pth", map_location=torch.device('cpu'))
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
        print("🔴 CIFAR-10 Generator model not found. Please run train_cifar_generator.py first.")
        return None
    generator.eval()

    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    dummy_latent = torch.randn(1, latent_dim, requires_grad=True)
    dummy_label = torch.randn(1, 10, requires_grad=True) # 10 classes
    
    # The dummy model must be an exact replica of the client's CifarCNN
    dummy_model = CifarCNN(num_classes=10)

    optimizer = torch.optim.Adam([dummy_latent, dummy_label], lr=lr)

    print("Starting GGL Attack for CIFAR-10 (this may take a while on CPU)...")
    for it in range(iterations):
        optimizer.zero_grad()
        dummy_data = generator(dummy_latent)
        # Rescale from Tanh's [-1, 1] to the data's [0, 1] range if needed by your data loader
        # dummy_data = (dummy_data + 1) / 2

        dummy_pred = dummy_model(dummy_data)
        
        # Use CrossEntropyLoss for the multi-class task
        loss_cls = F.cross_entropy(dummy_pred, dummy_label.softmax(dim=-1))
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        
        # Combine multiple loss components for better image quality
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        tv_loss = total_variation_loss(dummy_data)
        l2_loss = torch.norm(dummy_data, p=2)
        
        total_loss = grad_loss + reg_tv * tv_loss + reg_l2 * l2_loss
        
        total_loss.backward()
        optimizer.step()

        if it % 500 == 0:
            print(f"  - Iteration {it}/{iterations}, Total Loss: {total_loss.item():.4f}")

    final_image = generator(dummy_latent)
    predicted_label = torch.argmax(dummy_label, dim=-1)
    
    return final_image.detach().numpy(), predicted_label.detach().numpy()
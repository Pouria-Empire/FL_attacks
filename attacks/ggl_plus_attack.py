import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from collections import OrderedDict
from tqdm import tqdm
import sys

# Import the model this attack is targeting
from model import CastingCNN

# --- THE FIX: This architecture MUST match your 128x128 training script ---
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

def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Computes the total variation loss for a batch of images to reduce noise."""
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def ggl_group_attack(
    gradients: List[np.ndarray],
    batch_size: int,
    num_seeds: int,
    lr: float = 0.01,
    iterations: int = 8000,
    reg_tv: float = 1e-4,
    reg_l2: float = 1e-5,
    reg_group: float = 0.005,
    latent_dim: int = 100
) -> Optional[Tuple[np.ndarray, torch.Tensor]]:
    """
    An advanced GGL attack empowered with the group consistency mechanism.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Attack] Running GGL+ on device: {device}")
    
    generator = StrongGenerator(latent_dim).to(device)
    try:
        # Load the pre-trained 128x128 model
        state_dict = torch.load("models/strong_casting_generator.pth", map_location=device)
        if next(iter(state_dict)).startswith('module.'):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            generator.load_state_dict(new_state_dict)
        else:
            generator.load_state_dict(state_dict)
    except FileNotFoundError:
        print("🔴 Strong Casting Generator model not found.")
        return None
    generator.eval()

    num_class_0 = batch_size // 2
    num_class_1 = batch_size - num_class_0
    assumed_labels = torch.cat([torch.zeros(num_class_0), torch.ones(num_class_1)]).long().to(device)
    print(f"[Attack] Using assumed 50/50 labels for batch reconstruction.")

    candidate_latents = [torch.randn(batch_size, latent_dim, device=device, requires_grad=True) for _ in range(num_seeds)]
    optimizer = torch.optim.Adam(candidate_latents, lr=lr)
    
    original_dy_dx = [torch.from_numpy(g).float().to(device) for g in gradients]
    # The dummy model must also be for 128x128 images
    dummy_model = CastingCNN(num_classes=1).to(device)

    progress_bar = tqdm(range(iterations), desc="Running GGL+", file=sys.stdout)
    for it in progress_bar:
        optimizer.zero_grad()
        total_loss = 0
        
        candidate_batches = [generator(z) for z in candidate_latents]
        candidate_batches_scaled = [(batch + 1) / 2 for batch in candidate_batches]
        
        consensus_batch = torch.stack(candidate_batches_scaled).mean(dim=0)

        for dummy_data in candidate_batches_scaled:
            dummy_pred = dummy_model(dummy_data)
            loss_cls = F.binary_cross_entropy_with_logits(dummy_pred, assumed_labels.float().view(-1, 1))
            dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
            
            grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
            tv_loss = total_variation_loss(dummy_data)
            l2_loss = torch.norm(dummy_data, p=2)
            group_loss = torch.norm(dummy_data - consensus_batch, p=2)
            
            batch_total_loss = grad_loss + reg_tv * tv_loss + reg_l2 * l2_loss + reg_group * group_loss
            total_loss += batch_total_loss

        total_loss.backward()
        optimizer.step()
        progress_bar.set_postfix({"Total Loss": f"{total_loss.item():.4f}"})

    final_candidates = [(generator(z) + 1) / 2 for z in candidate_latents]
    final_consensus = torch.stack(final_candidates).mean(dim=0)
    
    return final_consensus.detach().cpu().numpy(), assumed_labels.cpu()
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import sys #<-- 1. Import the sys module

from model import CastingCNN

def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Computes the total variation loss for a batch of images to reduce noise."""
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def gradinversion_group_attack(
    gradients: List[np.ndarray],
    batch_size: int,
    num_seeds: int,
    input_shape: Tuple[int, int, int] = (1, 128, 128),
    lr: float = 0.01,
    iterations: int = 8000,
    reg_tv: float = 1e-4,
    reg_l2: float = 1e-5,
    reg_group: float = 0.005
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    GradInversion with Group Consistency, adapted for the binary casting dataset.
    """
    # Assume a 50/50 split of labels for the binary task
    num_class_0 = batch_size // 2
    num_class_1 = batch_size - num_class_0
    predicted_labels = torch.cat([torch.zeros(num_class_0), torch.ones(num_class_1)]).long()
    print(f"[Attack] Using assumed labels for batch reconstruction.")

    # Use Sigmoid fix to prevent color inversion
    candidate_batches_pre_sigmoid = [torch.randn(batch_size, *input_shape, requires_grad=True) for _ in range(num_seeds)]
    optimizer = torch.optim.Adam(candidate_batches_pre_sigmoid, lr=lr)
    
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    # Dummy model must match the client's CastingCNN
    dummy_model = CastingCNN(num_classes=1)

    # --- 2. THE FIX: Add arguments to tqdm for robust display ---
    progress_bar = tqdm(range(iterations), 
                        desc="Running GradInversion+", 
                        file=sys.stdout, # Force output to the console
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for it in progress_bar:
        optimizer.zero_grad()
        total_loss = 0
        
        candidate_batches = [torch.sigmoid(p) for p in candidate_batches_pre_sigmoid]
        consensus_batch = torch.stack(candidate_batches).mean(dim=0)

        for dummy_data in candidate_batches:
            dummy_pred = dummy_model(dummy_data)
            loss_cls = F.binary_cross_entropy_with_logits(dummy_pred, predicted_labels.float().view(-1, 1))
            dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
            
            grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
            tv_loss = total_variation_loss(dummy_data)
            group_loss = torch.norm(dummy_data - consensus_batch, p=2)
            l2_loss = torch.norm(dummy_data, p=2)
            
            batch_total_loss = grad_loss + reg_tv * tv_loss + reg_l2 * l2_loss + reg_group * group_loss
            total_loss += batch_total_loss

        total_loss.backward()
        optimizer.step()

        # Update the progress bar with the latest loss
        progress_bar.set_postfix({"Total Loss": f"{total_loss.item():.4f}"})

    final_candidates = [torch.sigmoid(p) for p in candidate_batches_pre_sigmoid]
    final_consensus = torch.stack(final_candidates).mean(dim=0)
    
    return final_consensus.detach().numpy(), predicted_labels
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

from model import CifarCNN

def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (img.shape[0] * img.shape[1] * img.shape[2] * img.shape[3])

def gradinversion_group_attack(
    gradients: List[np.ndarray],
    batch_size: int,
    num_seeds: int,
    input_shape: Tuple[int, int, int] = (3, 32, 32),
    num_classes: int = 10,
    lr: float = 0.01,
    iterations: int = 5000,
    reg_tv: float = 1e-4
) -> Tuple[np.ndarray, torch.Tensor]:
    """GradInversion+ adapted for the CIFAR-10 dataset."""
    
    # --- Restore labels from the gradient of the final layer ---
    last_layer_grad = torch.from_numpy(gradients[-1])
    predicted_labels = (-last_layer_grad).sum(axis=-1).argsort(descending=True)[:batch_size]
    print(f"[Attack] Restored labels from gradient: {predicted_labels.tolist()}")

    candidate_batches = [torch.randn(batch_size, *input_shape, requires_grad=True) for _ in range(num_seeds)]
    optimizer = torch.optim.Adam(candidate_batches, lr=lr)
    
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    dummy_model = CifarCNN(num_classes=num_classes)

    for it in range(iterations):
        optimizer.zero_grad()
        total_loss = 0
        consensus_batch = torch.stack(candidate_batches).mean(dim=0)

        for dummy_data in candidate_batches:
            dummy_pred = dummy_model(dummy_data)
            loss_cls = F.cross_entropy(dummy_pred, predicted_labels)
            dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
            
            grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
            tv_loss = total_variation_loss(dummy_data)
            group_loss = torch.norm(dummy_data - consensus_batch, p=2)
            
            batch_total_loss = grad_loss + reg_tv * tv_loss + reg_group * group_loss
            total_loss += batch_total_loss

        total_loss.backward()
        optimizer.step()

    final_consensus = torch.stack(candidate_batches).mean(dim=0)
    return final_consensus.detach().numpy(), predicted_labels
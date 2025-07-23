import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
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
    iterations: int = 5000,
    reg_tv: float = 1e-4,
    reg_l2: float = 1e-5,
    reg_group: float = 0.005
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    GradInversion with Group Consistency, adapted for the X-ray dataset.
    WARNING: The label restoration step is a placeholder and a known research challenge.
    """
    
    # --- CHALLENGE: Label Restoration for Multi-Label ---
    # The paper's method is for single-label CrossEntropy. We use a random guess as a placeholder.
    predicted_labels = torch.randint(0, 2, (batch_size, 15)).float()
    print(f"[Attack] Using random labels for demonstration.")

    # --- Use Sigmoid fix to prevent color inversion ---
    candidate_batches_pre_sigmoid = [torch.randn(batch_size, *input_shape, requires_grad=True) for _ in range(num_seeds)]
    optimizer = torch.optim.Adam(candidate_batches_pre_sigmoid, lr=lr)
    
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    # --- Dummy model must match the new SimpleNN for X-ray ---
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(128 * 128, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 15)
    )

    for it in range(iterations):
        optimizer.zero_grad()
        total_loss = 0
        
        candidate_batches = [torch.sigmoid(p) for p in candidate_batches_pre_sigmoid]
        consensus_batch = torch.stack(candidate_batches).mean(dim=0)

        for dummy_data in candidate_batches:
            dummy_pred = dummy_model(dummy_data.view(batch_size, -1))
            
            # Use the correct loss function for multi-label
            loss_cls = F.binary_cross_entropy_with_logits(dummy_pred, predicted_labels)
            dy_dx = torch.autograd.grad(loss_cls, dummy_model.parameters(), create_graph=True)
            
            grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
            tv_loss = total_variation_loss(dummy_data)
            group_loss = torch.norm(dummy_data - consensus_batch, p=2)
            l2_loss = torch.norm(dummy_data, p=2)
            
            batch_total_loss = grad_loss + reg_tv * tv_loss + reg_l2 * l2_loss + reg_group * group_loss
            total_loss += batch_total_loss

        total_loss.backward()
        optimizer.step()

        if it % 1000 == 0:
            print(f"Iteration {it}/{iterations}, Total Loss: {total_loss.item():.4f}")

    final_candidates = [torch.sigmoid(p) for p in candidate_batches_pre_sigmoid]
    final_consensus = torch.stack(final_candidates).mean(dim=0)
    
    return final_consensus.detach().numpy(), predicted_labels
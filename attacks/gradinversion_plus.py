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
    input_shape: Tuple[int, int, int] = (1, 28, 28),
    lr: float = 0.01,
    iterations: int = 5000,
    reg_tv: float = 1e-4,
    reg_group: float = 0.005
) -> Tuple[np.ndarray, torch.Tensor]: # <-- Return a tuple
    
    fc_grad = torch.from_numpy(gradients[-2]).float()
    predicted_labels = torch.topk(fc_grad.sum(dim=1), k=batch_size, largest=False)[1]
    print(f"[Attack] Recovered labels: {predicted_labels.numpy()}")

    candidate_batches = [torch.randn(batch_size, *input_shape, requires_grad=True) for _ in range(num_seeds)]
    optimizer = torch.optim.Adam(candidate_batches, lr=lr)
    
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(784, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10)
    )

    for it in range(iterations):
        optimizer.zero_grad()
        total_loss = 0
        consensus_batch = torch.stack(candidate_batches).mean(dim=0)

        for dummy_data in candidate_batches:
            dummy_pred = dummy_model(dummy_data.view(batch_size, -1))
            loss_cls = F.cross_entropy(dummy_pred, predicted_labels)
            dy_dx = torch.autograd.grad(loss_cls, dummy_model.parameters(), create_graph=True)
            grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
            tv_loss = total_variation_loss(dummy_data)
            group_loss = torch.norm(dummy_data - consensus_batch, p=2)
            batch_total_loss = grad_loss + reg_tv * tv_loss + reg_group * group_loss
            total_loss += batch_total_loss

        total_loss.backward()
        optimizer.step()

        if it % 1000 == 0:
            print(f"Iteration {it}/{iterations}, Total Loss: {total_loss.item():.4f}")

    final_consensus = torch.stack(candidate_batches).mean(dim=0)
    
    # Return both the images and the labels used to generate them
    return final_consensus.detach().numpy(), predicted_labels
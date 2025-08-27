# attacks/gradient_inversion.py (Grayscale Version)

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

# IMPORTANT: You must modify your CifarCNN in model.py to accept 1 input channel.
from model import CifarCNN 

def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / torch.numel(img)

def dlg_attack(
    gradients: List[np.ndarray], lr: float = 0.01, iterations: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    print("--- Launching DLG Attack (Grayscale) ---")
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    # Key Change for Grayscale: Dummy data has 1 channel.
    dummy_data = torch.randn(1, 1, 32, 32, requires_grad=True)
    dummy_label = torch.randn(1, 10, requires_grad=True)
    
    # IMPORTANT: CifarCNN must be the grayscale-compatible version.
    dummy_model = CifarCNN(num_classes=10) 
    optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=lr)

    for it in range(iterations):
        optimizer.zero_grad()
        dummy_pred = dummy_model(dummy_data)
        loss_cls = F.cross_entropy(dummy_pred, dummy_label.softmax(dim=-1))
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        grad_loss.backward()
        optimizer.step()
        if it % 500 == 0:
            print(f"  DLG Iteration {it}/{iterations}, Grad Loss: {grad_loss.item():.4f}")

    return dummy_data.detach().numpy(), torch.argmax(dummy_label, dim=-1).detach().numpy()

def idlg_attack(
    gradients: List[np.ndarray], lr: float = 0.01, iterations: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    print("--- Launching iDLG Attack (Grayscale) ---")
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    last_layer_grad = original_dy_dx[-1]
    recovered_label_idx = torch.argmin(last_layer_grad).item()
    recovered_label = torch.tensor([recovered_label_idx])
    print(f"  iDLG recovered label: {recovered_label_idx}")

    # Key Change for Grayscale: Dummy data has 1 channel.
    dummy_data = torch.randn(1, 1, 32, 32, requires_grad=True)
    
    # IMPORTANT: CifarCNN must be the grayscale-compatible version.
    dummy_model = CifarCNN(num_classes=10)
    optimizer = torch.optim.Adam([dummy_data], lr=lr)

    for it in range(iterations):
        optimizer.zero_grad()
        dummy_pred = dummy_model(dummy_data)
        loss_cls = F.cross_entropy(dummy_pred, recovered_label)
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        grad_loss.backward()
        optimizer.step()
        if it % 500 == 0:
            print(f"  iDLG Iteration {it}/{iterations}, Grad Loss: {grad_loss.item():.4f}")

    return dummy_data.detach().numpy(), recovered_label.numpy()

def mdlg_attack(
    gradients: List[np.ndarray], lr: float = 0.01, iterations: int = 2000,
    reg_tv: float = 1e-4, reg_l2: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    print("--- Launching mDLG Attack (Grayscale) ---")
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    # Key Change for Grayscale: Dummy data has 1 channel.
    dummy_data = torch.randn(1, 1, 32, 32, requires_grad=True)
    dummy_label = torch.randn(1, 10, requires_grad=True)
    
    # IMPORTANT: CifarCNN must be the grayscale-compatible version.
    dummy_model = CifarCNN(num_classes=10)
    optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=lr)

    for it in range(iterations):
        optimizer.zero_grad()
        dummy_pred = dummy_model(dummy_data)
        loss_cls = F.cross_entropy(dummy_pred, dummy_label.softmax(dim=-1))
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        tv_loss = total_variation_loss(dummy_data)
        l2_loss = torch.norm(dummy_data, p=2)
        total_loss = grad_loss + reg_tv * tv_loss + reg_l2 * l2_loss
        
        total_loss.backward()
        optimizer.step()
        if it % 500 == 0:
            print(f"  mDLG Iteration {it}/{iterations}, Loss: {total_loss.item():.4f}")

    return dummy_data.detach().numpy(), torch.argmax(dummy_label, dim=-1).detach().numpy()
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Computes the total variation loss for a batch of images to reduce noise."""
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def dlg_attack(
    gradients: List[np.ndarray],
    input_shape: Tuple[int, int, int] = (1, 128, 128),
    lr: float = 0.01,
    iterations: int = 5000,
    reg_tv: float = 1e-4
) -> Tuple[np.ndarray, None]:
    """
    Empowered DLG for the 128x128 X-ray dataset with TV regularization.
    WARNING: Performance on complex images is expected to be poor.
    """
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    # Optimize a "pre-image" to be passed through sigmoid to fix color inversion
    dummy_data_pre_sigmoid = torch.randn(1, *input_shape, requires_grad=True)
    dummy_logits = torch.randn((1, 15), requires_grad=True) # 15 classes for X-ray
    optimizer = torch.optim.Adam([dummy_data_pre_sigmoid, dummy_logits], lr=lr)

    # The dummy model MUST match your new SimpleNN for X-rays
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(128 * 128, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 15)
    )

    for it in range(iterations):
        optimizer.zero_grad()
        # Constrain the image to the [0, 1] range
        dummy_data = torch.sigmoid(dummy_data_pre_sigmoid)
        dummy_pred = dummy_model(dummy_data.view(1, -1))
        
        # Use the correct loss for multi-label classification
        loss_cls = F.binary_cross_entropy_with_logits(dummy_pred, torch.sigmoid(dummy_logits))
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        
        # Add TV Regularization to reduce noise
        tv_loss = total_variation_loss(dummy_data)
        total_loss = grad_loss + reg_tv * tv_loss
        
        total_loss.backward()
        optimizer.step()
        
        if it % 1000 == 0:
            print(f"Iteration {it}/{iterations}, Total Loss: {total_loss.item():.4f}, Grad Loss: {grad_loss.item():.4f}")

    # Return the final constrained image and None for the labels
    return torch.sigmoid(dummy_data_pre_sigmoid).detach().numpy(), None


def mdlg_attack(
    gradients: List[np.ndarray],
    input_shape: Tuple[int, int, int] = (1, 128, 128),
    lr: float = 0.01,
    iterations: int = 2000
) -> Tuple[np.ndarray, None]:
    """
    Empowered mDLG for the 128x128 X-ray dataset.
    WARNING: Performance on complex images is expected to be very poor.
    """
    tgt_grad_W = torch.from_numpy(gradients[0]).float()
    
    # Match the shape of the new model's first layer
    W_shape = (256, 128*128)
    W = torch.randn(W_shape, requires_grad=True)
    
    # Optimize a "pre-image" to be passed through sigmoid
    dummy_data_pre_sigmoid = torch.randn(1, *input_shape, requires_grad=True)
    opt = torch.optim.Adam([dummy_data_pre_sigmoid], lr=lr)

    for _ in range(iterations):
        opt.zero_grad()
        # Constrain the image to the [0, 1] range
        dummy_data = torch.sigmoid(dummy_data_pre_sigmoid)
        
        logits = torch.matmul(dummy_data.view(1, -1), W.t())
        loss = logits.norm()
        grad_W, = torch.autograd.grad(loss, [W], create_graph=True)
        grad_loss = F.mse_loss(grad_W, tgt_grad_W)
        grad_loss.backward()
        opt.step()

    # Return the final constrained image and None for the labels
    return torch.sigmoid(dummy_data_pre_sigmoid).detach().numpy(), None


def gradinversion_attack(
    gradients: List[np.ndarray],
    batch_size: int,
    input_shape: Tuple[int, int, int] = (1, 128, 128),
    lr: float = 0.1,
    iterations: int = 5000
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    GradInversion adapted for the X-ray dataset.
    WARNING: The label restoration step is a placeholder due to the multi-label setup.
    """
    # CHALLENGE: Label Restoration for multi-label is an open research problem.
    # The paper's method is for single-label CrossEntropy. We use a random guess as a placeholder.
    predicted_labels = torch.randint(0, 2, (batch_size, 15)).float()
    print(f"[Attack] Using random labels for demonstration.")

    dummy_data_pre_sigmoid = torch.randn(batch_size, *input_shape, requires_grad=True)
    optimizer = torch.optim.Adam([dummy_data_pre_sigmoid], lr=lr)
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(128*128, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 15)
    )

    for it in range(iterations):
        optimizer.zero_grad()
        dummy_data = torch.sigmoid(dummy_data_pre_sigmoid)
        dummy_pred = dummy_model(dummy_data.view(batch_size, -1))
        loss_cls = F.binary_cross_entropy_with_logits(dummy_pred, predicted_labels)
        dy_dx = torch.autograd.grad(loss_cls, dummy_model.parameters(), create_graph=True)
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        tv_loss = total_variation_loss(dummy_data)
        total_loss = grad_loss + 1e-4 * tv_loss
        total_loss.backward()
        optimizer.step()
        if it % 1000 == 0:
            print(f"Iteration {it}/{iterations}, Grad Loss: {grad_loss.item():.4f}")
            
    return torch.sigmoid(dummy_data_pre_sigmoid).detach().numpy(), predicted_labels

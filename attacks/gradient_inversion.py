import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

# Import the correct model to replicate in the attack
from model import CastingCNN

def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Computes the total variation loss for a batch of images to reduce noise."""
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def dlg_attack(
    gradients: List[np.ndarray],
    input_shape: Tuple[int, int, int] = (1, 300, 300),
    lr: float = 0.01,
    iterations: int = 8000,
    reg_tv: float = 1e-4 #<-- New parameter for TV regularization
) -> Tuple[np.ndarray, None]:
    """
    Empowered DLG for the 300x300 casting dataset with Total Variation regularization.
    """
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    # Optimize a "pre-image" and pass it through sigmoid to fix color inversion
    dummy_data_pre_sigmoid = torch.randn(1, *input_shape, requires_grad=True)
    dummy_logits = torch.randn((1, 1), requires_grad=True) 
    optimizer = torch.optim.Adam([dummy_data_pre_sigmoid, dummy_logits], lr=lr)

    # The dummy model must be an exact replica of the client's CastingCNN
    dummy_model = CastingCNN(num_classes=1)

    for it in range(iterations):
        optimizer.zero_grad()
        # Constrain the image to the [0, 1] range to prevent color inversion
        dummy_data = torch.sigmoid(dummy_data_pre_sigmoid)
        dummy_pred = dummy_model(dummy_data)
        
        # Use the correct loss function for binary classification
        loss_cls = F.binary_cross_entropy_with_logits(dummy_pred, torch.sigmoid(dummy_logits))
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        
        # Calculate the two components of the loss
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        tv_loss = total_variation_loss(dummy_data)
        
        # --- MORE OPTIMIZATION: Combine losses with the regularization term ---
        total_loss = grad_loss + reg_tv * tv_loss
        
        total_loss.backward()
        optimizer.step()

        if it % 1000 == 0:
            print(f"Iteration {it}/{iterations}, Total Loss: {total_loss.item():.4f}, Grad Loss: {grad_loss.item():.4f}")

    # Return the final constrained image
    return torch.sigmoid(dummy_data_pre_sigmoid).detach().numpy(), None

def mdlg_attack(
    gradients: List[np.ndarray],
    input_shape: Tuple[int, int, int] = (1, 128, 128), # Ensure default shape is correct
    lr: float = 0.01,
    iterations: int = 2000
) -> Tuple[np.ndarray, None]:
    """Empowered mDLG for the casting dataset."""
    tgt_grad_W = torch.from_numpy(gradients[0]).float()
    
    # --- THE FIX: Match the shape of the CastingCNN's first layer's weights ---
    W_shape = (32, 1, 3, 3) 
    W = torch.randn(W_shape, requires_grad=True)
    
    dummy_data_pre_sigmoid = torch.randn(1, *input_shape, requires_grad=True)
    opt = torch.optim.Adam([dummy_data_pre_sigmoid], lr=lr)

    for _ in range(iterations):
        opt.zero_grad()
        dummy_data = torch.sigmoid(dummy_data_pre_sigmoid)
        
        # Simulate a single conv layer to get a gradient
        logits = F.conv2d(dummy_data, W, padding=1, stride=2)
        loss = logits.norm()
        grad_W, = torch.autograd.grad(loss, [W], create_graph=True)
        grad_loss = F.mse_loss(grad_W, tgt_grad_W)
        grad_loss.backward()
        opt.step()

    return torch.sigmoid(dummy_data_pre_sigmoid).detach().numpy(), None
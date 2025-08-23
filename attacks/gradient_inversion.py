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

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

# Import the correct model to replicate in the attack
from model import CifarCNN

def dlg_attack(
    gradients: List[np.ndarray],
    input_shape: Tuple[int, int, int] = (3, 32, 32), # 3 channels for CIFAR-10
    num_classes: int = 10,
    lr: float = 0.01,
    iterations: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """DLG attack for the CIFAR-10 dataset."""
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    dummy_data = torch.randn(1, *input_shape, requires_grad=True)
    dummy_label = torch.randn(1, num_classes, requires_grad=True)
    optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=lr)

    dummy_model = CifarCNN(num_classes=num_classes)

    for it in range(iterations):
        optimizer.zero_grad()
        dummy_pred = dummy_model(dummy_data)
        loss_cls = F.cross_entropy(dummy_pred, dummy_label.softmax(dim=-1))
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        grad_loss.backward()
        optimizer.step()

    reconstructed_label = torch.argmax(dummy_label, dim=-1).detach().numpy()
    return dummy_data.detach().numpy(), reconstructed_label

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
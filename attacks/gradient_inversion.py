import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

def dlg_attack(
    gradients: List[np.ndarray],
    input_shape: Tuple[int, int, int] = (1, 128, 128),
    lr: float = 0.1,
    iterations: int = 2000,
) -> np.ndarray:
    """
    DLG adapted for the 128x128 X-ray dataset.
    WARNING: Performance on complex images is expected to be poor.
    """
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    dummy_data_pre_sigmoid = torch.randn(1, *input_shape, requires_grad=True)
    dummy_logits = torch.randn((1, 15), requires_grad=True) # 15 classes
    optimizer = torch.optim.Adam([dummy_data_pre_sigmoid, dummy_logits], lr=lr)

    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(128 * 128, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 15)
    )

    for it in range(iterations):
        optimizer.zero_grad()
        dummy_data = torch.sigmoid(dummy_data_pre_sigmoid)
        dummy_pred = dummy_model(dummy_data.view(1, -1))
        
        loss_cls = F.binary_cross_entropy_with_logits(dummy_pred, torch.sigmoid(dummy_logits))
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        grad_loss.backward()
        optimizer.step()
        if it % 500 == 0:
            print(f"Iteration {it}/{iterations}, Grad Loss: {grad_loss.item():.4f}")

    return torch.sigmoid(dummy_data_pre_sigmoid).detach().numpy()


def mdlg_attack(
    gradients: List[np.ndarray],
    input_shape: Tuple[int, int, int] = (1, 128, 128),
    lr: float = 0.01,
    iterations: int = 500
) -> np.ndarray:
    """
    mDLG adapted for the 128x128 X-ray dataset.
    WARNING: Performance on complex images is expected to be very poor.
    """
    tgt_grad_W = torch.from_numpy(gradients[0]).float() # Grad of fc1.weight
    
    # Match the shape of the new model's first layer
    W_shape = (256, 128*128)
    W = torch.randn(W_shape, requires_grad=True)

    dummy_data_pre_sigmoid = torch.randn(1, *input_shape, requires_grad=True)
    opt = torch.optim.Adam([dummy_data_pre_sigmoid], lr=lr)

    for _ in range(iterations):
        opt.zero_grad()
        dummy_data = torch.sigmoid(dummy_data_pre_sigmoid)
        
        logits = torch.matmul(dummy_data.view(1, -1), W.t())
        loss = logits.norm()
        grad_W, = torch.autograd.grad(loss, [W], create_graph=True)
        grad_loss = F.mse_loss(grad_W, tgt_grad_W)
        grad_loss.backward()
        opt.step()

    return torch.sigmoid(dummy_data_pre_sigmoid).detach().numpy()
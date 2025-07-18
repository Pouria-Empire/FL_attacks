# attacks/gradient_inversion.py
import torch
import numpy as np
from typing import Tuple, List

# In attacks/gradient_inversion.py

def dlg_attack(
    gradients: list,            # [grad_fc1_w, grad_fc1_b, grad_fc2_w, grad_fc2_b]
    input_shape: Tuple[int],    # (1, 1, 28, 28)
    lr: float = 0.1,
    iterations: int = 1000,
) -> np.ndarray:
    
    # --- We expect gradients from ALL layers now ---
    # gradients[0] = fc1.weight, gradients[1] = fc1.bias
    # gradients[2] = fc2.weight, gradients[3] = fc2.bias
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]

    # --- Dummy data and model parameters ---
    dummy_data = torch.randn(input_shape, requires_grad=True)
    
    # The BIG CHANGE: Instead of a dummy label, we create dummy logits
    # that we can optimize. This lets the attack find the correct label.
    dummy_logits = torch.randn((1, 10), requires_grad=True) # MNIST has 10 classes

    # Use a dummy model that matches the client's SimpleNN architecture
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(784, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    )
    
    # We only need the parameters for gradient calculation, so we can just use the
    # list of gradients we already have to get their shapes.
    dummy_params = list(dummy_model.parameters())

    # --- Optimizer ---
    # We now optimize BOTH the dummy_data and the dummy_logits
    optimizer = torch.optim.Adam([dummy_data, dummy_logits], lr=lr)

    for it in range(iterations):
        optimizer.zero_grad()

        # Reshape dummy data for the linear layer
        dummy_data_flat = dummy_data.view(1, -1)
        
        # Forward pass through our dummy model to get gradients
        # We use our learnable dummy_logits as the output
        dummy_pred = dummy_model(dummy_data_flat)
        loss_cls = torch.nn.functional.cross_entropy(dummy_pred, dummy_logits.softmax(dim=-1))
        
        # Calculate gradients w.r.t. model parameters
        dy_dx = torch.autograd.grad(loss_cls, dummy_params, create_graph=True)

        # --- The Gradient Matching Loss ---
        grad_loss = 0
        for gx, gy in zip(original_dy_dx, dy_dx):
            grad_loss += ((gx - gy) ** 2).sum()
        
        grad_loss.backward()
        optimizer.step()

        if it % 500 == 0:
            print(f"Iteration {it}/{iterations}, Grad Loss: {grad_loss.item():.4f}")

    # Return the optimized data
    return dummy_data.detach().numpy()


def mdlg_attack(gradients: list, input_shape: Tuple[int],
                lr: float = 0.01, iterations: int = 500) -> np.ndarray:
    tgt_grad_W = torch.from_numpy(gradients[0]).float()      # (64, 784)

    W = torch.randn_like(tgt_grad_W, requires_grad=True)
    dummy_data = torch.randn(input_shape, requires_grad=True)
    opt = torch.optim.Adam([dummy_data], lr=lr)

    for _ in range(iterations):
        opt.zero_grad()
        logits = torch.matmul(dummy_data.view(1, -1), W.t())
        loss = logits.norm()                                  # arbitrary, just needs grad
        grad_W, = torch.autograd.grad(loss, [W], create_graph=True)
        grad_loss = torch.nn.functional.mse_loss(grad_W, tgt_grad_W)
        grad_loss.backward()
        opt.step()

    return dummy_data.detach().numpy()


import torch.nn.functional as F

def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Computes the total variation loss for a batch of images to reduce noise."""
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def gradinversion_attack(
    gradients: List[np.ndarray],
    batch_size: int,
    input_shape: Tuple[int, int, int] = (1, 28, 28),
    lr: float = 0.1,
    iterations: int = 5000
) -> np.ndarray:
    """
    Performs a GradInversion attack to reconstruct a batch of images.
    Implements Batch Label Restoration and Fidelity Regularization from the paper.
    """
    # 1. Batch Label Restoration (from paper Section 3.2)
    # Assumes the second to last gradient is from the final FC layer's weights
    fc_grad = torch.from_numpy(gradients[-2]).float()
    predicted_labels = torch.topk(fc_grad.sum(dim=1), k=batch_size, largest=False)[1]
    print(f"[Attack] Recovered labels: {predicted_labels.numpy()}")

    # 2. Initialize dummy data for the entire batch
    dummy_data = torch.randn(batch_size, *input_shape, requires_grad=True)
    optimizer = torch.optim.Adam([dummy_data], lr=lr)
    
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(784, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    )

    for it in range(iterations):
        optimizer.zero_grad()
        
        # 3. Calculate Gradient Matching Loss for the batch
        dummy_pred = dummy_model(dummy_data.view(batch_size, -1))
        loss_cls = F.cross_entropy(dummy_pred, predicted_labels)
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))

        # 4. Add Fidelity Regularization (from paper Section 3.3)
        tv_loss = total_variation_loss(dummy_data)
        
        # 5. Combine losses and update
        total_loss = grad_loss + 1e-4 * tv_loss
        total_loss.backward()
        optimizer.step()

        if it % 1000 == 0:
            print(f"Iteration {it}/{iterations}, Grad Loss: {grad_loss.item():.4f}")

    return dummy_data.detach().numpy()
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

def dlg_attack(
    gradients: list,
    input_shape: Tuple[int] = (1, 1, 28, 28),
    lr: float = 0.1,
    iterations: int = 2000,
) -> np.ndarray:
    
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    # --- THE FIX: Optimize a "pre-image" and apply sigmoid ---
    dummy_data_pre_sigmoid = torch.randn(input_shape, requires_grad=True)
    dummy_logits = torch.randn((1, 10), requires_grad=True)
    optimizer = torch.optim.Adam([dummy_data_pre_sigmoid, dummy_logits], lr=lr)

    # Ensure dummy model matches the client's model perfectly
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(784, 64), 
        torch.nn.ReLU(), 
        torch.nn.Linear(64, 10),
        torch.nn.LogSoftmax(dim=1)
    )

    for it in range(iterations):
        optimizer.zero_grad()
        # Apply sigmoid to constrain the image to the [0, 1] range
        dummy_data = torch.sigmoid(dummy_data_pre_sigmoid)

        dummy_pred = dummy_model(dummy_data.view(1, -1))
        loss_cls = F.cross_entropy(dummy_pred, dummy_logits.softmax(dim=-1))
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        grad_loss.backward()
        optimizer.step()
        if it % 500 == 0:
            print(f"Iteration {it}/{iterations}, Grad Loss: {grad_loss.item():.4f}")

    # Return the final constrained image
    return torch.sigmoid(dummy_data_pre_sigmoid).detach().numpy()


def mdlg_attack(
    gradients: list,
    input_shape: Tuple[int] = (1, 1, 28, 28),
    lr: float = 0.01,
    iterations: int = 500
) -> np.ndarray:
    tgt_grad_W = torch.from_numpy(gradients[0]).float()
    W = torch.randn_like(tgt_grad_W, requires_grad=True)
    dummy_data_pre_sigmoid = torch.randn(input_shape, requires_grad=True)
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

def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
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
) -> Tuple[np.ndarray, torch.Tensor]:
    
    fc_grad = torch.from_numpy(gradients[-2]).float()
    predicted_labels = torch.topk(fc_grad.sum(dim=1), k=batch_size, largest=False)[1]
    print(f"[Attack] Recovered labels: {predicted_labels.numpy()}")

    dummy_data_pre_sigmoid = torch.randn(batch_size, *input_shape, requires_grad=True)
    optimizer = torch.optim.Adam([dummy_data_pre_sigmoid], lr=lr)
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    dummy_model = torch.nn.Sequential(torch.nn.Linear(784, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10), torch.nn.LogSoftmax(dim=1))

    for it in range(iterations):
        optimizer.zero_grad()
        dummy_data = torch.sigmoid(dummy_data_pre_sigmoid)
        dummy_pred = dummy_model(dummy_data.view(batch_size, -1))
        loss_cls = F.nll_loss(dummy_pred, predicted_labels)
        dy_dx = torch.autograd.grad(loss_cls, dummy_model.parameters(), create_graph=True)
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        tv_loss = total_variation_loss(dummy_data)
        total_loss = grad_loss + 1e-4 * tv_loss
        total_loss.backward()
        optimizer.step()

        if it % 1000 == 0:
            print(f"Iteration {it}/{iterations}, Grad Loss: {grad_loss.item():.4f}")
            
    return torch.sigmoid(dummy_data_pre_sigmoid).detach().numpy(), predicted_labels
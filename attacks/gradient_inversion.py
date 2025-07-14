# attacks/gradient_inversion.py
import torch
import numpy as np
from typing import Tuple

def dlg_attack(
    gradients: list,            # [grad_W, grad_b] from the victim
    input_shape: Tuple[int],    # (1,1,28,28)
    output_shape: Tuple[int],   # (1,10)
    lr: float = 0.1,
    iterations: int = 1000,
) -> np.ndarray:
    # === target gradients we want to reproduce ===============================
    tgt_grad_W = torch.from_numpy(gradients[0]).float()      # (64, 784)
    tgt_grad_b = torch.from_numpy(gradients[1]).float()      # (64,)

    out_dim, in_dim = tgt_grad_W.shape

    # === dummy model parameters (just for gradient calculation) ==============
    W = torch.randn(out_dim, in_dim, requires_grad=True)     # (64, 784)
    b = torch.randn(out_dim,              requires_grad=True)  # (64,)

    # === what we are trying to recover =======================================
    dummy_data  = torch.randn(input_shape, requires_grad=True)
    dummy_label = torch.randint(0, output_shape[1], (1,))    # class index

    opt = torch.optim.Adam([dummy_data], lr=lr)

    for _ in range(iterations):
        opt.zero_grad()

        # forward
        logits = torch.matmul(dummy_data.view(1, -1), W.t()) + b
        loss_cls = torch.nn.functional.cross_entropy(logits, dummy_label)

        # gradients w.r.t. *model parameters*
        grad_W, grad_b = torch.autograd.grad(
            loss_cls, [W, b], create_graph=True
        )

        # gradient‑matching loss
        grad_loss = torch.nn.functional.mse_loss(grad_W, tgt_grad_W) + \
                    torch.nn.functional.mse_loss(grad_b, tgt_grad_b)

        grad_loss.backward()
        opt.step()

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

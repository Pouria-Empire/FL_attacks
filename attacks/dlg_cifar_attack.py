import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

# Import the correct model to replicate in the attack
from model import CifarCNN

def dlg_cifar_attack(
    gradients: List[np.ndarray],
    input_shape: Tuple[int, int, int] = (3, 32, 32), # 3 channels for CIFAR-10
    num_classes: int = 10,
    lr: float = 0.01,
    iterations: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs a DLG attack on the CifarCNN model for the CIFAR-10 dataset.
    """
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    # Start with random guesses for the data and label
    dummy_data = torch.randn(1, *input_shape, requires_grad=True)
    dummy_label = torch.randn(1, num_classes, requires_grad=True)
    optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=lr)

    # The dummy model must be an exact replica of the client's CifarCNN
    dummy_model = CifarCNN(num_classes=num_classes)

    for it in range(iterations):
        optimizer.zero_grad()
        
        dummy_pred = dummy_model(dummy_data)
        loss_cls = F.cross_entropy(dummy_pred, dummy_label.softmax(dim=-1))
        
        # Calculate gradients from the dummy data and model
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        
        # Calculate the gradient matching loss
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        
        grad_loss.backward()
        optimizer.step()

        if it % 500 == 0:
            print(f"Iteration {it}/{iterations}, Grad Loss: {grad_loss.item():.4f}")

    reconstructed_label = torch.argmax(dummy_label, dim=-1).detach().numpy()
    return dummy_data.detach().numpy(), reconstructed_label
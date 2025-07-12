import torch
import numpy as np
from flwr.common import NDArrays

def dlg_attack(gradients: NDArrays, input_shape: tuple, output_shape: tuple, lr=0.1, iterations=1000):
    """
    Deep Leakage from Gradients (DLG) attack implementation
    """
    # Initialize dummy data and label
    dummy_data = torch.randn(input_shape, requires_grad=True)
    dummy_label = torch.randn(output_shape, requires_grad=True)
    
    optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=lr)
    
    for _ in range(iterations):
        optimizer.zero_grad()
        
        # Compute dummy gradients
        dummy_pred = torch.matmul(dummy_data, gradients[0].T) + gradients[1]
        dummy_loss = torch.nn.functional.cross_entropy(dummy_pred, dummy_label)
        dummy_grad = torch.autograd.grad(dummy_loss, [dummy_data, dummy_label], create_graph=True)
        
        # Compute gradient difference loss
        grad_diff = sum(
            (torch.norm(dummy_g - g) for dummy_g, g in zip(dummy_grad, gradients)
        ))
        
        grad_diff.backward()
        optimizer.step()
    
    return dummy_data.detach().numpy()

def mdlg_attack(gradients: NDArrays, input_shape: tuple, lr=0.01, iterations=500):
    """
    Modified Deep Leakage from Gradients (MDLG) attack implementation
    """
    # Initialize dummy data
    dummy_data = torch.randn(input_shape, requires_grad=True)
    optimizer = torch.optim.Adam([dummy_data], lr=lr)
    
    for _ in range(iterations):
        optimizer.zero_grad()
        
        # Compute dummy gradients
        dummy_grad = torch.autograd.grad(
            torch.norm(dummy_data), 
            [dummy_data], 
            create_graph=True
        )[0]
        
        # Compute gradient difference loss
        grad_diff = torch.norm(dummy_grad - gradients[0])
        grad_diff.backward()
        optimizer.step()
    
    return dummy_data.detach().numpy()
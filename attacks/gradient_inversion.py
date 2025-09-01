# attacks/gradient_inversion.py

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Type

def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]):
    """Helper function to load state_dict into a model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Computes the Total Variation Loss to reduce noise in the reconstructed image."""
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / torch.numel(img)

def dlg_attack(
    gradients: List[np.ndarray], model_class: Type[torch.nn.Module], 
    input_shape: Tuple[int, ...], num_classes: int, global_params: List[np.ndarray],
    lr: float, iterations: int
) -> Tuple[np.ndarray, np.ndarray]:
    
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    dummy_data = torch.randn(1, *input_shape, requires_grad=True)
    dummy_label = torch.randn(1, num_classes, requires_grad=True)
    
    dummy_model = model_class(num_classes=num_classes)
    # ✅ CRITICAL FIX: Synchronize the dummy model's weights with the global model.
    set_parameters(dummy_model, global_params)
    
    optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=lr)
    criterion = (lambda p, l: F.binary_cross_entropy_with_logits(p, l.sigmoid())) if num_classes == 1 else (lambda p, l: F.cross_entropy(p, l.softmax(dim=-1)))

    for it in range(iterations):
        optimizer.zero_grad()
        dummy_pred = dummy_model(dummy_data)
        loss_cls = criterion(dummy_pred, dummy_label)
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        grad_loss.backward()
        optimizer.step()
        with torch.no_grad():
            dummy_data.clamp_(-1, 1)

    reconstructed_label = (torch.sigmoid(dummy_label) > 0.5).int().detach().numpy() if num_classes == 1 else torch.argmax(dummy_label, dim=-1).detach().numpy()
    return dummy_data.detach().numpy(), reconstructed_label

def idlg_attack(
    gradients: List[np.ndarray], model_class: Type[torch.nn.Module], 
    input_shape: Tuple[int, ...], num_classes: int, global_params: List[np.ndarray],
    lr: float, iterations: int, reg_tv: float, reg_l2: float, is_multilabel: bool
) -> Tuple[np.ndarray, np.ndarray]:
    
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    last_layer_grad = original_dy_dx[-1]
    if is_multilabel:
        recovered_label = (last_layer_grad < 0).float().view(1, -1)
    elif num_classes > 1:
        recovered_label = torch.tensor([torch.argmin(last_layer_grad).item()])
    else:
        recovered_label = torch.tensor([[float(1 if last_layer_grad.item() < 0 else 0)]])
    
    dummy_data = torch.randn(1, *input_shape, requires_grad=True)
    dummy_model = model_class(num_classes=num_classes)
    # ✅ CRITICAL FIX: Synchronize the dummy model's weights.
    set_parameters(dummy_model, global_params)
    
    optimizer = torch.optim.Adam([dummy_data], lr=lr)
    if is_multilabel or num_classes == 1:
        criterion, label_for_loss = torch.nn.BCEWithLogitsLoss(), recovered_label
    else:
        criterion, label_for_loss = torch.nn.CrossEntropyLoss(), recovered_label.squeeze().long()

    for it in range(iterations):
        optimizer.zero_grad()
        dummy_pred = dummy_model(dummy_data)
        loss_cls = criterion(dummy_pred, label_for_loss)
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        tv_loss, l2_loss = total_variation_loss(dummy_data), torch.norm(dummy_data, p=2)
        total_loss = grad_loss + reg_tv * tv_loss + reg_l2 * l2_loss
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            dummy_data.clamp_(-1, 1)
            
    return dummy_data.detach().numpy(), recovered_label.numpy()

def mdlg_attack(
    gradients: List[np.ndarray], model_class: Type[torch.nn.Module], 
    input_shape: Tuple[int, ...], num_classes: int, global_params: List[np.ndarray],
    lr: float, iterations: int, reg_tv: float, reg_l2: float
) -> Tuple[np.ndarray, np.ndarray]:
    
    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    dummy_data = torch.randn(1, *input_shape, requires_grad=True)
    dummy_label = torch.randn(1, num_classes, requires_grad=True)
    dummy_model = model_class(num_classes=num_classes)
    # ✅ CRITICAL FIX: Synchronize the dummy model's weights.
    set_parameters(dummy_model, global_params)

    optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=lr)
    criterion = (lambda p, l: F.binary_cross_entropy_with_logits(p, l.sigmoid())) if num_classes == 1 else (lambda p, l: F.cross_entropy(p, l.softmax(dim=-1)))

    for it in range(iterations):
        optimizer.zero_grad()
        dummy_pred = dummy_model(dummy_data)
        loss_cls = criterion(dummy_pred, dummy_label)
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        tv_loss, l2_loss = total_variation_loss(dummy_data), torch.norm(dummy_data, p=2)
        total_loss = grad_loss + reg_tv * tv_loss + reg_l2 * l2_loss
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            dummy_data.clamp_(-1, 1)

    reconstructed_label = (torch.sigmoid(dummy_label) > 0.5).int().detach().numpy() if num_classes == 1 else torch.argmax(dummy_label, dim=-1).detach().numpy()
    return dummy_data.detach().numpy(), reconstructed_label
import torch
import numpy as np
import os
import torchvision
from typing import List, Dict, Optional, Tuple

# Import all attack functions
from attacks.gradient_inversion import gradinversion_attack, dlg_attack, mdlg_attack
from attacks.gradinversion_plus import gradinversion_group_attack
from attacks.ggl_attack import ggl_attack
from attacks.temporal_attack import temporal_attack

def reconstruct_data(
    gradients_list: List[List[np.ndarray]], 
    attack_params: Dict, 
    client_config: Dict
) -> Optional[Tuple[np.ndarray, Optional[torch.Tensor]]]:
    """
    Selects and runs the appropriate gradient inversion attack based on the config.
    """
    attack_type = attack_params.get("type", "dlg")
    print(f"[Attack] Attempting reconstruction using '{attack_type}' method.")
    
    if attack_type == "temporal":
        return temporal_attack(
            gradient_history=gradients_list, 
            lr=attack_params.get("attack_lr"), 
            iterations=attack_params.get("iterations")
        )

    # All other attacks operate on a single gradient from the list
    gradients = gradients_list[0]
    
    if attack_type == "gradinversion_plus":
        return gradinversion_group_attack(
            gradients=gradients, batch_size=client_config.get("batch_size", 8), 
            num_seeds=attack_params.get("num_seeds", 4), lr=attack_params.get("attack_lr"), 
            iterations=attack_params.get("iterations"), reg_tv=attack_params.get("reg_tv", 1e-4), 
            reg_l2=attack_params.get("reg_l2", 1e-5), reg_group=attack_params.get("reg_group", 0.005)
        )
    elif attack_type == "gradinversion":
        return gradinversion_attack(
            gradients=gradients, batch_size=client_config.get("batch_size", 8), 
            lr=attack_params.get("attack_lr"), iterations=attack_params.get("iterations")
        )
    elif attack_type == "ggl":
        return ggl_attack(
            gradients=gradients, lr=attack_params.get("attack_lr"), 
            iterations=attack_params.get("iterations")
        )
    elif attack_type == "dlg":
        return dlg_attack(
            gradients=gradients, lr=attack_params.get("attack_lr"), 
            iterations=attack_params.get("iterations")
        )
    elif attack_type == "mdlg":
        return mdlg_attack(
            gradients=gradients, lr=attack_params.get("attack_lr"), 
            iterations=attack_params.get("iterations")
        )
    else:
        print(f"Unknown reconstruction attack type: {attack_type}")
        return None

def save_reconstruction(
    data: np.ndarray, 
    predicted_labels: Optional[torch.Tensor], 
    client_id: int, 
    round_num: int, 
    reconstruction_dir: str,
    original_data: Optional[np.ndarray] = None, 
    original_labels: Optional[np.ndarray] = None
):
    """Saves the reconstructed image(s) and a comparison grid if original data is available."""
    recon_tensor = torch.from_numpy(data)
    
    if original_data is not None and original_labels is not None:
        original_tensor = torch.from_numpy(original_data)
        original_labels_tensor = torch.from_numpy(original_labels)
        
        # Align original images with the (potentially sorted) reconstructed ones
        if predicted_labels is not None and original_tensor.shape == recon_tensor.shape:
            sort_indices = torch.argsort(original_labels_tensor.sum(dim=-1))
            original_tensor_sorted = original_tensor[sort_indices]
            comparison_grid = torch.cat([original_tensor_sorted, recon_tensor])
        else:
            comparison_grid = torch.cat([original_tensor, recon_tensor])
            
        save_path = os.path.join(reconstruction_dir, f"comparison_client{client_id}_round{round_num}.png")
        torchvision.utils.save_image(comparison_grid, save_path, nrow=original_tensor.size(0))
        print(f"[Attack] Saved comparison grid to {save_path}")
    else:
        save_path = os.path.join(reconstruction_dir, f"reconstruction_client{client_id}_round{round_num}.png")
        torchvision.utils.save_image(recon_tensor, save_path)
        print(f"[Attack] Saved reconstruction grid to {save_path}")
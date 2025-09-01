# server/reconstruction.py

import torch
import numpy as np
import os
import torchvision
from typing import List, Dict, Optional, Tuple

# Import all attack functions
from attacks.gradient_inversion import dlg_attack, idlg_attack, mdlg_attack
from attacks.numerical_attacks import numerical_dlg_attack
# from attacks.ggl_cifar_attack import ggl_cifar_attack_strong # Uncomment when you use GGL

# Import all models needed for the dummy models in attacks
from model import CifarCNN, MedMNIST_CNN, CastingCNN, SensorMLP
from utils_data.sensor_data_util import load_and_preprocess_data
from attacks.ggl_medmnist_attack import ggl_medmnist_attack

def reconstruct_data(
    gradients_list: List[List[np.ndarray]],
    attack_params: Dict,
    client_config: Dict,
    data_type: str,
    main_config: Dict,
    global_params: List[np.ndarray]
) -> Optional[Tuple]:
    """
    Selects and runs the appropriate gradient inversion attack based on the config.
    """
    attack_type = attack_params.get("type", "mdlg")
    print(f"[Attack] Attempting reconstruction for '{data_type}' data using '{attack_type}' method.")

    raw_gradients = gradients_list[0]
    if len(raw_gradients) == 1 and raw_gradients[0].dtype == np.uint8:
        print(f"[Attack Failed] Received encrypted gradients.")
        return None, None

    # --- SETUP ATTACK ARGUMENTS ---
    attack_args = {
        "gradients": raw_gradients, "global_params": global_params,
        "lr": attack_params.get("attack_lr"), "iterations": attack_params.get("iterations"),
        "reg_tv": attack_params.get("reg_tv"), "reg_l2": attack_params.get("reg_l2")
    }

    # --- DATA TYPE SWITCHBOARD ---
    
    if data_type == "medmnist":
        attack_args.update({
            "model_class": MedMNIST_CNN, "input_shape": (1, 28, 28),
            "num_classes": main_config["data"]["num_classes"],
            "is_multilabel": main_config["data"].get("dataset_name", "") == "chestmnist"
        })
    
    elif data_type == "cifar10":
        attack_args.update({
            "model_class": CifarCNN, "input_shape": (1, 32, 32),
            "num_classes": main_config["data"]["num_classes"], "is_multilabel": False
        })

    elif data_type == "casting":
        attack_args.update({
            "model_class": CastingCNN, "input_shape": (1, 128, 128),
            "num_classes": main_config["data"]["num_classes"], "is_multilabel": False
        })

    elif data_type == "sensor":
        X, y, _ = load_and_preprocess_data(main_config["data"]["path"])
        num_features = X.shape[1]
        num_classes = len(np.unique(y))
        
        # Use the dedicated numerical attack
        return numerical_dlg_attack(
            gradients=raw_gradients, model_class=SensorMLP,
            num_features=num_features, num_classes=num_classes,
            global_params=global_params, lr=attack_params.get("attack_lr"),
            iterations=attack_params.get("iterations")
        )

    else:
        print(f"Unknown data type for reconstruction: {data_type}")
        return None, None

    # --- ATTACK TYPE SWITCHBOARD (for image data) ---
    
    if attack_type == "dlg":
        attack_args.pop("reg_tv"); attack_args.pop("reg_l2"); attack_args.pop("is_multilabel", None)
        return dlg_attack(**attack_args)
    elif attack_type == "idlg":
        return idlg_attack(**attack_args)
    elif attack_type == "mdlg":
        attack_args.pop("is_multilabel", None)
        return mdlg_attack(**attack_args)
    elif attack_type == "ggl":
            return ggl_medmnist_attack(
                gradients=raw_gradients,
                global_params=global_params,
                num_classes=main_config["data"]["num_classes"],
                num_restarts=attack_params.get("num_seeds", 4),
                lr=attack_params.get("attack_lr"),
                iterations=attack_params.get("iterations"),
                reg_tv=attack_params.get("reg_tv"),
                reg_l2=attack_params.get("reg_l2")
            )
    else:
        print(f"Attack type '{attack_type}' is not supported for {data_type}.")
        return None, None


def save_reconstruction(
    data: np.ndarray,
    predicted_labels: Optional[torch.Tensor],
    client_id: int,
    round_num: int,
    reconstruction_dir: str,
    data_type: str,
    original_data: Optional[np.ndarray] = None,
    original_labels: Optional[np.ndarray] = None
):
    """Saves the reconstructed data for inspection."""
    if data is None: return
    
    if data_type in ["image", "casting", "cifar10", "medmnist"]:
        if data.min() < 0: data = (data / 2 + 0.5)
        recon_tensor = torch.from_numpy(data.clip(0, 1))
        
        if original_data is not None:
            original_tensor = torch.from_numpy(original_data)
            comparison_grid = torch.cat([original_tensor, recon_tensor], dim=0)
            save_path = os.path.join(reconstruction_dir, f"comparison_client{client_id}_round{round_num}.png")
            torchvision.utils.save_image(comparison_grid, save_path, nrow=original_tensor.size(0))
            print(f"[Attack] Saved image comparison grid to {save_path}")
        else:
            save_path = os.path.join(reconstruction_dir, f"reconstruction_client{client_id}_round{round_num}.png")
            torchvision.utils.save_image(recon_tensor, save_path)
            print(f"[Attack] Saved reconstructed image to {save_path}")

    elif data_type == "sensor":
        save_path = os.path.join(reconstruction_dir, f"reconstruction_client{client_id}_round{round_num}.txt")
        with open(save_path, "w") as f:
            f.write("--- Original Data ---\n")
            if original_labels is not None: f.write(f"Label: {original_labels}\n")
            f.write("Features:\n" + str(original_data))
            f.write("\n\n--- Reconstructed Data ---\n")
            if predicted_labels is not None: f.write(f"Predicted Label: {predicted_labels}\n")
            f.write("Features:\n" + str(data))
        print(f"[Attack] Saved numerical reconstruction to {save_path}")
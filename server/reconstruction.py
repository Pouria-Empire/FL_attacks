import torch
import numpy as np
import os
import torchvision
from typing import List, Dict, Optional, Tuple

# Import all attack functions for both data types
from attacks.gradient_inversion import dlg_attack, mdlg_attack
from attacks.gradinversion_plus import gradinversion_group_attack
from attacks.ggl_attack import ggl_attack
from attacks.temporal_attack import temporal_attack
from attacks.numerical_attacks import numerical_dlg_attack
from utils_data.sensor_data_util import load_and_preprocess_data
from attacks.ggl_plus_attack import ggl_group_attack


def reconstruct_data(
    gradients_list: List[List[np.ndarray]],
    attack_params: Dict,
    client_config: Dict,
    data_type: str,
    main_config: Dict
) -> Optional[Tuple[np.ndarray, Optional[torch.Tensor]]]:
    """
    Selects and runs the appropriate gradient inversion attack based on the data type.
    """
    attack_type = attack_params.get("type", "dlg")
    print(f"[Attack] Attempting reconstruction for '{data_type}' data using '{attack_type}' method.")

    # This check gracefully handles encrypted payloads, preventing a crash.
    raw_gradients = gradients_list[0]
    if len(raw_gradients) == 1 and raw_gradients[0].dtype == np.uint8:
        print(f"[Attack Failed] Received encrypted gradients. Encryption defense was successful.")
        return None

    # --- IMAGE RECONSTRUCTION LOGIC ---
    if data_type == "image":
        # All image attacks operate on a single gradient from the list
        gradients = gradients_list[0]

        if attack_type == "dlg":
            return dlg_attack(
                gradients=gradients,
                lr=attack_params.get("attack_lr"),
                iterations=attack_params.get("iterations")
            )
        elif attack_type == "mdlg":
            return mdlg_attack(
                gradients=gradients,
                lr=attack_params.get("attack_lr"),
                iterations=attack_params.get("iterations")
            )
        elif attack_type == "ggl":
            return ggl_attack(
                gradients=gradients, lr=attack_params.get("attack_lr"),
                iterations=attack_params.get("iterations")
            )
        elif attack_type == "gradinversion_plus":
            return gradinversion_group_attack(
                gradients=gradients, batch_size=client_config.get("batch_size", 8),
                num_seeds=attack_params.get("num_seeds", 4), lr=attack_params.get("attack_lr"),
                iterations=attack_params.get("iterations"), reg_tv=attack_params.get("reg_tv", 1e-4),
                reg_l2=attack_params.get("reg_l2", 1e-5), reg_group=attack_params.get("reg_group", 0.005)
            )
        elif attack_type == "ggl_plus":
            batch_size = attack_params.get("attack_batch_size", client_config.get("batch_size"))
            print(f"[Attack] Using batch size: {batch_size} for GGL+ reconstruction.")
            return ggl_group_attack(
                gradients=gradients,
                batch_size=batch_size,
                num_seeds=attack_params.get("num_seeds", 4),
                lr=attack_params.get("attack_lr"),
                iterations=attack_params.get("iterations"),
                reg_tv=attack_params.get("reg_tv"),
                reg_l2=attack_params.get("reg_l2"),
                reg_group=attack_params.get("reg_group")
            )
        else:
            print(f"Image attack type '{attack_type}' is not supported in this configuration.")
            return None

    # --- NUMERICAL RECONSTRUCTION LOGIC ---
    elif data_type == "sensor":
        print("[Attack] Launching numerical DLG attack.")
        X, y, _ = load_and_preprocess_data(main_config["data"]["path"])
        num_features = X.shape[1]
        num_classes = len(np.unique(y))
        
        return numerical_dlg_attack(
            gradients_list[0],
            num_features=num_features,
            num_classes=num_classes,
            lr=attack_params.get("attack_lr"),
            iterations=attack_params.get("iterations")
        )

    else:
        print(f"Unknown data type for reconstruction: {data_type}")
        return None


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
    """Saves the reconstructed data based on its type (image or numerical)."""
    if data_type == "image":
        recon_tensor = torch.from_numpy(data)
        if original_data is not None:
            original_tensor = torch.from_numpy(original_data)
            
            if original_tensor.shape[0] != recon_tensor.shape[0]:
                print(f"🔴 WARNING: Mismatch between original ({original_tensor.shape[0]}) and reconstructed ({recon_tensor.shape[0]}) batch sizes.")
                save_path = os.path.join(reconstruction_dir, f"reconstruction_client{client_id}_round{round_num}.png")
                torchvision.utils.save_image(recon_tensor, save_path, nrow=recon_tensor.shape[0])
                return

            # Create the vertical grid: original images on top, reconstructed below
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
            if original_labels is not None:
                f.write(f"Label: {original_labels}\n")
            f.write("Features:\n")
            f.write(str(original_data))
            f.write("\n\n--- Reconstructed Data ---\n")
            if predicted_labels is not None:
                f.write(f"Predicted Label: {predicted_labels}\n")
            f.write("Features:\n")
            f.write(str(data))
        print(f"[Attack] Saved numerical reconstruction to {save_path}")
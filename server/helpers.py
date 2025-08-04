import yaml
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from torch.utils.data import DataLoader
import os

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
from scipy.spatial import distance

from model import SimpleNN, SensorMLP
from utils_data.chest_data_util import FINDINGS

def load_config() -> Dict[str, Any]:
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def test_and_log_misclassifications(
    model: torch.nn.Module,
    test_loader: DataLoader,
    is_backdoor_test: bool,
    target_label: int,
    is_image: bool
) -> Tuple[float, float]:
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = torch.nn.BCEWithLogitsLoss() if is_image else torch.nn.CrossEntropyLoss()
    log_file = "backdoor_misclassifications.log"

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            total_loss += criterion(outputs, labels).item() * data.size(0)
            total += labels.size(0)

            if is_image:
                predicted = torch.sigmoid(outputs) > 0.5
                correct_predictions = (predicted == labels.byte()).all(dim=1)
                correct += correct_predictions.sum().item()
            else: # Numerical data
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions = (predicted == labels)
                correct += correct_predictions.sum().item()
                if is_backdoor_test:
                    successful_attack_indices = (predicted == target_label).nonzero(as_tuple=False).squeeze()
                    if successful_attack_indices.numel() > 0:
                        if successful_attack_indices.dim() == 0:
                            successful_attack_indices = [successful_attack_indices.item()]
                        else:
                            successful_attack_indices = successful_attack_indices.tolist()
                        for idx_in_batch in successful_attack_indices:
                            true_label = labels[idx_in_batch].item()
                            if true_label != target_label:
                                with open(log_file, "a") as f:
                                    f.write(f"SUCCESSFUL MISCLASSIFICATION:\n")
                                    f.write(f"  - Original Label: {true_label}\n")
                                    f.write(f"  - Model Predicted: {target_label} (due to trigger)\n\n")

    return total_loss / total if total > 0 else 0, correct / total if total > 0 else 0

def safe_metrics_aggregation(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregates metrics and prints them clearly."""
    aggregated = {}
    if any("accuracy" in m for _, m in metrics):
        aggregated["accuracy"] = np.mean([m["accuracy"] for _, m in metrics if "accuracy" in m])
    if any("backdoor_asr" in m for _, m in metrics):
        aggregated["backdoor_asr"] = np.mean([m["backdoor_asr"] for _, m in metrics if "backdoor_asr" in m])
    
    print("\n[Round Metrics]")
    if "accuracy" in aggregated:
        print(f"  Normal Accuracy: {aggregated['accuracy']*100:.2f}%")
    if "backdoor_asr" in aggregated and aggregated["backdoor_asr"] > 0:
        print(f"  Attack Success Rate (ASR): {aggregated['backdoor_asr']*100:.2f}%")
        
    fit_durations = [m["fit_duration"] for _, m in metrics if "fit_duration" in m]
    if fit_durations:
        print(f"Avg. Client Fit Time: {np.mean(fit_durations):.4f} seconds")
            
    return aggregated

def evaluate_reconstruction(original_batch: np.ndarray, recon_batch: np.ndarray):
    """Calculates and prints a comprehensive set of similarity metrics for images."""
    psnr_scores, ssim_scores, mse_scores, cos_sims = [], [], [], []
    noise_batch, ssim_noise_scores = np.random.rand(*original_batch.shape), []
    for i in range(original_batch.shape[0]):
        original_img, recon_img, noise_img = original_batch[i].squeeze(), recon_batch[i].squeeze(), noise_batch[i].squeeze()
        data_range = original_img.max() - original_img.min()
        psnr_scores.append(psnr(original_img, recon_img, data_range=data_range))
        ssim_scores.append(ssim(original_img, recon_img, data_range=data_range))
        mse_scores.append(mse(original_img, recon_img))
        cos_sims.append(1 - distance.cosine(original_img.flatten(), recon_img.flatten()))
        ssim_noise_scores.append(ssim(original_img, noise_img, data_range=data_range))
    avg_psnr, avg_ssim, avg_mse, avg_cos_sim = np.mean(psnr_scores), np.mean(ssim_scores), np.mean(mse_scores), np.mean(cos_sims)
    avg_ssim_noise = np.mean(ssim_noise_scores)
    rdlv = (avg_ssim - avg_ssim_noise) / (1 - avg_ssim_noise) if (1 - avg_ssim_noise) != 0 else 0
    print("\n--- Image Reconstruction Quality Metrics ---")
    print(f"  MSE (↓ is better):      {avg_mse:.4f}")
    print(f"  PSNR (↑ is better):     {avg_psnr:.2f} dB")
    print(f"  SSIM (↑ is better):     {avg_ssim:.4f}")
    print(f"  Cosine Sim (↑ is better):{avg_cos_sim:.4f}")
    print(f"  RDLV (↑ is better):     {rdlv:.4f}")
    print("------------------------------------------")

def evaluate_reconstruction_numerical(original_data: np.ndarray, recon_data: np.ndarray):
    """Calculates and prints similarity metrics for numerical data."""
    mse_score = np.mean((original_data - recon_data)**2)
    mae_score = np.mean(np.abs(original_data - recon_data))
    cos_sim = 1 - distance.cosine(original_data.flatten(), recon_data.flatten()) if original_data.std() > 0 and recon_data.std() > 0 else 0

    print("\n--- Numerical Reconstruction Quality Metrics ---")
    print(f"  MSE (↓ is better):      {mse_score:.4f}")
    print(f"  MAE (↓ is better):      {mae_score:.4f}")
    print(f"  Cosine Sim (↑ is better):{cos_sim:.4f}")
    print("------------------------------------------")
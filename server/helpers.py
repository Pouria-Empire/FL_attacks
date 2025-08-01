import yaml
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from torch.utils.data import DataLoader
import os

# Import metrics
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
from scipy.spatial import distance

# Import project-specific modules
from model import SimpleNN, SensorMLP
from utils_data.chest_data_util import FINDINGS

def load_config() -> Dict[str, Any]:
    """Load the YAML configuration file."""
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]):
    """Sets the parameters of a PyTorch model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """Gets the parameters of a PyTorch model."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def test_and_log_misclassifications(
    model: torch.nn.Module, 
    test_loader: DataLoader, 
    is_backdoor_test: bool, 
    target_label_idx: int,
    is_image: bool
) -> Tuple[float, float]:
    """Tests the model and logs misclassifications, handles both data types."""
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
                if is_backdoor_test:
                    misclassified_indices = (~correct_predictions).nonzero(as_tuple=False).squeeze()
                    if misclassified_indices.numel() > 0:
                        if misclassified_indices.dim() == 0:
                            misclassified_indices = [misclassified_indices.item()]
                        else:
                            misclassified_indices = misclassified_indices.tolist()
                        try:
                            df = test_loader.dataset.dataset.df
                            for idx_in_batch in misclassified_indices:
                                original_idx = test_loader.dataset.indices[idx_in_batch]
                                filename = df.iloc[original_idx]['Image Index']
                                true_labels = df.iloc[original_idx]['Finding Labels']
                                with open(log_file, "a") as f:
                                    f.write(f"MISCLASSIFICATION on {filename}:\n  - True: {true_labels}, Predicted: {FINDINGS[target_label_idx]}\n")
                        except Exception:
                            pass
            else: # Numerical data
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions = (predicted == labels)
                correct += correct_predictions.sum().item()

    return total_loss / total if total > 0 else 0, correct / total if total > 0 else 0

def safe_metrics_aggregation(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregates metrics from clients."""
    aggregated = {}
    if any("accuracy" in m for _, m in metrics):
        aggregated["accuracy"] = np.mean([m["accuracy"] for _, m in metrics if "accuracy" in m])
        
    if any("backdoor_asr" in m for _, m in metrics):
        aggregated["backdoor_asr"] = np.mean([m["backdoor_asr"] for _, m in metrics if "backdoor_asr" in m])
    
    print("\n[Round Metrics]")
    if "accuracy" in aggregated: print(f"  Normal Accuracy: {aggregated['accuracy']*100:.2f}%")
    if "backdoor_asr" in aggregated: print(f"  Attack Success Rate (ASR): {aggregated['backdoor_asr']*100:.2f}%")
        
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
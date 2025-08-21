import yaml
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from torch.utils.data import DataLoader
import os

# Import metrics
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
from scipy.spatial import distance

# Import project-specific models
from model import CastingCNN, SensorMLP, CifarCNN

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
    target_label: int,
    data_type: str,
    class_names: List[str]
) -> Tuple[float, float]:
    """
    Tests the model, handles all data types, and logs backdoor misclassifications.
    """
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    
    is_binary_image = (data_type == "casting")
    criterion = torch.nn.BCEWithLogitsLoss() if is_binary_image else torch.nn.CrossEntropyLoss()
    log_file = "backdoor_misclassifications.log"
    
    successful_flips = 0
    total_non_target = 0

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            total += labels.size(0)

            if is_binary_image:
                labels_for_loss = labels.float().view(-1, 1)
                total_loss += criterion(outputs, labels_for_loss).item() * data.size(0)
                predicted = torch.sigmoid(outputs) > 0.5
                correct += (predicted == labels_for_loss).sum().item()
            else: # Sensor or CIFAR-10
                total_loss += criterion(outputs, labels).item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            
            if is_backdoor_test:
                for i in range(len(predicted)):
                    true_label = labels[i].item()
                    predicted_label = predicted[i].item()
                    
                    if true_label != target_label:
                        total_non_target += 1
                        if predicted_label == target_label:
                            successful_flips += 1
                            with open(log_file, "a") as f:
                                f.write(f"SUCCESSFUL MISCLASSIFICATION ({data_type.upper()}):\n")
                                f.write(f"  - Original Label: {class_names[true_label] if class_names else true_label}\n")
                                f.write(f"  - Model Predicted: {class_names[target_label] if class_names else target_label} (due to trigger)\n\n")
    
    if is_backdoor_test:
        accuracy = successful_flips / total_non_target if total_non_target > 0 else 0
        print(f"  - Backdoor ASR Details: {successful_flips} successful flips out of {total_non_target} vulnerable samples.")
    else:
        accuracy = correct / total if total > 0 else 0

    return total_loss / total if total > 0 else 0, accuracy

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
    
    if "backdoor_asr" in aggregated:
        print(f"  Attack Success Rate (ASR): {aggregated['backdoor_asr']*100:.2f}%")
        
    fit_durations = [m["fit_duration"] for _, m in metrics if "fit_duration" in m]
    if fit_durations:
        avg_fit_duration = np.mean(fit_durations)
        print(f"Avg. Client Fit Time: {avg_fit_duration:.4f} seconds")
            
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
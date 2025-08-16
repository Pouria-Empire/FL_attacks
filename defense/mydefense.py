import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from torch.utils.data import DataLoader

# Import the metric for image leakage calculation
from skimage.metrics import structural_similarity as ssim

# Import all model types to be used by the agent
from model import SimpleNN, SensorMLP, CastingCNN

from crypto_utils import chaotic_map_obfuscate

def chaotic_encryption(params: List[np.ndarray], key: float = 3.99) -> List[np.ndarray]:
    return chaotic_map_obfuscate(params, key)

# --- HELPER FUNCTIONS ---
def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]):
    """Sets the parameters of a PyTorch model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def test(model: torch.nn.Module, test_loader: DataLoader, is_image: bool) -> Tuple[float, float]:
    """Generic test function that handles both image and sensor data."""
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = torch.nn.BCEWithLogitsLoss() if is_image else torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            total += labels.size(0)
            
            if is_image:
                labels = labels.float().view(-1, 1)
                total_loss += criterion(outputs, labels).item() * data.size(0)
                predicted = torch.sigmoid(outputs) > 0.5
                correct += (predicted == labels).sum().item()
            else: # Sensor data
                total_loss += criterion(outputs, labels).item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

    return total_loss / total if total > 0 else 0, correct / total if total > 0 else 0

# --- 1. Client Utility Evaluation ---
def calculate_client_utility(
    global_model_params: List[np.ndarray],
    client_update_params: List[np.ndarray],
    verification_loader: DataLoader,
    model_class, 
    model_args: dict,
    is_image: bool
) -> float:
    """Calculates the utility of a client's update by measuring the change in accuracy."""
    model = model_class(**model_args)
    
    set_parameters(model, global_model_params)
    _, base_accuracy = test(model, verification_loader, is_image)

    set_parameters(model, client_update_params)
    _, new_accuracy = test(model, verification_loader, is_image)

    return new_accuracy - base_accuracy

# --- 2. Data Leakage Evaluation (MyFunc) ---
def my_func_image_leakage(original_image, reconstructed_image, alpha, beta, gamma) -> float:
    perceptual_similarity = ssim(
        original_image.squeeze(),
        reconstructed_image.squeeze(),
        data_range=original_image.max() - original_image.min()
    )
    return beta * perceptual_similarity


def my_func_numerical_leakage(original_data, reconstructed_data, alpha, beta, gamma) -> float:
    mse = np.mean((original_data - reconstructed_data) ** 2)
    quantitative_similarity = 1 / (1 + mse)
    return gamma * quantitative_similarity


# --- 3. The MyDefense Agent ---
class MyDefenseAgent:
    def __init__(self, config: dict, server_holdout_loader: DataLoader, model_class, model_args: dict):
        self.config = config
        self.params = config.get("mitigations", {}).get("mydefense_params", {})
        self.utility_threshold = self.params.get("utility_threshold", 0.0)
        self.leakage_threshold = self.params.get("leakage_threshold", 0.8)
        self.server_holdout_loader = server_holdout_loader
        self.trigger_chaotic_obfuscation_for_client = {}
        self.model_class = model_class
        self.model_args = model_args
        self.data_type = self.config.get("data", {}).get("type", "image")
        self.is_image = self.data_type == "image"

    def decide_and_defend(
        self,
        client_id: int,
        global_model_params: List[np.ndarray],
        clean_update_params: List[np.ndarray], # The de-obfuscated params
        reconstruction_result: Optional[Tuple[np.ndarray, torch.Tensor]],
        original_data: Optional[np.ndarray]
    ) -> bool:
        
        # --- Criterion 1: Client Contribution Utility (based on CLEAN params) ---
        client_utility = calculate_client_utility(
            global_model_params, clean_update_params, self.server_holdout_loader,
            self.model_class, self.model_args, self.is_image
        )
        print(f"  - Client {client_id} Contribution Utility Score: {client_utility:.4f}")
        if client_utility < self.utility_threshold:
            print(f"  - DECISION: Client contribution utility is too low. REJECTING update.")
            return False

        # --- Criterion 2: Data Leakage (based on reconstruction from RAW/OBFUSCATED params) ---
        if reconstruction_result is not None and original_data is not None:
            reconstructed_data, _ = reconstruction_result
            
            leakage_fn = my_func_image_leakage if self.is_image else my_func_numerical_leakage
            leakage_score = leakage_fn(
                original_data, reconstructed_data,
                alpha=self.params.get("alpha", 0.0),
                beta=self.params.get("beta", 1.0),
                gamma=self.params.get("gamma", 1.0)
            )
            print(f"  - Client {client_id} Data Leakage (MyFunc): {leakage_score:.4f}")
            
            if leakage_score > self.leakage_threshold:
                print(f"  - DECISION: High data leakage. Triggering chaotic obfuscation for Client {client_id}.")
                self.trigger_chaotic_obfuscation_for_client[client_id] = True

        print(f"  - DECISION: Client {client_id} update is acceptable. ACCEPTING.")
        return True
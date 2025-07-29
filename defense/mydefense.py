import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from skimage.metrics import structural_similarity as ssim

from model import SimpleNN

# --- HELPER FUNCTIONS ---
def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def test(model, test_loader):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            predicted = torch.sigmoid(outputs) > 0.5
            total += labels.size(0)
            correct += (predicted == labels.byte()).all(dim=1).sum().item()
    return total_loss / total if total > 0 else 0, correct / total if total > 0 else 0

# --- 1. Client Utility Evaluation ---
def calculate_client_utility(
    global_model_params: List[np.ndarray],
    client_update_params: List[np.ndarray],
    server_holdout_loader: torch.utils.data.DataLoader
) -> float:
    model = SimpleNN(num_classes=15)
    set_parameters(model, global_model_params)
    _, base_accuracy = test(model, server_holdout_loader)
    
    # Use a simple average to estimate impact
    temp_model_params = [(g + c) / 2 for g, c in zip(global_model_params, client_update_params)]
    set_parameters(model, temp_model_params)
    _, new_accuracy = test(model, server_holdout_loader)
    
    return new_accuracy - base_accuracy

# --- 2. Data Leakage Evaluation (MyFunc) ---
def my_func_image_leakage(
    original_image: np.ndarray,
    reconstructed_image: np.ndarray,
    alpha: float, beta: float, gamma: float
) -> float:
    perceptual_similarity = ssim(
        original_image.squeeze(),
        reconstructed_image.squeeze(),
        data_range=original_image.max() - original_image.min()
    )
    return beta * perceptual_similarity

# --- 3. Chaotic Encryption Simulation ---
def chaotic_encryption(params: List[np.ndarray], key: float = 3.99) -> List[np.ndarray]:
    encrypted_params = []
    x = 0.5
    for p in params:
        mask = np.zeros_like(p, dtype=np.float32)
        flat_mask = mask.flatten()
        for i in range(len(flat_mask)):
            x = key * x * (1 - x)
            flat_mask[i] = x
        mask = flat_mask.reshape(p.shape)
        encrypted_params.append(p + mask.astype(p.dtype))
    return encrypted_params

# --- 4. The MyDefense Agent ---
class MyDefenseAgent:
    def __init__(self, config: dict, server_holdout_loader: torch.utils.data.DataLoader):
        self.config = config
        self.params = config.get("mitigations", {}).get("mydefense_params", {})
        self.utility_threshold = self.params.get("utility_threshold", 0.0)
        self.leakage_threshold = self.params.get("leakage_threshold", 0.8)
        self.server_holdout_loader = server_holdout_loader
        self.trigger_chaotic_encryption_for_client = {}

    def decide_and_defend(
        self,
        client_id: int,
        global_model_params: List[np.ndarray],
        client_update_params: List[np.ndarray],
        reconstruction_result: Optional[Tuple[np.ndarray, torch.Tensor]],
        original_data: Optional[np.ndarray]
    ) -> bool:
        
        client_utility = calculate_client_utility(
            global_model_params, client_update_params, self.server_holdout_loader
        )
        print(f"  - Client {client_id} Utility Score: {client_utility:.4f}")

        # --- THE FIX: Change 'if reconstruction_result' to 'if reconstruction_result is not None' ---
        if reconstruction_result is not None and original_data is not None:
            reconstructed_images, _ = reconstruction_result
            leakage_score = my_func_image_leakage(
                original_data[0], reconstructed_images[0],
                alpha=self.params.get("alpha", 0.2),
                beta=self.params.get("beta", 0.6),
                gamma=self.params.get("gamma", 0.2)
            )
            print(f"  - Client {client_id} Data Leakage (MyFunc): {leakage_score:.4f}")
            
            if leakage_score > self.leakage_threshold:
                print(f"  - DECISION: High data leakage. Triggering chaotic encryption for Client {client_id}.")
                self.trigger_chaotic_encryption_for_client[client_id] = True

        if client_utility < self.utility_threshold:
            print(f"  - DECISION: Client {client_id} utility too low. REJECTING update.")
            return False

        print(f"  - DECISION: Client {client_id} update acceptable. ACCEPTING.")
        return True
import flwr as fl
import torch
import argparse
import yaml
import numpy as np
import os
import pickle
import time
from torch.utils.data import DataLoader
from typing import Tuple, List

# Import custom project modules
from model import CastingCNN
from utils_data.casting_data_util import get_client_data
from attacks.data_poisoning import PoisonedDataset
from attacks.model_poisoning import scaling_attack
from attacks.defenses import gradient_clipping, gradient_sparsification, add_differential_privacy
from crypto_utils import encrypt_params, decrypt_params
from defense.mydefense import chaotic_encryption

# --- HELPER FUNCTIONS ---
def load_config():
    """Loads the main config.yml file."""
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def train(model: torch.nn.Module, train_loader: DataLoader, epochs: int, lr: float):
    """Train for binary classification."""
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            labels = labels.float().view(-1, 1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return

def test(model: torch.nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
    """Validate for binary classification."""
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.float().view(-1, 1)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            predicted = torch.sigmoid(outputs) > 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / total if total > 0 else 0, correct / total if total > 0 else 0

# --- FLOWER CLIENT ---
class ImageFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, config: dict):
        self.cid = cid
        self.client_id_numeric = int(cid.replace("client", ""))
        self.config = config
        self.attack_config = config.get("attacks", {})
        self.client_config = config.get("clients", {})
        self.data_config = config.get("data", {})
        
        self.model = CastingCNN(num_classes=self.data_config["num_classes"])
        
        self.trainset, self.testset = get_client_data(
            cid=self.cid,
            total_clients=self.client_config["total"],
            data_path=self.data_config["path"],
            img_size=self.data_config["img_size"]
        )
        
        dp_params = self.attack_config.get("data_poisoning", {})
        is_dp_malicious = (dp_params.get("enable", False) and self.client_id_numeric in dp_params.get("malicious_clients", []))
        if is_dp_malicious:
            print(f"Client {self.client_id_numeric}: Applying data poisoning.")
            self.trainset = PoisonedDataset(
                dataset=self.trainset, 
                poison_frac=dp_params.get("poison_frac", 0.3), 
                target_label=dp_params.get("target_label", 1)
            )

        self.trainloader = DataLoader(self.trainset, batch_size=self.client_config["batch_size"], shuffle=True)
        self.testloader = DataLoader(self.testset, batch_size=self.client_config["batch_size"])

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        if len(parameters) == 1 and parameters[0].dtype == np.uint8:
            try:
                parameters = decrypt_params(parameters[0].tobytes())
            except Exception:
                return
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        fit_start_time = time.time()
        
        self.set_parameters(parameters)
        original_parameters = self.get_parameters({})
        
        gi_params = self.attack_config.get("gradient_inversion", {})
        is_gi_target = (gi_params.get("enable", False) and self.client_id_numeric == gi_params.get("target_client"))
        params_to_send, num_examples = None, 0
        metrics = {}

        if is_gi_target:
            attack_type = gi_params.get("type", "dlg")
            print(f"Client {self.client_id_numeric}: Acting as Gradient Inversion target ({attack_type}).")
            
            if attack_type in ["gradinversion", "gradinversion_plus"]:
                batch_data, batch_target = next(iter(self.trainloader))
            else: # dlg, mdlg, ggl
                single_item_loader = DataLoader(self.trainset, batch_size=1, shuffle=True)
                batch_data, batch_target = next(iter(single_item_loader))
            
            os.makedirs("client_data", exist_ok=True)
            with open(f"client_data/client_{self.client_id_numeric}_image_data.pkl", "wb") as f:
                pickle.dump({'data': batch_data.numpy(), 'label': batch_target.numpy()}, f)
            
            self.model.train()
            criterion = torch.nn.BCEWithLogitsLoss()
            output = self.model(batch_data)
            loss = criterion(output, batch_target.float().view(-1, 1))
            gradients = torch.autograd.grad(loss, self.model.parameters())
            params_to_send = [grad.cpu().numpy() for grad in gradients]
            num_examples = len(batch_data)
            metrics["attack"] = "gradient_inversion"
        else: # Standard or Model Poisoning Client
            train(self.model, self.trainloader, self.client_config["local_epochs"], self.client_config["learning_rate"])
            new_params = self.get_parameters({})
            update_delta = [new - old for new, old in zip(new_params, original_parameters)]
            
            params_to_defend = update_delta
            mp_params = self.attack_config.get("model_poisoning", {})
            if (mp_params.get("enable", False) and self.client_id_numeric in mp_params.get("malicious_clients", [])):
                params_to_defend = scaling_attack(params_to_defend, mp_params.get("scale_factor", -1.0))
            
            final_params = [original + defended for original, defended in zip(original_parameters, params_to_defend)]
            params_to_send = final_params
            num_examples = len(self.trainset)

        # --- APPLY DEFENSES ---
        defense_start_time = time.time()
        if config.get("apply_chaotic_encryption", False):
            params_to_send = chaotic_encryption(params_to_send)
        
        defense_type = config.get("defense_type")
        if defense_type == "clipping":
            params_to_send = gradient_clipping(params_to_send, config.get("clipping_norm"))
        elif defense_type == "sparsification":
            params_to_send = gradient_sparsification(params_to_send, config.get("sparsity"))
        elif defense_type == "dp":
            params_to_send = add_differential_privacy(params_to_send, config.get("clipping_norm"), config.get("noise_multiplier"))
        elif defense_type == "encryption":
            encrypted_bytes = encrypt_params(params_to_send)
            params_to_send = [np.frombuffer(encrypted_bytes, dtype=np.uint8)]
        
        defense_duration = time.time() - defense_start_time
        fit_duration = time.time() - fit_start_time
        
        metrics.update({
            "fit_duration": fit_duration, 
            "defense_duration": defense_duration,
            "logical_client_id": self.client_id_numeric,
            "data_type": "image"
        })
        
        return params_to_send, num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}
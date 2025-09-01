# clients/sensor_client.py

# Standard library imports
import os
import pickle
import time
from typing import Tuple, List

# Third-party library imports
import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Project-specific imports
from model import SensorMLP
from utils_data.sensor_data_util import get_client_data, load_and_preprocess_data
from attacks.data_poisoning import PoisonedDatasetWrapper
from attacks.model_poisoning import scaling_attack
from attacks.defenses import gradient_clipping, add_differential_privacy, gradient_sparsification
from crypto_utils import decrypt_params, encrypt_params, chaotic_map_obfuscate


def train(model: torch.nn.Module, train_loader: DataLoader, epochs: int, lr: float, cid: str):
    """The standard training loop for a sensor data client."""
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Client {cid} - Epoch {epoch+1}/{epochs}")
        for features, labels in pbar:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels.squeeze().long())
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})
    return

def test(model: torch.nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
    """The standard evaluation loop for a sensor data client."""
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            labels = labels.squeeze().long()
            total_loss += criterion(outputs, labels).item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / total if total > 0 else 0
    return avg_loss, accuracy

def _sizeof_parameters(params: List[np.ndarray]) -> int:
    """Helper function to calculate the total size of model parameters in bytes."""
    if not params: return 0
    if len(params) == 1 and isinstance(params[0], np.ndarray) and params[0].dtype == np.uint8:
        return int(params[0].nbytes)
    return int(sum(p.nbytes for p in params))


class SensorFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, config: dict):
        self.cid = cid
        self.client_id_numeric = int(cid.replace("client", ""))
        self.config = config
        self.attack_config = config.get("attacks", {})
        self.client_config = config.get("clients", {})
        self.data_config = config.get("data", {})
        self.data_type = "sensor"

        # --- Load Sensor Model and Data ---
        X, y, _ = load_and_preprocess_data(self.data_config["path"])
        self.num_features = X.shape[1]
        self.num_classes = len(np.unique(y))
        
        self.model = SensorMLP(input_features=self.num_features, num_classes=self.num_classes)
        
        self.trainset, self.testset = get_client_data(
            cid=self.cid, total_clients=self.client_config["total"],
            csv_path=self.data_config["path"]
        )
        
        # --- Attack and Dataloader Setup ---
        dp_params = self.attack_config.get("data_poisoning", {})
        if (dp_params.get("enable", False) and self.client_id_numeric in dp_params.get("malicious_clients", [])):
            print(f"Client {self.cid}: Applying data poisoning.")
            self.trainset = PoisonedDatasetWrapper(dataset=self.trainset, data_type=self.data_type, **dp_params)

        gi_params = self.attack_config.get("gradient_inversion", {})
        is_gi_target = (gi_params.get("enable", False) and self.client_id_numeric == gi_params.get("target_client"))
        batch_size = gi_params.get("attack_batch_size") if is_gi_target else self.client_config["batch_size"]
        if is_gi_target: print(f"Client {self.cid} (GI Target): Using attack batch size of {batch_size}")

        self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.testloader = DataLoader(self.testset, batch_size=self.client_config["batch_size"], num_workers=0)

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        if len(parameters) == 1 and parameters[0].dtype == np.uint8:
            try: parameters = decrypt_params(parameters[0].tobytes())
            except Exception: return
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        fit_start_time = time.time()
        self.set_parameters(parameters)
        metrics = {}
        bytes_down = _sizeof_parameters(parameters)

        gi_params = self.attack_config.get("gradient_inversion", {})
        is_gi_target = (gi_params.get("enable", False) and self.client_id_numeric == gi_params.get("target_client"))

        if is_gi_target:
            print(f"Client {self.cid}: Acting as GI Target. Processing one batch...")
            batch_data, batch_target = next(iter(self.trainloader))
            os.makedirs("client_data", exist_ok=True)
            with open(f"client_data/client_{self.client_id_numeric}_{self.data_type}_data.pkl", "wb") as f:
                pickle.dump({'data': batch_data.numpy(), 'label': batch_target.numpy()}, f)

            self.model.train()
            criterion = torch.nn.CrossEntropyLoss()
            output = self.model(batch_data)
            loss = criterion(output, batch_target.squeeze().long())
            
            gradients = torch.autograd.grad(loss, self.model.parameters())
            params_to_send = [grad.cpu().numpy() for grad in gradients]
            num_examples = len(batch_data)
            metrics["attack"] = "gradient_inversion"
        else:
            print(f"Client {self.cid}: Starting full local training...")
            train(self.model, self.trainloader, self.client_config["local_epochs"], self.client_config["learning_rate"], self.cid)
            
            new_params = self.get_parameters({})
            num_examples = len(self.trainset)
            update = [new - old for new, old in zip(new_params, parameters)]

            mp_params = self.attack_config.get("model_poisoning", {})
            if (mp_params.get("enable", False) and self.client_id_numeric in mp_params.get("malicious_clients", [])):
                update = scaling_attack(update, mp_params.get("scale_factor", -1.0))
            
            defense_type = config.get("defense_type")
            if defense_type == "clipping": update = gradient_clipping(update, config.get("clipping_norm"))
            elif defense_type == "dp": update = add_differential_privacy(update, config.get("clipping_norm"), config.get("noise_multiplier"))
            elif defense_type == "sparsification": update = gradient_sparsification(update, config.get("sparsity"))

            params_to_send = [orig + upd for orig, upd in zip(parameters, update)]

        if config.get("apply_chaotic_obfuscation", False):
            print(f"Client {self.cid}: Applying chaotic obfuscation as instructed by server.")
            params_to_send = chaotic_map_obfuscate(params=params_to_send, **config.get("mydefense_params", {}))
        
        if config.get("defense_type") == "encryption":
            print(f"Client {self.cid}: Encrypting update before sending.")
            encrypted_bytes = encrypt_params(params_to_send)
            params_to_send = [np.frombuffer(encrypted_bytes, dtype=np.uint8)]

        bytes_up = _sizeof_parameters(params_to_send)
        metrics.update({
            "fit_duration": time.time() - fit_start_time,
            "logical_client_id": self.client_id_numeric, "data_type": self.data_type,
            "was_chaotically_obfuscated": config.get("apply_chaotic_obfuscation", False),
            "bytes_down": bytes_down, "bytes_up": bytes_up,
        })
        return params_to_send, num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}
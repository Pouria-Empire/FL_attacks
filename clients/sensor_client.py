import flwr as fl
import torch
import argparse
import yaml
import numpy as np
import os
import pickle
import time
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List

# --- Import the components for the sensor dataset ---
from model import SensorMLP
from utils_data.sensor_data_util import get_client_data, load_and_preprocess_data
from attacks.numerical_attacks import PoisonedSensorDataset, numerical_dlg_attack
from attacks.model_poisoning import scaling_attack
from attacks.defenses import gradient_clipping, gradient_sparsification, add_differential_privacy
from crypto_utils import encrypt_params, decrypt_params
from defense.mydefense import chaotic_encryption

# --- HELPER FUNCTIONS ---
def load_config():
    with open("config.yml", "r") as f: return yaml.safe_load(f)

def train(model: torch.nn.Module, train_loader: DataLoader, epochs: int, lr: float):
    """Train for standard numerical classification."""
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(model: torch.nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
    """Validate for standard numerical classification."""
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            total_loss += criterion(outputs, labels).item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / total if total > 0 else 0, correct / total if total > 0 else 0

# --- FLOWER CLIENT ---
class SensorFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, config: dict):
        self.cid = cid
        self.client_id_numeric = int(cid.replace("client", ""))
        self.config = config
        self.attack_config = config.get("attacks", {})
        self.client_config = config.get("clients", {})
        self.data_config = config.get("data", {})
        
        X, y, _ = load_and_preprocess_data(self.data_config["path"])
        self.num_features = X.shape[1]
        self.num_classes = len(np.unique(y))
        
        self.model = SensorMLP(input_features=self.num_features, num_classes=self.num_classes)
        
        self.trainset, self.testset = get_client_data(
            cid=self.cid,
            total_clients=self.client_config["total"],
            csv_path=self.data_config["path"]
        )
        
        dp_params = self.attack_config.get("data_poisoning", {})
        is_dp_malicious = (dp_params.get("enable", False) and self.client_id_numeric in dp_params.get("malicious_clients", []))
        if is_dp_malicious:
            print(f"✅ Client {self.client_id_numeric}: CONFIRMED as malicious. Applying poisoning.")
            
            # --- DEBUGGING: Check a sample before and after ---
            poisoned_dataset = PoisonedSensorDataset(
                dataset=self.trainset, 
                poison_frac=dp_params.get("poison_frac", 0.3),
                target_label=dp_params.get("target_label", 0),
                trigger_noise_level=dp_params.get("trigger_noise_level", 0.1)
            )
            
            # Find an index that is guaranteed to be poisoned
            if len(poisoned_dataset.poison_indices) > 0:
                print("len(poisoned_dataset.poison_indices): "+str(len(poisoned_dataset.poison_indices)))
                check_idx = poisoned_dataset.poison_indices[0]
                
                # Get the "before" and "after" versions of this specific sample
                original_features, original_label = self.trainset[check_idx]
                poisoned_features, poisoned_label = poisoned_dataset[check_idx]

                if not torch.equal(original_features, poisoned_features):
                    print("✅ DEBUG: Data poisoning was successfully applied to a sample.")
                    print(f"   Original Label: {original_label.item()}, Poisoned Label: {poisoned_label.item()}")
                else:
                    print("🔴 DEBUG: WARNING! Data poisoning did NOT change the sample data.")
            else:
                print("🔴 DEBUG: WARNING! No samples were selected for poisoning.")
            
            # Now, assign the poisoned dataset to the client
            self.trainset = poisoned_dataset

        gi_params = self.attack_config.get("gradient_inversion", {})
        is_gi_target = (gi_params.get("enable", False) and self.client_id_numeric == gi_params.get("target_client"))

        if is_gi_target and "attack_batch_size" in gi_params:
            batch_size = gi_params["attack_batch_size"]
            print(f"Client {self.client_id_numeric} (Attacker): Using attack batch size of {batch_size}")
        else:
            batch_size = self.client_config["batch_size"]
        
        self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        self.testloader = DataLoader(self.testset, batch_size=self.client_config["batch_size"])

        self.trainloader = DataLoader(self.trainset, batch_size=self.client_config["batch_size"], shuffle=True)
        self.testloader = DataLoader(self.testset, batch_size=self.client_config["batch_size"])

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        if len(parameters) == 1 and parameters[0].dtype == np.uint8:
            try:
                print(f"Client {self.client_id_numeric}: Decrypting global model parameters.")
                parameters = decrypt_params(parameters[0].tobytes())
            except Exception as e:
                print(f"Client {self.client_id_numeric}: Could not decrypt parameters: {e}")
                return
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        start_time = time.time()
        self.set_parameters(parameters)
        
        gi_params = self.attack_config.get("gradient_inversion", {})
        is_gi_target = (gi_params.get("enable", False) and self.client_id_numeric == gi_params.get("target_client"))
        params_to_send, num_examples = None, 0
        metrics = {}
        metrics["data_type"] = "sensor"

        if is_gi_target:
            print(f"Client {self.client_id_numeric}: Acting as Gradient Inversion target (Numerical).")
            single_item_loader = DataLoader(self.trainset, batch_size=1, shuffle=True)
            features, label = next(iter(single_item_loader))
            
            os.makedirs("client_data", exist_ok=True)
            with open(f"client_data/client_{self.client_id_numeric}_sensor_data.pkl", "wb") as f:
                pickle.dump({'data': features.numpy(), 'label': label.numpy()}, f)
            
            self.model.train(); criterion = torch.nn.CrossEntropyLoss()
            output = self.model(features); loss = criterion(output, label)
            gradients = torch.autograd.grad(loss, self.model.parameters())
            params_to_send = [grad.cpu().numpy() for grad in gradients]; num_examples = len(features)
            metrics["attack"] = "gradient_inversion"
        else:
            train(self.model, self.trainloader, self.client_config["local_epochs"], self.client_config["learning_rate"])
            params_to_send = self.get_parameters({})
            num_examples = len(self.trainset)
            mp_params = self.attack_config.get("model_poisoning", {})
            if (mp_params.get("enable", False) and self.client_id_numeric in mp_params.get("malicious_clients", [])):
                params_to_send = scaling_attack(params_to_send, mp_params.get("scale_factor", -1.0))

        defense_type = config.get("defense_type")
        if config.get("apply_chaotic_encryption", False):
            print(f"Client {self.client_id_numeric}: Applying Chaotic Encryption as instructed.")
            params_to_send = chaotic_encryption(params_to_send)
        elif defense_type == "clipping":
            params_to_send = gradient_clipping(params_to_send, config.get("clipping_norm"))
        elif defense_type == "sparsification":
            params_to_send = gradient_sparsification(params_to_send, config.get("sparsity"))
        elif defense_type == "dp":
            params_to_send = add_differential_privacy(params_to_send, config.get("clipping_norm"), config.get("noise_multiplier"))
        elif defense_type == "encryption":
            print(f"Client {self.client_id_numeric}: Encrypting parameters.")
            encrypted_bytes = encrypt_params(params_to_send)
            params_to_send = [np.frombuffer(encrypted_bytes, dtype=np.uint8)]

        fit_duration = time.time() - start_time
        metrics["fit_duration"] = fit_duration
        metrics["logical_client_id"] = self.client_id_numeric
        metrics["data_type"] = "sensor"
        
        return params_to_send, num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}
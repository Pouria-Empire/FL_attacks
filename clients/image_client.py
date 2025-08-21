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

# --- Corrected Imports ---
from model import CastingCNN, CifarCNN # Import both image models
# Import from the specific, existing data utility files
from utils_data.casting_data_util import get_client_data as get_casting_data
from utils_data.cifar_data_util import get_client_data as get_cifar_data
from attacks.data_poisoning import PoisonedDataset
from attacks.model_poisoning import scaling_attack
from attacks.defenses import gradient_clipping, gradient_sparsification, add_differential_privacy
from crypto_utils import encrypt_params, decrypt_params, chaotic_map_obfuscate

# --- HELPER FUNCTIONS ---
def load_config():
    """Loads the main config.yml file."""
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def train(model: torch.nn.Module, train_loader: DataLoader, epochs: int, lr: float, num_classes: int):
    """Universal train function for binary or multi-class image classification."""
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss() if num_classes == 1 else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            if num_classes == 1:
                labels = labels.float().view(-1, 1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return

def test(model: torch.nn.Module, test_loader: DataLoader, num_classes: int) -> Tuple[float, float]:
    """Universal validate function for binary or multi-class image classification."""
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = torch.nn.BCEWithLogitsLoss() if num_classes == 1 else torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            original_labels = labels.clone() # Keep original labels for accuracy check
            if num_classes == 1:
                labels = labels.float().view(-1, 1)
            
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)

            if num_classes == 1:
                predicted = (torch.sigmoid(outputs) > 0.5)
                correct += (predicted.squeeze().long() == original_labels).sum().item()
            else:
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == original_labels).sum().item()
            
            total += labels.size(0)
            
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
        self.data_type = self.data_config.get("type", "casting")

        self.num_classes = self.data_config["num_classes"]
        
        # Select the correct model based on the data type from the config
        if self.data_type == "casting":
            self.model = CastingCNN(num_classes=self.num_classes)
            self.trainset, self.testset = get_casting_data(
                cid=self.cid,
                total_clients=self.client_config["total"],
                data_path=self.data_config["path"],
                img_size=self.data_config["img_size"]
            )
        elif self.data_type == "cifar10":
            self.model = CifarCNN(num_classes=self.num_classes)
            self.trainset, self.testset = get_cifar_data(
                cid=self.cid,
                total_clients=self.client_config["total"],
                data_path=self.data_config["path"],
                img_size=self.data_config["img_size"]
            )
        else:
            raise ValueError(f"Unsupported image data type: {self.data_type}")


        # Optional poisoning
        dp_params = self.attack_config.get("data_poisoning", {})
        is_dp_malicious = (dp_params.get("enable", False) and self.client_id_numeric in dp_params.get("malicious_clients", []))
        if is_dp_malicious:
            print(f"Client {self.client_id_numeric}: Applying data poisoning.")
            self.trainset = PoisonedDataset(
                dataset=self.trainset,
                poison_frac=dp_params.get("poison_frac", 0.3),
                target_label=dp_params.get("target_label", 1)
            )

        # Gradient inversion batch size logic
        gi_params = self.attack_config.get("gradient_inversion", {})
        is_gi_target = (gi_params.get("enable", False) and self.client_id_numeric == gi_params.get("target_client"))
        batch_size = gi_params.get("attack_batch_size") if is_gi_target and "attack_batch_size" in gi_params else self.client_config["batch_size"]
        if is_gi_target: print(f"Client {self.client_id_numeric} (Attacker): Using attack batch size of {batch_size}")

        self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
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
            print(f"Client {self.client_id_numeric}: Acting as Gradient Inversion target.")
            batch_data, batch_target = next(iter(self.trainloader))
            os.makedirs("client_data", exist_ok=True)
            with open(f"client_data/client_{self.client_id_numeric}_image_data.pkl", "wb") as f:
                pickle.dump({'data': batch_data.numpy(), 'label': batch_target.numpy()}, f)

            self.model.train()
            criterion = torch.nn.BCEWithLogitsLoss() if self.num_classes == 1 else torch.nn.CrossEntropyLoss()
            output = self.model(batch_data)
            loss = criterion(output, batch_target.float().view(-1, 1) if self.num_classes == 1 else batch_target)
            
            gradients = torch.autograd.grad(loss, self.model.parameters())
            params_to_send = [grad.cpu().numpy() for grad in gradients]
            num_examples = len(batch_data)
            metrics["attack"] = "gradient_inversion"
        else:
            train(self.model, self.trainloader, self.client_config["local_epochs"], self.client_config["learning_rate"], num_classes=self.num_classes)
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
        if config.get("apply_chaotic_obfuscation", False):
            params_to_send = chaotic_map_obfuscate(params_to_send, key=config.get("chaos_key", 3.99))
        
        defense_type = config.get("defense_type")
        if defense_type == "clipping":
            params_to_send = gradient_clipping(params_to_send, config.get("clipping_norm"))
        elif defense_type == "encryption":
            encrypted_bytes = encrypt_params(params_to_send)
            params_to_send = [np.frombuffer(encrypted_bytes, dtype=np.uint8)]
        
        defense_duration = time.time() - defense_start_time
        fit_duration = time.time() - fit_start_time
        
        metrics.update({
            "fit_duration": fit_duration,
            "defense_duration": defense_duration,
            "logical_client_id": self.client_id_numeric,
            "data_type": "image",
            "was_chaotically_obfuscated": config.get("apply_chaotic_obfuscation", False)
        })

        return params_to_send, num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, num_classes=self.num_classes)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}
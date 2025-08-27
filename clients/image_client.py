# clients/image_client.py (Fixed for ChestMNIST)

import flwr as fl
import torch
import numpy as np
import os
import pickle
import time
from torch.utils.data import DataLoader
from typing import Tuple, List
from tqdm import tqdm

# CHESTMNIST CHANGE: Import the correct models and data utils
from model import CifarCNN, MedMNIST_CNN
from utils_data.cifar_data_util import get_client_data as get_cifar_data
from utils_data.medmnist_data_util import get_client_data as get_medmnist_data

from attacks.data_poisoning import PoisonedDataset
from attacks.model_poisoning import scaling_attack
from crypto_utils import decrypt_params, encrypt_params, chaotic_map_obfuscate
from attacks.defenses import gradient_clipping, add_differential_privacy, gradient_sparsification


def train(model: torch.nn.Module, train_loader: DataLoader, epochs: int, lr: float, cid: str, multi_label: bool):
    """Universal train function that handles multi-class and multi-label."""
    model.train()
    # CHESTMNIST CHANGE: Use BCEWithLogitsLoss for multi-label tasks
    criterion = torch.nn.BCEWithLogitsLoss() if multi_label else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Client {cid} - Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            optimizer.zero_grad()
            # CHESTMNIST CHANGE: Ensure labels are float for BCE loss
            if multi_label:
                labels = labels.float()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})
    return

def test(model: torch.nn.Module, test_loader: DataLoader, multi_label: bool) -> Tuple[float, float]:
    """Universal test function that handles multi-class and multi-label."""
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    # CHESTMNIST CHANGE: Use BCEWithLogitsLoss for multi-label tasks
    criterion = torch.nn.BCEWithLogitsLoss() if multi_label else torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            
            if multi_label:
                # CHESTMNIST CHANGE: Logic for multi-label prediction and accuracy
                labels = labels.float()
                total_loss += criterion(outputs, labels).item() * images.size(0)
                # Get predictions by applying sigmoid and thresholding at 0.5
                predicted = torch.sigmoid(outputs) > 0.5
                # Accuracy is the fraction of samples where all labels are predicted correctly (strict)
                correct += (predicted == labels).all(dim=1).sum().item()
            else:
                # Original logic for multi-class
                total_loss += torch.nn.CrossEntropyLoss()(outputs, labels).item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels.squeeze().long()).sum().item()

            total += labels.size(0)
            
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / total if total > 0 else 0
    return avg_loss, accuracy

def _sizeof_parameters(params: List[np.ndarray]) -> int:
    if not params: return 0
    if len(params) == 1 and isinstance(params[0], np.ndarray) and params[0].dtype == np.uint8:
        return int(params[0].nbytes)
    return int(sum(p.nbytes for p in params))


class ImageFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, config: dict):
        self.cid = cid
        self.client_id_numeric = int(cid.replace("client", ""))
        self.config = config
        self.attack_config = config.get("attacks", {})
        self.client_config = config.get("clients", {})
        self.data_config = config.get("data", {})
        self.data_type = self.data_config.get("type")
        self.num_classes = self.data_config["num_classes"]
        
        # CHESTMNIST CHANGE: A flag to determine the task type
        self.is_multilabel = self.data_config.get("dataset_name", "") == "chestmnist"

        # CHESTMNIST CHANGE: Updated logic to handle different data types
        if self.data_type == "cifar10":
            self.model = CifarCNN(num_classes=self.num_classes)
            self.trainset, self.testset = get_cifar_data(
                cid=self.cid, total_clients=self.client_config["total"],
                data_path=self.data_config["path"], img_size=self.data_config["img_size"]
            )
        elif self.data_type == "medmnist":
            self.model = MedMNIST_CNN(num_classes=self.num_classes)
            self.trainset, self.testset = get_medmnist_data(
                cid=self.cid, total_clients=self.client_config["total"],
                data_path=self.data_config["path"], dataset_name=self.data_config["dataset_name"]
            )
        else:
            raise ValueError(f"Unsupported image data type: {self.data_type}")

        gi_params = self.attack_config.get("gradient_inversion", {})
        is_gi_target = (gi_params.get("enable", False) and self.client_id_numeric == gi_params.get("target_client"))
        batch_size = gi_params.get("attack_batch_size") if is_gi_target else self.client_config["batch_size"]
        
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
            # CHESTMNIST CHANGE: Use BCEWithLogitsLoss and correct label format for multi-label
            criterion = torch.nn.BCEWithLogitsLoss() if self.is_multilabel else torch.nn.CrossEntropyLoss()
            output = self.model(batch_data)
            
            labels_for_loss = batch_target.float() if self.is_multilabel else batch_target
            loss = criterion(output, labels_for_loss)
            
            gradients = torch.autograd.grad(loss, self.model.parameters())
            params_to_send = [grad.cpu().numpy() for grad in gradients]
            num_examples = len(batch_data)
            metrics["attack"] = "gradient_inversion"
        else:
            print(f"Client {self.cid}: Starting full local training...")
            train(self.model, self.trainloader, self.client_config["local_epochs"], self.client_config["learning_rate"], self.cid, self.is_multilabel)
            
            new_params = self.get_parameters({})
            num_examples = len(self.trainset)
            update = [new - old for new, old in zip(new_params, parameters)]

            # Apply attacks and defenses... (logic remains the same)
            mp_params = self.attack_config.get("model_poisoning", {})
            if (mp_params.get("enable", False) and self.client_id_numeric in mp_params.get("malicious_clients", [])):
                update = scaling_attack(update, mp_params.get("scale_factor", -1.0))
            
            defense_type = config.get("defense_type")
            if defense_type == "clipping": update = gradient_clipping(update, config.get("clipping_norm"))
            elif defense_type == "dp": update = add_differential_privacy(update, config.get("clipping_norm"), config.get("noise_multiplier"))
            elif defense_type == "sparsification": update = gradient_sparsification(update, config.get("sparsity"))

            params_to_send = [orig + upd for orig, upd in zip(parameters, update)]

        if config.get("apply_chaotic_obfuscation", False):
            params_to_send = chaotic_map_obfuscate(params_to_send, key=config.get("chaos_key", 3.99))
        
        if config.get("defense_type") == "encryption":
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
        loss, accuracy = test(self.model, self.testloader, self.is_multilabel)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}
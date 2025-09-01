# clients/image_client.py

import os
import pickle
import time
import random
from typing import Tuple, List

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CifarCNN, MedMNIST_CNN, CastingCNN
from utils_data.cifar_data_util import get_client_data as get_cifar_data
from utils_data.medmnist_data_util import get_client_data as get_medmnist_data
from utils_data.casting_data_util import get_client_data as get_casting_data
from attacks.data_poisoning import PoisonedDatasetWrapper
from attacks.model_poisoning import scaling_attack
from attacks.defenses import gradient_clipping, add_differential_privacy, gradient_sparsification
from crypto_utils import decrypt_params, encrypt_params, chaotic_map_obfuscate

def train(model: torch.nn.Module, train_loader: DataLoader, epochs: int, lr: float, cid: str, use_bce: bool):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss() if use_bce else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Client {cid} - Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            optimizer.zero_grad()
            if use_bce: labels = labels.float()
            else: labels = labels.squeeze().long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            pbar.set_postfix({"loss": loss.item()})
    return

def test(model: torch.nn.Module, test_loader: DataLoader, use_bce: bool) -> Tuple[float, float]:
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = torch.nn.BCEWithLogitsLoss() if use_bce else torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images); total += labels.size(0)
            if use_bce:
                labels = labels.float()
                total_loss += criterion(outputs, labels).item() * images.size(0)
                predicted = torch.sigmoid(outputs) > 0.5
                correct += (predicted == labels).all(dim=1).sum().item()
            else:
                labels = labels.squeeze().long()
                total_loss += torch.nn.CrossEntropyLoss()(outputs, labels).item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
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
        self.cid = cid; self.client_id_numeric = int(cid.replace("client", ""))
        self.config = config; self.attack_config = config.get("attacks", {})
        self.client_config = config.get("clients", {}); self.data_config = config.get("data", {})
        self.data_type = self.data_config.get("type"); self.num_classes = self.data_config["num_classes"]
        dataset_name = self.data_config.get("dataset_name", "")
        self.use_bce_loss = (dataset_name in ["chestmnist", "pneumoniamnist"]) or (self.data_type == "casting")

        if self.data_type == "medmnist":
            self.model = MedMNIST_CNN(num_classes=self.num_classes)
            self.trainset, self.testset = get_medmnist_data(
                cid=self.cid, total_clients=self.client_config["total"],
                data_path=self.data_config["path"], dataset_name=dataset_name
            )
        else: # Add other data types here
            raise ValueError(f"Unsupported image data type: {self.data_type}")

        # ✅ FIX: This is the correct __init__ logic for probabilistic attacks.
        # It sets up all the necessary attributes that are used later in the fit() method.
        dp_params = self.attack_config.get("data_poisoning", {})
        self.is_potential_data_poisoner = (dp_params.get("enable", False) and self.client_id_numeric in dp_params.get("malicious_clients", []))
        if self.is_potential_data_poisoner:
            print(f"Client {self.cid}: Configured as a POTENTIAL data poisoner.")
            self.clean_trainset = self.trainset # Save the original clean data
            self.poisoned_trainset = PoisonedDatasetWrapper(dataset=self.trainset, data_type=self.data_type, **dp_params)
            self.dp_attack_probability = dp_params.get("attack_probability", 1.0)

        mp_params = self.attack_config.get("model_poisoning", {})
        self.is_potential_model_poisoner = (mp_params.get("enable", False) and self.client_id_numeric in mp_params.get("malicious_clients", []))
        if self.is_potential_model_poisoner:
            print(f"Client {self.cid}: Configured as a POTENTIAL model poisoner.")
            self.mp_attack_probability = mp_params.get("attack_probability", 1.0)
            self.mp_scale_factor = mp_params.get("scale_factor", -1.0)

        self.testloader = DataLoader(self.testset, batch_size=self.client_config["batch_size"], num_workers=0)
        
    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        if len(parameters) == 1 and parameters[0].dtype == np.uint8:
            try: parameters = decrypt_params(parameters[0].tobytes())
            except Exception: return
        params_dict = zip(self.model_state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        self.set_parameters(parameters); fit_start_time = time.time()
        metrics = {}; bytes_down = _sizeof_parameters(parameters)

        active_trainset = self.trainset
        if self.is_potential_data_poisoner:
            if random.random() < self.dp_attack_probability:
                print(f"Client {self.cid}: BEHAVING MALICIOUSLY (Data Poisoning) this round.")
                active_trainset = self.poisoned_trainset
            else:
                print(f"Client {self.cid}: Behaving honestly (Data Poisoning) this round.")
                active_trainset = self.clean_trainset
        
        gi_params = self.attack_config.get("gradient_inversion", {})
        is_gi_target = (gi_params.get("enable", False) and self.client_id_numeric == gi_params.get("target_client"))
        batch_size = gi_params.get("attack_batch_size") if is_gi_target else self.client_config["batch_size"]
        trainloader_this_round = DataLoader(active_trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        if is_gi_target:
            batch_data, batch_target = next(iter(trainloader_this_round))
            os.makedirs("client_data", exist_ok=True)
            with open(f"client_data/client_{self.client_id_numeric}_{self.data_type}_data.pkl", "wb") as f:
                pickle.dump({'data': batch_data.numpy(), 'label': batch_target.numpy()}, f)
            self.model.train()
            criterion = torch.nn.BCEWithLogitsLoss() if self.use_bce_loss else torch.nn.CrossEntropyLoss()
            output = self.model(batch_data)
            labels_for_loss = batch_target.float() if self.use_bce_loss else batch_target.squeeze().long()
            loss = criterion(output, labels_for_loss)
            gradients = torch.autograd.grad(loss, self.model.parameters())
            params_to_send = [grad.cpu().numpy() for grad in gradients]
            num_examples = len(batch_data)
            metrics["attack"] = "gradient_inversion"
        else:
            train(self.model, trainloader_this_round, self.client_config["local_epochs"], self.client_config["learning_rate"], self.cid, self.use_bce_loss)
            new_params = self.get_parameters({})
            num_examples = len(active_trainset)
            update = [new - old for new, old in zip(new_params, parameters)]
            if self.is_potential_model_poisoner and random.random() < self.mp_attack_probability:
                print(f"Client {self.cid}: BEHAVING MALICIOUSLY (Model Poisoning) this round.")
                update = scaling_attack(update, self.mp_scale_factor)
            elif self.is_potential_model_poisoner:
                print(f"Client {self.cid}: Behaving honestly (Model Poisoning) this round.")
            defense_type = config.get("defense_type")
            if defense_type == "clipping": update = gradient_clipping(update, config.get("clipping_norm"))
            elif defense_type == "dp": update = add_differential_privacy(update, config.get("clipping_norm"), config.get("noise_multiplier"))
            elif defense_type == "sparsification": update = gradient_sparsification(update, config.get("sparsity"))
            params_to_send = [orig + upd for orig, upd in zip(parameters, update)]
        
        if config.get("apply_chaotic_obfuscation", False):
            params_to_send = chaotic_map_obfuscate(params=params_to_send, **config.get("mydefense_params", {}))
        if config.get("defense_type") == "encryption":
            encrypted_bytes = encrypt_params(params_to_send)
            params_to_send = [np.frombuffer(encrypted_bytes, dtype=np.uint8)]
        
        bytes_up = _sizeof_parameters(params_to_send)
        metrics.update({
            "fit_duration": time.time() - fit_start_time, "logical_client_id": self.client_id_numeric,
            "data_type": self.data_type, "was_chaotically_obfuscated": config.get("apply_chaotic_obfuscation", False),
            "bytes_down": bytes_down, "bytes_up": bytes_up,
        })
        return params_to_send, num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, self.use_bce_loss)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}
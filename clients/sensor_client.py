import flwr as fl
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, List
import time
import os
import pickle

# --- Import the new components for the sensor dataset ---
from model import SensorMLP
from utils_data.sensor_data_util import get_client_data, load_and_preprocess_data
from attacks.numerical_attacks import PoisonedSensorDataset
from attacks.model_poisoning import scaling_attack
from attacks.defenses import gradient_clipping, gradient_sparsification, add_differential_privacy

def train(model: torch.nn.Module, train_loader: DataLoader, epochs: int, lr: float):
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
        if (dp_params.get("enable", False) and self.client_id_numeric in dp_params.get("malicious_clients", [])):
            print(f"Client {self.client_id_numeric}: Applying numerical data poisoning.")
            self.trainset = PoisonedSensorDataset(
                dataset=self.trainset, poison_frac=dp_params.get("poison_frac", 0.3),
                target_label=dp_params.get("target_label", 0),
                trigger_value=dp_params.get("trigger_value", 5.0)
            )

        self.trainloader = DataLoader(self.trainset, batch_size=self.client_config["batch_size"], shuffle=True)
        self.testloader = DataLoader(self.testset, batch_size=self.client_config["batch_size"])

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        start_time = time.time()
        self.set_parameters(parameters)
        
        gi_params = self.attack_config.get("gradient_inversion", {})
        is_gi_target = (gi_params.get("enable", False) and self.client_id_numeric == gi_params.get("target_client"))
        params_to_send, num_examples = None, 0
        
        # --- THE FIX: Initialize the metrics dictionary ---
        metrics = {}

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
        if defense_type == "clipping":
            params_to_send = gradient_clipping(params_to_send, config.get("clipping_norm"))
        
        fit_duration = time.time() - start_time
        metrics["fit_duration"] = fit_duration
        metrics["logical_client_id"] = self.client_id_numeric
        metrics["data_type"] = "sensor"
        
        return params_to_send, num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}

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

from model import SimpleNN
from chest_data_util import get_client_data
from attacks.data_poisoning import PoisonedDataset
from attacks.model_poisoning import scaling_attack
from attacks.defenses import gradient_clipping, gradient_sparsification, add_differential_privacy

def load_config():
    with open("config.yml", "r") as f: return yaml.safe_load(f)

def train(model: torch.nn.Module, train_loader: DataLoader, epochs: int, lr: float):
    """Train for multi-label classification."""
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return

def test(model: torch.nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
    """Validate for multi-label classification."""
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

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, config: dict):
        self.cid = cid
        self.client_id_numeric = int(cid.replace("client", ""))
        self.config = config
        self.attack_config = config.get("attacks", {})
        self.client_config = config.get("clients", {})
        self.data_config = config.get("data", {})
        self.model = SimpleNN(num_classes=15)
        
        self.trainset, self.testset = get_client_data(
            cid=self.cid,
            total_clients=self.client_config["total"],
            data_path=self.data_config["path"],
            train_list_file=self.data_config["train_list"],
            test_list_file=self.data_config["test_list"]
        )
        
        dp_params = self.attack_config.get("data_poisoning", {})
        is_dp_malicious = (dp_params.get("enable", False) and self.client_id_numeric in dp_params.get("malicious_clients", []))
        if is_dp_malicious:
            print(f"Client {self.client_id_numeric}: Applying data poisoning.")
            self.trainset = PoisonedDataset(dataset=self.trainset, poison_frac=dp_params.get("poison_frac", 0.1), target_label_idx=dp_params.get("target_label_idx", 7))

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

        if is_gi_target:
            attack_type = gi_params.get("type", "dlg")
            print(f"Client {self.client_id_numeric}: Acting as Gradient Inversion target ({attack_type}).")
            if attack_type in ["gradinversion", "gradinversion_plus"]:
                batch_data, batch_target = next(iter(self.trainloader))
            else:
                single_item_loader = DataLoader(self.trainset, batch_size=1, shuffle=True)
                batch_data, batch_target = next(iter(single_item_loader))
            os.makedirs("client_data", exist_ok=True)
            with open(f"client_data/client_{self.client_id_numeric}_data.pkl", "wb") as f:
                pickle.dump({'data': batch_data.numpy(), 'label': batch_target.numpy()}, f)
            self.model.train()
            criterion = torch.nn.BCEWithLogitsLoss()
            output = self.model(batch_data)
            loss = criterion(output, batch_target)
            gradients = torch.autograd.grad(loss, self.model.parameters())
            params_to_send = [grad.cpu().numpy() for grad in gradients]
            num_examples = len(batch_data)
        else:
            train(self.model, self.trainloader, self.client_config["local_epochs"], self.client_config["learning_rate"])
            params_to_send = self.get_parameters({})
            num_examples = len(self.trainset)
            mp_params = self.attack_config.get("model_poisoning", {})
            if (mp_params.get("enable", False) and self.client_id_numeric in mp_params.get("malicious_clients", [])):
                attack_type = mp_params.get("type", "scaling")
                print(f"Client {self.client_id_numeric}: Applying model poisoning ({attack_type}).")
                if attack_type == "scaling":
                    params_to_send = scaling_attack(params_to_send, mp_params.get("scale_factor", -1.0))

        defense_type = config.get("defense_type")
        if defense_type == "clipping":
            params_to_send = gradient_clipping(params_to_send, config.get("clipping_norm"))
        elif defense_type == "sparsification":
            params_to_send = gradient_sparsification(params_to_send, config.get("sparsity"))
        elif defense_type == "dp":
            params_to_send = add_differential_privacy(params_to_send, config.get("clipping_norm"), config.get("noise_multiplier"))
        elif defense_type == "encryption":
            print(f"Client {self.client_id_numeric}: Simulating high cost of Encryption.")
            time.sleep(config.get("encryption_delay", 3.0))

        fit_duration = time.time() - start_time
        metrics = {"fit_duration": fit_duration}
        if is_gi_target:
            metrics["attack"] = "gradient_inversion"
        
        return params_to_send, num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=str, required=True)
    args = parser.parse_args()
    config = load_config()
    client = FlowerClient(args.cid, config)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client, grpc_max_message_length=1024*1024*1024)

if __name__ == "__main__":
    main()
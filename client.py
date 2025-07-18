import flwr as fl
import torch
import argparse
import yaml
import numpy as np
import os
import pickle
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from typing import Tuple

# Import attack and model definitions
from model import SimpleNN
from attacks.data_poisoning import PoisonedDataset
from attacks.model_poisoning import add_noise, sign_flip, scaling_attack


# --------------------------------------------------------------------------
# UTILITY FUNCTIONS (included to make the script self-contained)
# --------------------------------------------------------------------------

def load_data(data_dir: str) -> Tuple[Dataset, Dataset]:
    """Load the MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train_data, test_data

def get_client_data(cid: str, total_clients: int, data_path: str) -> Tuple[Subset, Dataset]:
    """Load the full dataset and return the partition for a specific client."""
    train_data_full, test_data = load_data(data_path)
    
    client_id_numeric = int(cid.replace("client", ""))
    client_idx = client_id_numeric - 1
    len_train = len(train_data_full)
    
    indices = list(range(
        client_idx * (len_train // total_clients),
        (client_idx + 1) * (len_train // total_clients)
    ))
    
    return Subset(train_data_full, indices), test_data

# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------

def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def train(model, train_loader, epochs, lr):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return

def test(model, test_loader):
    model.eval()
    test_loss, correct = 0, 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)


# --------------------------------------------------------------------------
# FLOWER CLIENT
# --------------------------------------------------------------------------

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, attack_config, client_config):
        self.cid_string = cid
        self.client_id_numeric = int(cid.replace("client", ""))
        self.attack_config = attack_config
        self.client_config = client_config
        self.model = SimpleNN()

        self.train_data_subset, self.test_data = get_client_data(
            cid=self.cid_string,
            total_clients=self.client_config["total"],
            data_path="./data"
        )
        
        dp_params = self.attack_config.get("data_poisoning", {})
        is_dp_malicious = (dp_params.get("enable", False) and
                           self.client_id_numeric in dp_params.get("malicious_clients", []))
        if is_dp_malicious:
            print(f"Client {self.client_id_numeric}: Applying data poisoning to its dataset.")
            self.train_data_subset = PoisonedDataset(
                dataset=self.train_data_subset,
                poison_frac=dp_params.get("poison_frac", 0.1),
                target_label=dp_params.get("target_label", 0)
            )
        
        self.train_loader = DataLoader(self.train_data_subset, batch_size=self.client_config["batch_size"], shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.client_config["batch_size"])

    def get_parameters(self, config):
        return [val.cpu().numpy().astype('float32') for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        gi_params = self.attack_config.get("gradient_inversion", {})
        is_gi_target = (gi_params.get("enable", False) and
                        self.client_id_numeric == gi_params.get("target_client"))

        if is_gi_target:
            attack_type = gi_params.get("type", "dlg")
            print(f"Client {self.client_id_numeric}: Acting as Gradient Inversion target ({attack_type}).")
            
            # --- THE FIX: Use the existing, properly shuffled train_loader ---
            # This ensures a different batch is used in each round.
            batch_data, batch_target = next(iter(self.train_loader))
            # ---

            # Ensure the batch size for saving matches the actual batch used
            if len(batch_data) != self.client_config["batch_size"] and attack_type == "gradinversion":
                 print(f"Warning: Batch size mismatch. Expected {self.client_config['batch_size']}, got {len(batch_data)}.")

            os.makedirs("client_data", exist_ok=True)
            with open(f"client_data/client_{self.client_id_numeric}_data.pkl", "wb") as f:
                pickle.dump({'data': batch_data.numpy(), 'label': batch_target.numpy()}, f)

            self.model.train()
            criterion = torch.nn.CrossEntropyLoss()
            output = self.model(batch_data)
            loss = criterion(output, batch_target)
            
            gradients = torch.autograd.grad(loss, self.model.parameters())
            gradients_as_numpy = [grad.cpu().numpy() for grad in gradients]

            return gradients_as_numpy, len(batch_data), {"attack": "gradient_inversion"}
        
        train(self.model, self.train_loader, self.client_config["local_epochs"], self.client_config["learning_rate"])
        updated_params = self.get_parameters({})

        mp_params = self.attack_config.get("model_poisoning", {})
        is_mp_malicious = (mp_params.get("enable", False) and
                           self.client_id_numeric in mp_params.get("malicious_clients", []))

        if is_mp_malicious:
            attack_type = mp_params.get("type", "scaling")
            print(f"Client {self.client_id_numeric}: Applying model poisoning ({attack_type}).")
            if attack_type == "scaling":
                updated_params = scaling_attack(updated_params, mp_params.get("scale_factor", -1.0))
            elif attack_type == "sign_flip":
                updated_params = sign_flip(updated_params)
            elif attack_type == "noise":
                updated_params = add_noise(updated_params, mp_params.get("noise_scale", 0.5))
        
        return updated_params, len(self.train_data_subset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = test(self.model, self.test_loader)
        return float(loss), len(self.test_data), {"accuracy": float(acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=str, required=True, help="Client ID (e.g., client1, client2)")
    args = parser.parse_args()

    config = load_config()
    attack_config = config.get("attacks", {})
    client_config = config.get("clients", {})

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(args.cid, attack_config, client_config).to_client(),
        grpc_max_message_length=1024*1024*1024
    )

if __name__ == "__main__":
    main()
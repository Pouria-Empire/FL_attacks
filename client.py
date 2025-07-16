import flwr as fl
import torch
import argparse
from torch.utils.data import DataLoader, Subset
import yaml
import numpy as np
import os
import pickle

# Import all necessary components
from model import SimpleNN
from utils import load_data, get_parameters, set_parameters
from attacks.data_poisoning import PoisonedDataset
from attacks.model_poisoning import add_noise, sign_flip, scaling_attack


def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

# The standard train function
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

# The standard test function
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


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, attack_config):
        self.cid_string = cid
        self.client_id_numeric = int(cid.replace("client", ""))
        self.attack_config = attack_config
        self.model = SimpleNN()
        self.config = load_config()
        train_data_full, test_data = load_data(self.config["data"]["path"])
        
        dp_params = self.attack_config.get("data_poisoning", {})
        is_dp_malicious = (dp_params.get("enable", False) and
                           self.client_id_numeric in dp_params.get("malicious_clients", []))
        if is_dp_malicious:
            print(f"Client {self.client_id_numeric}: Applying data poisoning to its dataset.")
            train_data_full = PoisonedDataset(
                dataset=train_data_full,
                poison_frac=dp_params.get("poison_frac", 0.1),
                target_label=dp_params.get("target_label", 0)
            )

        n_clients = self.config["clients"]["total"]
        client_idx = self.client_id_numeric - 1
        indices = list(range(client_idx * (len(train_data_full) // n_clients),
                       (client_idx + 1) * (len(train_data_full) // n_clients)))

        self.train_data_subset = Subset(train_data_full, indices)
        self.test_data = test_data
        self.train_loader = DataLoader(self.train_data_subset, batch_size=self.config["clients"]["batch_size"], shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.config["clients"]["batch_size"])

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
            print(f"Client {self.client_id_numeric}: Acting as Gradient Inversion target. Sending raw gradients.")
            single_batch_loader = DataLoader(self.train_data_subset, batch_size=1, shuffle=True)
            data, target = next(iter(single_batch_loader))

            # --- ADD THIS BLOCK TO SAVE THE ORIGINAL DATA ---
            os.makedirs("client_data", exist_ok=True)
            with open(f"client_data/client_{self.client_id_numeric}_data.pkl", "wb") as f:
                pickle.dump({'data': data.numpy(), 'label': target.numpy()}, f)
            # --- END OF BLOCK ---

            self.model.train()
            criterion = torch.nn.CrossEntropyLoss()
            output = self.model(data)
            loss = criterion(output, target)
            
            gradients = torch.autograd.grad(loss, self.model.parameters())
            gradients_as_numpy = [grad.cpu().numpy() for grad in gradients]

            return gradients_as_numpy, 1, {"attack": "gradient_inversion", "logical_client_id": self.client_id_numeric}
        
        train(self.model, self.train_loader, epochs=self.config["clients"]["local_epochs"], lr=self.config["clients"]["learning_rate"])
        
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
        
        return updated_params, len(self.train_loader.dataset), {"logical_client_id": self.client_id_numeric}
        

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = test(self.model, self.test_loader)
        return float(loss), len(self.test_data), {"accuracy": float(acc), "logical_client_id": self.client_id_numeric}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=str, required=True, help="Client ID (e.g., client1, client2)")
    args = parser.parse_args()

    config = load_config()
    attack_config = config.get("attacks", {})

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(args.cid, attack_config).to_client(),
        grpc_max_message_length=1024*1024*1024
    )

if __name__ == "__main__":
    main()
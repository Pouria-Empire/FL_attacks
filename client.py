import flwr as fl
import torch
import argparse
from torch.utils.data import DataLoader, Subset
from model import SimpleNN
from utils import load_data, get_parameters, set_parameters
import yaml
import numpy as np

# Import attacks
from attacks.model_poisoning import add_noise, sign_flip, scaling_attack
from attacks.data_poisoning import PoisonedDataset
from attacks.backdoor import BackdoorAttack

def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def train(model, train_loader, epochs, lr):
    """Train the model and return average loss and accuracy"""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    running_loss = 0.0
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return float(avg_loss), float(accuracy)

def test(model, test_loader):
    """Evaluate the model and return loss and accuracy"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction='sum'
            ).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return float(test_loss), float(accuracy)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, attack_config):
        self.cid = cid
        self.attack_config = attack_config
        self.client_id_num = int(cid.replace("client", ""))  # Extract numeric id
        
        self.model = SimpleNN()
        self.config = load_config()
        
        # Load and split data
        train_data, test_data = load_data(self.config["data"]["path"])
        total_samples = len(train_data)
        samples_per_client = total_samples // self.config["clients"]["total"]
        client_id = self.client_id_num - 1  # Convert to 0-based index
        
        # Split data between clients
        start_idx = client_id * samples_per_client
        end_idx = (client_id + 1) * samples_per_client
        if client_id == self.config["clients"]["total"] - 1:
            end_idx = total_samples
        
        indices = list(range(start_idx, end_idx))
        self.train_data = Subset(train_data, indices)
        self.test_data = test_data
        
        # Apply data poisoning attacks if enabled and this client is malicious
        if attack_config.get("data_poisoning", {}).get("enable", False):
            if self.client_id_num in attack_config["data_poisoning"]["malicious_clients"]:
                self.train_data = PoisonedDataset(
                    self.train_data,
                    poison_frac=0.3,
                    target_label=attack_config["data_poisoning"]["target_label"]
                )
                print(f"Client {self.cid} is using poisoned data.")
        
        # Apply backdoor attack if enabled and this client is malicious
        if attack_config.get("backdoor", {}).get("enable", False):
            if self.client_id_num in attack_config["backdoor"]["malicious_clients"]:
                self.train_data = BackdoorAttack(
                    self.train_data,
                    target_label=attack_config["backdoor"]["target_label"]
                )
                print(f"Client {self.cid} is using backdoor data.")
        
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.config["clients"]["batch_size"],
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_data,
            batch_size=self.config["clients"]["batch_size"]
        )

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        epochs = self.config["clients"]["local_epochs"]
        lr = self.config["clients"]["learning_rate"]
        
        train_loss, train_accuracy = train(self.model, self.train_loader, epochs, lr)
        
        metrics = {
            "loss": train_loss,
            "accuracy": train_accuracy,
            "phase": "train",  # Add phase identifier
            "num_examples": len(self.train_data)
        }
        print(f"[Client {self.cid} Training] Loss: {train_loss:.4f}, Accuracy: {train_accuracy*100:.2f}%")
        
        # Apply model poisoning if enabled and this client is malicious
        if self.attack_config.get("model_poisoning", {}).get("enable", False):
            if self.client_id_num in self.attack_config["model_poisoning"]["malicious_clients"]:
                poison_type = self.attack_config["model_poisoning"]["type"]
                params = get_parameters(self.model)
                
                if poison_type == "noise":
                    poisoned_params = add_noise(params, noise_scale=0.5)
                elif poison_type == "sign_flip":
                    poisoned_params = sign_flip(params)
                elif poison_type == "scaling":
                    poisoned_params = scaling_attack(params, scale_factor=-1.0)
                else:
                    poisoned_params = params
                
                set_parameters(self.model, poisoned_params)
                print(f"Client {self.cid} applied model poisoning: {poison_type}")
        
        return get_parameters(self.model), len(self.train_data), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader)
        
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "phase": "eval",  # Add phase identifier
            "num_examples": len(self.test_data)
        }
        print(f"[Client {self.cid} Evaluation] Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")
        return loss, len(self.test_data), metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=str, required=True, help="Client ID (e.g., client1, client2)")
    args = parser.parse_args()
    
    try:
        config = load_config()
        attack_config = config.get("attacks", {})
        fl.client.start_client(
            server_address="127.0.0.1:8080",
            client=FlowerClient(args.cid, attack_config).to_client()
        )
    except Exception as e:
        print(f"Error in client {args.cid}: {str(e)}")

if __name__ == "__main__":
    main()
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
    """Train with proper parameter types"""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return running_loss/len(train_loader), correct/total

def test(model, test_loader):
    """Consistent evaluation"""
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    return test_loss/len(test_loader.dataset), correct/len(test_loader.dataset)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, attack_config):
        self.cid = cid
        self.attack_config = attack_config
        self.model = SimpleNN()
        self.config = load_config()
        
        # Load and prepare data
        train_data, test_data = load_data(self.config["data"]["path"])
        n_clients = self.config["clients"]["total"]
        client_idx = int(cid.replace("client", "")) - 1
        
        # Split data
        n_samples = len(train_data)
        per_client = n_samples // n_clients
        indices = range(client_idx * per_client, 
                       (client_idx + 1) * per_client if client_idx != n_clients - 1 else n_samples)
        
        self.train_data = Subset(train_data, list(indices))
        self.test_data = test_data
        
        # Apply attacks if configured
        if attack_config.get("data_poisoning", {}).get("enable", False):
            if client_idx + 1 in attack_config["data_poisoning"]["malicious_clients"]:
                self.train_data = PoisonedDataset(self.train_data, 
                                                target_label=attack_config["data_poisoning"]["target_label"])
        
        if attack_config.get("backdoor", {}).get("enable", False):
            if client_idx + 1 in attack_config["backdoor"]["malicious_clients"]:
                self.train_data = BackdoorAttack(self.train_data,
                                               target_label=attack_config["backdoor"]["target_label"])
        
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
        """Ensure float32 numpy arrays"""
        return [val.cpu().numpy().astype('float32') for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Handle both numpy and tensor inputs"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
                     for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = train(self.model, self.train_loader, 
                         self.config["clients"]["local_epochs"],
                         self.config["clients"]["learning_rate"])
        
        # Model poisoning if configured
        if self.attack_config.get("model_poisoning", {}).get("enable", False):
            if int(self.cid.replace("client", "")) in self.attack_config["model_poisoning"]["malicious_clients"]:
                params = get_parameters(self.model)
                poison_type = self.attack_config["model_poisoning"]["type"]
                
                if poison_type == "noise":
                    poisoned = add_noise(params, 0.5)
                elif poison_type == "sign_flip":
                    poisoned = sign_flip(params)
                elif poison_type == "scaling":
                    poisoned = scaling_attack(params, -1.0)
                
                set_parameters(self.model, poisoned)
        
        return self.get_parameters({}), len(self.train_data), {
            "loss": loss, "accuracy": acc, "phase": "train"}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = test(self.model, self.test_loader)
        return loss, len(self.test_data), {
            "loss": loss, "accuracy": acc, "phase": "eval"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=str, required=True, 
                       help="Client ID (e.g., client1, client2)")
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
import flwr as fl
import torch
import argparse
from torch.utils.data import DataLoader, Subset
from model import SimpleNN
from utils import load_data, get_parameters, set_parameters
import yaml
import numpy as np
import os
import pickle

def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

# We now need BOTH training functions.
# This one is for the DLG target.
def train_single_batch(model, data_batch, lr):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    data, target = data_batch
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    total = target.size(0)

    return loss.item(), correct, total

# This one is for the honest clients.
def train(model, train_loader, epochs, lr):
    """Train the model on the training set for a number of epochs."""
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_loss, correct, total = 0.0, 0, 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
    final_loss = epoch_loss / total if total > 0 else 0.0
    final_acc = correct / total if total > 0 else 0.0
    return final_loss, final_acc


def test(model, test_loader):
    # This function is fine
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
        # This method is fine as it was
        self.cid_string = cid
        self.client_id_numeric = int(cid.replace("client", ""))
        self.attack_config = attack_config
        self.model = SimpleNN()
        self.config = load_config()

        train_data_full, test_data = load_data(self.config["data"]["path"])
        n_clients = self.config["clients"]["total"]
        client_idx = self.client_id_numeric - 1
        n_samples = len(train_data_full)
        per_client = n_samples // n_clients
        indices = list(range(client_idx * per_client,
                       (client_idx + 1) * per_client if client_idx != n_clients - 1 else n_samples))

        self.train_data_subset = Subset(train_data_full, indices)
        self.test_data = test_data

        self.train_loader = DataLoader(
            self.train_data_subset,
            batch_size=self.config["clients"]["batch_size"],
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_data,
            batch_size=self.config["clients"]["batch_size"]
        )

    def get_parameters(self, config):
        return [val.cpu().numpy().astype('float32') for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # --- START of CHANGE: Different logic for target vs. honest clients ---
        gradient_inversion_params = self.attack_config.get("gradient_inversion", {})
        is_dlg_target = (gradient_inversion_params.get("enable", False) and
                         self.client_id_numeric == gradient_inversion_params.get("target_client"))

        if is_dlg_target:
            # DLG TARGET CLIENT: Train on ONE batch to create a leaky gradient.
            print(f"Client {self.client_id_numeric}: Acting as DLG target. Training on one batch.")
            
            initial_params = get_parameters(self.model)
            
            # Use a batch size of 1 for the attack, regardless of config
            single_batch_loader = DataLoader(self.train_data_subset, batch_size=1, shuffle=True)
            data_batch = next(iter(single_batch_loader))

            # Save the data needed for the server-side attack
            os.makedirs("client_data", exist_ok=True)
            data_to_save = {
                'initial_params': initial_params,
                'data_batch': data_batch[0].cpu().numpy(),
                'target_batch': data_batch[1].cpu().numpy()
            }
            with open(f"client_data/client_{self.client_id_numeric}_data.pkl", 'wb') as f:
                pickle.dump(data_to_save, f)

            # Perform training on only that single batch
            loss, correct, total = train_single_batch(
                self.model, data_batch, self.config["clients"]["learning_rate"]
            )
            
            num_examples = total
            accuracy = correct / total
            metrics = {"loss": float(loss), "accuracy": float(accuracy)}

        else:
            # HONEST CLIENT: Perform full local training.
            loss, accuracy = train(
                self.model,
                self.train_loader,
                epochs=self.config["clients"]["local_epochs"],
                lr=self.config["clients"]["learning_rate"]
            )
            num_examples = len(self.train_loader.dataset)
            metrics = {"loss": float(loss), "accuracy": float(accuracy)}

        # Add common metrics
        metrics["phase"] = "train"
        metrics["logical_client_id"] = self.client_id_numeric
        
        return self.get_parameters({}), num_examples, metrics
        # --- END of CHANGE ---

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = test(self.model, self.test_loader)
        return float(loss), len(self.test_data), {
            "loss": float(loss),
            "accuracy": float(acc),
            "phase": "eval",
            "logical_client_id": self.client_id_numeric
        }


def main():
    # This function is fine, no changes needed
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
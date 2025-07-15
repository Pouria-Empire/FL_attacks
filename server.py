import flwr as fl
from flwr.common import Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Any, Optional
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
import pickle

# Import all necessary components
from model import SimpleNN
from utils import get_parameters, load_data
from attacks.gradient_inversion import dlg_attack, mdlg_attack
from attacks.data_poisoning import PoisonedDataset

def load_config() -> Dict[str, Any]:
    """Load the YAML configuration file."""
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]):
    """Helper function to set model parameters from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict)

def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    """A local test function for the server to use for evaluation."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    correct, total_loss = 0, 0.0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = correct / len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader.dataset)
    return avg_loss, accuracy

def safe_metrics_aggregation(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """A safe way to aggregate metrics, ignoring missing keys."""
    aggregated = {}
    # Aggregate custom backdoor ASR metric
    if any("backdoor_asr" in m for _, m in metrics):
        aggregated["backdoor_asr"] = sum(m["backdoor_asr"] for _, m in metrics if "backdoor_asr" in m) / len(metrics)

    # Aggregate standard evaluation accuracy
    if any("accuracy" in m for _, m in metrics):
         aggregated["accuracy"] = sum(m["accuracy"] for _, m in metrics if "accuracy" in m) / len(metrics)

    print("\n[Round Metrics]")
    if "accuracy" in aggregated:
        print(f"Eval Accuracy: {aggregated['accuracy']*100:.2f}%")
    if "backdoor_asr" in aggregated:
        print(f"Attack Success Rate (ASR): {aggregated['backdoor_asr']*100:.2f}%")
    
    return aggregated

class SecureFedAvg(FedAvg):
    def __init__(self, attack_config: Dict[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_config = attack_config
        self.reconstruction_dir = "reconstructions"
        os.makedirs(self.reconstruction_dir, exist_ok=True)

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        model = SimpleNN()
        params = get_parameters(model)
        return ndarrays_to_parameters(params)

    def aggregate_fit(self, server_round: int, results: List[Tuple[Any, Any]], failures: List[Any]) -> Optional[Parameters]:
        """Aggregate results and perform gradient inversion if enabled."""
        gi_params = self.attack_config.get("gradient_inversion", {})
        if not gi_params.get("enable", False):
            # If GI is not enabled, just do standard aggregation
            return super().aggregate_fit(server_round, results, failures)

        # Gradient Inversion logic
        target_client_id = gi_params.get("target_client", 1)
        target_fit_res = None
        honest_clients_results = []
        for client, fit_res in results:
            # The client sends a specific metric if it's a GI target
            if fit_res.metrics.get("attack") == "gradient_inversion":
                target_fit_res = fit_res
            else:
                honest_clients_results.append((client, fit_res))
        
        if target_fit_res:
            print(f"\n[Attack] Intercepted update from gradient inversion target: client {target_client_id}")
            try:
                gradients = parameters_to_ndarrays(target_fit_res.parameters)
                reconstructed = self._reconstruct_data(gradients, gi_params)
                if reconstructed is not None:
                    # You would need to fetch original data if you want a comparison image
                    self._save_reconstruction(reconstructed, target_client_id, server_round)
            except Exception as e:
                print(f"[Attack Failed] Gradient Inversion error: {str(e)}")
        
        # Aggregate only the updates from honest clients
        if not honest_clients_results:
            return None # No honest clients to aggregate, return current parameters
        
        return super().aggregate_fit(server_round, honest_clients_results, failures)

    def _reconstruct_data(self, gradients: List[np.ndarray], attack_params: Dict) -> Optional[np.ndarray]:
        """Call the appropriate reconstruction function based on config."""
        attack_type = attack_params.get("type", "dlg")
        print(f"[Attack] Attempting reconstruction using '{attack_type}' method.")

        if attack_type == "dlg":
            return dlg_attack(
                gradients=gradients,
                input_shape=(1, 1, 28, 28),
                lr=attack_params.get("attack_lr", 0.01),
                iterations=attack_params.get("iterations", 5000)
            )
        elif attack_type == "mdlg":
            return mdlg_attack(
                gradients=[gradients[0]], 
                input_shape=(1, 1, 28, 28),
                lr=attack_params.get("attack_lr", 0.01),
                iterations=attack_params.get("iterations", 500)
            )
        else:
            print(f"Unknown reconstruction attack type: {attack_type}")
            return None

    def _save_reconstruction(self, data: np.ndarray, client_id: int, round_num: int):
        """Save the reconstructed image."""
        img_array = data
        img = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        img = (img * 255).astype(np.uint8).squeeze()
        
        recon_path = f"{self.reconstruction_dir}/reconstruction_client{client_id}_round{round_num}.png"
        Image.fromarray(img).save(recon_path)
        print(f"[Attack] Saved reconstruction to {recon_path}")


def evaluate_backdoor(server_round: int, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    """Evaluate the global model's backdoor Attack Success Rate (ASR)."""
    attack_config = load_config().get("attacks", {}).get("data_poisoning", {})
    if not attack_config.get("enable"):
        return None 

    model = SimpleNN()
    
    # The 'parameters' variable is already a list of numpy arrays, so we pass it directly
    set_parameters(model, parameters) # <-- REMOVED parameters_to_ndarrays()
    
    _, test_data = load_data("./data")

    backdoor_test_set = PoisonedDataset(
        dataset=test_data,
        poison_frac=1.0,
        target_label=attack_config.get("target_label", 0)
    )
    backdoor_loader = torch.utils.data.DataLoader(backdoor_test_set, batch_size=64)
    
    loss, accuracy = test(model, backdoor_loader)
    
    # The print statement was moved to the metrics aggregation function
    # to avoid printing during the initial server evaluation round.
    
    return loss, {"backdoor_asr": accuracy}


def main():
    config = load_config()
    
    strategy = SecureFedAvg(
        attack_config=config.get("attacks", {}),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config["server"]["min_clients"],
        min_evaluate_clients=config["server"]["min_clients"],
        min_available_clients=config["server"]["min_clients"],
        # Use a custom function to aggregate backdoor and standard eval metrics
        evaluate_metrics_aggregation_fn=safe_metrics_aggregation,
        # Activate the backdoor evaluation after each round
        evaluate_fn=evaluate_backdoor, 
    )
    
    fl.server.start_server(
        server_address=config["server"]["address"],
        config=fl.server.ServerConfig(num_rounds=config["server"]["rounds"]),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
import flwr as fl
from flwr.common import Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Any, Optional
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
import torchvision
import pickle

from model import SimpleNN
from utils import get_parameters, load_data
from attacks.gradinversion_plus import gradinversion_group_attack
from attacks.data_poisoning import PoisonedDataset

def load_config() -> Dict[str, Any]:
    with open("config.yml", "r") as f: return yaml.safe_load(f)

def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict)

def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    correct, total_loss = 0, 0.0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1); correct += pred.eq(target).sum().item()
    return total_loss / len(test_loader.dataset), correct / len(test_loader.dataset)

def safe_metrics_aggregation(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    aggregated = {}
    if any("backdoor_asr" in m for _, m in metrics):
        aggregated["backdoor_asr"] = np.mean([m["backdoor_asr"] for _, m in metrics if "backdoor_asr" in m])
    if any("accuracy" in m for _, m in metrics):
         aggregated["accuracy"] = np.mean([m["accuracy"] for _, m in metrics if "accuracy" in m])
    print("\n[Round Metrics]")
    if "accuracy" in aggregated: print(f"Eval Accuracy: {aggregated['accuracy']*100:.2f}%")
    if "backdoor_asr" in aggregated: print(f"Attack Success Rate (ASR): {aggregated['backdoor_asr']*100:.2f}%")
    return aggregated

class SecureFedAvg(FedAvg):
    def __init__(self, attack_config: Dict[str, Any], client_config: Dict[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_config = attack_config
        self.client_config = client_config
        self.reconstruction_dir = "reconstructions"
        os.makedirs(self.reconstruction_dir, exist_ok=True)

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        model = SimpleNN()
        return ndarrays_to_parameters(get_parameters(model))

    def aggregate_fit(self, server_round: int, results: List[Tuple[Any, Any]], failures: List[Any]) -> Optional[Parameters]:
        gi_params = self.attack_config.get("gradient_inversion", {})
        if not gi_params.get("enable", False):
            return super().aggregate_fit(server_round, results, failures)

        target_client_id = gi_params.get("target_client", 1)
        target_fit_res, honest_clients_results = None, []
        for client_proxy, fit_res in results:
            if fit_res.metrics.get("attack") == "gradient_inversion":
                target_fit_res = fit_res
            else:
                honest_clients_results.append((client_proxy, fit_res))
        
        if target_fit_res:
            print(f"\n[Attack] Intercepted update from gradient inversion target: client {target_client_id}")
            try:
                gradients = parameters_to_ndarrays(target_fit_res.parameters)
                reconstruction_result = self._reconstruct_data(gradients, gi_params)
                if reconstruction_result is not None:
                    reconstructed_images, predicted_labels = reconstruction_result
                    original_data, original_labels = None, None
                    data_path = f"client_data/client_{target_client_id}_data.pkl"
                    if os.path.exists(data_path):
                        with open(data_path, "rb") as f:
                            saved_data = pickle.load(f)
                        original_data, original_labels = saved_data['data'], saved_data['label']
                        os.remove(data_path)
                    
                    # --- THE FIX: Use 'target_client_id' instead of 'client_id' ---
                    self._save_reconstruction(reconstructed_images, predicted_labels, target_client_id, server_round, original_data, original_labels)
                    
            except Exception as e:
                print(f"[Attack Failed] Gradient Inversion error: {str(e)}")
        
        if not honest_clients_results: return None
        return super().aggregate_fit(server_round, honest_clients_results, failures)

    def _reconstruct_data(self, gradients: List[np.ndarray], attack_params: Dict) -> Optional[Tuple[np.ndarray, torch.Tensor]]:
        attack_type = attack_params.get("type", "dlg")
        print(f"[Attack] Attempting reconstruction using '{attack_type}' method.")
        if attack_type == "gradinversion_plus":
            return gradinversion_group_attack(gradients=gradients, batch_size=self.client_config.get("batch_size", 8), num_seeds=attack_params.get("num_seeds", 4), lr=attack_params.get("attack_lr"), iterations=attack_params.get("iterations"), reg_tv=attack_params.get("reg_tv", 1e-4), reg_group=attack_params.get("reg_group", 0.005))
        return None

    def _save_reconstruction(self, data: np.ndarray, predicted_labels: torch.Tensor, client_id: int, round_num: int, original_data: Optional[np.ndarray] = None, original_labels: Optional[np.ndarray] = None):
        recon_tensor = torch.from_numpy(data)
        if original_data is not None and original_labels is not None:
            original_tensor = torch.from_numpy(original_data)
            original_labels_tensor = torch.from_numpy(original_labels)
            
            sort_indices = torch.argsort(original_labels_tensor)
            original_tensor_sorted = original_tensor[sort_indices]
            
            comparison_grid = torch.cat([original_tensor_sorted, recon_tensor])
            save_path = f"{self.reconstruction_dir}/comparison_client{client_id}_round{round_num}.png"
            torchvision.utils.save_image(comparison_grid, save_path, nrow=original_tensor.size(0))
            print(f"[Attack] Saved comparison grid to {save_path}")
        else:
            save_path = f"{self.reconstruction_dir}/reconstruction_client{client_id}_round{round_num}.png"
            torchvision.utils.save_image(recon_tensor, save_path)
            print(f"[Attack] Saved reconstruction grid to {save_path}")

def evaluate_backdoor(server_round: int, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    attack_config = load_config().get("attacks", {}).get("data_poisoning", {})
    if not attack_config.get("enable"): return None 
    model = SimpleNN()
    set_parameters(model, parameters)
    _, test_data = load_data("./data")
    backdoor_test_set = PoisonedDataset(dataset=test_data, poison_frac=1.0, target_label=attack_config.get("target_label", 0))
    backdoor_loader = torch.utils.data.DataLoader(backdoor_test_set, batch_size=64)
    loss, accuracy = test(model, backdoor_loader)
    return loss, {"backdoor_asr": accuracy}

def main():
    config = load_config()
    strategy = SecureFedAvg(
        attack_config=config.get("attacks", {}),
        client_config=config.get("clients", {}),
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=config["server"]["min_clients"],
        min_evaluate_clients=config["server"]["min_clients"],
        min_available_clients=config["server"]["min_clients"],
        evaluate_metrics_aggregation_fn=safe_metrics_aggregation,
        fit_metrics_aggregation_fn=safe_metrics_aggregation,
        evaluate_fn=evaluate_backdoor, 
    )
    
    fl.server.start_server(
        server_address=config["server"]["address"],
        config=fl.server.ServerConfig(num_rounds=config["server"]["rounds"]),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
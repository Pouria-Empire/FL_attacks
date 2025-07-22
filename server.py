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

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from scipy.spatial import distance

from model import SimpleNN
from utils import get_parameters, load_data
from attacks.gradient_inversion import gradinversion_attack, dlg_attack, mdlg_attack
from attacks.gradinversion_plus import gradinversion_group_attack
from attacks.ggl_attack import ggl_attack
from attacks.data_poisoning import PoisonedDataset

# Import your custom modules
from model import SimpleNN
from attacks.data_poisoning import PoisonedDataset

def load_config() -> Dict[str, Any]:
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict)

def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    # This function is defined in client.py and should be available if you run utils.py
    # For now, let's assume it's available. If not, you need to define it or import it.
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


def safe_metrics_aggregation(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    aggregated = {}
    if any("backdoor_asr" in m for _, m in metrics):
        aggregated["backdoor_asr"] = np.mean([m["backdoor_asr"] for _, m in metrics if "backdoor_asr" in m])
    if any("accuracy" in m for _, m in metrics):
         aggregated["accuracy"] = np.mean([m["accuracy"] for _, m in metrics if "accuracy" in m])
    
    print("\n[Round Metrics]")
    if "accuracy" in aggregated:
        print(f"Eval Accuracy: {aggregated['accuracy']*100:.2f}%")
    if "backdoor_asr" in aggregated:
        print(f"Attack Success Rate (ASR): {aggregated['backdoor_asr']*100:.2f}%")
        
    if any("fit_duration" in m for _, m in metrics):
        avg_fit_duration = np.mean([m["fit_duration"] for num, m in metrics if "fit_duration" in m and num > 0])
        if not np.isnan(avg_fit_duration):
            print(f"Avg. Client Fit Time: {avg_fit_duration:.4f} seconds")

    return aggregated

class SecureFedAvg(FedAvg):
    def __init__(self, mitigation_config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.mitigation_config = mitigation_config

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        config = {}
        if self.mitigation_config.get("enable", False):
            defense_type = self.mitigation_config.get("defense_type")
            config["defense_type"] = defense_type
            
            if defense_type == "clipping":
                config.update(self.mitigation_config.get("clipping_params", {}))
            elif defense_type == "sparsification":
                config.update(self.mitigation_config.get("sparsification_params", {}))
            elif defense_type == "dp":
                config.update(self.mitigation_config.get("dp_params", {}))
            elif defense_type == "encryption":
                config.update(self.mitigation_config.get("encryption_params", {}))
        
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        
        for _, fit_ins in fit_ins_list:
            fit_ins.config.update(config)
            
        return fit_ins_list

    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], failures: List[BaseException]) -> Optional[Parameters]:
        if self.mitigation_config.get("enable", False) and self.mitigation_config.get("defense_type") == "median":
            if not results:
                return None
            print("\n[Mitigation] Applying coordinate-wise median aggregation.")
            all_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            stacked_params = zip(*all_params)
            median_params = [np.median(np.stack(layer_params), axis=0) for layer_params in stacked_params]
            return ndarrays_to_parameters(median_params), {}

        return super().aggregate_fit(server_round, results, failures)

def main():
    config = load_config()
    # Need to load data here for the backdoor evaluation
    _, test_data = load_data(config["data"]["path"])

    def evaluate_backdoor(server_round: int, parameters: List[np.ndarray], conf: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        attack_config = config.get("attacks", {}).get("data_poisoning", {})
        if not attack_config.get("enable"):
            return None
        
        model = SimpleNN()
        set_parameters(model, parameters)
        
        backdoor_test_set = PoisonedDataset(dataset=test_data, poison_frac=1.0, target_label=attack_config.get("target_label", 0))
        backdoor_loader = torch.utils.data.DataLoader(backdoor_test_set, batch_size=64)
        loss, accuracy = test(model, backdoor_loader)
        return loss, {"backdoor_asr": accuracy}

    strategy = SecureFedAvg(
        mitigation_config=config.get("mitigations", {}),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
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
import flwr as fl
from flwr.common import (
    #FitIns,
    NDArrays,  # <-- FIX: Added the missing import
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Any
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
import pickle

# Import model and attacks
from model import SimpleNN
from utils import get_parameters
from attacks.gradient_inversion import dlg_attack, mdlg_attack

def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def safe_metrics_aggregation(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    aggregated = {}
    train_metrics = [m for _, m in metrics if m.get("phase") == "train"]
    eval_metrics = [m for _, m in metrics if m.get("phase") == "eval"]

    def avg_metric(metric_list, name):
        total = sum(m.get(name, 0) * m.get("num_examples", 1) for m in metric_list)
        examples = sum(m.get("num_examples", 1) for m in metric_list)
        return total / examples if examples > 0 else 0.0

    if train_metrics:
        aggregated["train_loss"] = avg_metric(train_metrics, "loss")
        aggregated["train_accuracy"] = avg_metric(train_metrics, "accuracy")
    
    if eval_metrics:
        aggregated["eval_loss"] = avg_metric(eval_metrics, "loss")
        aggregated["eval_accuracy"] = avg_metric(eval_metrics, "accuracy")

    print("\n[Round Metrics]")
    if "train_loss" in aggregated:
        print(f"Train Loss: {aggregated['train_loss']:.4f} | Accuracy: {aggregated['train_accuracy']*100:.2f}%")
    if "eval_loss" in aggregated:
        print(f"Eval Loss: {aggregated['eval_loss']:.4f} | Accuracy: {aggregated['eval_accuracy']*100:.2f}%")
    
    return aggregated


class SecureFedAvg(FedAvg):
    def __init__(self, attack_config: Dict[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_config = attack_config
        self.reconstruction_dir = "reconstructions"
        os.makedirs(self.reconstruction_dir, exist_ok=True)

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        model = SimpleNN()
        params = get_parameters(model)
        return ndarrays_to_parameters(params)

    def aggregate_fit(self, server_round, results, failures):
        
        attack_params = self.attack_config.get("gradient_inversion", {})
        is_attack_enabled = attack_params.get("enable", False)
        target_client_id = attack_params.get("target_client", 1)

        # --- DLG Attack Logic ---
        if is_attack_enabled:
            target_fit_res = None
            for client, fit_res in results:
                # Find the target client's result
                if fit_res.metrics.get("logical_client_id") == target_client_id:
                    print(f"\n[DLG] Intercepted gradients from target client {target_client_id}")
                    target_fit_res = fit_res
                    break
            
            if target_fit_res:
                try:
                    # The parameters are now raw gradients
                    gradients_ndarrays = parameters_to_ndarrays(target_fit_res.parameters)
                    
                    data_file = f"client_data/client_{target_client_id}_data.pkl"
                    with open(data_file, 'rb') as f:
                        client_data = pickle.load(f)
                    
                    reconstructed = self._reconstruct_data(gradients_ndarrays, attack_params)
                    
                    if reconstructed is not None:
                        self._save_reconstruction(
                            reconstructed,
                            client_id=target_client_id,
                            round_num=server_round,
                            original_data=client_data["data_batch"],
                            original_label=client_data["target_batch"]
                        )
                    os.remove(data_file)
                except Exception as e:
                    print(f"[DLG Failed] Error on client {target_client_id}: {str(e)}")

        # --- Standard Aggregation for honest clients ---
        honest_clients_results = [
            (client, fit_res) for client, fit_res in results 
            if not is_attack_enabled or fit_res.metrics.get("logical_client_id") != target_client_id
        ]
        
        if not honest_clients_results:
             # If only the target client responded, there's nothing to aggregate.
             # Return the current server model without changes.
            return self.current_parameters, {}

        # Proceed with standard FedAvg on the honest clients
        aggregated, metrics = super().aggregate_fit(server_round, honest_clients_results, failures)
        
        return aggregated, metrics

    def _compute_gradients(self, initial_params: NDArrays, updated_params: Parameters, lr: float) -> List[np.ndarray]:
        # This method is no longer needed for the DLG attack but might be useful for other attacks.
        updated_weights = parameters_to_ndarrays(updated_params)
        return [(init - upd) / lr for init, upd in zip(initial_params, updated_weights)]

    def _reconstruct_data(self, gradients: List[np.ndarray], attack_params: Dict) -> np.ndarray:
        attack_type = attack_params.get("type", "dlg")
        print(f"[Attack] Attempting reconstruction using '{attack_type}' method.")

        if attack_type == "dlg":
            return dlg_attack(
                gradients=gradients,
                input_shape=(1, 1, 28, 28),
                lr=attack_params.get("attack_lr", 0.1),
                iterations=attack_params.get("iterations", 1000)
            )
        elif attack_type == "mdlg":
             # mDLG only uses the first gradient (fc1.weight)
            return mdlg_attack(
                gradients=[gradients[0]], 
                input_shape=(1, 1, 28, 28),
                lr=attack_params.get("attack_lr", 0.01),
                iterations=attack_params.get("iterations", 500)
            )
        else:
            print(f"Unknown attack type: {attack_type}")
            return None

    def _save_reconstruction(self, data, client_id, round_num, original_data=None, original_label=None):
        img_array = data
        img = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        img = (img * 255).astype(np.uint8).squeeze()
        
        recon_path = f"{self.reconstruction_dir}/client{client_id}_round{round_num}.png"
        Image.fromarray(img).save(recon_path)
        
        if original_data is not None:
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(original_data[0].squeeze(), cmap='gray')
            plt.title(f"Original (Label: {original_label[0]})")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(img, cmap='gray')
            plt.title("Reconstructed")
            plt.axis('off')
            
            comp_path = f"{self.reconstruction_dir}/comparison_client{client_id}_round{round_num}.png"
            plt.savefig(comp_path)
            plt.close()
            print(f"[DLG] Saved comparison to {comp_path}")
        else:
            print(f"[DLG] Saved reconstruction to {recon_path}")

def main():
    config = load_config()
    strategy = SecureFedAvg(
        attack_config=config.get("attacks", {}),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config["server"]["min_clients"],
        min_evaluate_clients=config["server"]["min_clients"],
        min_available_clients=config["server"]["min_clients"],
        evaluate_metrics_aggregation_fn=safe_metrics_aggregation,
        fit_metrics_aggregation_fn=safe_metrics_aggregation
    )
    
    fl.server.start_server(
        server_address=config["server"]["address"],
        config=fl.server.ServerConfig(num_rounds=config["server"]["rounds"]),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
import flwr as fl
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Optional, Any, Union
from flwr.common import Scalar, Parameters, NDArrays, parameters_to_ndarrays
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from collections import OrderedDict
import torch
from model import SimpleNN  # Import your model class
from utils import get_parameters  # Import parameter extraction function
from flwr.common import ndarrays_to_parameters  # Import parameter conversion
# Import attacks
from attacks.gradient_inversion import dlg_attack, mdlg_attack

def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def weighted_metrics_aggregation(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Improved metrics aggregation with weighted averaging"""
    if not metrics:
        return {}
    
    # Initialize metrics storage
    aggregated = {
        "train_loss": 0.0,
        "train_accuracy": 0.0,
        "eval_loss": 0.0,
        "eval_accuracy": 0.0
    }
    train_count = 0
    eval_count = 0
    
    for num_examples, m in metrics:
        phase = m.get("phase", "train")
        
        if phase == "train":
            if "loss" in m:
                aggregated["train_loss"] += m["loss"] * num_examples
            if "accuracy" in m:
                aggregated["train_accuracy"] += m["accuracy"] * num_examples
            train_count += num_examples
        elif phase == "eval":
            if "loss" in m:
                aggregated["eval_loss"] += m["loss"] * num_examples
            if "accuracy" in m:
                aggregated["eval_accuracy"] += m["accuracy"] * num_examples
            eval_count += num_examples
    
    # Calculate weighted averages
    if train_count > 0:
        aggregated["train_loss"] /= train_count
        aggregated["train_accuracy"] /= train_count
    if eval_count > 0:
        aggregated["eval_loss"] /= eval_count
        aggregated["eval_accuracy"] /= eval_count
    
    # Print results only if we have metrics
    print("\n[Round Summary]")
    if train_count > 0:
        print(f"Train Loss: {aggregated['train_loss']:.4f} | Accuracy: {aggregated['train_accuracy']*100:.2f}%")
    if eval_count > 0:
        print(f"Eval Loss: {aggregated['eval_loss']:.4f} | Accuracy: {aggregated['eval_accuracy']*100:.2f}%")
    
    return {k: v for k, v in aggregated.items() if (("train" in k and train_count > 0) or ("eval" in k and eval_count > 0))}

class ImprovedFedAvg(FedAvg):
    def __init__(self, attack_config: Dict[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_config = attack_config
        self.global_model = None

    def initialize_parameters(self, client_manager):
        """Initialize global model parameters"""
        model = SimpleNN()  # Make sure SimpleNN is imported
        self.global_model = ndarrays_to_parameters(get_parameters(model))
        return self.global_model

    def aggregate_fit(self, server_round, results, failures):
        """Improved aggregation with better error handling"""
        # First get aggregated parameters and metrics
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)
        
        if not results:
            return aggregated_params, metrics

        # Store global model for gradient computation
        if aggregated_params is not None:
            self.global_model = aggregated_params

        # Gradient inversion attack if enabled
        if self.attack_config.get("gradient_inversion", {}).get("enable", False):
            self._perform_gradient_inversion(server_round, results)
        
        return aggregated_params, metrics

    def _perform_gradient_inversion(self, server_round, results):
        """Handle gradient inversion attacks safely"""
        target_client_id = self.attack_config["gradient_inversion"].get("target_client", 1)
        learning_rate = self.attack_config["gradient_inversion"].get("learning_rate", 0.01)
        
        for client, fit_res in results:
            cid = client.cid
            try:
                # Handle both UUID and clientX formats
                if cid.startswith("client"):
                    client_id = int(cid.replace("client", ""))
                else:
                    continue  # Skip UUID clients for attacks
                
                if client_id == target_client_id and self.global_model is not None:
                    # Perform reconstruction
                    updated_params = fit_res.parameters
                    gradients = self._compute_gradients(updated_params, learning_rate)
                    reconstructed = self._reconstruct_data(gradients)
                    
                    if reconstructed is not None:
                        self._save_reconstruction(reconstructed, client_id, server_round)
            except Exception as e:
                print(f"Skipping gradient inversion for {cid}: {str(e)}")
                continue

    def _compute_gradients(self, updated_params, lr):
        """Compute gradients from parameter updates"""
        updated = parameters_to_ndarrays(updated_params)
        global_weights = parameters_to_ndarrays(self.global_model)
        return [(w - u) / lr for w, u in zip(global_weights, updated)]

    def _reconstruct_data(self, gradients):
        """Reconstruct data using configured attack method"""
        attack_type = self.attack_config["gradient_inversion"].get("type", "mdlg")
        if attack_type == "dlg":
            return dlg_attack(gradients, (1, 1, 28, 28), (1, 10))
        return mdlg_attack(gradients, (1, 1, 28, 28))

    def _save_reconstruction(self, data, client_id, round_num):
        """Save and visualize reconstructed data"""
        img_data = data[0, 0]
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255
        img_data = img_data.astype(np.uint8)
        
        os.makedirs("reconstructions", exist_ok=True)
        img_path = f"reconstructions/client{client_id}_round{round_num}.png"
        Image.fromarray(img_data).save(img_path)
        print(f"Saved reconstruction to {img_path}")

def main():
    config = load_config()
    attack_config = config.get("attacks", {})
    
    strategy = ImprovedFedAvg(
        attack_config=attack_config,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config["server"]["min_clients"],
        min_evaluate_clients=config["server"]["min_clients"],
        min_available_clients=config["server"]["min_clients"],
        evaluate_metrics_aggregation_fn=weighted_metrics_aggregation,
        fit_metrics_aggregation_fn=weighted_metrics_aggregation
    )
    
    fl.server.start_server(
        server_address=config["server"]["address"],
        config=fl.server.ServerConfig(num_rounds=config["server"]["rounds"]),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
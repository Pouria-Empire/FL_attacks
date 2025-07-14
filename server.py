import flwr as fl
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Optional, Any, Union
from flwr.common import Scalar, Parameters, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch

# Import model and attacks
from model import SimpleNN
from utils import get_parameters
from attacks.gradient_inversion import dlg_attack, mdlg_attack

def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def safe_metrics_aggregation(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Robust metrics aggregation"""
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
        self.global_model = None

    def initialize_parameters(self, client_manager):
        """Initialize global model parameters"""
        model = SimpleNN()
        params = get_parameters(model)
        self.global_model = ndarrays_to_parameters(params)
        return self.global_model

    def aggregate_fit(self, server_round, results, failures):
        """Main aggregation with attack handling"""
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        
        if not results:
            return aggregated, metrics

        if aggregated:
            self.global_model = aggregated

        if self.attack_config.get("gradient_inversion", {}).get("enable", False):
            self._perform_gradient_attack(server_round, results)

        return aggregated, metrics

    def _perform_gradient_attack(self, server_round, results):
        """UUID-only gradient attack with shape validation"""
        target_pos = self.attack_config["gradient_inversion"].get("target_client", 1) - 1
        
        if len(results) <= target_pos or not self.global_model:
            return

        client, res = results[target_pos]
        try:
            gradients = self._compute_gradients(res.parameters, 
                        self.attack_config["gradient_inversion"].get("learning_rate", 0.01))
            
            # Expected shapes for SimpleNN (784->64->10)
            expected_shapes = {
                'dlg': [torch.Size([64, 784]), torch.Size([64])],
                'mdlg': [torch.Size([64, 784])]
            }
            
            attack_type = self.attack_config["gradient_inversion"].get("type", "dlg")
            req_shapes = expected_shapes[attack_type]
            
            if all(g.shape == s for g, s in zip(gradients[:len(req_shapes)], req_shapes)):
                reconstructed = dlg_attack(
                    [g.numpy() for g in gradients[:2]],   # 0: weight, 1: bias
                    (1, 1, 28, 28),
                    (1, 10)
                )
                if reconstructed is not None:
                    self._save_reconstruction(reconstructed, target_pos+1, server_round)
        
        except Exception as e:
            print(f"Attack failed on client {client.cid}: {str(e)}")

    def _compute_gradients(self, updated_params, lr):
        """Compute gradients with automatic shape correction"""
        updated = parameters_to_ndarrays(updated_params)
        global_weights = parameters_to_ndarrays(self.global_model)
        
        gradients = []
        for i, (g, u) in enumerate(zip(global_weights, updated)):
            grad = (g - u) / lr
            
            # Auto-reshape based on parameter position
            if i == 0:  # fc1 weights (64, 784)
                grad = grad.reshape(64, 784)
            elif i == 1:  # fc1 bias (64)
                grad = grad.reshape(64)
            elif i == 2:  # fc2 weights (10, 64)
                grad = grad.reshape(10, 64)
            elif i == 3:  # fc2 bias (10)
                grad = grad.reshape(10)
                
            gradients.append(torch.from_numpy(grad).float())
        
        return gradients

    def _reconstruct_data(self, gradients, attack_type):
        """Handle gradient shapes correctly for MNIST reconstruction"""
        if attack_type == "dlg":
            # For MNIST: fc1 weights are (64, 784), fc2 are (10, 64)
            fc1_grad = gradients[0].numpy()  # Shape should be (64, 784)
            fc1_grad = fc1_grad.reshape(64, 784) if fc1_grad.shape != (64, 784) else fc1_grad
            fc2_grad = gradients[1].numpy()  # Shape should be (10, 64)
            fc2_grad = fc2_grad.reshape(10, 64) if fc2_grad.shape != (10, 64) else fc2_grad
            
            return dlg_attack(
                [fc1_grad, fc2_grad],  # Weight and bias gradients
                (1, 1, 28, 28),        # MNIST input shape
                (1, 10)                 # Output classes
            )
        else:
            # For MDLG just use first layer gradients
            fc1_grad = gradients[0].numpy()
            fc1_grad = fc1_grad.reshape(64, 784) if fc1_grad.shape != (64, 784) else fc1_grad
            return mdlg_attack([fc1_grad], (1, 1, 28, 28))

    def _save_reconstruction(self, data, client_id, round_num):
        """Save reconstructed image"""
        img = (data[0, 0] - data[0, 0].min()) / (data[0, 0].max() - data[0, 0].min()) * 255
        img = img.astype(np.uint8)
        
        os.makedirs("reconstructions", exist_ok=True)
        path = f"reconstructions/client{client_id}_round{round_num}.png"
        Image.fromarray(img).save(path)
        print(f"[Attack] Saved reconstruction to {path}")

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
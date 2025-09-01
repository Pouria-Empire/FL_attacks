# server/main.py

import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
from typing import Dict, List, Tuple, Optional

from server.helpers import load_config, set_parameters, test_and_log_misclassifications, safe_metrics_aggregation
from server.strategy import SecureFedAvg
from model import CastingCNN, SensorMLP, CifarCNN, MedMNIST_CNN
from utils_data.casting_data_util import load_data as load_casting_data
from utils_data.sensor_data_util import load_sensor_data, load_and_preprocess_data 
from utils_data.cifar_data_util import load_data as load_cifar_data
from utils_data.medmnist_data_util import load_data as load_medmnist_data

# ✅ FIX: Import the correct, new class name 'PoisonedDatasetWrapper'.
from attacks.data_poisoning import PoisonedDatasetWrapper
from attacks.numerical_attacks import PoisonedSensorDataset


def main():
    """Load data, start Flower server with all features."""
    config = load_config()
    data_config = config["data"]
    data_type = data_config.get("type", "casting")

    # --- Load data based on type for server-side evaluation ---
    if data_type == "casting":
        trainset, testset = load_casting_data(data_config["path"], data_config["img_size"])
        server_holdout = Subset(trainset, list(range(min(200, len(trainset)))))
    elif data_type == "sensor":
        trainset, testset = load_sensor_data(data_config["path"])
        server_holdout = Subset(trainset, list(range(min(100, len(trainset)))))
    elif data_type == "cifar10":
        trainset, testset = load_cifar_data(data_config["path"], data_config["img_size"])
        server_holdout = Subset(trainset, list(range(500)))
    elif data_type == "medmnist":
        trainset, testset = load_medmnist_data(data_config["dataset_name"], data_config["path"])
        server_holdout = Subset(trainset, list(range(500)))
    else:
        raise ValueError(f"Invalid data type in config.yml: {data_type}")

    testloader = DataLoader(testset, batch_size=128)
    server_holdout_loader = DataLoader(server_holdout, batch_size=32)

    def get_evaluate_fn(config: dict):
        """Return an evaluation function for server-side evaluation."""
        attack_config = config.get("attacks", {}).get("data_poisoning", {})
        
        if os.path.exists("backdoor_misclassifications.log"):
            os.remove("backdoor_misclassifications.log")
        
        def evaluate(server_round: int, parameters: fl.common.NDArrays, conf: dict) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            if data_type == "medmnist":
                model = MedMNIST_CNN(num_classes=data_config["num_classes"])
            else:
                raise ValueError(f"Unknown data_type for evaluation: {data_type}")

            set_parameters(model, parameters)
            
            # ✅ FIX: Added the missing 'data_type' argument to this function call.
            loss, accuracy = test_and_log_misclassifications(
                model=model,
                test_loader=testloader,
                is_backdoor_test=False, 
                target_label=0,
                data_type=data_type, # <-- This was missing
                num_classes=data_config["num_classes"],
                class_names=data_config.get("class_names", [])
            )
            
            backdoor_asr = 0.0
            if attack_config.get("enable", False):
                target_label = attack_config.get("target_label", 1)
                
                backdoor_test_set = PoisonedDatasetWrapper(dataset=testset, poison_frac=1.0, target_label=target_label, data_type=data_type)
                backdoor_loader = DataLoader(backdoor_test_set, batch_size=128)
                
                # ✅ FIX: Also added the missing 'data_type' argument to this second call.
                _, backdoor_asr = test_and_log_misclassifications(
                    model=model,
                    test_loader=backdoor_loader,
                    is_backdoor_test=True, 
                    target_label=target_label,
                    data_type=data_type, # <-- This was missing
                    num_classes=data_config["num_classes"],
                    class_names=data_config.get("class_names", [])
                )
            
            return loss, {"accuracy": accuracy, "backdoor_asr": backdoor_asr}
        return evaluate
    strategy = SecureFedAvg(
        config=config,
        server_holdout_loader=server_holdout_loader,
        test_loader=testloader,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config["server"]["min_clients"],
        min_evaluate_clients=config["server"]["min_clients"],
        min_available_clients=config["server"]["min_clients"],
        evaluate_fn=get_evaluate_fn(config),
        fit_metrics_aggregation_fn=safe_metrics_aggregation,
        evaluate_metrics_aggregation_fn=safe_metrics_aggregation,
    )
    
    fl.server.start_server(
        server_address=config["server"]["address"],
        config=fl.server.ServerConfig(num_rounds=config["server"]["rounds"]),
        strategy=strategy,
        grpc_max_message_length=1024*1024*1024
    )

if __name__ == "__main__":
    main()
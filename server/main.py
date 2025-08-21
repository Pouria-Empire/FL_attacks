import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Optional
import os
import numpy as np

# Import from the new server package and other modules
from server.helpers import load_config, set_parameters, test_and_log_misclassifications, safe_metrics_aggregation
from server.strategy import SecureFedAvg
from model import CastingCNN, SensorMLP, CifarCNN

# Import all data loaders
from utils_data.casting_data_util import load_data as load_casting_data
from utils_data.sensor_data_util import load_sensor_data, load_and_preprocess_data
from utils_data.cifar_data_util import load_data as load_cifar_data
from attacks.data_poisoning import PoisonedDataset
from attacks.numerical_attacks import PoisonedSensorDataset


def main():
    """Load data, start Flower server with all features."""
    config = load_config()
    data_config = config["data"]
    data_type = data_config.get("type", "casting")

    # --- Load data based on type for server-side evaluation ---
    if data_type == "casting":
        print("Server loading CASTING dataset...")
        trainset, testset = load_casting_data(
            data_config["path"],
            data_config["img_size"]
        )
        testloader = DataLoader(testset, batch_size=32)
        server_holdout = Subset(trainset, list(range(min(200, len(trainset)))))
        server_holdout_loader = DataLoader(server_holdout, batch_size=32)
        
    elif data_type == "sensor":
        print("Server loading SENSOR dataset...")
        trainset, testset = load_sensor_data(data_config["path"])
        testloader = DataLoader(testset, batch_size=32)
        server_holdout = Subset(trainset, list(range(min(100, len(trainset)))))
        server_holdout_loader = DataLoader(server_holdout, batch_size=32)

    elif data_type == "cifar10":
        print("Server loading CIFAR-10 dataset...")
        trainset, testset = load_cifar_data(
            data_config["path"],
            data_config["img_size"]
        )
        testloader = DataLoader(testset, batch_size=64)
        server_holdout = Subset(trainset, list(range(500)))
        server_holdout_loader = DataLoader(server_holdout, batch_size=64)
    else:
        raise ValueError(f"Invalid data type in config.yml: {data_type}")

    def get_evaluate_fn(config: dict):
        """Return an evaluation function for server-side evaluation."""
        attack_config = config.get("attacks", {}).get("data_poisoning", {})
        
        if os.path.exists("backdoor_misclassifications.log"):
            os.remove("backdoor_misclassifications.log")
        
        def evaluate(server_round: int, parameters: fl.common.NDArrays, conf: dict) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            # Select the correct model based on the data type
            if data_type == "casting":
                model = CastingCNN(num_classes=data_config["num_classes"])
            elif data_type == "cifar10":
                model = CifarCNN(num_classes=data_config["num_classes"])
            else: # sensor
                X, y, _ = load_and_preprocess_data(data_config["path"])
                model = SensorMLP(input_features=X.shape[1], num_classes=len(np.unique(y)))

            set_parameters(model, parameters)
            
            loss, accuracy = test_and_log_misclassifications(
                model, testloader, False, 0, 
                data_type=data_type,
                class_names=data_config.get("class_names", [])
            )
            
            backdoor_asr = 0.0
            if attack_config.get("enable", False):
                print(f"\n--- Testing Backdoor (Round {server_round}) ---")
                target_label = attack_config.get("target_label", 1)
                
                # Create a triggered test set for ASR calculation
                if data_type in ["casting", "cifar10"]:
                    backdoor_test_set = PoisonedDataset(dataset=testset, poison_frac=1.0, target_label=target_label)
                else: # sensor
                    backdoor_test_set = PoisonedSensorDataset(
                        dataset=testset, poison_frac=1.0, target_label=target_label,
                        trigger_noise_level=attack_config.get("trigger_noise_level", 0.1)
                    )
                backdoor_loader = DataLoader(backdoor_test_set, batch_size=32)
                
                with open("backdoor_misclassifications.log", "a") as f:
                    f.write(f"--- MISCLASSIFICATIONS FOR ROUND {server_round} ---\n")

                _, backdoor_asr = test_and_log_misclassifications(
                    model, backdoor_loader, True, target_label, 
                    data_type=data_type, 
                    class_names=data_config.get("class_names", [])
                )
            
            print(f"Server-side evaluation round {server_round} complete.")
            return loss, {"accuracy": accuracy, "backdoor_asr": backdoor_asr}
        return evaluate

    # Define strategy
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
    
    # Start Flower server
    fl.server.start_server(
        server_address=config["server"]["address"],
        config=fl.server.ServerConfig(num_rounds=config["server"]["rounds"]),
        strategy=strategy,
        grpc_max_message_length=1024*1024*1024
    )

if __name__ == "__main__":
    main()
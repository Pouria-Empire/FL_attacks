import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Optional
import os
import numpy as np

# Import from the new server package and other modules
from server.helpers import load_config, set_parameters, test_and_log_misclassifications, safe_metrics_aggregation
from server.strategy import SecureFedAvg
from model import SimpleNN, SensorMLP

# Import both data loaders
from utils_data.chest_data_util import load_data as load_image_data, ChestXRayDataset
from utils_data.sensor_data_util import load_sensor_data, load_and_preprocess_data
from attacks.numerical_attacks import PoisonedSensorDataset

def main():
    """Load data, start Flower server with all features."""
    config = load_config()
    data_config = config["data"]
    data_type = data_config.get("type", "image")

    # Load data based on type for server-side evaluation
    if data_type == "image":
        # ... (image data loading logic)
        pass
    elif data_type == "sensor":
        print("Server loading SENSOR dataset...")
        trainset, testset = load_sensor_data(data_config["path"])
        testloader = DataLoader(testset, batch_size=32)
        server_holdout = Subset(trainset, list(range(min(100, len(trainset)))))
        server_holdout_loader = DataLoader(server_holdout, batch_size=32)
    else:
        raise ValueError(f"Invalid data type in config.yml: {data_type}")

    def get_evaluate_fn(config: dict):
        """Return an evaluation function for server-side evaluation."""
        attack_config = config.get("attacks", {}).get("data_poisoning", {})
        
        if os.path.exists("backdoor_misclassifications.log"):
            os.remove("backdoor_misclassifications.log")
        
        def evaluate(server_round: int, parameters: fl.common.NDArrays, conf: dict) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            # Select the correct model based on the data type
            if data_type == "image":
                model = SimpleNN(num_classes=15)
            else:
                X, y, _ = load_and_preprocess_data(data_config["path"])
                model = SensorMLP(input_features=X.shape[1], num_classes=len(np.unique(y)))

            set_parameters(model, parameters)
            
            # 1. Evaluate Normal Accuracy
            loss, accuracy = test_and_log_misclassifications(model, testloader, False, 0, is_image=(data_type=="image"))
            
            backdoor_asr = 0.0
            if attack_config.get("enable", False):
                target_label = attack_config.get("target_label", 0)
                
                # Create a triggered test set for ASR calculation
                backdoor_test_set = PoisonedSensorDataset(
                    dataset=testset, 
                    poison_frac=1.0, # Trigger all images
                    target_label=target_label,
                    trigger_noise_level=attack_config.get("trigger_noise_level", 0.1)
                )
                backdoor_loader = DataLoader(backdoor_test_set, batch_size=32)
                
                with open("backdoor_misclassifications.log", "a") as f:
                    f.write(f"--- MISCLASSIFICATIONS FOR ROUND {server_round} ---\n")

                # 2. Evaluate Attack Success Rate
                _, backdoor_asr = test_and_log_misclassifications(model, backdoor_loader, True, target_label, False)
            
            print(f"\nServer-side evaluation round {server_round} complete.")
            return loss, {"accuracy": accuracy, "backdoor_asr": backdoor_asr}
        return evaluate

    # Define strategy
    strategy = SecureFedAvg(
        config=config,
        server_holdout_loader=server_holdout_loader,
        fraction_fit=1.0,
        fraction_evaluate=1.0, # <-- Re-enable server-side evaluation
        min_fit_clients=config["server"]["min_clients"],
        min_evaluate_clients=config["server"]["min_clients"], # <-- Also needed
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
    if os.path.exists("results.log"):
        os.remove("results.log")
    main()
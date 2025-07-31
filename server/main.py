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
from utils_data.sensor_data_util import get_client_data as get_sensor_data, load_and_preprocess_data, load_sensor_data
from attacks.data_poisoning import PoisonedDataset

def main():
    """Load data, start Flower server with all features."""
    config = load_config()
    data_config = config["data"]
    data_type = data_config.get("type", "image") # Default to image if not specified

    # --- Load data based on type for server-side evaluation ---
    if data_type == "image":
        print("Server loading IMAGE dataset...")
        trainset, testset = load_image_data(
            data_config["path"],
            data_config["train_list"],
            data_config["test_list"]
        )
        testloader = DataLoader(testset, batch_size=32)
        
        # Create a dedicated holdout set to avoid indexing errors
        holdout_df = trainset.df.iloc[:200].reset_index(drop=True)
        server_holdout_set = ChestXRayDataset(
            data_path=data_config["path"],
            df=holdout_df,
            transform=trainset.transform
        )
        server_holdout_loader = DataLoader(server_holdout_set, batch_size=32)
        
    elif data_type == "sensor":
        print("Server loading SENSOR dataset...")
        trainset, testset = load_sensor_data(data_config["path"])
        testloader = DataLoader(testset, batch_size=32)
        
        # For sensor data, create a holdout from the train set
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
            
            loss, accuracy = test_and_log_misclassifications(model, testloader, is_backdoor_test=False, target_label_idx=0, is_image=(data_type=="image"))
            
            backdoor_asr = 0.0
            if attack_config.get("enable", False) and data_type == "image": # Backdoor only for images for now
                target_idx = attack_config.get("target_label_idx", 7)
                backdoor_test_set = PoisonedDataset(dataset=testset, poison_frac=1.0, target_label_idx=target_idx)
                backdoor_loader = DataLoader(backdoor_test_set, batch_size=32)
                with open("backdoor_misclassifications.log", "a") as f:
                    f.write(f"--- MISCLASSIFICATIONS FOR ROUND {server_round} ---\n")
                _, backdoor_asr = test_and_log_misclassifications(model, backdoor_loader, True, target_idx, True)
            
            print(f"\nServer-side evaluation round {server_round} complete.")
            return loss, {"accuracy": accuracy, "backdoor_asr": backdoor_asr}
        return evaluate

    # Define strategy
    strategy = SecureFedAvg(
        config=config,
        server_holdout_loader=server_holdout_loader,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=config["server"]["min_clients"],
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
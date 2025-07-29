import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Optional
import os

# Import from the new server package and other modules
from server.helpers import load_config, set_parameters, test_and_log_misclassifications, safe_metrics_aggregation
from server.strategy import SecureFedAvg
from model import SimpleNN
from chest_data_util import load_data, ChestXRayDataset
from attacks.data_poisoning import PoisonedDataset

def main():
    """Load data, start Flower server with all features."""
    config = load_config()
    data_config = config["data"]
    
    # Load data needed for server-side evaluation and defense agent
    trainset, testset = load_data(
        data_config["path"],
        data_config["train_list"],
        data_config["test_list"]
    )
    testloader = DataLoader(testset, batch_size=32)
    
    # Create a small, clean holdout set for the MyDefense agent's utility calculation
    server_holdout = Subset(trainset, list(range(150)))
    server_holdout_loader = DataLoader(server_holdout, batch_size=32)

    def get_evaluate_fn(config: dict):
        """Return an evaluation function for server-side evaluation."""
        attack_config = config.get("attacks", {}).get("data_poisoning", {})
        
        # Clear the log file at the beginning of each server run
        if os.path.exists("backdoor_misclassifications.log"):
            os.remove("backdoor_misclassifications.log")
        
        def evaluate(server_round: int, parameters: fl.common.NDArrays, conf: dict) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            model = SimpleNN(num_classes=15)
            set_parameters(model, parameters)
            
            # 1. Evaluate Normal Accuracy on the clean test set
            loss, accuracy = test_and_log_misclassifications(model, testloader, is_backdoor_test=False, target_label_idx=0)
            
            # 2. Evaluate Attack Success Rate on a triggered test set (if enabled)
            backdoor_asr = 0.0
            if attack_config.get("enable", False):
                target_idx = attack_config.get("target_label_idx", 7)
                
                # Create a test set where ALL images have the trigger
                backdoor_test_set = PoisonedDataset(
                    dataset=testset, 
                    poison_frac=1.0, # Trigger all images
                    target_label_idx=target_idx
                )
                backdoor_loader = DataLoader(backdoor_test_set, batch_size=32)
                
                # Log which round is being tested
                with open("backdoor_misclassifications.log", "a") as f:
                    f.write(f"--- MISCLASSIFICATIONS FOR ROUND {server_round} ---\n")

                # Run the test and get the ASR
                _, backdoor_asr = test_and_log_misclassifications(model, backdoor_loader, is_backdoor_test=True, target_label_idx=target_idx)
            
            print(f"\nServer-side evaluation round {server_round} complete.")
            return loss, {"accuracy": accuracy, "backdoor_asr": backdoor_asr}
        return evaluate

    # Define strategy
    strategy = SecureFedAvg(
        config=config,
        server_holdout_loader=server_holdout_loader,
        fraction_fit=1.0,
        fraction_evaluate=0.0, # Disable client-side evaluation; server handles it
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
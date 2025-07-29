import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import os
import torch
import pickle
import torchvision

from model import SimpleNN
from server.helpers import get_parameters, evaluate_reconstruction
from server.reconstruction import reconstruct_data, save_reconstruction
from defense.mydefense import MyDefenseAgent
from crypto_utils import encrypt_params, decrypt_params

class SecureFedAvg(FedAvg):
    def __init__(self, config: dict, server_holdout_loader: torch.utils.data.DataLoader, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.attack_config = config.get("attacks", {})
        self.client_config = config.get("clients", {})
        self.mitigation_config = config.get("mitigations", {})
        self.reconstruction_dir = "reconstructions"
        os.makedirs(self.reconstruction_dir, exist_ok=True)
        self.global_parameters = None
        self.gradient_history = {}
        self.cid_to_logical_id = {}
        
        if self.mitigation_config.get("enable", False) and self.mitigation_config.get("defense_type") == "mydefense":
            self.defense_agent = MyDefenseAgent(config, server_holdout_loader)
        else:
            self.defense_agent = None

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        model = SimpleNN(num_classes=15)
        self.global_parameters = get_parameters(model)
        return fl.common.ndarrays_to_parameters(self.global_parameters)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        config = {}
        if self.defense_agent:
            client_configs = []
            num_clients_to_sample = int(self.min_fit_clients)
            sampled_clients = client_manager.sample(
                num_clients=num_clients_to_sample, min_num_clients=num_clients_to_sample
            )
            for client in sampled_clients:
                logical_id = self.cid_to_logical_id.get(client.cid)
                client_config = {}
                if logical_id and self.defense_agent.trigger_chaotic_encryption_for_client.get(logical_id, False):
                    print(f"\n[Mitigation] Instructing Client {logical_id} to apply Chaotic Encryption.")
                    client_config["apply_chaotic_encryption"] = True
                    self.defense_agent.trigger_chaotic_encryption_for_client[logical_id] = False
                client_configs.append((client, fl.common.FitIns(parameters, client_config)))
            return client_configs

        if self.mitigation_config.get("enable", False):
            defense_type = self.mitigation_config.get("defense_type")
            config["defense_type"] = defense_type
            if defense_type == "encryption":
                print("\n[Mitigation] Encrypting global model for distribution.")
                params_np = parameters_to_ndarrays(parameters)
                encrypted_bytes = encrypt_params(params_np)
                parameters = ndarrays_to_parameters([np.frombuffer(encrypted_bytes, dtype=np.uint8)])
            elif defense_type != "none":
                param_key = f"{defense_type}_params"
                if param_key in self.mitigation_config:
                    config.update(self.mitigation_config[param_key])
        
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in fit_ins_list:
            fit_ins.config.update(config)
        return fit_ins_list

    def aggregate_fit(self, server_round: int, results: List[Tuple[Any, Any]], failures: List[Any]):
        # Learn the mapping from connection CID to logical ID
        for client_proxy, fit_res in results:
            if "logical_client_id" in fit_res.metrics:
                self.cid_to_logical_id[client_proxy.cid] = fit_res.metrics["logical_client_id"]
        
        # Decrypt results if encryption is enabled
        decrypted_results = []
        if self.mitigation_config.get("enable", False) and self.mitigation_config.get("defense_type") == "encryption":
            print("[Mitigation] Decrypting client updates.")
            for client_proxy, fit_res in results:
                raw_params = parameters_to_ndarrays(fit_res.parameters)
                if len(raw_params) == 1 and raw_params[0].dtype == np.uint8:
                    try:
                        decrypted_params = decrypt_params(raw_params[0].tobytes())
                        fit_res.parameters = ndarrays_to_parameters(decrypted_params)
                        decrypted_results.append((client_proxy, fit_res))
                    except Exception as e:
                        print(f"Could not decrypt update from {client_proxy.cid}: {e}")
                else:
                    decrypted_results.append((client_proxy, fit_res))
        else:
            decrypted_results = results
        
        # If MyDefense is active, use it to filter clients
        if self.defense_agent:
            print("\n--- MyDefense Agent Analyzing Round ---")
            accepted_results = []
            for client_proxy, fit_res in decrypted_results:
                client_id = self.cid_to_logical_id.get(client_proxy.cid)
                if client_id is None: continue

                client_update_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                reconstruction_result, original_data = None, None
                
                if self.attack_config.get("gradient_inversion", {}).get("enable", False) and client_id == self.attack_config["gradient_inversion"]["target_client"]:
                    print(f"-> Analyzing Gradient Inversion target: Client {client_id}")
                    reconstruction_result = self._reconstruct_data([client_update_params], self.attack_config["gradient_inversion"])
                    data_path = f"client_data/client_{client_id}_data.pkl"
                    if os.path.exists(data_path):
                        with open(data_path, "rb") as f: saved_data = pickle.load(f)
                        original_data, original_labels = saved_data['data'], saved_data['label']
                        os.remove(data_path)
                    
                    if reconstruction_result is not None and original_data is not None:
                        reconstructed_images, predicted_labels = reconstruction_result
                        evaluate_reconstruction(original_data, reconstructed_images)
                        self._save_reconstruction(reconstructed_images, predicted_labels, client_id, server_round, original_data, original_labels)

                if self.defense_agent.decide_and_defend(client_id, self.global_parameters, client_update_params, reconstruction_result, original_data):
                    accepted_results.append((client_proxy, fit_res))

            if not accepted_results:
                print("--- MyDefense Result: All updates rejected. ---")
                return fl.common.ndarrays_to_parameters(self.global_parameters), {}
            print(f"--- MyDefense Result: Aggregating {len(accepted_results)} of {len(results)} updates. ---")
            aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, accepted_results, failures)
        
        else: # Standard pipeline without MyDefense
            if self.mitigation_config.get("enable", False) and self.mitigation_config.get("defense_type") == "median":
                if not decrypted_results: return None, {}
                print("\n[Mitigation] Applying coordinate-wise median aggregation.")
                all_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in decrypted_results]
                stacked_params = zip(*all_params)
                median_params = [np.median(np.stack(layer_params), axis=0) for layer_params in stacked_params]
                return ndarrays_to_parameters(median_params), {}
            
            gi_params = self.attack_config.get("gradient_inversion", {})
            if gi_params.get("enable", False):
                target_client_id = gi_params.get("target_client", 1)
                target_fit_res, honest_clients_results = None, []
                for client_proxy, fit_res in decrypted_results:
                    logical_id = fit_res.metrics.get("logical_client_id")
                    if logical_id == target_client_id and fit_res.metrics.get("attack") == "gradient_inversion":
                        target_fit_res = fit_res
                    else:
                        honest_clients_results.append((client_proxy, fit_res))
                if target_fit_res:
                    try:
                        gradients = parameters_to_ndarrays(target_fit_res.parameters)
                        reconstruction_result = self._reconstruct_data([gradients], gi_params)
                        if reconstruction_result is not None:
                            reconstructed_images, predicted_labels = reconstruction_result if isinstance(reconstruction_result, tuple) else (reconstruction_result, None)
                            original_data, original_labels = None, None
                            data_path = f"client_data/client_{target_client_id}_data.pkl"
                            if os.path.exists(data_path):
                                with open(data_path, "rb") as f: saved_data = pickle.load(f)
                                original_data, original_labels = saved_data['data'], saved_data['label']
                                os.remove(data_path)
                            if original_data is not None:
                                evaluate_reconstruction(original_data, reconstructed_images)
                            self._save_reconstruction(reconstructed_images, predicted_labels, target_client_id, server_round, original_data, original_labels)
                    except Exception as e:
                        print(f"[Attack Failed] Gradient Inversion error: {str(e)}")
                if not honest_clients_results: return None, {}
                aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, honest_clients_results, failures)
            else:
                aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, decrypted_results, failures)

        if aggregated_params:
            self.global_parameters = fl.common.parameters_to_ndarrays(aggregated_params)
        return aggregated_params, aggregated_metrics
    
    def _reconstruct_data(self, gradients_list: List[List[np.ndarray]], attack_params: Dict):
        """Wrapper to call the refactored reconstruction function."""
        return reconstruct_data(gradients_list, attack_params, self.client_config)

    def _save_reconstruction(self, data: np.ndarray, predicted_labels: Optional[torch.Tensor], client_id: int, round_num: int, original_data: Optional[np.ndarray] = None, original_labels: Optional[np.ndarray] = None):
        """Wrapper to call the refactored save function."""
        save_reconstruction(data, predicted_labels, client_id, round_num, self.reconstruction_dir, original_data, original_labels)
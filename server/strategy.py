import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import os
import torch
import pickle
from collections import OrderedDict

# Import all project modules
from model import SensorMLP, CastingCNN, CifarCNN
from server.helpers import get_parameters, evaluate_reconstruction, evaluate_reconstruction_numerical
from server.reconstruction import reconstruct_data, save_reconstruction
from defense.mydefense import MyDefenseAgent
from crypto_utils import encrypt_params, decrypt_params, chaotic_map_deobfuscate
from utils_data.sensor_data_util import load_and_preprocess_data

class SecureFedAvg(FedAvg):
    def __init__(self, config: dict, server_holdout_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.attack_config = config.get("attacks", {})
        self.client_config = config.get("clients", {})
        self.mitigation_config = config.get("mitigations", {})
        self.test_loader = test_loader
        self.reconstruction_dir = "reconstructions"
        os.makedirs(self.reconstruction_dir, exist_ok=True)
        self.global_parameters = None
        self.gradient_history = {}
        self.cid_to_logical_id = {}
        
        if self.mitigation_config.get("enable", False) and self.mitigation_config.get("defense_type") == "mydefense":
            data_type = self.config.get("data", {}).get("type", "image")
            if data_type == "casting":
                model_class = CastingCNN
                model_args = {"num_classes": self.config["data"]["num_classes"]}
            elif data_type == "cifar10":
                model_class = CifarCNN
                model_args = {"num_classes": self.config["data"]["num_classes"]}
            else: # sensor
                X, y, _ = load_and_preprocess_data(self.config["data"]["path"])
                model_class = SensorMLP
                model_args = {"input_features": X.shape[1], "num_classes": len(np.unique(y))}
            
            self.defense_agent = MyDefenseAgent(config, server_holdout_loader, model_class, model_args)
        else:
            self.defense_agent = None

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        data_type = self.config.get("data", {}).get("type", "image")
        if data_type == "casting":
            model = CastingCNN(num_classes=self.config["data"]["num_classes"])
        elif data_type == "cifar10":
            model = CifarCNN(num_classes=self.config["data"]["num_classes"])
        elif data_type == "sensor":
            X, y, _ = load_and_preprocess_data(self.config["data"]["path"])
            model = SensorMLP(input_features=X.shape[1], num_classes=len(np.unique(y)))
        else:
            raise ValueError(f"Unknown data type for initialization: {data_type}")
            
        self.global_parameters = get_parameters(model)
        return fl.common.ndarrays_to_parameters(self.global_parameters)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        config = {}
        if self.defense_agent:
            client_configs = []
            num_clients_to_sample = int(self.min_fit_clients)
            sampled_clients = client_manager.sample(num_clients=num_clients_to_sample, min_num_clients=num_clients_to_sample)
            for client in sampled_clients:
                logical_id = self.cid_to_logical_id.get(client.cid)
                client_config = {}
                if logical_id and self.defense_agent.trigger_chaotic_obfuscation_for_client.get(logical_id, False):
                    print(f"\n[Mitigation] Instructing Client {logical_id} to apply Chaotic Obfuscation.")
                    client_config["apply_chaotic_obfuscation"] = True
                    client_config.update(self.mitigation_config.get("mydefense_params", {}))
                    self.defense_agent.trigger_chaotic_obfuscation_for_client[logical_id] = False
                client_configs.append((client, fl.common.FitIns(parameters, client_config)))
            return client_configs

        if self.mitigation_config.get("enable", False):
            defense_type = self.mitigation_config.get("defense_type")
            config["defense_type"] = defense_type
            if defense_type == "encryption":
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
        for client_proxy, fit_res in results:
            if "logical_client_id" in fit_res.metrics:
                self.cid_to_logical_id[client_proxy.cid] = fit_res.metrics["logical_client_id"]

        if self.defense_agent:
            aggregated_params, aggregated_metrics = self.defense_agent_aggregation(server_round, results, failures)
        else:
            processed_results = self.process_results(results)
            aggregated_params, aggregated_metrics = self.standard_aggregation(server_round, processed_results, failures)
        
        # 🔎 New: collect and log uplink/downlink usage
        total_up = sum((fit_res.metrics or {}).get("bytes_up", 0) for _, fit_res in results)
        total_down = sum((fit_res.metrics or {}).get("bytes_down", 0) for _, fit_res in results)

        print(f"[Round {server_round}] Total uplink: {total_up} bytes | Total downlink: {total_down} bytes")

        # Ensure aggregated_metrics is a dict, then attach totals
        if aggregated_metrics is None:
            aggregated_metrics = {}
        aggregated_metrics.update({
            "total_bytes_up": int(total_up),
            "total_bytes_down": int(total_down),
        })

        if aggregated_params:
            self.global_parameters = fl.common.parameters_to_ndarrays(aggregated_params)
        return aggregated_params, aggregated_metrics

    def process_results(self, results: List[Tuple[Any, Any]]):
        processed_results = []
        defense_type = self.mitigation_config.get("defense_type")
        for client_proxy, fit_res in results:
            raw_params = parameters_to_ndarrays(fit_res.parameters)
            if defense_type == "encryption" and len(raw_params) == 1 and raw_params[0].dtype == np.uint8:
                try:
                    decrypted_params = decrypt_params(raw_params[0].tobytes())
                    fit_res.parameters = ndarrays_to_parameters(decrypted_params)
                except Exception as e:
                    print(f"Could not decrypt update from {client_proxy.cid}: {e}")
            elif fit_res.metrics.get("was_chaotically_obfuscated"):
                chaos_params = self.mitigation_config.get("mydefense_params", {})
                deobfuscated_params = chaotic_map_deobfuscate(raw_params, key=chaos_params.get("chaos_key"), seed=chaos_params.get("chaos_seed"))
                fit_res.parameters = ndarrays_to_parameters(deobfuscated_params)
            processed_results.append((client_proxy, fit_res))
        return processed_results
    
    def defense_agent_aggregation(self, server_round, results, failures):
        print("\n--- MyDefense Agent Analyzing Round ---")
        accepted_results = []
        for client_proxy, fit_res in results:
            client_id = self.cid_to_logical_id.get(client_proxy.cid)
            if client_id is None: continue
            
            raw_update_params = parameters_to_ndarrays(fit_res.parameters)
            clean_update_params = raw_update_params
            
            if fit_res.metrics.get("was_chaotically_obfuscated"):
                print(f"[Mitigation] De-obfuscating update from client {client_proxy.cid} for utility analysis.")
                chaos_params = self.mitigation_config.get("mydefense_params", {})
                clean_update_params = chaotic_map_deobfuscate(
                    raw_update_params, key=chaos_params.get("chaos_key"), seed=chaos_params.get("chaos_seed")
                )

            reconstruction_result, original_data = None, None
            
            if (self.attack_config.get("gradient_inversion", {}).get("enable", False) and 
                client_id == self.attack_config["gradient_inversion"]["target_client"] and
                fit_res.metrics.get("attack") == "gradient_inversion"):
                
                data_type = fit_res.metrics.get("data_type", "cifar10")
                print(f"-> Analyzing GI target: Client {client_id} ({data_type} data)")
                
                reconstruction_result = self._reconstruct_data([raw_update_params], self.attack_config["gradient_inversion"], data_type)
                
                data_path = f"client_data/client_{client_id}_{data_type}_data.pkl"
                if os.path.exists(data_path):
                    with open(data_path, "rb") as f: saved_data = pickle.load(f)
                    original_data, original_labels = saved_data['data'], saved_data['label']
                    os.remove(data_path)
                
                if reconstruction_result is not None and original_data is not None:
                    reconstructed_images, predicted_labels = reconstruction_result
                    evaluate_reconstruction(original_data, reconstructed_images)
                    self._save_reconstruction(reconstructed_images, predicted_labels, client_id, server_round, data_type, original_data, original_labels)

            if self.defense_agent.decide_and_defend(client_id, self.global_parameters, clean_update_params, reconstruction_result, original_data):
                fit_res.parameters = ndarrays_to_parameters(clean_update_params)
                accepted_results.append((client_proxy, fit_res))
        
        if not accepted_results:
            print("--- MyDefense Result: All updates rejected. ---")
            aggregated_params, aggregated_metrics = fl.common.ndarrays_to_parameters(self.global_parameters), {}
        else:
            print(f"--- MyDefense Result: Aggregating {len(accepted_results)} of {len(results)} updates. ---")
            aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, accepted_results, failures)
        
        return aggregated_params, aggregated_metrics

    def standard_aggregation(self, server_round, results, failures):
        gi_params = self.attack_config.get("gradient_inversion", {})
        if gi_params.get("enable", False):
            target_client_id = gi_params.get("target_client", 1)
            target_fit_res, honest_clients_results = None, []
            for client_proxy, fit_res in results:
                logical_id = self.cid_to_logical_id.get(client_proxy.cid)
                if logical_id == target_client_id and fit_res.metrics.get("attack") == "gradient_inversion":
                    target_fit_res = fit_res
                else:
                    honest_clients_results.append((client_proxy, fit_res))
            if target_fit_res:
                try:
                    gradients = parameters_to_ndarrays(target_fit_res.parameters)
                    data_type = target_fit_res.metrics.get("data_type", "image")
                    reconstruction_result = self._reconstruct_data([gradients], gi_params, data_type)
                    if reconstruction_result is not None:
                        reconstructed_data, predicted_labels = reconstruction_result if isinstance(reconstruction_result, tuple) else (reconstruction_result, None)
                        original_data, original_labels = None, None
                        data_path = f"client_data/client_{target_client_id}_{data_type}_data.pkl"
                        if os.path.exists(data_path):
                            with open(data_path, "rb") as f: saved_data = pickle.load(f)
                            original_data, original_labels = saved_data['data'], saved_data['label']
                            os.remove(data_path)
                        if original_data is not None:
                            if data_type == "image" or data_type == "casting" or data_type == "cifar10":
                                evaluate_reconstruction(original_data, reconstructed_data)
                            else:
                                evaluate_reconstruction_numerical(original_data, reconstructed_data)
                        self._save_reconstruction(reconstructed_data, predicted_labels, target_client_id, server_round, data_type, original_data, original_labels)
                except Exception as e:
                    print(f"[Attack Failed] Gradient Inversion error: {str(e)}")
            if not honest_clients_results: return None, {}
            return super().aggregate_fit(server_round, honest_clients_results, failures)
        
        if self.mitigation_config.get("enable", False) and self.mitigation_config.get("defense_type") == "median":
            if not results: return None, {}
            all_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            median_params = [np.median(np.stack(layer_params), axis=0) for layer_params in zip(*all_params)]
            return ndarrays_to_parameters(median_params), {}
        return super().aggregate_fit(server_round, results, failures)
    
    def _reconstruct_data(self, gradients_list: List[List[np.ndarray]], attack_params: Dict, data_type: str):
        """Wrapper to call the refactored reconstruction function."""
        return reconstruct_data(gradients_list, attack_params, self.client_config, data_type, self.config)

    def _save_reconstruction(self, data: np.ndarray, predicted_labels: Optional[torch.Tensor], client_id: int, round_num: int, data_type: str, original_data: Optional[np.ndarray] = None, original_labels: Optional[np.ndarray] = None):
        """Wrapper to call the refactored save function."""
        save_reconstruction(data, predicted_labels, client_id, round_num, self.reconstruction_dir, data_type, original_data, original_labels)
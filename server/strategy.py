# server/strategy.py

import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import os
import torch
import pickle
import time

from model import CifarCNN, MedMNIST_CNN
from server.helpers import get_parameters
from server.reconstruction import reconstruct_data, save_reconstruction
from defense.mydefense import MyDefenseAgent
from crypto_utils import decrypt_params, chaotic_map_deobfuscate

class SecureFedAvg(FedAvg):
    def __init__(self, config: dict, server_holdout_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.attack_config = config.get("attacks", {})
        self.client_config = config.get("clients", {})
        self.mitigation_config = config.get("mitigations", {})
        self.reconstruction_dir = "reconstructions"
        os.makedirs(self.reconstruction_dir, exist_ok=True)
        self.global_parameters = None
        self.cid_to_logical_id = {}
        
        if self.mitigation_config.get("enable", False) and self.mitigation_config.get("defense_type") == "mydefense":
            data_type = self.config.get("data", {}).get("type", "cifar10")
            num_classes = self.config.get("data", {}).get("num_classes", 10)
            model_class = CifarCNN if data_type == "cifar10" else MedMNIST_CNN
            model_args = {"num_classes": num_classes}
            self.defense_agent = MyDefenseAgent(config, server_holdout_loader, model_class, model_args)
        else:
            self.defense_agent = None

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        data_type = self.config.get("data", {}).get("type", "cifar10")
        num_classes = self.config.get("data", {}).get("num_classes", 10)
        if data_type == "cifar10": model = CifarCNN(num_classes=num_classes)
        elif data_type == "medmnist": model = MedMNIST_CNN(num_classes=num_classes)
        else: raise ValueError(f"Unsupported data type for init: {data_type}")
        self.global_parameters = get_parameters(model)
        return fl.common.ndarrays_to_parameters(self.global_parameters)
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[Any, Any]], failures: List[Any]):
        for client_proxy, fit_res in results:
            if "logical_client_id" in fit_res.metrics:
                self.cid_to_logical_id[client_proxy.cid] = fit_res.metrics["logical_client_id"]

        if self.defense_agent:
            aggregated_params, aggregated_metrics = self.defense_agent_aggregation(server_round, results, failures)
        else:
            aggregated_params, aggregated_metrics = self.standard_aggregation(server_round, results, failures)
        
        if aggregated_metrics:
            recon_time = aggregated_metrics.get("reconstruction_duration_sec")
            agg_time = aggregated_metrics.get("aggregation_duration_sec")
            print("\n--- Server-Side Metrics ---")
            if recon_time is not None: print(f"  Reconstruction Duration: {recon_time:.4f} seconds")
            if agg_time is not None: print(f"  Aggregation Duration: {agg_time:.4f} seconds")
            total_up = sum((fit_res.metrics or {}).get("bytes_up", 0) for _, fit_res in results)
            total_down = sum((fit_res.metrics or {}).get("bytes_down", 0) for _, fit_res in results)
            print(f"  Communication: {total_down} bytes (downlink), {total_up} bytes (uplink)")
            print("---------------------------\n")

        if aggregated_params: self.global_parameters = fl.common.parameters_to_ndarrays(aggregated_params)
        return aggregated_params, aggregated_metrics

    def _process_single_result(self, fit_res: fl.common.FitRes):
        raw_params = parameters_to_ndarrays(fit_res.parameters)
        if self.mitigation_config.get("defense_type") == "encryption" and len(raw_params) == 1 and raw_params[0].dtype == np.uint8:
            try: return decrypt_params(raw_params[0].tobytes())
            except Exception as e:
                print(f"Could not decrypt update: {e}"); return None
        elif fit_res.metrics.get("was_chaotically_obfuscated"):
            chaos_params = self.mitigation_config.get("mydefense_params", {})
            return chaotic_map_deobfuscate(raw_params, key=chaos_params.get("chaos_key", 3.99))
        return raw_params

    def defense_agent_aggregation(self, server_round: int, results: List[Tuple[Any, Any]], failures: List[Any]):
        print("\n--- MyDefense Agent Analyzing Round ---")
        accepted_results_for_aggregation = []
        aggregated_metrics = {}; recon_performed = False
        
        for client_proxy, fit_res in results:
            client_id = self.cid_to_logical_id.get(client_proxy.cid)
            if client_id is None: continue

            reconstruction_result = None
            # MYDEFENSE LOGIC 1: Perform reconstruction on the RAW update.
            if self.attack_config.get("gradient_inversion", {}).get("enable", False) and client_id == self.attack_config["gradient_inversion"]["target_client"]:
                recon_start_time = time.time()
                recon_performed = True
                print(f"-> Analyzing GI target: Client {client_id}")
                raw_params = parameters_to_ndarrays(fit_res.parameters)
                data_type = fit_res.metrics.get("data_type")
                reconstruction_result = self._reconstruct_data(raw_params, self.attack_config["gradient_inversion"], data_type)
                aggregated_metrics["reconstruction_duration_sec"] = time.time() - recon_start_time
                
                original_data, original_labels = None, None
                data_path = f"client_data/client_{client_id}_{data_type}_data.pkl"
                print(data_path)
                if os.path.exists(data_path):
                    with open(data_path, "rb") as f: saved_data = pickle.load(f)
                    original_data, original_labels = saved_data['data'], saved_data['label']
                    os.remove(data_path)

                if reconstruction_result and original_data is not None:
                    reconstructed_data, predicted_labels = reconstruction_result
                    self._save_reconstruction(data=reconstructed_data, predicted_labels=predicted_labels, client_id=client_id, round_num=server_round, data_type=data_type, original_data=original_data, original_labels=original_labels)

            # MYDEFENSE LOGIC 2: Get the CLEAN update for utility checks and aggregation.
            clean_update_params = self._process_single_result(fit_res)
            if clean_update_params is None:
                print(f"  - DECISION: Could not process update from Client {client_id}. REJECTING."); continue

            # MYDEFENSE LOGIC 3: Agent decides, which internally measures leakage and triggers future defenses.
            if self.defense_agent.decide_and_defend(client_id, self.global_parameters, clean_update_params, reconstruction_result, original_data):
                fit_res.parameters = ndarrays_to_parameters(clean_update_params)
                accepted_results_for_aggregation.append((client_proxy, fit_res))
        
        agg_start_time = time.time()
        if not accepted_results_for_aggregation:
            print("--- MyDefense Result: All updates rejected. ---")
            aggregated_params = fl.common.ndarrays_to_parameters(self.global_parameters)
        else:
            print(f"--- MyDefense Result: Aggregating {len(accepted_results_for_aggregation)} of {len(results)} updates. ---")
            aggregated_params, _ = super().aggregate_fit(server_round, accepted_results_for_aggregation, failures)
        
        aggregated_metrics["aggregation_duration_sec"] = time.time() - agg_start_time
        return aggregated_params, aggregated_metrics

    def standard_aggregation(self, server_round: int, results: List[Tuple[Any, Any]], failures: List[Any]):
        aggregated_metrics = {}
        gi_params = self.attack_config.get("gradient_inversion", {})
        
        if gi_params.get("enable", False):
            target_client_id = gi_params.get("target_client", 1)
            target_tuple = next(((p, r) for p, r in results if self.cid_to_logical_id.get(p.cid) == target_client_id), None)
            honest_clients_results = [(p, r) for p, r in results if self.cid_to_logical_id.get(p.cid) != target_client_id]

            if target_tuple:
                target_proxy, target_fit_res = target_tuple
                recon_start_time = time.time()
                try:
                    raw_params = parameters_to_ndarrays(target_fit_res.parameters)
                    data_type = target_fit_res.metrics.get("data_type")
                    reconstruction_result = self._reconstruct_data(raw_params, gi_params, data_type)
                    aggregated_metrics["reconstruction_duration_sec"] = time.time() - recon_start_time
                    
                    if reconstruction_result:
                        reconstructed_data, predicted_labels = reconstruction_result
                        
                        # ✅ FIX: Added the missing logic to load the original data file.
                        original_data, original_labels = None, None
                        data_path = f"client_data/client_{target_client_id}_{data_type}_data.pkl"
                        if os.path.exists(data_path):
                            with open(data_path, "rb") as f: saved_data = pickle.load(f)
                            original_data, original_labels = saved_data['data'], saved_data['label']
                            os.remove(data_path)

                        self._save_reconstruction(
                            data=reconstructed_data, 
                            predicted_labels=predicted_labels, 
                            client_id=target_client_id, 
                            round_num=server_round, 
                            data_type=data_type, 
                            original_data=original_data, 
                            original_labels=original_labels
                        )
                except Exception as e: print(f"[Attack Failed] Gradient Inversion error: {e}")

            if not honest_clients_results: return None, aggregated_metrics
            
            clean_honest_results = []
            for p, r in honest_clients_results:
                clean_params = self._process_single_result(r)
                if clean_params is not None:
                    r.parameters = ndarrays_to_parameters(clean_params)
                    clean_honest_results.append((p,r))

            agg_start_time = time.time()
            aggregated_params, metrics_from_super = super().aggregate_fit(server_round, clean_honest_results, failures)
            aggregated_metrics["aggregation_duration_sec"] = time.time() - agg_start_time
            if metrics_from_super: aggregated_metrics.update(metrics_from_super)
            return aggregated_params, aggregated_metrics
        
        # Fallback for when GI attack is not enabled
        clean_results = []
        for p, r in results:
            clean_params = self._process_single_result(r)
            if clean_params is not None:
                r.parameters = ndarrays_to_parameters(clean_params)
                clean_results.append((p,r))
                
        agg_start_time = time.time()
        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, clean_results, failures)
        if aggregated_metrics is None: aggregated_metrics = {}
        aggregated_metrics["aggregation_duration_sec"] = time.time() - agg_start_time
        return aggregated_params, aggregated_metrics
    
    def _reconstruct_data(self, gradients, attack_params, data_type):
        return reconstruct_data([gradients], attack_params, self.client_config, data_type, self.config,self.global_parameters)

    def _save_reconstruction(self, data, predicted_labels, client_id, round_num, data_type, original_data=None, original_labels=None):
        save_reconstruction(data, predicted_labels, client_id, round_num, self.reconstruction_dir, data_type, original_data, original_labels)
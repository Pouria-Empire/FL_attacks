import flwr as fl
from flwr.common import Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Any, Optional
import yaml
import numpy as np
import os
import torch
import pickle
import torchvision

# Import metrics
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
from scipy.spatial import distance

# Import custom project modules
from model import SimpleNN
from chest_data_util import load_data, ChestXRayDataset, FINDINGS
from attacks.data_poisoning import PoisonedDataset
from attacks.gradient_inversion import dlg_attack, mdlg_attack
from attacks.gradinversion_plus import gradinversion_group_attack
from attacks.ggl_attack import ggl_attack
from attacks.temporal_attack import temporal_attack
from crypto_utils import encrypt_params, decrypt_params

# --- HELPER FUNCTIONS ---
def load_config() -> Dict[str, Any]:
    with open("config.yml", "r") as f: return yaml.safe_load(f)

def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    
def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def test_and_log_misclassifications(
    model: torch.nn.Module, 
    test_loader: torch.utils.data.DataLoader, 
    is_backdoor_test: bool, 
    target_label_idx: int
) -> Tuple[float, float]:
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = torch.nn.BCEWithLogitsLoss()
    log_file = "backdoor_misclassifications.log"
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            predicted = torch.sigmoid(outputs) > 0.5
            total += labels.size(0)
            correct_predictions = (predicted == labels.byte()).all(dim=1)
            correct += correct_predictions.sum().item()
            if is_backdoor_test:
                misclassified_indices = (~correct_predictions).nonzero(as_tuple=False).squeeze()
                try:
                    df = test_loader.dataset.dataset.df
                    for idx in misclassified_indices.tolist():
                        original_idx = test_loader.dataset.indices[idx]
                        filename = df.iloc[original_idx]['Image Index']
                        true_labels = df.iloc[original_idx]['Finding Labels']
                        with open(log_file, "a") as f:
                            f.write(f"MISCLASSIFICATION on {filename}:\n  - True: {true_labels}, Predicted: {FINDINGS[target_label_idx]}\n")
                except Exception:
                    pass
    return total_loss / total if total > 0 else 0, correct / total if total > 0 else 0

def safe_metrics_aggregation(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    aggregated = {}
    if any("accuracy" in m for _, m in metrics):
        aggregated["accuracy"] = np.mean([m["accuracy"] for _, m in metrics if "accuracy" in m])
    if any("backdoor_asr" in m for _, m in metrics):
        aggregated["backdoor_asr"] = np.mean([m["backdoor_asr"] for _, m in metrics if "backdoor_asr" in m])
    print("\n[Round Metrics]")
    if "accuracy" in aggregated: print(f"  Normal Accuracy: {aggregated['accuracy']*100:.2f}%")
    if "backdoor_asr" in aggregated: print(f"  Attack Success Rate (ASR): {aggregated['backdoor_asr']*100:.2f}%")
    fit_durations = [m["fit_duration"] for _, m in metrics if "fit_duration" in m]
    if fit_durations:
        print(f"Avg. Client Fit Time: {np.mean(fit_durations):.4f} seconds")
    return aggregated

def evaluate_reconstruction(original_batch: np.ndarray, recon_batch: np.ndarray):
    psnr_scores, ssim_scores, mse_scores, cos_sims = [], [], [], []
    noise_batch, ssim_noise_scores = np.random.rand(*original_batch.shape), []
    for i in range(original_batch.shape[0]):
        original_img, recon_img, noise_img = original_batch[i].squeeze(), recon_batch[i].squeeze(), noise_batch[i].squeeze()
        data_range = original_img.max() - original_img.min()
        psnr_scores.append(psnr(original_img, recon_img, data_range=data_range))
        ssim_scores.append(ssim(original_img, recon_img, data_range=data_range))
        mse_scores.append(mse(original_img, recon_img))
        cos_sims.append(1 - distance.cosine(original_img.flatten(), recon_img.flatten()))
        ssim_noise_scores.append(ssim(original_img, noise_img, data_range=data_range))
    avg_psnr, avg_ssim, avg_mse, avg_cos_sim = np.mean(psnr_scores), np.mean(ssim_scores), np.mean(mse_scores), np.mean(cos_sims)
    avg_ssim_noise = np.mean(ssim_noise_scores)
    rdlv = (avg_ssim - avg_ssim_noise) / (1 - avg_ssim_noise) if (1 - avg_ssim_noise) != 0 else 0
    print("\n--- Reconstruction Quality Metrics ---")
    print(f"  MSE (↓ is better):      {avg_mse:.4f}")
    print(f"  PSNR (↑ is better):     {avg_psnr:.2f} dB")
    print(f"  SSIM (↑ is better):     {avg_ssim:.4f}")
    print(f"  Cosine Sim (↑ is better):{avg_cos_sim:.4f}")
    print(f"  RDLV (↑ is better):     {rdlv:.4f}")
    print("------------------------------------")

# --- STRATEGY CLASS WITH ALL FEATURES ---
class SecureFedAvg(FedAvg):
    def __init__(self, config: dict, **kwargs):
        self.config = config
        self.attack_config = config.get("attacks", {})
        self.client_config = config.get("clients", {})
        self.mitigation_config = config.get("mitigations", {})
        self.reconstruction_dir = "reconstructions"
        os.makedirs(self.reconstruction_dir, exist_ok=True)
        self.gradient_history = {}
        super().__init__(**kwargs)

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        model = SimpleNN(num_classes=15)
        return fl.common.ndarrays_to_parameters(get_parameters(model))

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        config = {}
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
                if fit_res.metrics.get("attack") == "gradient_inversion":
                    target_fit_res = fit_res
                else:
                    honest_clients_results.append((client_proxy, fit_res))
            if target_fit_res:
                try:
                    gradients = parameters_to_ndarrays(target_fit_res.parameters)
                    attack_type = gi_params.get("type")
                    if attack_type == "temporal":
                        if target_client_id not in self.gradient_history: self.gradient_history[target_client_id] = []
                        self.gradient_history[target_client_id].append(gradients)
                        history_size = gi_params.get("history_size", 3)
                        if len(self.gradient_history[target_client_id]) >= history_size:
                            reconstruction_result = self._reconstruct_data(self.gradient_history[target_client_id], gi_params)
                            self.gradient_history[target_client_id] = []
                        else:
                            print(f"\n[Attack] Collected gradient {len(self.gradient_history[target_client_id])}/{history_size} for temporal attack.")
                            reconstruction_result = None
                    else:
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
            return super().aggregate_fit(server_round, honest_clients_results, failures)
        
        return super().aggregate_fit(server_round, decrypted_results, failures)

    def _reconstruct_data(self, gradients_list: List[List[np.ndarray]], attack_params: Dict):
        attack_type = attack_params.get("type", "dlg")
        print(f"[Attack] Attempting reconstruction using '{attack_type}' method.")
        if attack_type == "temporal":
            return temporal_attack(gradient_history=gradients_list, lr=attack_params.get("attack_lr"), iterations=attack_params.get("iterations"))
        
        gradients = gradients_list[0] # Other attacks use a single gradient
        if attack_type == "gradinversion_plus":
            return gradinversion_group_attack(gradients=gradients, batch_size=self.client_config.get("batch_size", 8), num_seeds=attack_params.get("num_seeds", 4), lr=attack_params.get("attack_lr"), iterations=attack_params.get("iterations"))
        elif attack_type == "ggl":
            return ggl_attack(gradients=gradients, lr=attack_params.get("attack_lr"), iterations=attack_params.get("iterations"))
        # DLG and mDLG are highly experimental for this dataset
        elif attack_type == "dlg":
            return dlg_attack(gradients=gradients, lr=attack_params.get("attack_lr"), iterations=attack_params.get("iterations"))
        elif attack_type == "mdlg":
            return mdlg_attack(gradients=gradients, lr=attack_params.get("attack_lr"), iterations=attack_params.get("iterations"))
        else:
            print(f"Unknown reconstruction attack type: {attack_type}")
            return None

    def _save_reconstruction(self, data: np.ndarray, predicted_labels: Optional[torch.Tensor], client_id: int, round_num: int, original_data: Optional[np.ndarray] = None, original_labels: Optional[np.ndarray] = None):
        recon_tensor = torch.from_numpy(data)
        if original_data is not None:
            original_tensor = torch.from_numpy(original_data)
            comparison_grid = torch.cat([original_tensor, recon_tensor])
            save_path = f"{self.reconstruction_dir}/comparison_client{client_id}_round{round_num}.png"
            torchvision.utils.save_image(comparison_grid, save_path, nrow=original_tensor.size(0))
            print(f"[Attack] Saved comparison grid to {save_path}")
        else:
            save_path = f"{self.reconstruction_dir}/reconstruction_client{client_id}_round{round_num}.png"
            torchvision.utils.save_image(recon_tensor, save_path)
            print(f"[Attack] Saved reconstruction grid to {save_path}")

def main():
    config = load_config()
    data_config = config["data"]
    _, testset = load_data(data_config["path"], data_config["train_list"], data_config["test_list"])
    testloader = torch.utils.data.DataLoader(testset, batch_size=32)

    def get_evaluate_fn(config: dict):
        attack_config = config.get("attacks", {}).get("data_poisoning", {})
        if os.path.exists("backdoor_misclassifications.log"):
            os.remove("backdoor_misclassifications.log")
        def evaluate(server_round: int, parameters: fl.common.NDArrays, conf: dict):
            model = SimpleNN(num_classes=15)
            set_parameters(model, parameters)
            loss, accuracy = test_and_log_misclassifications(model, testloader, False, 0)
            backdoor_asr = 0.0
            if attack_config.get("enable", False):
                target_idx = attack_config.get("target_label_idx", 7)
                backdoor_test_set = PoisonedDataset(dataset=testset, poison_frac=1.0, target_label_idx=target_idx)
                backdoor_loader = DataLoader(backdoor_test_set, batch_size=32)
                with open("backdoor_misclassifications.log", "a") as f:
                    f.write(f"--- MISCLASSIFICATIONS FOR ROUND {server_round} ---\n")
                _, backdoor_asr = test_and_log_misclassifications(model, backdoor_loader, True, target_idx)
            return loss, {"accuracy": accuracy, "backdoor_asr": backdoor_asr}
        return evaluate

    strategy = SecureFedAvg(
        config=config,
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=config["server"]["min_clients"],
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
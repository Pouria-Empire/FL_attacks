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

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
from scipy.spatial import distance

from model import SimpleNN
from utils import load_data # <-- ONLY load_data is imported from utils
from attacks.data_poisoning import PoisonedDataset
from attacks.gradient_inversion import gradinversion_attack, dlg_attack, mdlg_attack
from attacks.gradinversion_plus import gradinversion_group_attack
from attacks.ggl_attack import ggl_attack

# --- HELPER FUNCTIONS NOW LOCAL TO SERVER ---
def load_config() -> Dict[str, Any]:
    with open("config.yml", "r") as f: return yaml.safe_load(f)

def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict)
    
def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    correct, total_loss = 0, 0.0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data); total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1); correct += pred.eq(target).sum().item()
    return total_loss / len(test_loader.dataset), correct / len(test_loader.dataset)

def safe_metrics_aggregation(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    aggregated = {}
    if any("backdoor_asr" in m for _, m in metrics):
        aggregated["backdoor_asr"] = np.mean([m["backdoor_asr"] for _, m in metrics if "backdoor_asr" in m])
    if any("accuracy" in m for _, m in metrics):
         aggregated["accuracy"] = np.mean([m["accuracy"] for _, m in metrics if "accuracy" in m])
    print("\n[Round Metrics]")
    if "accuracy" in aggregated: print(f"Eval Accuracy: {aggregated['accuracy']*100:.2f}%")
    if "backdoor_asr" in aggregated: print(f"Attack Success Rate (ASR): {aggregated['backdoor_asr']*100:.2f}%")
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

class SecureFedAvg(FedAvg):
    def __init__(self, attack_config: Dict[str, Any], client_config: Dict[str, Any], mitigation_config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.attack_config = attack_config
        self.client_config = client_config
        self.mitigation_config = mitigation_config
        self.reconstruction_dir = "reconstructions"
        os.makedirs(self.reconstruction_dir, exist_ok=True)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager):
        config = {}
        if self.mitigation_config.get("enable", False):
            defense_type = self.mitigation_config.get("defense_type")
            config["defense_type"] = defense_type
            if defense_type == "clipping": config.update(self.mitigation_config.get("clipping_params", {}))
            elif defense_type == "sparsification": config.update(self.mitigation_config.get("sparsification_params", {}))
            elif defense_type == "dp": config.update(self.mitigation_config.get("dp_params", {}))
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in fit_ins_list:
            fit_ins.config.update(config)
        return fit_ins_list

    def aggregate_fit(self, server_round: int, results: List[Tuple[Any, Any]], failures: List[Any]):
        if self.mitigation_config.get("enable", False) and self.mitigation_config.get("defense_type") == "median":
            if not results: return None
            print("\n[Mitigation] Applying coordinate-wise median aggregation.")
            all_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            stacked_params = zip(*all_params)
            median_params = [np.median(np.stack(layer_params), axis=0) for layer_params in stacked_params]
            return ndarrays_to_parameters(median_params), {}

        gi_params = self.attack_config.get("gradient_inversion", {})
        if not gi_params.get("enable", False):
            return super().aggregate_fit(server_round, results, failures)

        target_client_id = gi_params.get("target_client", 1)
        target_fit_res, honest_clients_results = None, []
        for client_proxy, fit_res in results:
            if fit_res.metrics.get("attack") == "gradient_inversion":
                target_fit_res = fit_res
            else:
                honest_clients_results.append((client_proxy, fit_res))
        
        if target_fit_res:
            try:
                gradients = parameters_to_ndarrays(target_fit_res.parameters)
                reconstruction_result = self._reconstruct_data(gradients, gi_params)
                if reconstruction_result is not None:
                    if isinstance(reconstruction_result, tuple):
                        reconstructed_images, predicted_labels = reconstruction_result
                    else:
                        reconstructed_images, predicted_labels = reconstruction_result, None
                    original_data, original_labels = None, None
                    data_path = f"client_data/client_{target_client_id}_data.pkl"
                    if os.path.exists(data_path):
                        with open(data_path, "rb") as f:
                            saved_data = pickle.load(f)
                        original_data, original_labels = saved_data['data'], saved_data['label']
                        os.remove(data_path)
                    if original_data is not None:
                        evaluate_reconstruction(original_data, reconstructed_images)
                    self._save_reconstruction(reconstructed_images, predicted_labels, target_client_id, server_round, original_data, original_labels)
            except Exception as e:
                print(f"[Attack Failed] Gradient Inversion error: {str(e)}")
        
        if not honest_clients_results: return None
        return super().aggregate_fit(server_round, honest_clients_results, failures)

    def _reconstruct_data(self, gradients: List[np.ndarray], attack_params: Dict):
        attack_type = attack_params.get("type", "dlg")
        print(f"[Attack] Attempting reconstruction using '{attack_type}' method.")
        if attack_type == "gradinversion_plus":
            return gradinversion_group_attack(gradients=gradients, batch_size=self.client_config.get("batch_size", 8), num_seeds=attack_params.get("num_seeds", 4), lr=attack_params.get("attack_lr"), iterations=attack_params.get("iterations"), reg_tv=attack_params.get("reg_tv", 1e-4), reg_l2=attack_params.get("reg_l2", 1e-5), reg_group=attack_params.get("reg_group", 0.005))
        elif attack_type == "gradinversion":
            return gradinversion_attack(gradients=gradients, batch_size=self.client_config.get("batch_size", 8), lr=attack_params.get("attack_lr"), iterations=attack_params.get("iterations"))
        elif attack_type == "ggl":
            return ggl_attack(gradients=gradients, lr=attack_params.get("attack_lr"), iterations=attack_params.get("iterations"))
        elif attack_type == "dlg":
            return dlg_attack(gradients=gradients, lr=attack_params.get("attack_lr"), iterations=attack_params.get("iterations"))
        elif attack_type == "mdlg":
            return mdlg_attack(gradients=gradients, lr=attack_params.get("attack_lr"), iterations=attack_params.get("iterations"))
        else:
            print(f"Unknown reconstruction attack type: {attack_type}")
            return None

    def _save_reconstruction(self, data: np.ndarray, predicted_labels: Optional[torch.Tensor], client_id: int, round_num: int, original_data: Optional[np.ndarray] = None, original_labels: Optional[np.ndarray] = None):
        recon_tensor = torch.from_numpy(data)
        if original_data is not None and original_labels is not None:
            original_tensor, original_labels_tensor = torch.from_numpy(original_data), torch.from_numpy(original_labels)
            if predicted_labels is not None:
                sort_indices = torch.argsort(original_labels_tensor)
                original_tensor_sorted = original_tensor[sort_indices]
                comparison_grid = torch.cat([original_tensor_sorted, recon_tensor])
            else:
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
    _, test_data = load_data(config["data"]["path"])

    def server_evaluate_backdoor(server_round: int, parameters: List[np.ndarray], conf: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        attack_config = config.get("attacks", {}).get("data_poisoning", {})
        if not attack_config.get("enable"): return None
        model = SimpleNN(); set_parameters(model, parameters)
        backdoor_test_set = PoisonedDataset(dataset=test_data, poison_frac=1.0, target_label=attack_config.get("target_label", 0))
        backdoor_loader = torch.utils.data.DataLoader(backdoor_test_set, batch_size=64)
        loss, accuracy = test(model, backdoor_loader)
        return loss, {"backdoor_asr": accuracy}

    strategy = SecureFedAvg(
        attack_config=config.get("attacks", {}),
        client_config=config.get("clients", {}),
        mitigation_config=config.get("mitigations", {}),
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=config["server"]["min_clients"],
        min_evaluate_clients=config["server"]["min_clients"],
        min_available_clients=config["server"]["min_clients"],
        evaluate_metrics_aggregation_fn=safe_metrics_aggregation,
        fit_metrics_aggregation_fn=safe_metrics_aggregation,
        evaluate_fn=server_evaluate_backdoor, 
    )
    
    fl.server.start_server(
        server_address=config["server"]["address"],
        config=fl.server.ServerConfig(num_rounds=config["server"]["rounds"]),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
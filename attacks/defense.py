from flwr.server.strategy import FedAvg
import numpy as np

class SecureFedAvg(FedAvg):
    """Defense strategy with gradient clipping and noise"""
    
    def __init__(self, clip_threshold=1.0, noise_scale=0.01, **kwargs):
        super().__init__(**kwargs)
        self.clip_threshold = clip_threshold
        self.noise_scale = noise_scale
        
    def aggregate_fit(self, server_round, results, failures):
        # Apply defenses before aggregation
        secured_results = []
        for client, parameters, num_examples, metrics in results:
            # Gradient clipping
            clipped_params = [np.clip(p, -self.clip_threshold, self.clip_threshold) 
                            for p in parameters]
            # Add differential privacy noise
            noisy_params = [p + np.random.normal(0, self.noise_scale, p.shape) 
                          for p in clipped_params]
            secured_results.append((client, noisy_params, num_examples, metrics))
        
        return super().aggregate_fit(server_round, secured_results, failures)
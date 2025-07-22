import numpy as np
from typing import List

def gradient_clipping(params: List[np.ndarray], clipping_norm: float) -> List[np.ndarray]:
    """Clips the parameters' L2 norm to a maximum value."""
    l2_norm = np.sqrt(sum(np.sum(p * p) for p in params))
    scaling_factor = min(1.0, clipping_norm / (l2_norm + 1e-6))
    return [scaling_factor * p for p in params]

def gradient_sparsification(params: List[np.ndarray], sparsity: float) -> List[np.ndarray]:
    """Keeps the top `sparsity` percentage of gradients and sets the rest to zero."""
    sparsified_params = []
    for p in params:
        if p.size == 0:
            sparsified_params.append(p)
            continue
        threshold = np.quantile(np.abs(p), 1 - sparsity)
        sparsified_p = np.where(np.abs(p) >= threshold, p, 0)
        sparsified_params.append(sparsified_p)
    return sparsified_params

def add_differential_privacy(params: List[np.ndarray], clipping_norm: float, noise_multiplier: float) -> List[np.ndarray]:
    """Clips and adds Gaussian noise for Differential Privacy."""
    clipped_params = gradient_clipping(params, clipping_norm)
    noisy_params = []
    for p in clipped_params:
        noise = np.random.normal(0, clipping_norm * noise_multiplier, p.shape)
        noisy_params.append(p + noise.astype(p.dtype))
    return noisy_params
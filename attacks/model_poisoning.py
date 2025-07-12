import numpy as np
from typing import List
from flwr.common import NDArrays

def add_noise(parameters: NDArrays, noise_scale=0.5) -> NDArrays:
    """Add Gaussian noise to model parameters"""
    return [param + noise_scale * np.random.randn(*param.shape) for param in parameters]

def sign_flip(parameters: NDArrays) -> NDArrays:
    """Flip signs of model parameters"""
    return [-param for param in parameters]

def scaling_attack(parameters: NDArrays, scale_factor=-1.0) -> NDArrays:
    """Scale model parameters by negative factor"""
    return [scale_factor * param for param in parameters]
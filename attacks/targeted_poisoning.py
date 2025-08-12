import numpy as np
import torch

def find_high_intensity_regions(image: np.ndarray, threshold_percentile: int = 90) -> np.ndarray:
    """
    Finds a binary mask of the most intense pixels in an image.
    These regions are likely to be the most important features.
    """
    threshold = np.percentile(image, threshold_percentile)
    # Create a mask where high-intensity pixels are 1, others are 0
    mask = (image > threshold).astype(float)
    return mask

def create_targeted_poisoning_update(
    original_update: List[np.ndarray],
    reconstructed_image: np.ndarray,
    poison_strength: float = 2.0
) -> List[np.ndarray]:
    """
    Crafts a model poisoning update that targets the weights connected
    to the most important parts of the reconstructed image.
    """
    print("  - Crafting targeted model poisoning update...")
    
    # 1. Find the most important regions of the reconstructed image
    # We use a high percentile to isolate the most prominent features
    importance_mask = find_high_intensity_regions(reconstructed_image, 95)
    
    # 2. Corrupt the first layer's weights based on this mask
    # This is a simplified but effective way to target the early feature extractors
    poisoned_update = []
    for i, layer_params in enumerate(original_update):
        if i == 0: # Target the first convolutional layer's weights
            # Flatten the importance mask to match the weights
            flat_mask = torch.tensor(importance_mask).flatten().numpy()
            
            # Create a malicious noise vector that is shaped by the importance mask
            # This ensures the noise is strongest where the image features are most important
            noise = np.random.randn(*layer_params.shape) * poison_strength
            targeted_noise = noise * flat_mask[:noise.size].reshape(noise.shape) # Apply mask
            
            # Add the targeted noise to the original update
            poisoned_layer = layer_params - targeted_noise.astype(layer_params.dtype)
            poisoned_update.append(poisoned_layer)
        else:
            # Keep other layers the same
            poisoned_update.append(layer_params)
            
    print("  - Targeted update crafted successfully.")
    return poisoned_update
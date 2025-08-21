import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from collections import OrderedDict

# Import the model this attack is designed for
from model import MultiModalNet

# --- 1. Define the Multi-Modal Generator ---
class MultiModalGenerator(nn.Module):
    """A Conditional GAN generator for 128x128 images, conditioned on sensor data."""
    def __init__(self, latent_dim=100, num_sensor_features=19, channels=1):
        super(MultiModalGenerator, self).__init__()
        # The input to the generator is the latent vector plus the sensor features
        input_dim = latent_dim + num_sensor_features
        
        self.init_size = 128 // 16  # Initial size for upsampling (8x8)
        self.l1 = nn.Sequential(nn.Linear(input_dim, 256 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2), nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(), # Output is in range [-1, 1]
        )

    def forward(self, z, sensor_data):
        # Concatenate the latent vector and the sensor data to form the input
        gen_input = torch.cat((z, sensor_data), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# --- 2. The GGL Attack for Multi-Modal Data ---
def ggl_multimodal_attack(
    gradients: List[np.ndarray],
    num_sensor_features: int,
    num_classes: int,
    lr: float = 0.01,
    iterations: int = 5000,
    latent_dim: int = 100
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Reconstructs both an image and sensor data from the gradients of a MultiModalNet.
    """
    generator = MultiModalGenerator(latent_dim, num_sensor_features)
    try:
        # Load the pre-trained multi-modal generator
        state_dict = torch.load("models/multimodal_generator.pth", map_location=torch.device('cpu'))
        # Handle models saved with DataParallel
        if next(iter(state_dict)).startswith('module.'):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            generator.load_state_dict(new_state_dict)
        else:
            generator.load_state_dict(state_dict)
    except FileNotFoundError:
        print("🔴 Multi-Modal Generator model not found. Please run the training script first.")
        return None
    generator.eval()

    original_dy_dx = [torch.from_numpy(g).float() for g in gradients]
    
    # We will optimize for the latent vector, the SENSOR DATA, and the label
    dummy_latent = torch.randn(1, latent_dim, requires_grad=True)
    dummy_sensor = torch.randn(1, num_sensor_features, requires_grad=True)
    dummy_logits = torch.randn((1, num_classes), requires_grad=True)
    
    # The dummy model MUST be an exact replica of the client's MultiModalNet
    dummy_model = MultiModalNet(num_sensor_features=num_sensor_features, num_classes=num_classes)

    optimizer = torch.optim.Adam([dummy_latent, dummy_sensor, dummy_logits], lr=lr)

    print("[Attack] Starting multi-modal GGL attack...")
    for it in range(iterations):
        optimizer.zero_grad()
        # The generator now needs the dummy sensor data to create an image
        dummy_image = generator(dummy_latent, dummy_sensor)
        dummy_image = (dummy_image + 1) / 2 # Rescale from Tanh's [-1, 1] to [0, 1]

        # The dummy model needs both dummy inputs
        dummy_pred = dummy_model(dummy_image, dummy_sensor)
        
        loss_cls = F.cross_entropy(dummy_pred, dummy_logits.softmax(dim=-1))
        dy_dx = torch.autograd.grad(loss_cls, list(dummy_model.parameters()), create_graph=True)
        
        grad_loss = sum(((gx - gy) ** 2).sum() for gx, gy in zip(original_dy_dx, dy_dx))
        
        grad_loss.backward()
        optimizer.step()

        if it % 500 == 0:
            print(f"Iteration {it}/{iterations}, Grad Loss: {grad_loss.item():.4f}")

    # Generate the final reconstructed image using the optimized latent and sensor vectors
    final_image = generator(dummy_latent, dummy_sensor)
    final_image = (final_image + 1) / 2
    
    predicted_label = torch.argmax(dummy_logits, dim=-1).item()
    
    # Return all three reconstructed components
    return final_image.detach().numpy(), dummy_sensor.detach().numpy(), predicted_label
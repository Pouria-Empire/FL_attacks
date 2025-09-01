# train_medmnist_generator.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm

import medmnist
from medmnist import INFO

# --- 1. GAN Model Definitions for MedMNIST (28x28 grayscale) ---

class Generator(nn.Module):
    """A DCGAN-style generator for 28x28x1 MedMNIST images."""
    def __init__(self, latent_dim=100, channels=1, img_size=28):
        super(Generator, self).__init__()
        self.init_size = img_size // 4  # Initial size (7x7)
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2), # 7 -> 14
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), # 14 -> 28
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),  # Output is in range [-1, 1]
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    """A DCGAN-style discriminator for 28x28x1 MedMNIST images."""
    def __init__(self, channels=1, img_size=28):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False), # 28 -> 14
            *discriminator_block(16, 32),              # 14 -> 7
            *discriminator_block(32, 64),              # 7 -> 4 (rounding down)
            *discriminator_block(64, 128),             # 4 -> 2
        )

        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# --- 2. MAIN TRAINING SCRIPT ---
def main():
    # --- Configuration ---
    epochs = 100 # GANs need many epochs for good quality
    lr = 0.0002
    latent_dim = 100
    batch_size = 128
    dataset_name = "pneumoniamnist"
    data_path = "./data/medmnist"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Normalize data to [-1, 1] to match the generator's Tanh output
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    train_dataset = DataClass(split='train', transform=transform, download=True, root=data_path)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # --- Models and Optimizers ---
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()

    print(f"Starting MedMNIST ({dataset_name}) Generator Training...")
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            
            # --- Train Discriminator ---
            imgs = imgs.to(device)
            valid = torch.ones(imgs.size(0), 1, device=device, requires_grad=False)
            fake = torch.zeros(imgs.size(0), 1, device=device, requires_grad=False)
            
            optimizer_d.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim, device=device)
            gen_imgs = generator(z)
            
            d_loss = (adversarial_loss(discriminator(imgs), valid) + adversarial_loss(discriminator(gen_imgs.detach()), fake)) / 2
            d_loss.backward()
            optimizer_d.step()
            
            # --- Train Generator ---
            optimizer_g.zero_grad()
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_g.step()
            
        print(f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    # Save the trained generator
    os.makedirs("models", exist_ok=True)
    save_path = "models/medmnist_generator.pth"
    torch.save(generator.state_dict(), save_path)
    print(f"\n✅ MedMNIST Generator training complete. Model saved to '{save_path}'")

if __name__ == "__main__":
    main()
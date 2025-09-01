import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm

# --- 1. MODEL DEFINITIONS (DCGAN for CIFAR-10) ---
# This architecture is specifically designed for 32x32x3 color images.

class Generator(nn.Module):
    """A DCGAN-style generator for 32x32x3 CIFAR-10 images."""
    def __init__(self, latent_dim=100, channels=3):
        super(Generator, self).__init__()
        self.init_size = 32 // 4  # Initial size for upsampling (8x8)
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2), # 8 -> 16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), # 16 -> 32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(), # Output is in range [-1, 1]
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    """A DCGAN-style discriminator for 32x32x3 CIFAR-10 images."""
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False), # 32 -> 16
            *discriminator_block(16, 32),              # 16 -> 8
            *discriminator_block(32, 64),              # 8 -> 4
            *discriminator_block(64, 128),             # 4 -> 2
        )

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# --- 2. MAIN TRAINING SCRIPT ---
def main():
    # --- Configuration ---
    epochs = 50 # GANs need more epochs for quality
    lr = 0.0002
    latent_dim = 100
    batch_size = 64
    data_path = "./data/cifar10" # Folder to store the downloaded data
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Preparing CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalize to [-1, 1] for Tanh
    ])
    
    train_set = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"Dataset prepared with {len(train_set)} images.")
    
    # --- Models and Optimizers ---
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()

    print("Starting CIFAR-10 Generator Training...")
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            imgs = imgs.to(device)
            valid = torch.ones(imgs.size(0), 1, device=device, requires_grad=False)
            fake = torch.zeros(imgs.size(0), 1, device=device, requires_grad=False)
            
            # --- Train Discriminator ---
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
    save_path = "models/cifar_generator.pth"
    torch.save(generator.state_dict(), save_path)
    print(f"\n✅ CIFAR-10 Generator training complete. Model saved to '{save_path}'")

if __name__ == "__main__":
    main()
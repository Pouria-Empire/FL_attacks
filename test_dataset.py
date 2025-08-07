import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import time
import pandas as pd
from PIL import Image
import glob

# --- 1. MODEL DEFINITIONS (DCGAN for 128x128) ---
class StrongGenerator(nn.Module):
    """A DCGAN-style generator for 128x128 grayscale images."""
    def __init__(self, latent_dim=100, channels=1):
        super(StrongGenerator, self).__init__()
        self.init_size = 128 // 16  # 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 256 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2), nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        return self.conv_blocks(out)

class StrongDiscriminator(nn.Module):
    """A DCGAN-style discriminator for 128x128 grayscale images."""
    def __init__(self, channels=1):
        super(StrongDiscriminator, self).__init__()
        def block(in_feat, out_feat, bn=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            if bn: layers.append(nn.BatchNorm2d(out_feat, 0.8))
            return layers
        self.model = nn.Sequential(
            *block(channels, 16, bn=False), # 128 -> 64
            *block(16, 32),  # 64 -> 32
            *block(32, 64),  # 32 -> 16
            *block(64, 128), # 16 -> 8
            *block(128, 256) # 8 -> 4
        )
        ds_size = 128 // 2 ** 5
        self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, 1), nn.Sigmoid())
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        return self.adv_layer(out)

# --- 2. DATA UTILITIES ---
class CastingDataset(Dataset):
    """Custom Dataset that loads images from the casting dataset structure."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        # Find all images in the subdirectories ('ok_front', 'def_front')
        for class_folder in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_folder)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    self.image_paths.append(os.path.join(class_path, img_file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        # Labels are not needed for GAN training, so we return 0 as a placeholder
        if self.transform:
            image = self.transform(image)
        return image, 0

# --- 3. MAIN TRAINING SCRIPT ---
def main():
    # --- Configuration ---
    epochs = 40 # GANs need more epochs for quality on complex data
    lr = 0.0002
    latent_dim = 100
    batch_size = 64
    data_path = "./data/casting_dataset/train"
    img_size = 128
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Preparing dataset...")
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1] for Tanh
    ])
    
    train_set = CastingDataset(root_dir=data_path, transform=transform)
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"Dataset prepared with {len(train_set)} images.")
    
    # --- Models and Optimizers ---
    generator = StrongGenerator(latent_dim).to(device)
    discriminator = StrongDiscriminator().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()

    print("Starting Strong Generator Training...")
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            imgs = imgs.to(device)
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)
            
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
    save_path = "models/strong_casting_generator.pth"
    torch.save(generator.state_dict(), save_path)
    print(f"\n✅ Strong Casting Generator training complete. Model saved to '{save_path}'")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from torchvision import transforms
from typing import Tuple
import glob
from tqdm import tqdm
import time

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
            *block(channels, 16, bn=False), *block(16, 32),
            *block(32, 64), *block(64, 128), *block(128, 256)
        )
        ds_size = 128 // 2 ** 5
        self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, 1), nn.Sigmoid())
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        return self.adv_layer(out)

# --- 2. DATA UTILITIES ---
class ChestXRayDataset(Dataset):
    """Custom Dataset that loads images from pre-computed paths."""
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.df.iloc[idx]['ImagePath']
        image = Image.open(img_path).convert('L')
        label = 0 # Labels not needed for GAN training
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    # --- Configuration ---
    epochs = 30
    lr = 0.0002
    latent_dim = 100
    batch_size = 64
    data_path = "./data_200"
    train_list_file = "train_val_list.txt"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading (Corrected Logic) ---
    print("Preparing dataset...")
    # Find all image paths in the subdirectories and create a mapping from filename to full path
    all_image_paths = {os.path.basename(p): p for p in glob.glob(os.path.join(data_path, "images", "images", "*.png"))}
    
    # Read the training list (which contains only filenames)
    with open(os.path.join(data_path, train_list_file), 'r') as f:
        # Strip paths from the list, just in case
        train_files = [os.path.basename(line.strip()) for line in f]
    
    # Create a list of full paths for the training images
    train_image_full_paths = [all_image_paths[fname] for fname in train_files if fname in all_image_paths]
    
    if not train_image_full_paths:
        raise ValueError("No matching image files found. Check your paths and train_val_list.txt.")

    df_train = pd.DataFrame({'ImagePath': train_image_full_paths})
    
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    train_set = ChestXRayDataset(df=df_train, transform=transform)
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
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)
            
            optimizer_d.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim, device=device)
            gen_imgs = generator(z)
            d_loss = (adversarial_loss(discriminator(imgs), valid) + adversarial_loss(discriminator(gen_imgs.detach()), fake)) / 2
            d_loss.backward()
            optimizer_d.step()
            
            optimizer_g.zero_grad()
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_g.step()
            
        print(f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    os.makedirs("models", exist_ok=True)
    save_path = "models/strong_xray_generator.pth"
    torch.save(generator.state_dict(), save_path)
    print(f"\n✅ Training complete! Model saved to '{save_path}'")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

# We will use the data loading utility we already created
from chest_data_util import load_data

class XRayGenerator(nn.Module):
    """A DCGAN generator for 128x128 grayscale X-ray images."""
    def __init__(self, latent_dim=100, channels=1):
        super(XRayGenerator, self).__init__()
        self.model = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State size. (512) x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State size. (256) x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State size. (128) x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State size. (64) x 32 x 32
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # State size. (32) x 64 x 64
            nn.ConvTranspose2d(32, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final state size. (channels) x 128 x 128
        )
    def forward(self, z):
        return self.model(z.view(z.size(0), -1, 1, 1))

class XRayDiscriminator(nn.Module):
    """A DCGAN discriminator for 128x128 grayscale X-ray images."""
    def __init__(self, channels=1):
        super(XRayDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # Input is (channels) x 128 x 128
            nn.Conv2d(channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (32) x 64 x 64
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (64) x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (128) x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (256) x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (512) x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, img):
        return self.model(img)

def main():
    # --- Configuration ---
    epochs = 50 # GANs need more epochs
    lr = 0.0002
    latent_dim = 100
    batch_size = 64
    data_path = "./data"
    train_list = "train_val_list.txt"
    test_list = "test_list.txt" # Not used, but required by load_data
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    train_set, _ = load_data(data_path, train_list, test_list)
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    # --- Models and Optimizers ---
    generator = XRayGenerator(latent_dim).to(device)
    discriminator = XRayDiscriminator().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()

    print("Starting X-ray Generator Training...")
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            valid = torch.ones(imgs.size(0), 1, 1, 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, 1, 1, device=device)
            
            # Train Discriminator
            optimizer_d.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim, device=device)
            gen_imgs = generator(z)
            d_loss = (adversarial_loss(discriminator(imgs), valid) + adversarial_loss(discriminator(gen_imgs.detach()), fake)) / 2
            d_loss.backward()
            optimizer_d.step()
            
            # Train Generator
            optimizer_g.zero_grad()
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_g.step()

        print(f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    os.makedirs("models", exist_ok=True)
    torch.save(generator.state_dict(), "models/xray_generator.pth")
    print("\n✅ X-ray Generator training complete. Model saved to 'models/xray_generator.pth'")

if __name__ == "__main__":
    main()
# In train_generator.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import os

# Use the same Generator class from ggl_attack.py
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784), nn.Tanh()
        )
    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, img):
        return self.model(img.view(img.size(0), -1))

# Set device (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Force GPU even if fallback occurs (optional)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.cuda.set_device(0)  # Explicitly select GPU 0

def main():
    # Config
    epochs = 15
    lr = 0.0002
    latent_dim = 100
    batch_size = 64
    
    # Load Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Init models and move to GPU
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
    adversarial_loss = nn.BCELoss()

    print("Starting Generator Training...")
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # Move data to GPU
            imgs = imgs.to(device)
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)
            
            # Train Discriminator
            optimizer_d.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs), valid)
            z = torch.randn(imgs.size(0), latent_dim, device=device)  # Move noise to GPU
            gen_imgs = generator(z)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()
            
            # Train Generator
            optimizer_g.zero_grad()
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_g.step()

        print(f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    # Save the trained generator
    os.makedirs("models", exist_ok=True)
    torch.save(generator.state_dict(), "models/generator.pth")
    print("\n✅ Generator training complete. Model saved to 'models/generator.pth'")

if __name__ == "__main__":
    main()
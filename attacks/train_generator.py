import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import os

class StrongGenerator(nn.Module):
    """A more powerful DCGAN-style generator for MNIST."""
    def __init__(self, latent_dim=100):
        super(StrongGenerator, self).__init__()
        self.init_size = 28 // 4  # 7
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class StrongDiscriminator(nn.Module):
    """A DCGAN-style discriminator for MNIST."""
    def __init__(self):
        super(StrongDiscriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        ds_size = 28 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

def main():
    epochs, lr, latent_dim, batch_size = 30, 0.0002, 100, 64
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    generator, discriminator = StrongGenerator(latent_dim), StrongDiscriminator()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()

    print("Starting Strong Generator Training...")
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            valid, fake = torch.ones(imgs.size(0), 1), torch.zeros(imgs.size(0), 1)
            optimizer_d.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim)
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
    torch.save(generator.state_dict(), "models/strong_generator.pth")
    print("\n✅ Strong Generator training complete. Model saved to 'models/strong_generator.pth'")

if __name__ == "__main__":
    main()
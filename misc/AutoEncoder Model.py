import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ------- CNN Modules -------


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # [B, 32, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # [B, 64, H/4, W/4]
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(64 * 8 * 8, latent_dim)  # assuming input is 32x32

    def forward(self, x):
        x = self.conv(x)
        z = self.fc(x)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 64 * 8 * 8), nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [B, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # [B, 3, 32, 32]
            nn.Sigmoid(),  # use sigmoid if input is normalized to [0,1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 8, 8)
        return self.deconv(x)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, z):
        return self.classifier(z)


# ------- Wrapper -------


class AutoEncoderWithClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.classifier = AuxiliaryClassifier(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        pred = self.classifier(z)
        return recon, pred, z


# ------- Training -------


def train(model, dataloader, num_epochs=10, lr=1e-3, alpha=0.3):
    model.to(device)
    model.train()

    recon_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_recon, total_cls = 0.0, 0.0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            recon, pred, _ = model(x)
            loss_recon = recon_criterion(recon, x)
            loss_cls = cls_criterion(pred, y)
            loss = (1 - alpha) * loss_recon + alpha * loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += loss_recon.item()
            total_cls += loss_cls.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Recon Loss: {total_recon:.4f}, Aux Loss: {total_cls:.4f}"
        )


# ------- Example Setup -------

# Example for CelebA (resized 32x32) or CIFAR-10
latent_dim = 128
num_classes = 10  # adjust for dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy data (replace with real image dataset like CelebA or CIFAR-10)
x = torch.rand(5000, 3, 32, 32)  # 500 RGB images of size 32x32
y = torch.randint(0, num_classes, (5000,))
dataloader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)

model = AutoEncoderWithClassifier(latent_dim, num_classes)
train(model, dataloader, num_epochs=100, alpha=0.4)

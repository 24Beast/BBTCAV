import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim, H, W):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # [B, 32, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # [B, 64, H/4, W/4]
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(int(H // 4) * int(W // 4) * 64, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        z = self.fc(x)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim, H, W):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, int(H // 4) * int(W // 4) * 64), nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [B, 32, H//2, W//2]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # [B, 3, H, W]
            nn.Sigmoid(),  # use sigmoid if input is normalized to [0,1]
        )
        self.H = H
        self.W = W

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, int(self.H // 4), int(self.W // 4))
        return self.deconv(x)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, z):
        return self.classifier(z)


# ------- Wrapper -------


class AutoEncoderWithClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes, H=32, W=32):
        super().__init__()
        self.encoder = Encoder(latent_dim, H, W)
        self.decoder = Decoder(latent_dim, H, W)
        self.classifier = AuxiliaryClassifier(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        pred = self.classifier(z)
        return recon, pred, z

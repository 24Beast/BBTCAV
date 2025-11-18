import torch
import torch.nn as nn
from torchvision import transforms, models


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
    def __init__(self, latent_dim, H, W, base_channels=32, depth=3):
        """
        Args:
            latent_dim: Dimension of latent vector.
            H, W: Input height and width.
            base_channels: Number of channels in first conv layer.
            depth: Number of downsampling layers.
        """
        super().__init__()
        layers = []
        in_ch = 3
        out_ch = base_channels

        # Build variable-depth conv stack
        for i in range(depth):
            layers += [
                nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
            out_ch *= 2  # double channels each step

        self.conv = nn.Sequential(*layers)

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, H, W)
            conv_out = self.conv(dummy)
            flattened_dim = conv_out.numel()

        self.fc = nn.Linear(flattened_dim, latent_dim)
        self._conv_out_shape = conv_out.shape[1:]  # (C, H', W')

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        z = self.fc(x)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim, conv_out_shape, base_channels=32, depth=3):
        """
        Args:
            latent_dim: Latent vector dimension.
            conv_out_shape: (C, H', W') from encoder.
            base_channels: Must match encoder's base_channels.
            depth: Number of upsampling layers.
        """
        super().__init__()
        C, H_out, W_out = conv_out_shape
        self._conv_out_shape = conv_out_shape
        self.H_out, self.W_out = H_out, W_out

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, C * H_out * W_out),
            nn.ReLU(inplace=True),
        )

        layers = []
        in_ch = C
        out_ch = in_ch // 2
        for i in range(depth - 1):
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
            out_ch = max(out_ch // 2, base_channels)

        layers += [
            nn.ConvTranspose2d(in_ch, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        ]

        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, *self._conv_out_shape)
        return self.deconv(x)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, z):
        return self.classifier(z)


# ------- Wrapper -------


class AutoEncoderWithClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes, H=32, W=32, depth=3, base_channels=32):
        super().__init__()
        self.encoder = Encoder(latent_dim, H, W, depth, base_channels)
        self.decoder = Decoder(latent_dim, H, W, depth, base_channels)
        self.classifier = AuxiliaryClassifier(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        pred = self.classifier(z)
        return recon, pred, z


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim, H=32, W=32, depth=3, base_channels=32):
        super().__init__()
        self.encoder = Encoder(latent_dim, H, W, base_channels, depth)
        self.decoder = Decoder(
            latent_dim, self.encoder._conv_out_shape, base_channels, depth
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

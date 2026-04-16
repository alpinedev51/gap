import torch
import torch.nn as nn


class DAE(nn.Module):
    def __init__(self, in_channels=3, base_filters=32, num_layers=3):
        """
        Denoising Autoencoder wrapper.

        Args:
            in_channels (int): Input image channels (3 for RGB).
            base_filters (int): Number of filters in the first layer.
            num_layers (int): Number of conv/transpose-conv blocks.
        """
        super(DAE, self).__init__()
        self.encoder = Encoder(in_channels, base_filters, num_layers)
        self.decoder = Decoder(in_channels, base_filters, num_layers)

    def add_noise(self, x, noise_factor, noise_type="gaussian"):
        """Adds noise to a batch of images."""
        if noise_type == "gaussian":
            noise = torch.randn_like(x) * noise_factor
            x_noisy = x + noise
        elif noise_type == "salt_and_pepper":
            mask = torch.rand_like(x)
            x_noisy = x.clone()
            x_noisy[mask < (noise_factor / 2)] = 0
            x_noisy[mask > (1 - noise_factor / 2)] = 1
        else:
            return x

        return torch.clamp(x_noisy, 0.0, 1.0)

    def forward(self, x, noise_factor=0.0):
        if self.training and noise_factor > 0:
            x = self.add_noise(x, noise_factor)

        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def get_latent_features(self, x):
        """Helper for using the model as a feature extractor."""
        with torch.no_grad():
            return self.encoder(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, base_filters, num_layers):
        super(Encoder, self).__init__()
        layers = []
        curr_channels = in_channels

        for i in range(num_layers):
            out_channels = base_filters * (2**i)
            layers.extend(
                [
                    nn.Conv2d(
                        curr_channels, out_channels, kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                ]
            )
            curr_channels = out_channels

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, out_channels, base_filters, num_layers):
        super(Decoder, self).__init__()
        layers = []
        curr_channels = base_filters * (2 ** (num_layers - 1))

        for i in range(num_layers - 1, -1, -1):
            target_channels = base_filters * (2 ** (i - 1)) if i > 0 else out_channels

            layers.append(
                nn.ConvTranspose2d(
                    curr_channels,
                    target_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )

            if i > 0:
                layers.extend([nn.BatchNorm2d(target_channels), nn.ReLU(True)])
                curr_channels = target_channels
            else:
                layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

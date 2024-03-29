import torch
import torch.nn as nn

# Model definition


class Model(nn.Module):
    def __init__(self):
        "Convolutional Autoencoder definition"
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x14x14
            nn.Conv2d(16, 32, 3, padding=1),  # 32x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x7x7
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,
                               output_padding=1),  # 16x7x7
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1,
                               output_padding=1),  # 1x14x14
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        "Computes the forward pass"
        "in: x, f(x) -> h, g(h) -> r, out: r"
        return self.decoder(self.encoder(x))

    def encode(self, x):
        "Encodes sample x into the learned encoding h"
        "in: x, f(x) -> h, out: h"
        return self.encoder(x)

    def decode(self, h):
        "Decodes encoding h into the learned reconstruction r"
        "in: h, g(h) -> r, out: r"
        return self.decoder(h)

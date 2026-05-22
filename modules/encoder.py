import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[512, 256], latent_dim=128):
        super().__init__()
        # BatchNorm + L2-norm applied to the raw input before encoding (GRID).
        # BatchNorm learns per-feature statistics; L2-norm puts all inputs on the unit sphere.
        # Together they replace dataset-wide normalisation while preserving column-specific variance.
        self.input_norm = nn.BatchNorm1d(input_dim)

        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x_norm = F.normalize(self.input_norm(x), p=2, dim=-1)
        return self.network(x_norm), x_norm  # (latent, normalised_input)


class Decoder(nn.Module):
    def __init__(self, output_dim=768, hidden_dims=[256, 512], latent_dim=128):
        super().__init__()
        layers = []
        dims = [latent_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return self.network(z)

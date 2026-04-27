"""Feed-forward variational autoencoder for vector dimensionality reduction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: tuple[int, ...] = (256, 128)


def vae_elbo_loss(
    recon: Tensor,
    x: Tensor,
    mu: Tensor,
    logvar: Tensor,
    *,
    recon_loss: str = "mse",
    beta: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns ``(total, recon_term, kl_term)``."""
    if recon_loss == "mse":
        r = F.mse_loss(recon, x, reduction="mean")
    elif recon_loss == "bce":
        r = F.binary_cross_entropy(recon, x, reduction="mean")
    else:
        raise ValueError("recon_loss must be 'mse' or 'bce'.")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = r + beta * kl
    return total, r, kl


class VAE(nn.Module):
    """MLP VAE: Gaussian latent, Bernoulli or Gaussian decoder (MSE recon)."""

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        enc_layers: list[nn.Module] = []
        prev = config.input_dim
        for h in config.hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, config.latent_dim)
        self.fc_logvar = nn.Linear(prev, config.latent_dim)

        dec_layers: list[nn.Module] = []
        prev = config.latent_dim
        for h in reversed(config.hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        dec_layers.append(nn.Linear(prev, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

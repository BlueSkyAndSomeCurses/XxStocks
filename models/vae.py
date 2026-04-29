"""Feed-forward variational autoencoder for vector dimensionality reduction."""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import polars as pl


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

FEATURE_COLUMN_RE = re.compile(r"^text_embed_(\d+)$")


def sorted_feature_columns(columns: list[str], text_embed_only: bool = True) -> list[str]:
    if text_embed_only:
        matched_columns: list[tuple[int, str]] = []
        for column in columns:
            match = FEATURE_COLUMN_RE.match(column)
            if match is not None:
                matched_columns.append((int(match.group(1)), column))
        matched_columns.sort(key=lambda item: item[0])
        return [column for _, column in matched_columns]
    else:
        return list(columns)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dataframe_to_tensor(data: pl.DataFrame, feature_cols: list[str]) -> Tensor:
    features = data.select(list(feature_cols)).to_numpy()
    return torch.as_tensor(features, dtype=torch.float32)


def train_vae_on_dataframe(
    data: pl.DataFrame,
    config: VAEConfig | None = None,
    *,
    batch_size: int = 256,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    beta: float = 1.0,
    device: str | torch.device | None = None,
    shuffle: bool = True,
    text_embed_only: bool = True,
) -> tuple[VAE, list[str]]:
    feature_cols = sorted_feature_columns(data.columns, text_embed_only=text_embed_only)
    if not feature_cols:
        raise ValueError("No columns matching 'feature_<number>' were found.")

    resolved_device = resolve_device(device)
    input_dim = len(feature_cols)
    if config is None:
        config = VAEConfig(input_dim=input_dim)
    elif config.input_dim != input_dim:
        raise ValueError(
            f"VAEConfig.input_dim={config.input_dim} does not match the number of feature columns {input_dim}."
        )

    model = VAE(config).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    x_tensor = dataframe_to_tensor(data, feature_cols)
    loader = DataLoader(
        TensorDataset(x_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )

    model.train()
    for _ in range(epochs):
        for (batch_x,) in loader:
            batch_x = batch_x.to(resolved_device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_x)
            loss, _, _ = vae_elbo_loss(recon, batch_x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

    return model, feature_cols


def encode_with_vae(
    model: VAE,
    data: pl.DataFrame,
    feature_cols: list[str] | None = None,
    *,
    batch_size: int = 1024,
    device: str | torch.device | None = None,
    latent_prefix: str = "latent_",
    text_embed_only: bool = True,
) -> pl.DataFrame:
    if feature_cols is None:
        feature_cols = sorted_feature_columns(data.columns, text_embed_only=text_embed_only)
    else:
        feature_cols = list(feature_cols)

    if not feature_cols:
        raise ValueError("No feature columns were provided or detected.")

    expected_dim = model.config.input_dim
    if expected_dim != len(feature_cols):
        raise ValueError(
            f"Model expects {expected_dim} feature columns, but {len(feature_cols)} were provided."
        )

    if device is None:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
    else:
        model_device = resolve_device(device)

    x_tensor = dataframe_to_tensor(data, feature_cols)
    loader = DataLoader(
        TensorDataset(x_tensor),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model.eval()
    encoded_batches: list[Tensor] = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(model_device)
            mu, _logvar = model.encode(batch_x)
            encoded_batches.append(mu.cpu())

    latent_values = torch.cat(encoded_batches, dim=0)
    latent_columns = [f"{latent_prefix}{index}" for index in range(latent_values.shape[1])]
    latent_frame = pl.DataFrame(
        latent_values.numpy(),
        schema=latent_columns,
        orient="row",
    )
    feature_set = set(feature_cols)
    remaining_columns = [column for column in data.columns if column not in feature_set]
    base_frame = data.select(remaining_columns)
    return pl.concat([base_frame, latent_frame], how="horizontal")


def train_and_encode_vae_dataframe(
    data: pl.DataFrame,
    config: VAEConfig | None = None,
    *,
    text_embed_only: bool = True,
    **kwargs,
) -> tuple[VAE, pl.DataFrame]:
    """Train a VAE on feature columns and return the model plus encoded dataframe."""
    model, feature_cols = train_vae_on_dataframe(data, config=config, text_embed_only=text_embed_only, **kwargs)
    encoded_data = encode_with_vae(
        model,
        data,
        feature_cols=feature_cols,
        batch_size=kwargs.get("batch_size", 1024),
        device=kwargs.get("device"),
        text_embed_only=text_embed_only,
    )
    return model, encoded_data


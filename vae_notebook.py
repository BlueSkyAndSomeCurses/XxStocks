import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # VAE dimensionality reduction

    Trains :class:`models.vae.VAE` on dense vectors (synthetic Gaussian mixture by default).
    Swap in rows from ``text_encoder_embeddings_30m.parquet`` if you want to compress BERT bins.
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    import torch
    from torch import optim
    from torch.utils.data import DataLoader, TensorDataset

    from models.vae import VAE, VAEConfig, vae_elbo_loss

    return (
        DataLoader,
        TensorDataset,
        VAE,
        VAEConfig,
        np,
        optim,
        pl,
        torch,
        vae_elbo_loss,
    )


@app.cell
def _(torch):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    return (device,)


@app.cell
def _(mo):
    from pathlib import Path

    bert_path = "data/final_data/train/text_encoder_embeddings_30m.parquet"
    use_bert = Path(bert_path).exists()
    mo.md(f"Using **{'BERT parquet' if use_bert else 'synthetic data'}**.")
    return bert_path, use_bert


@app.cell
def _(bert_path, np, pl, torch, use_bert):
    if use_bert:
        df = pl.read_parquet(bert_path)
        text_cols = sorted(
            (c for c in df.columns if c.startswith("text_embed_")),
            key=lambda c: int(c.removeprefix("text_embed_")),
        )
        x_np = df.select(text_cols).to_numpy().astype(np.float32)
        # Subsample for quick demo in the notebook
        n = min(10_000, x_np.shape[0])
        x_np = x_np[:n]
    else:
        rng = np.random.default_rng(0)
        n, dim = 4096, 64
        centers = rng.normal(size=(8, dim)).astype(np.float32)
        idx = rng.integers(0, 8, size=n)
        x_np = centers[idx] + rng.normal(scale=0.3, size=(n, dim)).astype(np.float32)

    x = torch.tensor(x_np)
    input_dim = x.size(1)
    latent_dim = 16
    return input_dim, latent_dim, x


@app.cell
def _(VAEConfig, input_dim, latent_dim):
    cfg = VAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=(256, 128),
    )
    cfg
    return (cfg,)


@app.cell
def _(DataLoader, TensorDataset, VAE, cfg, device, optim, vae_elbo_loss, x):
    model = VAE(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(x),
        batch_size=256,
        shuffle=True,
        drop_last=True,
    )

    model.train()
    for epoch in range(30):
        total_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, _, _ = vae_elbo_loss(recon, xb, mu, logvar, beta=1.0)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= len(loader.dataset)
        print(f"epoch {epoch + 1}  loss={total_loss:.5f}")
    return loader, model


@app.cell
def _(device, model, torch, x):
    model.eval()
    with torch.no_grad():
        mu_eval, _logvar = model.encode(x.to(device))
    mu_cpu = mu_eval.cpu()
    latent_mean = mu_cpu.mean(dim=0)
    latent_std = mu_cpu.std(dim=0)
    print("latent dim means (first 5):", latent_mean[:5].tolist())
    print("latent dim std  (first 5):", latent_std[:5].tolist())
    return


@app.cell
def _(loader, model):
    model.eval()
    for (x_orig,) in loader:
        print(x_orig.shape)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

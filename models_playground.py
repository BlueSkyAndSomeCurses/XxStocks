# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.21.1",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from dataset.loader import SPYDataset
    from dataset.preprocessing import (
        augment_dataset,
        downsample_to_interval,
        time_train_test_split,
        split_features_target,
    )

    from models.lstm import LSTMModel
    from models.sarimax import direction_accuracy, forecast_sarimax, train_sarimax, save_sarimax_model, load_sarimax_model
    from models.binary_fincast import BinaryFinCast, BinaryFinCastConfig

    return (
        BinaryFinCast,
        BinaryFinCastConfig,
        DataLoader,
        LSTMModel,
        SPYDataset,
        augment_dataset,
        downsample_to_interval,
        forecast_sarimax,
        mo,
        nn,
        optim,
        pl,
        save_sarimax_model,
        split_features_target,
        time_train_test_split,
        torch,
        train_sarimax,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data loading
    """)
    return


@app.cell
def _(augment_dataset, downsample_to_interval, pl):
    CVS_PATH = "./data/1_min_SPY_2008-2021.csv"
    df = pl.read_csv(CVS_PATH)
    df = downsample_to_interval(df, interval="30m")
    df_augmented = augment_dataset(df)
    return (df_augmented,)


@app.cell
def _(df_augmented, time_train_test_split):
    train_df, test_df = time_train_test_split(df_augmented, test_ratio=0.2)
    return test_df, train_df


@app.cell
def _(split_features_target, test_df, train_df):
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    return X_test, X_train, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SARIMAX
    """)
    return


@app.cell
def _(X_train, train_sarimax, y_train):
    sarimax_fit = train_sarimax(y_train, X_train, order=(1, 1, 1))
    return (sarimax_fit,)


@app.cell
def _(sarimax_fit, save_sarimax_model):
    save_sarimax_model(sarimax_fit, "sarimax_model")
    return


@app.cell
def _(X_test, forecast_sarimax, sarimax_fit):
    arimax_forecast = forecast_sarimax(sarimax_fit, X_test)
    return (arimax_forecast,)


@app.cell
def _(arimax_forecast):
    print(arimax_forecast)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LSTM
    """)
    return


@app.cell
def _(DataLoader, SPYDataset, test_df, train_df):
    window_size = 32
    batch_size = 128

    train_dataset = SPYDataset(train_df, window_size=window_size)
    test_dataset = SPYDataset(test_df, window_size=window_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, train_dataset, train_loader


@app.cell
def _(torch):
    def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=5):
        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        counter = 0
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device).view(-1, 1)

                optimizer.zero_grad()
                logits, _, _ = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device).view(-1, 1)
                    logits_val, _, _ = model(x_val)
                    v_loss = criterion(logits_val, y_val)
                    total_val_loss += v_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.6f} | Val Loss {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    model.load_state_dict(best_model_state)
                    break

        return train_losses, val_losses

    return (train_model,)


@app.cell
def _(torch):
    def evaluate_model(model, data_loader, device):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward pass
                logits, _, _ = model(x_batch)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                y_true = y_batch.view(-1, 1)

                correct += (preds == y_true).sum().item()
                total += y_true.numel()

        accuracy = float(correct / max(total, 1))
        return accuracy

    return (evaluate_model,)


@app.cell
def _(LSTMModel, nn, optim, torch, train_dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = train_dataset.X.shape[1]
    model = LSTMModel(input_dim=input_dim, hidden_dim=64, layer_dim=2, output_dim=1).to(
        device
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50
    return criterion, device, epochs, input_dim, model, optimizer


@app.cell
def _(
    criterion,
    device,
    epochs,
    model,
    optimizer,
    test_loader,
    train_loader,
    train_model,
):
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, device, epochs=epochs, patience=5
    )
    return


@app.cell
def _(device, evaluate_model, model, test_loader):
    accuracy = evaluate_model(model, test_loader, device)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Binary Fincast
    """)
    return


@app.cell
def _(BinaryFinCast, nn, torch):
    class BinaryFinCastAdapter(nn.Module):
        def __init__(self, base_model: BinaryFinCast, freq_id: int = 0):
            super().__init__()
            self.base_model = base_model
            self.freq_id = freq_id

        def forward(self, x):
            freq_id = torch.full((x.size(0),), self.freq_id, dtype=torch.long, device=x.device)
            out = self.base_model(x=x, freq_id=freq_id)
            logits = out["logits"].view(-1, 1)
            return logits, None, None

    return (BinaryFinCastAdapter,)


@app.cell
def _(
    BinaryFinCast,
    BinaryFinCastAdapter,
    BinaryFinCastConfig,
    device,
    input_dim,
    nn,
    optim,
):
    fincast_config = BinaryFinCastConfig(
        input_dim=input_dim,
        patch_len=8,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    )
    fincast_backbone = BinaryFinCast(fincast_config).to(device)
    fincast_model = BinaryFinCastAdapter(fincast_backbone).to(device)
    fincast_criterion = nn.BCEWithLogitsLoss()
    fincast_optimizer = optim.Adam(fincast_model.parameters(), lr=1e-4)
    fincast_epochs = 50
    return fincast_criterion, fincast_epochs, fincast_model, fincast_optimizer


@app.cell
def _(
    device,
    fincast_criterion,
    fincast_epochs,
    fincast_model,
    fincast_optimizer,
    test_loader,
    train_loader,
    train_model,
):
    fincast_train_losses, fincast_val_losses = train_model(
        fincast_model,
        train_loader,
        test_loader,
        fincast_criterion,
        fincast_optimizer,
        device,
        epochs=fincast_epochs,
        patience=5,
    )
    return


@app.cell
def _(device, evaluate_model, fincast_model, test_loader):
    fincast_accuracy = evaluate_model(fincast_model, test_loader, device)
    print(f"Binary FinCast Test Accuracy: {fincast_accuracy * 100:.2f}%")
    return


if __name__ == "__main__":
    app.run()

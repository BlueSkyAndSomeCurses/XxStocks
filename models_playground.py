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

    from dataset.loader import SPYDataset
    from dataset.preprocessing import (
        augment_dataset,
        time_train_test_split,
        split_features_target,
    )

    from models.lstm import LSTMModel
    from models.sarimax import direction_accuracy, forecast_sarimax, train_sarimax

    return (
        DataLoader,
        LSTMModel,
        SPYDataset,
        augment_dataset,
        forecast_sarimax,
        mo,
        nn,
        np,
        optim,
        pl,
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
def _(augment_dataset, pl):
    CVS_PATH = "./data/1_min_SPY_2008-2021.csv"
    df = pl.read_csv(CVS_PATH)
    df_augmented = augment_dataset(df)
    return (df_augmented,)


@app.cell
def _(df_augmented, time_train_test_split):
    train_df, test_df = time_train_test_split(df_augmented, test_ratio=0.2)
    return test_df, train_df


@app.cell
def _(split_features_target, test_df, train_df):
    X_train, y_train = split_features_target(train_df)
    X_train, y_train = split_features_target(test_df)
    return X_train, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SARIMAX
    """)
    return


@app.cell
def _(X_train, forecast_sarimax, np, test_df, train_sarimax, y_train):
    sarimax_fit = train_sarimax(y_train, X_train, order=(1, 1, 1))
    sarimax_forecast = np.asarray(
        forecast_sarimax(sarimax_fit, steps=test_df.height), dtype=float
    )
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
def _(LSTMModel, mo, nn, optim, torch, train_dataset, train_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = train_dataset.X.shape[1]
    model = LSTMModel(input_dim=input_dim, hidden_dim=64, layer_dim=2, output_dim=1).to(
        device
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 3
    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.view(-1, 1).to(device)

            optimizer.zero_grad()
            logits, _, _ = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        train_losses.append(running_loss / max(batches, 1))

    mo.md(
        f"Torch training finished on **{device}**. Last train loss: **{train_losses[-1]:.6f}**"
    )
    return device, model


@app.cell
def _(device, model, test_loader, torch):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits, _, _ = model(x_batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            y_true = y_batch.view(-1, 1)

            correct += (preds == y_true).sum().item()
            total += y_true.numel()

    test_accuracy = float(correct / max(total, 1))
    test_accuracy
    return


if __name__ == "__main__":
    app.run()

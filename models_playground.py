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
    import numpy as np
    import polars as pl
    import torch
    from torch import nn, optim

    from dataset.loader import make_dataloader
    from dataset.preprocessing import (
        augment_dataset,
        downsample_to_interval,
        prepare_arima_data,
        split_features_target,
        time_train_test_split,
    )
    from models.fincast import (
        BinaryFinCast,
        BinaryFinCastConfig,
        ContinuousFinCast,
        ContinuousFinCastConfig,
    )
    from models.evaluation import evaluate_predictions
    from models.lstm import LSTMModel
    from models.sarimax import forecast_sarimax, train_sarimax

    return (
        BinaryFinCast,
        BinaryFinCastConfig,
        ContinuousFinCast,
        ContinuousFinCastConfig,
        LSTMModel,
        augment_dataset,
        downsample_to_interval,
        evaluate_predictions,
        forecast_sarimax,
        make_dataloader,
        nn,
        np,
        optim,
        pl,
        prepare_arima_data,
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
    CSV_PATH = "./data/1_min_SPY_2008-2021.csv"
    df = pl.read_csv(CSV_PATH)
    df = downsample_to_interval(df, interval="30m")
    df_augmented = augment_dataset(df)
    return (df_augmented,)


@app.cell
def _(df_augmented, time_train_test_split):
    train_df, test_df = time_train_test_split(df_augmented, test_ratio=0.2)
    print(train_df.shape, test_df.shape)
    return test_df, train_df


@app.cell
def _(prepare_arima_data, test_df, train_df):
    y_train_cont, X_train_cont = prepare_arima_data(train_df, task="continuous")
    y_test_cont, X_test_cont = prepare_arima_data(test_df, task="continuous")

    y_train_bin, X_train_bin = prepare_arima_data(train_df, task="binary")
    y_test_bin, X_test_bin = prepare_arima_data(test_df, task="binary")
    return (
        X_test_bin,
        X_test_cont,
        X_train_bin,
        X_train_cont,
        y_test_bin,
        y_test_cont,
        y_train_bin,
        y_train_cont,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SARIMAX pipelines (continuous and binary)
    """)
    return


@app.cell
def _(
    X_test_cont,
    X_train_cont,
    forecast_sarimax,
    np,
    train_sarimax,
    y_train_cont,
):
    sarimax_cont = train_sarimax(y_train_cont, X_train_cont, order=(1, 0, 1))
    pred_cont = np.asarray(forecast_sarimax(sarimax_cont, X_test_cont)).reshape(-1)
    return (pred_cont,)


@app.cell
def _(evaluate_predictions, pred_cont, y_test_cont):
    metrics_sarimax_cont = evaluate_predictions(
        y_test_cont.to_numpy(), pred_cont, task="continuous"
     )
    print("SARIMAX continuous metrics:")
    print(metrics_sarimax_cont)
    return


@app.cell
def _(
    X_test_bin,
    X_train_bin,
    forecast_sarimax,
    np,
    train_sarimax,
    y_train_bin,
):
    sarimax_bin = train_sarimax(y_train_bin, X_train_bin, order=(1, 0, 1))
    raw_pred_bin = np.asarray(forecast_sarimax(sarimax_bin, X_test_bin)).reshape(-1)
    pred_bin = np.where(raw_pred_bin >= 0, 1, -1)
    return (pred_bin,)


@app.cell
def _(evaluate_predictions, pred_bin, y_test_bin):
    metrics_sarimax_bin = evaluate_predictions(
        y_test_bin.to_numpy(), pred_bin, task="binary"
     )
    print("SARIMAX binary metrics:")
    print(metrics_sarimax_bin)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deep learning pipelines (binary and continuous)
    """)
    return


@app.cell
def _(make_dataloader, test_df, torch, train_df):
    window_size = 32
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader_bin = make_dataloader(
        train_df, task="binary", window_size=window_size, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader_bin = make_dataloader(
        test_df, task="binary", window_size=window_size, batch_size=batch_size, shuffle=False, drop_last=False
    )

    train_loader_cont = make_dataloader(
        train_df, task="continuous", window_size=window_size, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader_cont = make_dataloader(
        test_df, task="continuous", window_size=window_size, batch_size=batch_size, shuffle=False, drop_last=False
    )

    input_dim = train_loader_bin.dataset.X.shape[1]
    print(f"Using {device=}, {input_dim=}")
    return (
        device,
        input_dim,
        test_loader_bin,
        test_loader_cont,
        train_loader_bin,
        train_loader_cont,
    )


@app.cell
def _(evaluate_predictions, np, torch):
    def _extract_primary_output(model_out):
        if isinstance(model_out, tuple):
            return model_out[0]
        return model_out

    def train_generic(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        task="binary",
        epochs=20,
        patience=5,
     ):
        best_val = float("inf")
        stale_epochs = 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).view(-1, 1)

                optimizer.zero_grad()
                out = _extract_primary_output(model(x_batch))

                if task == "binary":
                    y_for_loss = (y_batch + 1.0) / 2.0
                else:
                    y_for_loss = y_batch

                loss = criterion(out, y_for_loss)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / max(len(train_loader), 1)

            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device).view(-1, 1)
                    out_val = _extract_primary_output(model(x_val))
                    if task == "binary":
                        y_for_loss = (y_val + 1.0) / 2.0
                    else:
                        y_for_loss = y_val
                    total_val_loss += criterion(out_val, y_for_loss).item()

            avg_val_loss = total_val_loss / max(len(val_loader), 1)
            print(f"Epoch {epoch + 1}: train={avg_train_loss:.6f}, val={avg_val_loss:.6f}")

            if avg_val_loss < best_val:
                best_val = avg_val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print("Early stopping triggered.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

    def evaluate_torch_model(model, data_loader, device, task="binary"):
        model.eval()
        all_true = []
        all_pred = []

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(device)
                out = _extract_primary_output(model(x_batch)).view(-1)

                if task == "binary":
                    preds = torch.where(
                        out >= 0.0,
                        torch.tensor(1.0, device=out.device),
                        torch.tensor(-1.0, device=out.device),
                    )
                else:
                    preds = out

                all_true.append(y_batch.view(-1).cpu().numpy())
                all_pred.append(preds.cpu().numpy())

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        return evaluate_predictions(y_true, y_pred, task=task)

    return evaluate_torch_model, train_generic


@app.cell
def _(
    LSTMModel,
    device,
    input_dim,
    nn,
    optim,
    test_loader_bin,
    train_generic,
    train_loader_bin,
):
    lstm_binary = LSTMModel(input_dim=input_dim, hidden_dim=64, layer_dim=2, output_dim=1).to(device)
    criterion_binary = nn.BCEWithLogitsLoss()
    optimizer_binary = optim.Adam(lstm_binary.parameters(), lr=1e-3)

    train_generic(
        lstm_binary,
        train_loader_bin,
        test_loader_bin,
        criterion_binary,
        optimizer_binary,
        device,
        task="binary",
        epochs=20,
        patience=5,
     )
    return (lstm_binary,)


@app.cell
def _(device, evaluate_torch_model, lstm_binary, test_loader_bin):
    metrics_lstm_binary = evaluate_torch_model(
        lstm_binary, test_loader_bin, device, task="binary"
     )
    print("LSTM binary metrics:")
    print(metrics_lstm_binary)
    return


@app.cell
def _(
    LSTMModel,
    device,
    input_dim,
    nn,
    optim,
    test_loader_cont,
    train_generic,
    train_loader_cont,
):
    lstm_continuous = LSTMModel(input_dim=input_dim, hidden_dim=64, layer_dim=2, output_dim=1).to(device)
    criterion_continuous = nn.MSELoss()
    optimizer_continuous = optim.Adam(lstm_continuous.parameters(), lr=1e-3)

    train_generic(
        lstm_continuous,
        train_loader_cont,
        test_loader_cont,
        criterion_continuous,
        optimizer_continuous,
        device,
        task="continuous",
        epochs=20,
        patience=5,
     )
    return (lstm_continuous,)


@app.cell
def _(device, evaluate_torch_model, lstm_continuous, test_loader_cont):
    metrics_lstm_continuous = evaluate_torch_model(
        lstm_continuous, test_loader_cont, device, task="continuous"
     )
    print("LSTM continuous metrics:")
    print(metrics_lstm_continuous)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## FinCast pipelines (binary and continuous)
    """)
    return


@app.cell
def _(nn, torch):
    class FinCastAdapter(nn.Module):
        def __init__(self, base_model: nn.Module, output_key: str, freq_id: int = 0):
            super().__init__()
            self.base_model = base_model
            self.output_key = output_key
            self.freq_id = freq_id

        def forward(self, x):
            freq_id = torch.full(
                (x.size(0),), self.freq_id, dtype=torch.long, device=x.device
            )
            out = self.base_model(x=x, freq_id=freq_id)
            return out[self.output_key].view(-1, 1), None, None

    return (FinCastAdapter,)


@app.cell
def _(
    BinaryFinCast,
    BinaryFinCastConfig,
    ContinuousFinCast,
    ContinuousFinCastConfig,
    FinCastAdapter,
    device,
    input_dim,
    nn,
    optim,
):
    fincast_bin_config = BinaryFinCastConfig(
        input_dim=input_dim,
        patch_len=8,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    )
    fincast_bin_backbone = BinaryFinCast(fincast_bin_config).to(device)
    fincast_bin_model = FinCastAdapter(
        fincast_bin_backbone, output_key="logits", freq_id=0
    ).to(device)
    fincast_bin_criterion = nn.BCEWithLogitsLoss()
    fincast_bin_optimizer = optim.Adam(fincast_bin_model.parameters(), lr=1e-4)

    fincast_cont_config = ContinuousFinCastConfig(
        input_dim=input_dim,
        patch_len=8,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    )
    fincast_cont_backbone = ContinuousFinCast(fincast_cont_config).to(device)
    fincast_cont_model = FinCastAdapter(
        fincast_cont_backbone, output_key="prediction", freq_id=0
    ).to(device)
    fincast_cont_criterion = nn.MSELoss()
    fincast_cont_optimizer = optim.Adam(fincast_cont_model.parameters(), lr=1e-4)
    return (
        fincast_bin_criterion,
        fincast_bin_model,
        fincast_bin_optimizer,
        fincast_cont_criterion,
        fincast_cont_model,
        fincast_cont_optimizer,
    )


@app.cell
def _(
    device,
    fincast_bin_criterion,
    fincast_bin_model,
    fincast_bin_optimizer,
    fincast_cont_criterion,
    fincast_cont_model,
    fincast_cont_optimizer,
    test_loader_bin,
    test_loader_cont,
    train_generic,
    train_loader_bin,
    train_loader_cont,
):
    train_generic(
        fincast_bin_model,
        train_loader_bin,
        test_loader_bin,
        fincast_bin_criterion,
        fincast_bin_optimizer,
        device,
        task="binary",
        epochs=20,
        patience=5,
    )

    train_generic(
        fincast_cont_model,
        train_loader_cont,
        test_loader_cont,
        fincast_cont_criterion,
        fincast_cont_optimizer,
        device,
        task="continuous",
        epochs=20,
        patience=5,
    )
    return


@app.cell
def _(
    device,
    evaluate_torch_model,
    fincast_bin_model,
    fincast_cont_model,
    test_loader_bin,
    test_loader_cont,
):
    metrics_fincast_binary = evaluate_torch_model(
        fincast_bin_model, test_loader_bin, device, task="binary"
     )
    print("Binary FinCast metrics:")
    print(metrics_fincast_binary)

    metrics_fincast_continuous = evaluate_torch_model(
        fincast_cont_model, test_loader_cont, device, task="continuous"
     )
    print("Continuous FinCast metrics:")
    print(metrics_fincast_continuous)
    return


if __name__ == "__main__":
    app.run()

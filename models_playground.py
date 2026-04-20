# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.21.1",
#     "numpy==2.4.3",
#     "polars==1.39.3",
#     "pyzmq>=27.1.0",
#     "scikit-learn==1.8.0",
#     "scipy>=1.13",
#     "statsmodels==0.14.6",
#     "torch==2.10.0",
# ]
# ///

import marimo

__generated_with = "0.22.4"
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
        get_bag_of_words,
        get_category_features,
        combine_numerical_and_text_data,
    )
    from models.fincast import (
        BinaryFinCast,
        BinaryFinCastConfig,
        ContinuousFinCast,
        ContinuousFinCastConfig,
    )
    from models.evaluation import evaluate_predictions
    from models.lstm import LSTMModel
    from models.sarimax import forecast_sarimax, rolling_one_step_forecast, train_sarimax

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn import set_config

    from statsmodels.tsa.statespace.sarimax import SARIMAXResults

    return (
        BinaryFinCast,
        BinaryFinCastConfig,
        ContinuousFinCast,
        ContinuousFinCastConfig,
        LSTMModel,
        PCA,
        Pipeline,
        StandardScaler,
        augment_dataset,
        combine_numerical_and_text_data,
        downsample_to_interval,
        evaluate_predictions,
        forecast_sarimax,
        make_dataloader,
        nn,
        np,
        optim,
        pl,
        prepare_arima_data,
        rolling_one_step_forecast,
        set_config,
        time_train_test_split,
        torch,
        train_sarimax,
    )


@app.cell
def _(set_config):
    set_config(transform_output="polars")
    return


@app.cell
def _(augment_dataset, downsample_to_interval, pl):
    CSV_PATH = "./data/1_min_SPY_2008-2021.csv"
    df = pl.read_csv(CSV_PATH)
    df = downsample_to_interval(df, interval="30m")
    df_augmented = augment_dataset(df)


    dictionary = pl.read_csv("data/final_data/dictionary/cleaned_dict.csv")
    category_text_data = pl.read_parquet("data/final_data/train/bow_2stages_30m.parquet")
    return category_text_data, df_augmented


@app.cell
def _(category_text_data, combine_numerical_and_text_data, df_augmented):
    combined_data = combine_numerical_and_text_data(df_augmented, category_text_data)
    return (combined_data,)


@app.cell
def _(combined_data):
    data_len = combined_data.height
    step = int(data_len * 0.2)
    return data_len, step


@app.cell
def _(data_len, step):
    data_len, step
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SARIMAX pipelines (continuous and binary)
    """)
    return


@app.cell
def _():
    sarimax_model_cont_path = "data/models_checkpoints/sarmix_cont_3"
    sarimax_model_bin_path = "data/models_checkpoints/sarmix_bin_3"
    return sarimax_model_bin_path, sarimax_model_cont_path


@app.cell
def _(PCA, Pipeline, StandardScaler):
    feature_processor = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50))])
    return (feature_processor,)


@app.cell
def _(
    combined_data,
    data_len,
    evaluate_predictions,
    feature_processor,
    prepare_arima_data,
    rolling_one_step_forecast,
    sarimax_model_cont_path,
    step,
    train_sarimax,
):
    for train_end_idx in range(step, data_len, step):

        if data_len - train_end_idx < 1000:
            continue

        train_part = combined_data.slice(0, train_end_idx)
        test_part = combined_data.slice(train_end_idx, step)


        print("Step", train_end_idx, step)

        y_train_cont, X_train_cont = prepare_arima_data(train_part, task="continuous")
        y_test_cont, X_test_cont = prepare_arima_data(test_part, task="continuous")

        X_train_transformed = feature_processor.fit_transform(X_train_cont)
        X_test_transformed = feature_processor.transform(X_test_cont)

        sarimax_cont = train_sarimax(y_train_cont, X_train_transformed, order=(3, 0, 3), disp=10)

        sarimax_cont.save(sarimax_model_cont_path)

        pred_cont = rolling_one_step_forecast(sarimax_cont, y_test_cont, X_test_transformed)

        metrics_sarimax_cont = evaluate_predictions(y_test_cont.to_numpy(), pred_cont, task="continuous")

        print("SARIMAX continuous metrics:")
        print(metrics_sarimax_cont)
    return


@app.cell
def _(
    combined_data,
    data_len,
    evaluate_predictions,
    feature_processor,
    prepare_arima_data,
    rolling_one_step_forecast,
    sarimax_model_bin_path,
    step,
    train_sarimax,
):
    for train_end_idx_bin in range(step, data_len, step):

        if data_len - train_end_idx_bin < 1000:
            continue

        train_part_bin = combined_data.slice(0, train_end_idx_bin)
        test_part_bin = combined_data.slice(train_end_idx_bin, train_end_idx_bin + step)

        print("Step", train_end_idx_bin, step)

        y_train_bin, X_train_bin = prepare_arima_data(train_part_bin, task="binary")
        y_test_bin, X_test_bin = prepare_arima_data(test_part_bin, task="binary")

        X_train_transformed_bin = feature_processor.fit_transform(X_train_bin)
        X_test_transformed_bin = feature_processor.transform(X_test_bin)

        sarimax_bin = train_sarimax(y_train_bin, X_train_transformed_bin, order=(3, 0, 3), disp=15)

        sarimax_bin.save(sarimax_model_bin_path)

        pred_bin = rolling_one_step_forecast(sarimax_bin, y_test_bin, X_test_transformed_bin)

        metrics_sarimax_bin = evaluate_predictions(y_test_bin.to_numpy(), pred_bin, task="binary")
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
def _(combined_data, time_train_test_split):
    train_df, test_df = time_train_test_split(combined_data, test_ratio=0.2)
    print(f"DL split – train: {train_df.height} rows, test: {test_df.height} rows")
    return test_df, train_df


@app.cell
def _(make_dataloader, test_df, torch, train_df):
    window_size = 32
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

    train_loader_bin = make_dataloader(
        train_df, task="binary", window_size=window_size, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader_bin = make_dataloader(
        test_df, task="binary", window_size=window_size, batch_size=batch_size, shuffle=False, drop_last=False,
        scaler=train_loader_bin.dataset.scaler,
    )

    train_loader_cont = make_dataloader(
        train_df, task="continuous", window_size=window_size, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader_cont = make_dataloader(
        test_df, task="continuous", window_size=window_size, batch_size=batch_size, shuffle=False, drop_last=False,
        scaler=train_loader_cont.dataset.scaler,
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
def _(
    combined_data,
    data_len,
    evaluate_predictions,
    feature_processor,
    prepare_arima_data,
    rolling_one_step_forecast,
    sarimax_model_bin_path,
    step,
    train_sarimax,
):
    # NOTE: this cell is a duplicate of the binary SARIMAX block above –
    # kept for backwards compatibility with previously cached marimo state.
    for train_end_idx_bin in range(step, data_len, step):

        if data_len - train_end_idx_bin < 1000:
            continue

        train_part_bin = combined_data.slice(0, train_end_idx_bin)
        test_part_bin = combined_data.slice(train_end_idx_bin, train_end_idx_bin + step)

        print("Step", train_end_idx_bin, step)

        y_train_bin, X_train_bin = prepare_arima_data(train_part_bin, task="binary")
        y_test_bin, X_test_bin = prepare_arima_data(test_part_bin, task="binary")

        X_train_transformed_bin = feature_processor.fit_transform(X_train_bin)
        X_test_transformed_bin = feature_processor.transform(X_test_bin)

        sarimax_bin = train_sarimax(y_train_bin, X_train_transformed_bin, order=(3, 0, 3), disp=15)

        sarimax_bin.save(sarimax_model_bin_path)

        pred_bin = rolling_one_step_forecast(sarimax_bin, y_test_bin, X_test_transformed_bin)

        metrics_sarimax_bin = evaluate_predictions(y_test_bin.to_numpy(), pred_bin, task="binary")
        print("SARIMAX binary metrics:")
        print(metrics_sarimax_bin)
    return


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
        epochs=40,
        patience=15,
    )
    return (lstm_binary,)


@app.cell
def _(lstm_binary, torch):
    torch.save(lstm_binary.state_dict(), "data/models_checkpoints/lstm_binary_2")
    return


@app.cell
def _(LSTMModel, device, input_dim, torch):
    lstm_binary_trained = LSTMModel(input_dim=input_dim, hidden_dim=64, layer_dim=2, output_dim=1).to(device)
    lstm_binary_trained.load_state_dict(torch.load("data/models_checkpoints/lstm_binary_2", weights_only=True))
    return


@app.cell
def _(device, evaluate_torch_model, lstm_binary, test_loader_bin):
    metrics_lstm_binary = evaluate_torch_model(lstm_binary, test_loader_bin, device, task="binary")
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
        epochs=40,
        patience=15,
    )
    return (lstm_continuous,)


@app.cell
def _(lstm_continuous, torch):
    torch.save(lstm_continuous.state_dict(), "data/models_checkpoints/lstm_continuous_2")
    return


@app.cell
def _(LSTMModel, device, input_dim, torch):
    lstm_continuous_trained = LSTMModel(input_dim=input_dim, hidden_dim=64, layer_dim=2, output_dim=1).to(device)
    lstm_continuous_trained.load_state_dict(torch.load("data/models_checkpoints/lstm_continuous_2", weights_only=True))
    return (lstm_continuous_trained,)


@app.cell
def _(device, evaluate_torch_model, lstm_continuous_trained, test_loader_cont):
    metrics_lstm_continuous = evaluate_torch_model(lstm_continuous_trained, test_loader_cont, device, task="continuous")
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
            freq_id = torch.full((x.size(0),), self.freq_id, dtype=torch.long, device=x.device)
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
    fincast_bin_model = FinCastAdapter(fincast_bin_backbone, output_key="logits", freq_id=0).to(device)
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
    fincast_cont_model = FinCastAdapter(fincast_cont_backbone, output_key="prediction", freq_id=0).to(device)
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
    metrics_fincast_binary = evaluate_torch_model(fincast_bin_model, test_loader_bin, device, task="binary")
    print("Binary FinCast metrics:")
    print(metrics_fincast_binary)

    metrics_fincast_continuous = evaluate_torch_model(fincast_cont_model, test_loader_cont, device, task="continuous")
    print("Continuous FinCast metrics:")
    print(metrics_fincast_continuous)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

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
#     "tqdm==4.67.3",
# ]
# ///

import marimo

__generated_with = "0.23.4"
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
        combine_numerical_and_bert_embeddings,
        split_text_embeddings_and_features,
    )
    from models.fincast import (
        BinaryFinCast,
        BinaryFinCastConfig,
        ContinuousFinCast,
        ContinuousFinCastConfig,
    )
    from models.evaluation import evaluate_predictions
    from models.lstm import LSTMModel, rolling_one_step_forecast_lstm
    from models.sarimax import forecast_sarimax, rolling_one_step_forecast, train_sarimax

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn import set_config

    from models.vae import encode_with_vae, train_and_encode_vae_dataframe, VAEConfig

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
        VAEConfig,
        augment_dataset,
        combine_numerical_and_bert_embeddings,
        combine_numerical_and_text_data,
        downsample_to_interval,
        encode_with_vae,
        evaluate_predictions,
        make_dataloader,
        nn,
        np,
        optim,
        pl,
        prepare_arima_data,
        rolling_one_step_forecast,
        rolling_one_step_forecast_lstm,
        set_config,
        split_features_target,
        split_text_embeddings_and_features,
        time_train_test_split,
        torch,
        train_and_encode_vae_dataframe,
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
    encoded_text = pl.read_parquet(
        "data/final_data/train/text_encoder_embeddings_30m.parquet"
    )
    return category_text_data, df_augmented, encoded_text


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


@app.cell
def _(data_len):
    train_size = int(data_len * 0.8)
    return (train_size,)


@app.cell
def _(train_size):
    train_size
    return


@app.cell
def _(combine_numerical_and_bert_embeddings, df_augmented, encoded_text):
    combined_with_encoded_text = combine_numerical_and_bert_embeddings(
        df_augmented, encoded_text
    )
    return (combined_with_encoded_text,)


@app.cell
def _(combined_data):
    combined_data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SARIMAX pipelines (continuous and binary)
    """)
    return


@app.cell
def _(
    StandardScaler,
    df_augmented,
    evaluate_predictions,
    prepare_arima_data,
    rolling_one_step_forecast,
    train_sarimax,
    train_size,
):
    def train_sarima_numeric_only(model_path: str, task: str):
        df_augmented_wo_date = df_augmented.drop("date")

        train_part = df_augmented_wo_date.head(train_size)
        test_part = df_augmented_wo_date.tail(df_augmented_wo_date.height - train_size)

        y_train, X_train = prepare_arima_data(train_part, task=task)
        y_test, X_test = prepare_arima_data(test_part, task=task)

        standard_scaler = StandardScaler()
        X_train = standard_scaler.fit_transform(X_train)

        X_test = standard_scaler.transform(X_test)

        sarimax = train_sarimax(y_train, X_train, order=(3, 0, 3), disp=7)

        sarimax.save(f"{model_path}_{task}_80_20")

        pred_cont = rolling_one_step_forecast(sarimax, y_test, X_test)

        metrics_sarimax = evaluate_predictions(y_test.to_numpy(), pred_cont, task=task)

        print(f"SARIMAX no text {task} metrics:")
        print(metrics_sarimax)

    return


@app.cell
def _():
    # train_sarima_numeric_only("data/models_checkpoints/sarimax_baseline", task="continuous") # already run
    return


@app.cell
def _():
    # train_sarima_numeric_only("data/models_checkpoints/sarimax_baseline", task="binary")
    return


@app.cell
def _(
    PCA,
    Pipeline,
    StandardScaler,
    combined_data,
    evaluate_predictions,
    prepare_arima_data,
    rolling_one_step_forecast,
    train_sarimax,
    train_size,
):
    def train_sarima_categories_pca(model_path: str, task: str):
        train_part = combined_data.head(train_size)
        test_part = combined_data.tail(combined_data.height - train_size)

        y_train, X_train = prepare_arima_data(train_part, task=task)
        y_test, X_test = prepare_arima_data(test_part, task=task)

        feature_processor = Pipeline(
            [("scaler", StandardScaler()), ("pca", PCA(n_components=50))]
        )

        X_train = feature_processor.fit_transform(X_train)

        X_test = feature_processor.transform(X_test)

        sarimax = train_sarimax(y_train, X_train, order=(3, 0, 3), disp=7)

        sarimax.save(f"{model_path}_{task}_window__80_20")

        pred_cont = rolling_one_step_forecast(sarimax, y_test, X_test)

        metrics_sarimax = evaluate_predictions(y_test.to_numpy(), pred_cont, task=task)

        print(f"SARIMAX categories pca {task} metrics:")
        print(metrics_sarimax)

    return


@app.cell
def _():
    # train_sarima_categories_pca("data/models_checkpoints/sarima_cats_pca", task="binary")
    return


@app.cell
def _():
    # train_sarima_categories_pca("data/models_checkpoints/sarima_cats_pca", task="continuous")
    return


@app.cell
def _(VAEConfig):
    vae_config_cats = VAEConfig(
        input_dim=43,
        latent_dim=16,
        hidden_dims=(128, 64),
    )
    return (vae_config_cats,)


@app.cell
def _(
    StandardScaler,
    combined_data,
    df_augmented,
    encode_with_vae,
    evaluate_predictions,
    pl,
    prepare_arima_data,
    rolling_one_step_forecast,
    train_and_encode_vae_dataframe,
    train_sarimax,
    train_size,
    vae_config_cats,
):
    def train_sarima_categories_vae(model_path: str, task: str):
        df_augmented_wo_date = df_augmented.drop("date")

        train_part = combined_data.head(train_size)
        test_part = combined_data.tail(combined_data.height - train_size)

        y_train, X_train = prepare_arima_data(train_part, task=task)
        y_test, X_test = prepare_arima_data(test_part, task=task)

        textual_columns = [
            column_name for column_name in X_train.columns if "Stage" in column_name
        ]
        name_mapping = {
            col_name: f"text_embed_{i}" for i, col_name in enumerate(textual_columns)
        }

        X_train_numeric, X_train_text = (
            X_train.drop(textual_columns),
            X_train.select(textual_columns).rename(name_mapping),
        )
        X_test_numeric, X_test_text = (
            X_test.drop(textual_columns),
            X_test.select(textual_columns).rename(name_mapping),
        )
        print(X_train_text)

        standard_scaler = StandardScaler()
        X_train_numeric = standard_scaler.fit_transform(X_train_numeric)
        X_test_numeric = standard_scaler.transform(X_test_numeric)

        model, train_reduced = train_and_encode_vae_dataframe(
            data=X_train_text, config=vae_config_cats
        )
        test_reduced = encode_with_vae(model, X_test_text)

        X_train = pl.concat([X_train_numeric, train_reduced], how="horizontal")
        X_test = pl.concat([X_test_numeric, test_reduced], how="horizontal")

        sarimax = train_sarimax(y_train, X_train, order=(3, 0, 3), disp=7)

        sarimax.save(f"{model_path}_{task}_window__80_20")

        pred_cont = rolling_one_step_forecast(sarimax, y_test, X_test)

        metrics_sarimax = evaluate_predictions(y_test.to_numpy(), pred_cont, task=task)

        print(f"SARIMAX categories vae {task} metrics:")
        print(metrics_sarimax)

    return


@app.cell
def _(
    PCA,
    StandardScaler,
    combined_with_encoded_text,
    evaluate_predictions,
    pl,
    prepare_arima_data,
    rolling_one_step_forecast,
    split_text_embeddings_and_features,
    train_sarimax,
    train_size,
):
    def train_sarima_embedings_pca(model_path: str, task: str):
        train_part = combined_with_encoded_text.head(train_size)
        test_part = combined_with_encoded_text.tail(
            combined_with_encoded_text.height - train_size
        )

        y_train, X_train = prepare_arima_data(train_part, task=task)
        y_test, X_test = prepare_arima_data(test_part, task=task)

        train_part_num, train_part_text = split_text_embeddings_and_features(X_train)
        test_part_num, test_part_text = split_text_embeddings_and_features(X_test)

        pca = PCA(n_components=30)
        train_part_text = pca.fit_transform(train_part_text)

        standard_scaler = StandardScaler()
        train_part_num = standard_scaler.fit_transform(train_part_num)
        train_part = pl.concat([train_part_num, train_part_text], how="horizontal")

        test_part_text = pca.transform(test_part_text)
        test_part_num = standard_scaler.transform(test_part_num)

        test_part = pl.concat([test_part_num, test_part_text], how="horizontal")

        sarimax = train_sarimax(y_train, train_part, order=(3, 0, 3), disp=7)

        sarimax.save(f"{model_path}_{task}_window__80_20")

        pred_cont = rolling_one_step_forecast(sarimax, y_test, test_part)

        metrics_sarimax = evaluate_predictions(y_test.to_numpy(), pred_cont, task=task)

        print(f"SARIMAX embedings pca {task} metrics:")
        print(metrics_sarimax)

    return (train_sarima_embedings_pca,)


@app.cell
def _(train_sarima_embedings_pca):
    train_sarima_embedings_pca(
        "data/models_checkpoints/sarimax_embedings_pca", "continuous"
    )
    return


@app.cell
def _(train_sarima_embedings_pca):
    train_sarima_embedings_pca("data/models_checkpoints/sarimax_embedings_pca", "binary")
    return


@app.cell
def _(VAEConfig):
    vae_config = VAEConfig(
        input_dim=768,
        latent_dim=16,
        hidden_dims=(256, 128),
    )
    return (vae_config,)


@app.cell
def _(
    StandardScaler,
    combined_with_encoded_text,
    encode_with_vae,
    evaluate_predictions,
    pl,
    prepare_arima_data,
    rolling_one_step_forecast,
    split_text_embeddings_and_features,
    train_and_encode_vae_dataframe,
    train_sarimax,
    train_size,
    vae_config,
):
    def train_sarima_embedings_vae(model_path: str, task: str):

        train_part_vae = combined_with_encoded_text.head(train_size)
        test_part_vae = combined_with_encoded_text.tail(
            combined_with_encoded_text.height - train_size
        )

        y_train, X_train = prepare_arima_data(train_part_vae, task=task)
        y_test, X_test = prepare_arima_data(test_part_vae, task=task)

        train_part_vae_num, train_part_vae_text = split_text_embeddings_and_features(
            X_train
        )
        test_part_vae_num, test_part_vae_text = split_text_embeddings_and_features(X_test)

        model, train_encoded = train_and_encode_vae_dataframe(
            data=train_part_vae_text, config=vae_config, text_embed_only=True
        )

        standard_scaler = StandardScaler()
        train_part_vae_num = standard_scaler.fit_transform(train_part_vae_num)
        train_part_vae = pl.concat([train_part_vae_num, train_encoded], how="horizontal")

        test_encoded = encode_with_vae(model, test_part_vae_text)
        test_part_vae_num = standard_scaler.transform(test_part_vae_num)

        test_part_vae = pl.concat([test_part_vae_num, test_encoded], how="horizontal")

        sarimax = train_sarimax(y_train, train_part_vae, order=(3, 0, 3))

        sarimax.save(f"{model_path}_{task}_window__80_20")

        pred_cont = rolling_one_step_forecast(sarimax, y_test, test_part_vae)

        metrics_sarimax = evaluate_predictions(y_test.to_numpy(), pred_cont, task=task)

        print(f"SARIMAX embedings vae {task} metrics:")
        print(metrics_sarimax)

    return (train_sarima_embedings_vae,)


@app.cell
def _(train_sarima_embedings_vae):
    train_sarima_embedings_vae(
        model_path="data/models_checkpoints/sarimax_embedings_vae", task="continuous"
    )
    return


@app.cell
def _(train_sarima_embedings_vae):
    train_sarima_embedings_vae(
        model_path="data/models_checkpoints/sarimax_embedings_vae", task="binary"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deep learning pipelines (binary and continuous)
    """)
    return


@app.cell
def _(torch):
    window_size = 32
    batch_size = 516
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cuda"
    )
    return batch_size, device, window_size


@app.cell
def _(np, pl):
    def to_numpy(matrix):
        if isinstance(matrix, pl.DataFrame):
            return np.nan_to_num(
                matrix.to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
        return np.nan_to_num(
            np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )

    return (to_numpy,)


@app.cell
def _(
    LSTMModel,
    PCA,
    Pipeline,
    StandardScaler,
    batch_size,
    combined_data,
    combined_with_encoded_text,
    device,
    df_augmented,
    encode_with_vae,
    evaluate_lstm_rolling,
    make_dataloader,
    nn,
    np,
    optim,
    split_features_target,
    split_text_embeddings_and_features,
    time_train_test_split,
    to_numpy,
    torch,
    train_and_encode_vae_dataframe,
    train_generic,
    vae_config,
    vae_config_cats,
    window_size,
):
    ## LSTM pipelines (mirroring SARIMAX preprocessing: numeric-only, PCA, VAE)


    def train_lstm_numeric_only(model_path: str, task: str = "binary"):
        train_part, test_part = time_train_test_split(df_augmented, test_ratio=0.2)

        X_train, y_train = split_features_target(train_part, task=task)
        X_test, y_test = split_features_target(test_part, task=task)

        # Scale features only
        scaler = StandardScaler()
        X_train_scaled = to_numpy(scaler.fit_transform(X_train))
        X_test_scaled = to_numpy(scaler.transform(X_test))

        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        y_train_np = y_train.to_numpy().astype(np.float32)
        y_test_np = y_test.to_numpy().astype(np.float32)

        train_loader = make_dataloader(
            X=X_train_scaled,
            y=y_train_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        test_loader = make_dataloader(
            X=X_test_scaled,
            y=y_test_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        input_dim_lstm = X_train_scaled.shape[1]

        if task == "binary":
            model = LSTMModel(
                input_dim=input_dim_lstm, hidden_dim=256, layer_dim=5, output_dim=1
            ).to(device)
            criterion = nn.BCEWithLogitsLoss()
        else:
            model = LSTMModel(
                input_dim=input_dim_lstm, hidden_dim=256, layer_dim=5, output_dim=1
            ).to(device)
            criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        train_generic(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            task=task,
            epochs=40,
            patience=15,
        )

        torch.save(model.state_dict(), f"{model_path}_{task}")

        X_forecast = np.vstack([X_train_scaled[-window_size:], X_test_scaled])
        y_forecast = np.concatenate([y_train_np[-window_size:], y_test_np])

        metrics_lstm = evaluate_lstm_rolling(
            model,
            X_forecast,
            y_forecast,
            window_size=window_size,
            device=device,
            task=task,
        )
        print(f"LSTM numeric only {task} metrics:")
        print(metrics_lstm)


    def train_lstm_categories_pca(model_path: str, task: str = "binary"):
        """Train LSTM on combined data (numeric + text) with PCA dimension reduction."""
        train_part, test_part = time_train_test_split(combined_data, test_ratio=0.2)

        X_train, y_train = split_features_target(train_part, task=task)
        X_test, y_test = split_features_target(test_part, task=task)

        # Apply scaling + PCA to features only
        feature_processor = Pipeline(
            [("scaler", StandardScaler()), ("pca", PCA(n_components=50))]
        )
        X_train_processed = to_numpy(feature_processor.fit_transform(X_train))
        X_test_processed = to_numpy(feature_processor.transform(X_test))

        y_train_np = y_train.to_numpy().astype(np.float32)
        y_test_np = y_test.to_numpy().astype(np.float32)

        train_loader = make_dataloader(
            X=X_train_processed,
            y=y_train_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        test_loader = make_dataloader(
            X=X_test_processed,
            y=y_test_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        input_dim_lstm = X_train_processed.shape[1]

        if task == "binary":
            model = LSTMModel(
                input_dim=input_dim_lstm, hidden_dim=256, layer_dim=5, output_dim=1
            ).to(device)
            criterion = nn.BCEWithLogitsLoss()
        else:
            model = LSTMModel(
                input_dim=input_dim_lstm, hidden_dim=256, layer_dim=5, output_dim=1
            ).to(device)
            criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        train_generic(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            task=task,
            epochs=40,
            patience=15,
        )

        torch.save(model.state_dict(), f"{model_path}_{task}")

        metrics_lstm = evaluate_lstm_rolling(
            model,
            X_test_processed,
            y_test_np,
            window_size=window_size,
            device=device,
            task=task,
        )
        print(f"LSTM categories PCA {task} metrics:")
        print(metrics_lstm)


    def train_lstm_categories_vae(model_path: str, task: str = "binary"):
        """Train LSTM on combined data with VAE encoding of text features."""
        train_part, test_part = time_train_test_split(combined_data, test_ratio=0.2)

        X_train, y_train = split_features_target(train_part, task=task)
        X_test, y_test = split_features_target(test_part, task=task)

        # Split numeric and text features
        textual_columns = [
            column_name for column_name in X_train.columns if "Stage" in column_name
        ]
        name_mapping = {
            col_name: f"text_embed_{i}" for i, col_name in enumerate(textual_columns)
        }

        X_train_numeric = X_train.drop(textual_columns)
        X_train_text = X_train.select(textual_columns).rename(name_mapping)
        X_test_numeric = X_test.drop(textual_columns)
        X_test_text = X_test.select(textual_columns).rename(name_mapping)

        # Scale numeric features only
        scaler_numeric = StandardScaler()
        X_train_numeric_scaled = to_numpy(scaler_numeric.fit_transform(X_train_numeric))
        X_test_numeric_scaled = to_numpy(scaler_numeric.transform(X_test_numeric))

        # Encode text with VAE
        vae_model, X_train_text_encoded = train_and_encode_vae_dataframe(
            data=X_train_text, config=vae_config_cats
        )
        X_test_text_encoded = encode_with_vae(vae_model, X_test_text)

        # Combine numeric and encoded text
        X_train_combined = np.hstack(
            [X_train_numeric_scaled, X_train_text_encoded.to_numpy().astype(np.float32)]
        )
        X_test_combined = np.hstack(
            [X_test_numeric_scaled, X_test_text_encoded.to_numpy().astype(np.float32)]
        )

        y_train_np = y_train.to_numpy().astype(np.float32)
        y_test_np = y_test.to_numpy().astype(np.float32)

        train_loader = make_dataloader(
            X=X_train_combined,
            y=y_train_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        test_loader = make_dataloader(
            X=X_test_combined,
            y=y_test_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        input_dim_lstm = X_train_combined.shape[1]

        if task == "binary":
            model = LSTMModel(
                input_dim=input_dim_lstm, hidden_dim=256, layer_dim=5, output_dim=1
            ).to(device)
            criterion = nn.BCEWithLogitsLoss()
        else:
            model = LSTMModel(
                input_dim=input_dim_lstm, hidden_dim=256, layer_dim=5, output_dim=1
            ).to(device)
            criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        train_generic(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            task=task,
            epochs=40,
            patience=15,
        )

        torch.save(model.state_dict(), f"{model_path}_{task}")

        metrics_lstm = evaluate_lstm_rolling(
            model,
            X_test_combined,
            y_test_np,
            window_size=window_size,
            device=device,
            task=task,
        )
        print(f"LSTM categories VAE {task} metrics:")
        print(metrics_lstm)


    def train_lstm_embeddings_pca(model_path: str, task: str = "binary"):
        """Train LSTM on BERT embeddings with PCA dimension reduction."""
        train_part, test_part = time_train_test_split(
            combined_with_encoded_text, test_ratio=0.2
        )

        X_train, y_train = split_features_target(train_part, task=task)
        X_test, y_test = split_features_target(test_part, task=task)

        # Split numeric and text embeddings
        X_train_numeric, X_train_text = split_text_embeddings_and_features(X_train)
        X_test_numeric, X_test_text = split_text_embeddings_and_features(X_test)

        X_train_numeric_np = X_train_numeric.to_numpy().astype(np.float32)
        X_train_text_np = X_train_text.to_numpy().astype(np.float32)
        X_test_numeric_np = X_test_numeric.to_numpy().astype(np.float32)
        X_test_text_np = X_test_text.to_numpy().astype(np.float32)

        # Apply PCA to text embeddings only
        pca = PCA(n_components=30)
        X_train_text_pca = pca.fit_transform(X_train_text_np)
        X_test_text_pca = pca.transform(X_test_text_np)

        # Scale numeric features only
        scaler_numeric = StandardScaler()
        X_train_numeric_scaled = to_numpy(scaler_numeric.fit_transform(X_train_numeric_np))
        X_test_numeric_scaled = to_numpy(scaler_numeric.transform(X_test_numeric_np))

        # Combine numeric and PCA-reduced text
        X_train_combined = np.hstack([X_train_numeric_scaled, X_train_text_pca])
        X_test_combined = np.hstack([X_test_numeric_scaled, X_test_text_pca])

        y_train_np = y_train.to_numpy().astype(np.float32)
        y_test_np = y_test.to_numpy().astype(np.float32)

        train_loader = make_dataloader(
            X=X_train_combined,
            y=y_train_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        test_loader = make_dataloader(
            X=X_test_combined,
            y=y_test_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        input_dim_lstm = X_train_combined.shape[1]

        if task == "binary":
            model = LSTMModel(
                input_dim=input_dim_lstm, hidden_dim=256, layer_dim=5, output_dim=1
            ).to(device)
            criterion = nn.BCEWithLogitsLoss()
        else:
            model = LSTMModel(
                input_dim=input_dim_lstm, hidden_dim=256, layer_dim=5, output_dim=1
            ).to(device)
            criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        train_generic(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            task=task,
            epochs=40,
            patience=15,
        )

        torch.save(model.state_dict(), f"{model_path}_{task}")

        metrics_lstm = evaluate_lstm_rolling(
            model,
            X_test_combined,
            y_test_np,
            window_size=window_size,
            device=device,
            task=task,
        )
        print(f"LSTM embeddings PCA {task} metrics:")
        print(metrics_lstm)


    def train_lstm_embeddings_vae(model_path: str, task: str = "binary"):
        """Train LSTM on BERT embeddings with VAE encoding."""
        train_part, test_part = time_train_test_split(
            combined_with_encoded_text, test_ratio=0.2
        )

        X_train, y_train = split_features_target(train_part, task=task)
        X_test, y_test = split_features_target(test_part, task=task)

        # Split numeric and text embeddings
        X_train_numeric, X_train_text = split_text_embeddings_and_features(X_train)
        X_test_numeric, X_test_text = split_text_embeddings_and_features(X_test)

        X_train_numeric_np = X_train_numeric.to_numpy().astype(np.float32)
        X_test_numeric_np = X_test_numeric.to_numpy().astype(np.float32)

        # Scale numeric features only
        scaler_numeric = StandardScaler()
        X_train_numeric_scaled = to_numpy(scaler_numeric.fit_transform(X_train_numeric_np))
        X_test_numeric_scaled = to_numpy(scaler_numeric.transform(X_test_numeric_np))

        # Encode embeddings with VAE
        vae_model, X_train_text_vae = train_and_encode_vae_dataframe(
            data=X_train_text, config=vae_config, text_embed_only=True
        )
        X_test_text_vae = encode_with_vae(vae_model, X_test_text)

        # Combine numeric and VAE-encoded text
        X_train_combined = np.hstack(
            [X_train_numeric_scaled, X_train_text_vae.to_numpy().astype(np.float32)]
        )
        X_test_combined = np.hstack(
            [X_test_numeric_scaled, X_test_text_vae.to_numpy().astype(np.float32)]
        )

        y_train_np = y_train.to_numpy().astype(np.float32)
        y_test_np = y_test.to_numpy().astype(np.float32)

        train_loader = make_dataloader(
            X=X_train_combined,
            y=y_train_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        test_loader = make_dataloader(
            X=X_test_combined,
            y=y_test_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        input_dim_lstm = X_train_combined.shape[1]

        if task == "binary":
            model = LSTMModel(
                input_dim=input_dim_lstm, hidden_dim=256, layer_dim=5, output_dim=1
            ).to(device)
            criterion = nn.BCEWithLogitsLoss()
        else:
            model = LSTMModel(
                input_dim=input_dim_lstm, hidden_dim=256, layer_dim=5, output_dim=1
            ).to(device)
            criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        train_generic(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            task=task,
            epochs=40,
            patience=15,
        )

        torch.save(model.state_dict(), f"{model_path}_{task}")

        metrics_lstm = evaluate_lstm_rolling(
            model,
            X_test_combined,
            y_test_np,
            window_size=window_size,
            device=device,
            task=task,
        )
        print(f"LSTM embeddings VAE {task} metrics:")
        print(metrics_lstm)


    return (
        train_lstm_categories_pca,
        train_lstm_embeddings_pca,
        train_lstm_embeddings_vae,
        train_lstm_numeric_only,
    )


@app.cell
def _(train_lstm_embeddings_vae):
    # Example: Train LSTM with BERT embeddings + VAE
    train_lstm_embeddings_vae("data/models_checkpoints/lstm_embeddings_vae", task="binary")
    train_lstm_embeddings_vae(
        "data/models_checkpoints/lstm_embeddings_vae", task="continuous"
    )
    return


@app.cell
def _(train_lstm_embeddings_pca):
    # Example: Train LSTM with BERT embeddings + PCA
    train_lstm_embeddings_pca("data/models_checkpoints/lstm_embeddings_pca", task="binary")
    train_lstm_embeddings_pca(
        "data/models_checkpoints/lstm_embeddings_pca", task="continuous"
    )
    return


@app.cell
def _():
    # Example: Train LSTM with categories + VAE
    # train_lstm_categories_vae("data/models_checkpoints/lstm_categories_vae", task="binary")
    # train_lstm_categories_vae(
    # "data/models_checkpoints/lstm_categories_vae", task="continuous"
    # )
    return


@app.cell
def _(train_lstm_categories_pca):
    # Example: Train LSTM with categories + PCA
    train_lstm_categories_pca("data/models_checkpoints/lstm_categories_pca", task="binary")
    train_lstm_categories_pca(
        "data/models_checkpoints/lstm_categories_pca", task="continuous"
    )
    return


@app.cell
def _(train_lstm_numeric_only):
    # Example: Train LSTM numeric only (continuous task)
    train_lstm_numeric_only("data/models_checkpoints/lstm_numeric", task="continuous")
    return


@app.cell
def _(train_lstm_numeric_only):
    # Example: Train LSTM numeric only (binary task)
    train_lstm_numeric_only("data/models_checkpoints/lstm_numeric", task="binary")
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
def _(evaluate_predictions, rolling_one_step_forecast_lstm):
    def evaluate_lstm_rolling(model, X_test, y_test, window_size, device, task="binary"):
        """Evaluate LSTM using rolling one-step-ahead forecast (proper time series evaluation)."""
        preds = rolling_one_step_forecast_lstm(
            model, X_test, y_test, window_size=window_size, device=device, task=task
        )
        return evaluate_predictions(y_test[window_size:], preds, task=task)


    def evaluate_fincast_rolling(
        model, X_test, y_test, window_size, device, task="binary"
     ):
        """Evaluate FinCast using rolling one-step-ahead forecast."""
        preds = rolling_one_step_forecast_lstm(
            model, X_test, y_test, window_size=window_size, device=device, task=task
        )
        return evaluate_predictions(y_test[window_size:], preds, task=task)

    return evaluate_fincast_rolling, evaluate_lstm_rolling


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


@app.cell
def _(
    BinaryFinCast,
    BinaryFinCastConfig,
    ContinuousFinCast,
    ContinuousFinCastConfig,
    FinCastAdapter,
    PCA,
    Pipeline,
    StandardScaler,
    batch_size,
    combined_data,
    combined_with_encoded_text,
    device,
    df_augmented,
    encode_with_vae,
    evaluate_fincast_rolling,
    make_dataloader,
    nn,
    np,
    optim,
    split_features_target,
    split_text_embeddings_and_features,
    time_train_test_split,
    to_numpy,
    torch,
    train_and_encode_vae_dataframe,
    train_generic,
    vae_config,
    vae_config_cats,
    window_size,
):
    class _InlineFinCastAdapter(nn.Module):
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


    def _make_fincast_model(task: str, input_dim: int):
        if task == "binary":
            config = BinaryFinCastConfig(
                input_dim=input_dim,
                patch_len=8,
                d_model=128,
                n_heads=4,
                n_layers=4,
                dropout=0.1,
            )
            backbone = BinaryFinCast(config).to(device)
            output_key = "logits"
            criterion = nn.BCEWithLogitsLoss()
            lr = 1e-4
        else:
            config = ContinuousFinCastConfig(
                input_dim=input_dim,
                patch_len=8,
                d_model=128,
                n_heads=4,
                n_layers=4,
                dropout=0.1,
            )
            backbone = ContinuousFinCast(config).to(device)
            output_key = "prediction"
            criterion = nn.MSELoss()
            lr = 1e-4

        adapter_cls = (
            FinCastAdapter if "FinCastAdapter" in globals() else _InlineFinCastAdapter
        )
        model = adapter_cls(backbone, output_key=output_key, freq_id=0).to(device)
        return model, criterion, lr


    def train_fincast_numeric_only(model_path: str, task: str = "binary"):
        train_part, test_part = time_train_test_split(df_augmented, test_ratio=0.2)

        X_train, y_train = split_features_target(train_part, task=task)
        X_test, y_test = split_features_target(test_part, task=task)

        scaler = StandardScaler()
        X_train_scaled = to_numpy(scaler.fit_transform(X_train))
        X_test_scaled = to_numpy(scaler.transform(X_test))

        y_train_np = y_train.to_numpy().astype(np.float32)
        y_test_np = y_test.to_numpy().astype(np.float32)

        train_loader = make_dataloader(
            X=X_train_scaled,
            y=y_train_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        test_loader = make_dataloader(
            X=X_test_scaled,
            y=y_test_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        model, criterion, lr = _make_fincast_model(task, X_train_scaled.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_generic(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            task=task,
            epochs=40,
            patience=15,
        )

        torch.save(model.state_dict(), f"{model_path}_{task}")

        X_forecast = np.vstack([X_train_scaled[-window_size:], X_test_scaled])
        y_forecast = np.concatenate([y_train_np[-window_size:], y_test_np])
        metrics_fincast = evaluate_fincast_rolling(
            model,
            X_forecast,
            y_forecast,
            window_size=window_size,
            device=device,
            task=task,
        )
        print(f"FinCast numeric only {task} metrics:")
        print(metrics_fincast)


    def train_fincast_categories_pca(model_path: str, task: str = "binary"):
        train_part, test_part = time_train_test_split(combined_data, test_ratio=0.2)

        X_train, y_train = split_features_target(train_part, task=task)
        X_test, y_test = split_features_target(test_part, task=task)

        feature_processor = Pipeline(
            [("scaler", StandardScaler()), ("pca", PCA(n_components=50))]
        )
        X_train_processed = to_numpy(feature_processor.fit_transform(X_train))
        X_test_processed = to_numpy(feature_processor.transform(X_test))

        y_train_np = y_train.to_numpy().astype(np.float32)
        y_test_np = y_test.to_numpy().astype(np.float32)

        train_loader = make_dataloader(
            X=X_train_processed,
            y=y_train_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        test_loader = make_dataloader(
            X=X_test_processed,
            y=y_test_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        model, criterion, lr = _make_fincast_model(task, X_train_processed.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_generic(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            task=task,
            epochs=40,
            patience=15,
        )

        torch.save(model.state_dict(), f"{model_path}_{task}")

        X_forecast = np.vstack([X_train_processed[-window_size:], X_test_processed])
        y_forecast = np.concatenate([y_train_np[-window_size:], y_test_np])
        metrics_fincast = evaluate_fincast_rolling(
            model,
            X_forecast,
            y_forecast,
            window_size=window_size,
            device=device,
            task=task,
        )
        print(f"FinCast categories PCA {task} metrics:")
        print(metrics_fincast)


    def train_fincast_categories_vae(model_path: str, task: str = "binary"):
        train_part, test_part = time_train_test_split(combined_data, test_ratio=0.2)

        X_train, y_train = split_features_target(train_part, task=task)
        X_test, y_test = split_features_target(test_part, task=task)

        textual_columns = [
            column_name for column_name in X_train.columns if "Stage" in column_name
        ]
        name_mapping = {
            col_name: f"text_embed_{i}" for i, col_name in enumerate(textual_columns)
        }

        X_train_numeric = X_train.drop(textual_columns)
        X_train_text = X_train.select(textual_columns).rename(name_mapping)
        X_test_numeric = X_test.drop(textual_columns)
        X_test_text = X_test.select(textual_columns).rename(name_mapping)

        scaler_numeric = StandardScaler()
        X_train_numeric_scaled = to_numpy(scaler_numeric.fit_transform(X_train_numeric))
        X_test_numeric_scaled = to_numpy(scaler_numeric.transform(X_test_numeric))

        vae_model, X_train_text_encoded = train_and_encode_vae_dataframe(
            data=X_train_text, config=vae_config_cats
        )
        X_test_text_encoded = encode_with_vae(vae_model, X_test_text)

        X_train_combined = np.hstack(
            [X_train_numeric_scaled, X_train_text_encoded.to_numpy().astype(np.float32)]
        )
        X_test_combined = np.hstack(
            [X_test_numeric_scaled, X_test_text_encoded.to_numpy().astype(np.float32)]
        )

        y_train_np = y_train.to_numpy().astype(np.float32)
        y_test_np = y_test.to_numpy().astype(np.float32)

        train_loader = make_dataloader(
            X=X_train_combined,
            y=y_train_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        test_loader = make_dataloader(
            X=X_test_combined,
            y=y_test_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        model, criterion, lr = _make_fincast_model(task, X_train_combined.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_generic(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            task=task,
            epochs=40,
            patience=15,
        )

        torch.save(model.state_dict(), f"{model_path}_{task}")

        X_forecast = np.vstack([X_train_combined[-window_size:], X_test_combined])
        y_forecast = np.concatenate([y_train_np[-window_size:], y_test_np])
        metrics_fincast = evaluate_fincast_rolling(
            model,
            X_forecast,
            y_forecast,
            window_size=window_size,
            device=device,
            task=task,
        )
        print(f"FinCast categories VAE {task} metrics:")
        print(metrics_fincast)


    def train_fincast_embeddings_pca(model_path: str, task: str = "binary"):
        train_part, test_part = time_train_test_split(
            combined_with_encoded_text, test_ratio=0.2
        )

        X_train, y_train = split_features_target(train_part, task=task)
        X_test, y_test = split_features_target(test_part, task=task)

        X_train_numeric, X_train_text = split_text_embeddings_and_features(X_train)
        X_test_numeric, X_test_text = split_text_embeddings_and_features(X_test)

        X_train_numeric_np = X_train_numeric.to_numpy().astype(np.float32)
        X_train_text_np = X_train_text.to_numpy().astype(np.float32)
        X_test_numeric_np = X_test_numeric.to_numpy().astype(np.float32)
        X_test_text_np = X_test_text.to_numpy().astype(np.float32)

        pca = PCA(n_components=30)
        X_train_text_pca = pca.fit_transform(X_train_text_np)
        X_test_text_pca = pca.transform(X_test_text_np)

        scaler_numeric = StandardScaler()
        X_train_numeric_scaled = to_numpy(scaler_numeric.fit_transform(X_train_numeric_np))
        X_test_numeric_scaled = to_numpy(scaler_numeric.transform(X_test_numeric_np))

        X_train_combined = np.hstack([X_train_numeric_scaled, X_train_text_pca])
        X_test_combined = np.hstack([X_test_numeric_scaled, X_test_text_pca])

        y_train_np = y_train.to_numpy().astype(np.float32)
        y_test_np = y_test.to_numpy().astype(np.float32)

        train_loader = make_dataloader(
            X=X_train_combined,
            y=y_train_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        test_loader = make_dataloader(
            X=X_test_combined,
            y=y_test_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        model, criterion, lr = _make_fincast_model(task, X_train_combined.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_generic(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            task=task,
            epochs=40,
            patience=15,
        )

        torch.save(model.state_dict(), f"{model_path}_{task}")

        X_forecast = np.vstack([X_train_combined[-window_size:], X_test_combined])
        y_forecast = np.concatenate([y_train_np[-window_size:], y_test_np])
        metrics_fincast = evaluate_fincast_rolling(
            model,
            X_forecast,
            y_forecast,
            window_size=window_size,
            device=device,
            task=task,
        )
        print(f"FinCast embeddings PCA {task} metrics:")
        print(metrics_fincast)


    def train_fincast_embeddings_vae(model_path: str, task: str = "binary"):
        train_part, test_part = time_train_test_split(
            combined_with_encoded_text, test_ratio=0.2
        )

        X_train, y_train = split_features_target(train_part, task=task)
        X_test, y_test = split_features_target(test_part, task=task)

        X_train_numeric, X_train_text = split_text_embeddings_and_features(X_train)
        X_test_numeric, X_test_text = split_text_embeddings_and_features(X_test)

        X_train_numeric_np = X_train_numeric.to_numpy().astype(np.float32)
        X_test_numeric_np = X_test_numeric.to_numpy().astype(np.float32)

        scaler_numeric = StandardScaler()
        X_train_numeric_scaled = to_numpy(scaler_numeric.fit_transform(X_train_numeric_np))
        X_test_numeric_scaled = to_numpy(scaler_numeric.transform(X_test_numeric_np))

        vae_model, X_train_text_vae = train_and_encode_vae_dataframe(
            data=X_train_text, config=vae_config, text_embed_only=True
        )
        X_test_text_vae = encode_with_vae(vae_model, X_test_text)

        X_train_combined = np.hstack(
            [X_train_numeric_scaled, X_train_text_vae.to_numpy().astype(np.float32)]
        )
        X_test_combined = np.hstack(
            [X_test_numeric_scaled, X_test_text_vae.to_numpy().astype(np.float32)]
        )

        y_train_np = y_train.to_numpy().astype(np.float32)
        y_test_np = y_test.to_numpy().astype(np.float32)

        train_loader = make_dataloader(
            X=X_train_combined,
            y=y_train_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        test_loader = make_dataloader(
            X=X_test_combined,
            y=y_test_np,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        model, criterion, lr = _make_fincast_model(task, X_train_combined.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_generic(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            task=task,
            epochs=40,
            patience=15,
        )

        torch.save(model.state_dict(), f"{model_path}_{task}")

        X_forecast = np.vstack([X_train_combined[-window_size:], X_test_combined])
        y_forecast = np.concatenate([y_train_np[-window_size:], y_test_np])
        metrics_fincast = evaluate_fincast_rolling(
            model,
            X_forecast,
            y_forecast,
            window_size=window_size,
            device=device,
            task=task,
        )
        print(f"FinCast embeddings VAE {task} metrics:")
        print(metrics_fincast)

    return (
        train_fincast_categories_pca,
        train_fincast_embeddings_pca,
        train_fincast_embeddings_vae,
        train_fincast_numeric_only,
    )


@app.cell
def _(train_fincast_numeric_only):
    train_fincast_numeric_only(
        "data/models_checkpoints/fincast_numeric_cont", task="continuous"
    )
    train_fincast_numeric_only(
        "data/models_checkpoints/fincast_numeric_binary", task="binary"
    )
    return


@app.cell
def _(train_fincast_categories_pca):
    train_fincast_categories_pca(
        "data/models_checkpoints/fincast_cats_pca_cont", task="continuous"
    )
    train_fincast_categories_pca(
        "data/models_checkpoints/fincast_cats_pca_bin", task="binary"
    )
    return


@app.cell
def _(train_fincast_embeddings_pca):
    train_fincast_embeddings_pca(
        "data/models_checkpoints/fincast_emb_pca_cont", task="continuous"
    )
    train_fincast_embeddings_pca(
        "data/models_checkpoints/fincast_emb_pca_bin", task="binary"
    )
    return


@app.cell
def _(train_fincast_embeddings_vae):
    train_fincast_embeddings_vae(
        "data/models_checkpoints/fincast_emb_pca_cont", task="continuous"
    )
    train_fincast_embeddings_vae(
        "data/models_checkpoints/fincast_emb_pca_bin", task="binary"
    )
    return


@app.cell
def _(train_fincast_embeddings_vae):
    train_fincast_embeddings_vae(
        "data/models_checkpoints/fincast_emb_pca_bin", task="binary"
    )
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

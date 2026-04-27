"""Sliding-window dataset: scaled OHLCV-style features + per-bar BERT embeddings."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.loader import _FeatureScaler
from dataset.preprocessing import (
    _drop_leaky_columns,
    add_prediction_targets,
    _resolve_target_col,
)


class BertTimeSeriesDataset(Dataset):
    """Each sample is ``window_size`` numeric rows, BERT vector at the last bar, and target."""

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        window_size: int = 16,
        task: str = "binary",
        text_embed_prefix: str = "text_embed_",
        scaler: _FeatureScaler | None = None,
        numeric_extra_drop: list[str] | None = None,
    ) -> None:
        prepared = add_prediction_targets(data)
        target_col = _resolve_target_col(task)

        text_cols = sorted(
            (c for c in prepared.columns if c.startswith(text_embed_prefix)),
            key=lambda c: int(c.removeprefix(text_embed_prefix)),
        )
        if not text_cols:
            raise ValueError(
                f"No columns starting with {text_embed_prefix!r}; "
                "join BERT parquet before building this dataset."
            )

        extra_drop = list(
            numeric_extra_drop
            or ["date", "TimeBin", "TradeDate", "target_binary", "target_continuous"]
        )
        X_num_df = _drop_leaky_columns(prepared, extra=extra_drop + text_cols)
        X_txt_np = prepared.select(text_cols).to_numpy().astype(np.float32)
        y_np = prepared.select(target_col).to_numpy().reshape(-1).astype(np.float32)

        X_np = X_num_df.to_numpy().astype(np.float32)
        if scaler is None:
            scaler = _FeatureScaler.fit(X_np)
        self.scaler = scaler
        X_scaled = scaler.transform(X_np)

        self.X_num = torch.tensor(X_scaled, dtype=torch.float32)
        self.X_txt = torch.tensor(X_txt_np, dtype=torch.float32)
        self.y = torch.tensor(y_np, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self) -> int:
        return self.X_num.size(0) - self.window_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = self.window_size
        x_seq = self.X_num[index : index + w]
        bert = self.X_txt[index + w - 1]
        y = self.y[index + w]
        return x_seq, bert, y


def make_bert_ts_dataloader(
    data: pl.DataFrame,
    *,
    task: str = "binary",
    window_size: int = 16,
    batch_size: int = 128,
    shuffle: bool = False,
    drop_last: bool = False,
    scaler: _FeatureScaler | None = None,
    text_embed_prefix: str = "text_embed_",
) -> DataLoader:
    ds = BertTimeSeriesDataset(
        data,
        window_size=window_size,
        task=task,
        text_embed_prefix=text_embed_prefix,
        scaler=scaler,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

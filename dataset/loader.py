import torch
import polars as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataset.preprocessing import split_features_target


class _FeatureScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

    @classmethod
    def fit(cls, X: np.ndarray) -> "_FeatureScaler":
        return cls(mean=X.mean(axis=0), std=X.std(axis=0, ddof=0))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X.astype(np.float32) - self.mean) / self.std


class SPYDataset(Dataset):

    def __init__(
        self,
        data: pl.DataFrame,
        window_size: int = 16,
        task: str = "binary",
        scaler: _FeatureScaler | None = None,
    ):
        X_df, y_df = split_features_target(data, task=task)

        X_np = X_df.to_numpy()
        if scaler is None:
            scaler = _FeatureScaler.fit(X_np)
        self.scaler = scaler

        X_scaled = scaler.transform(X_np)

        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.y = torch.tensor(
            np.asarray(y_df.to_numpy()).reshape(-1), dtype=torch.float32
        )

        self.window_size = window_size
        self.task = task

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, index):
        x_slice = self.X[index : index + self.window_size]
        y_val = self.y[index + self.window_size]
        return x_slice, y_val


class BinarySPYDataset(SPYDataset):
    def __init__(
        self,
        data: pl.DataFrame,
        window_size: int = 16,
        scaler: _FeatureScaler | None = None,
    ):
        super().__init__(data=data, window_size=window_size, task="binary", scaler=scaler)


class ContinuousSPYDataset(SPYDataset):
    def __init__(
        self,
        data: pl.DataFrame,
        window_size: int = 16,
        scaler: _FeatureScaler | None = None,
    ):
        super().__init__(
            data=data, window_size=window_size, task="continuous", scaler=scaler
        )


def make_dataloader(
    data: pl.DataFrame,
    task: str,
    window_size: int = 16,
    batch_size: int = 128,
    shuffle: bool = False,
    drop_last: bool = False,
    scaler: _FeatureScaler | None = None,
) -> DataLoader:
    if task == "binary":
        dataset = BinarySPYDataset(data=data, window_size=window_size, scaler=scaler)
    elif task in {"continuous", "non-binary", "non_binary"}:
        dataset = ContinuousSPYDataset(
            data=data, window_size=window_size, scaler=scaler
        )
    else:
        raise ValueError("task must be either 'binary' or 'continuous'.")

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )

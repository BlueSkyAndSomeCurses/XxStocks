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
    ):
        X_df, y_df = split_features_target(data, task=task)

        X_np = X_df.to_numpy()

        self.X = torch.tensor(X_np, dtype=torch.float32)
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


class ArrayWindowDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        window_size: int = 16,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y).reshape(-1), dtype=torch.float32)
        self.window_size = window_size

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
    ):
        super().__init__(data=data, window_size=window_size, task="binary")


class ContinuousSPYDataset(SPYDataset):
    def __init__(
        self,
        data: pl.DataFrame,
        window_size: int = 16,
    ):
        super().__init__(
            data=data, window_size=window_size, task="continuous"
        )


def make_dataloader(
    data: pl.DataFrame | None = None,
    task: str | None = None,
    window_size: int = 16,
    batch_size: int = 128,
    shuffle: bool = False,
    drop_last: bool = False,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> DataLoader:
    if X is not None and y is not None:
        dataset = ArrayWindowDataset(X=X, y=y, window_size=window_size)
    else:
        if data is None or task is None:
            raise ValueError("Provide either X and y, or data and task.")
        if task == "binary":
            dataset = BinarySPYDataset(data=data, window_size=window_size)
        elif task in {"continuous", "non-binary", "non_binary"}:
            dataset = ContinuousSPYDataset(
                data=data, window_size=window_size
            )
        else:
            raise ValueError("task must be either 'binary' or 'continuous'.")

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )

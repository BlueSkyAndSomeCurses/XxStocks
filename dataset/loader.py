import torch
import polars as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataset.preprocessing import split_features_target


class SPYDataset(Dataset):
    def __init__(self, data: pl.DataFrame, window_size: int = 16, task: str = "binary"):
        X_df, y_df = split_features_target(data, task=task)

        self.X = torch.tensor(X_df.to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y_df.to_numpy()).reshape(-1), dtype=torch.float32)

        self.window_size = window_size
        self.task = task

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, index):
        x_slice = self.X[index : index + self.window_size]

        y_val = self.y[index + self.window_size]

        return x_slice, y_val


class BinarySPYDataset(SPYDataset):
    def __init__(self, data: pl.DataFrame, window_size: int = 16):
        super().__init__(data=data, window_size=window_size, task="binary")


class ContinuousSPYDataset(SPYDataset):
    def __init__(self, data: pl.DataFrame, window_size: int = 16):
        super().__init__(data=data, window_size=window_size, task="continuous")


def make_dataloader(
    data: pl.DataFrame,
    task: str,
    window_size: int = 16,
    batch_size: int = 128,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    if task == "binary":
        dataset = BinarySPYDataset(data=data, window_size=window_size)
    elif task in {"continuous", "non-binary", "non_binary"}:
        dataset = ContinuousSPYDataset(data=data, window_size=window_size)
    else:
        raise ValueError("task must be either 'binary' or 'continuous'.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

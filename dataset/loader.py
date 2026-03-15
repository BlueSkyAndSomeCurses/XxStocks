import torch
import polars as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataset.preprocessing import split_features_target


class SPYDataset(Dataset):
    def __init__(self, data: pl.DataFrame, window_size: int = 16):
        X_df, y_df = split_features_target(data)

        self.X = torch.tensor(X_df.to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(y_df.to_numpy(), dtype=torch.float32)

        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, index):
        x_slice = self.X[index : index + self.window_size]

        y_val = self.y[index + self.window_size]

        return x_slice, y_val

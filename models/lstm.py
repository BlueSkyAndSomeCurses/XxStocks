import numpy as np
import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        out, (hn, cn) = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hn, cn


def rolling_one_step_forecast_lstm(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window_size: int,
    device: str = "cpu",
    task: str = "binary",
):
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same length.")
    if len(X_test) <= window_size:
        raise ValueError(f"Test set must have more than {window_size} samples.")

    rolling_window = X_test[:window_size].copy()
    preds = np.empty(len(y_test) - window_size, dtype=np.float32)

    with torch.no_grad():
        pred_idx = 0
        for t in range(window_size, len(y_test)):
            x_window = torch.from_numpy(rolling_window).unsqueeze(0).to(torch.float32).to(device)

            out, _, _ = model(x_window)
            pred = out.squeeze().cpu().numpy()
            preds[pred_idx] = pred
            pred_idx += 1

            if t < len(X_test):
                rolling_window = np.vstack([rolling_window[1:], X_test[t]])
    
    return preds

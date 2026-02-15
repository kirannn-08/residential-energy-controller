import numpy as np
import torch
from torch.utils.data import Dataset


class PVDataset(Dataset):
    """
    Sliding-window dataset for PV power forecasting.
    Each sample predicts the next-step PV power.
    """

    def __init__(self, pv_series, window_size):
        """
        pv_series: 1D array-like of PV power (kW)
        window_size: number of past time steps used as input
        """
        self.window_size = window_size

        pv_series = np.array(pv_series, dtype=np.float32)

        self.mean = pv_series.mean()
        self.std = pv_series.std() + 1e-6

        self.series = (pv_series - self.mean) / self.std

    def __len__(self):
        return len(self.series) - self.window_size

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.window_size]
        y = self.series[idx + self.window_size]

        x = torch.tensor(x).unsqueeze(0)  # (channels=1, seq_len)
        y = torch.tensor(y).unsqueeze(0)  # (1,)

        return x, y
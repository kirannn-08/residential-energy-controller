import numpy as np
import torch
from torch.utils.data import Dataset


class LoadDataset(Dataset):
    """
    Sliding-window dataset for load forecasting.
    Supports multichannel input and multichannel output.
    """

    def __init__(self, load_matrix, window_size):
        """
        load_matrix: shape (time, channels)
        window_size: number of past steps
        """
        self.window_size = window_size

        data = np.array(load_matrix, dtype=np.float32)

        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0) + 1e-6

        self.data = (data - self.mean) / self.std

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window_size]
        y = self.data[idx + self.window_size]

        x = torch.tensor(x.T)       # (channels, seq_len)
        y = torch.tensor(y)         # (channels,)

        return x, y
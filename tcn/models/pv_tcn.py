import torch
import torch.nn as nn
from .tcn_base import TCN


class PVTCN(nn.Module):
    """
    TCN-based PV power forecaster
    """

    def __init__(self, input_channels=1, channel_sizes=[32, 32, 64],
                 kernel_size=3, dropout=0.2):
        super().__init__()

        self.tcn = TCN(input_channels, channel_sizes, kernel_size, dropout)
        self.fc = nn.Linear(channel_sizes[-1], 1)

    def forward(self, x):
        features = self.tcn(x)
        last_features = features[:, :, -1]
        return self.fc(last_features)
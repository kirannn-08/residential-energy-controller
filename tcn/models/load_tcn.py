import torch
import torch.nn as nn
from .tcn_base import TCN


class LoadTCN(nn.Module):
    """
    Multi-output Load TCN.
    Predicts next-step load for each room.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        channel_sizes=[16, 16, 32],
        kernel_size=3,
        dropout=0.2
    ):
        super().__init__()

        self.tcn = TCN(
            input_channels=input_channels,
            channel_sizes=channel_sizes,
            kernel_size=kernel_size,
            dropout=dropout
        )

        self.fc = nn.Linear(channel_sizes[-1], output_channels)

    def forward(self, x):
        features = self.tcn(x)
        last_features = features[:, :, -1]
        return self.fc(last_features)
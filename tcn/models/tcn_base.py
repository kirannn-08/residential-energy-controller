import torch
import torch.nn as nn
from .temporal_block import TemporalBlock


class TCN(nn.Module):
    """
    Temporal Convolutional Network
    """

    def __init__(self, input_channels, channel_sizes, kernel_size=3, dropout=0.2):
        super().__init__()

        layers = []

        for i in range(len(channel_sizes)):
            dilation = 2 ** i
            in_ch = input_channels if i == 0 else channel_sizes[i - 1]
            out_ch = channel_sizes[i]

            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
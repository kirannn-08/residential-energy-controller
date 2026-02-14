import torch
import torch.nn as nn
from .temporal_block import TemporalBlock

class TCN(nn.Module):
    """
   TCN consisting of stacked temporal blocks with increasing dialation
    """

    def __init__(
            self,
            input_channels,
            channel_sizes,
            kernel_size=3,
            dropout=0.2
        ):
        super().__init__()

        layers = []

        for i in range (len(channel_sizes)):
            dilation = 2 ** i

            input_channels = (
                input_channels if i == 0 else channel_sizes[i-1]
            )
            out_channels = channel_sizes[i]

            layers.append(
                TemporalBlock(
                    in_channels= input_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dialation=dilation,
                    dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)

        def forward(self, x):
            """
        Input shape:
            x: (batch, channels, sequence_length)

        Output shape:
            (batch, last_channel, sequence_length)
            """
        return self.network( x )
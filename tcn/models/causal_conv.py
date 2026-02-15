import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """
    1D convolution that preserves causality.
    Output at time t depends only on inputs <= t.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out
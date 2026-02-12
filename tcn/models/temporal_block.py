import torch
import torch.nn as nn 

from causal_conv import CasualConv1d

class TemporalBlock(nn.Module):
    """
    creating a single TCN residual block with :-
    - Two causal convolutional 
    - Dialation for long memory
    - Residual connection for stable training 
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dialation,
            dropout=0.2
    ):
        super().__init__()

        self.conv1 = CasualConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dialation=dialation,
        )
        self.relu1= nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CasualConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dialation=dialation,
        )
        self.relu2= nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size =1 )
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        """
        input shape:
        x: (batch, in_channels , seq_len)

        output shape:
        x: (batch, in_channels , seq_len)
        """
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)


        out = self.conv2(x)
        out = self.relu2(out)
        out = self.dropout2(out)

        #residual path 
        res = x if self.downsample is None else self.downsample(x)

        return out + res
import torch 
import torch.nn as nn

from .tcn_base import TCN


class PVTCN(nn.Module):
    """tcn based solar pv forecaster.
    predicts next step real power in kw
    """
    def __init__(
            self,
            input_channels =1,
            channel_sizes =[32,32,64],
            kernel_sizes = 3,
            dropout=0.2
    ):
        super().__init__()
        self.tcn = TCN(
            input_channels=input_channels,
            channel_sizes=channel_sizes,
            kernel_size=kernel_sizes,
            dropout=dropout
        )
        self.fc = nn.Linear(channel_sizes[-1],[1])
        
        def forward(self, x):
            """
            Input:
                x: (batch, channels, sequence_length)

            Output:
                y: (batch, 1)
            """
            features = self.tcn(x)

            # Take the last time step
            last_features = features[:, :, -1]

            return self.fc(last_features)
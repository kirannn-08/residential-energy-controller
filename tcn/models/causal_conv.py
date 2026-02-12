import torch 
import torch.nn as nn

class CasualConv1d(nn.Module):
    """ using 1d convolution that preserves casuality .
     Output at time t depends only on inputs <= t.
    """

    def __init__(self, in_channels , out_channels , kernel_size , dialation):
        super().__init__()

        #amount of padding needed to keep the sequence length 

        self.padding = (kernel_size -1 ) * dialation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dialation,
            padding=self.padding
        )
    def forward(self, x):
        """
        input shape: 
        x: (batch_size, channels , sequence_length)

        output shape 
            (batch_size, channels , sequence_length)
        """

        out = self.conv(x)

        # removing future time steps to enforce casuality int plant 
        if self.padding > 0:
            out = out[:, :, :-self.padding]

        return out
    
    """implementing causality makes sure that our system o/p at time t
        never depends on future inputs , hence making our model to not train on 
        predicted outcomes rather on actual available and present data.
    """
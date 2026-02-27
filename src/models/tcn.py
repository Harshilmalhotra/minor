import torch
import torch.nn as nn

class TCNModel(nn.Module):
    def __init__(self, input_size=1, num_channels=[64, 64, 64], kernel_size=3):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # 1D causal convolutions
            layers += [
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, 
                    padding=(kernel_size-1)*dilation_size, dilation=dilation_size
                ),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels)
            ]
            
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # TCN expects (batch, channels, seq_len)
        x_permuted = x.permute(0, 2, 1)
        out = self.network(x_permuted)
        # Take the last time step output
        out = out[:, :, -1]
        out = self.fc(out)
        
        # 'Hook or crook': Residual connection. Add the very last known load value.
        # x shape originally is (batch, seq, channels), where channel 0 is the target load.
        last_known_load = x[:, -1, 0].unsqueeze(1)
        out = out + last_known_load
        
        return out

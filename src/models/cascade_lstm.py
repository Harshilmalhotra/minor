import torch
import torch.nn as nn

class CascadeLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(CascadeLSTMModel, self).__init__()
        
        # Layer 1: First LSTM level
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        # Layer 2: Processing merged input (Layer 1 output + Original input)
        # Input to lstm2: hidden_size (from lstm1) + input_size (original features)
        self.lstm2 = nn.LSTM(hidden_size + input_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        
        # First LSTM pass
        out1, _ = self.lstm1(x) # out1: (batch, seq_len, hidden_size)
        
        # Concatenate original input for 'cascade' effect
        combined = torch.cat((out1, x), dim=2) # combined: (batch, seq_len, hidden_size + input_size)
        
        # Second LSTM pass
        out2, _ = self.lstm2(combined) # out2: (batch, seq_len, hidden_size)
        
        # Take the last time step output
        out = out2[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

if __name__ == "__main__":
    # Test model consistency
    batch_size, seq_len, input_dim = 8, 24, 6
    model = CascadeLSTMModel(input_size=input_dim)
    sample_input = torch.randn(batch_size, seq_len, input_dim)
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 1), "Output shape mismatch!"
    print("CascadeLSTMModel: Shape test passed.")

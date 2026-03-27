import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(BiLSTMModel, self).__init__()
        
        self.bilstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(0.3)
        # Multiply hidden_size by 2 because it's bidirectional
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.bilstm(x)
        
        # Take the last time step output from both directions
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

if __name__ == "__main__":
    batch_size, seq_len, input_dim = 8, 24, 6
    model = BiLSTMModel(input_size=input_dim)
    sample_input = torch.randn(batch_size, seq_len, input_dim)
    output = model(sample_input)
    assert output.shape == (batch_size, 1), "Output shape mismatch!"
    print("BiLSTMModel: Shape test passed.")

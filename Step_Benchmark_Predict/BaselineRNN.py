import torch.nn as nn
"""Baseline RNN model."""


class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=3, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size

# RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

# output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # [batch_size, seq_len, hidden_size]
        out = self.fc(rnn_out)  # [batch_size, seq_len, output_size]
        return out

import torch
import torch.nn as nn
"""Baseline GRU model."""


class LSTMmodel(nn.Module):  # GRUmodel
    def __init__(self, input_size, hidden_size=128, output_size=3, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        dropout=dropout if num_layers > 1 else 0,  # dropout only when stacking layers
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)  # [B, T, H]
        return self.fc(gru_out)  # [B, T, O]
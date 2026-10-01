import torch.nn as nn
"""Enhanced LSTM model."""


class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=3, num_layers=1, dropout=0.2, forget_bias=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        # init weights
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                # forget gate bias = 1 to help gradients flow
                if 'lstm.bias_ih' in name or 'lstm.bias_hh' in name:
                    param.data[self.hidden_size:2*self.hidden_size].fill_(self.forget_bias)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [B, T, H]
        return self.fc(lstm_out)  # [B, T, O]

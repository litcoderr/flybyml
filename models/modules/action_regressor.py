import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=n_features, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            dropout=dropout, 
            bidirectional=True, # Alfred
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 16) # 10 controls + 6 camera args
                                                 # TODO: employ upper bound, lower bound of each factor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out = (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x)
        y_pred = self.linear(out[:,-1])
        return y_pred
    
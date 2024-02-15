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
            bidirectional=False, # Alfred uses bidirectional lstm
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 16) # 10 controls + 6 camera args
                                                 # TODO: employ upper bound, lower bound of each factor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, 520]

        return:
            [batch, seq_len, 16]
        """
        out, _ = self.lstm(x)

        shape = out.shape
        out = out.reshape(-1, shape[2])
        y_pred = self.linear(out)
        y_pred = y_pred.reshape(shape[0], shape[1], -1)
        return y_pred
    
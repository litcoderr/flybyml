import torch
import torch.nn as nn

from model.action_regressor import LSTMRegressor

class BaseWithoutVisNetwork(nn.Module):
    """
    Network without visual observations, which contains only LSTM.
    """
    def __init__(self, args):
        super().__init__()

        self.action_regressor = LSTMRegressor(
            args.dsensory+args.dinst, 
            args.temporal_network.hidden_size, 
            args.temporal_network.num_layers, 
            args.dropout
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, batch, prev_context=None):
        # [b, seq_len, 8]
        inp_t = torch.cat([batch['sensory_observations'], batch['instructions']], dim=2)
        # [b, seq_len, 16]
        out_act_t, context = self.action_regressor(inp_t, prev_context)

        return torch.cat((self.sigmoid(out_act_t[:, :, :10]), out_act_t[:, :, 10:]), dim=2), context
        
import torch
import torch.nn as nn

from models.modules.encoder import ResNetEncoder
from models.modules.action_regressor import LSTMRegressor

class BaseNetwork(nn.Module):
    """Network which passes the input image through CNN and concatenates
    actions, state vector with CNN's output and passes that through LSTM.
    """
    def __init__(self, args):
        super().__init__()
        if args.vis_encoder == "resnet50":
            self.vis_encoder = ResNetEncoder()
            self.vis_fc = nn.Sequential(
                nn.Linear(self.vis_encoder.output_shape[0], args.dframe),
                nn.ReLU(True),
            )
        elif args.vis_encoder == "resnet50_clip":
            # TODO: define clip encoder
            pass

        self.action_regressor = LSTMRegressor(args.dframe+args.dsensory+args.dinst, args.hidden_size, args.num_layers, args.dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, batch, prev_context=None):
        frames = batch['visual_observations']

        # [b, seq_len, 512]
        vis_feat_t = self.vis_fc(self.vis_encoder(frames))
        # [b, seq_len, 520]
        inp_t = torch.cat([vis_feat_t, batch['sensory_observations'], batch['instructions']], dim=2)
        # [b, seq_len, 16]
        out_act_t, context = self.action_regressor(inp_t, prev_context)

        return torch.cat((self.sigmoid(out_act_t[:, :, :10]), out_act_t[:, :, 10:]), dim=2), context

class BaseWithoutVisNetwork(nn.Module):
    """Network without visual observations, which contains only LSTM.
    """
    def __init__(self, args):
        super().__init__()

        self.action_regressor = LSTMRegressor(args.dsensory+args.dinst, args.hidden_size, args.num_layers, args.dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, batch, prev_context=None):
        # [b, seq_len, 8]
        inp_t = torch.cat([batch['sensory_observations'], batch['instructions']], dim=2)
        # [b, seq_len, 16]
        out_act_t, context = self.action_regressor(inp_t, prev_context)

        return torch.cat((self.sigmoid(out_act_t[:, :, :10]), out_act_t[:, :, 10:]), dim=2), context
        
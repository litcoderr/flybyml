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

        self.action_regressor = LSTMRegressor(args.dframe+args.dact+args.demb, args.hidden_size, args.num_layers, args.dropout)
        
    def forward(self, batch):
        frames = batch['visual_observations']

        vis_feat_t = self.vis_fc(self.vis_encoder(frames))
        # TODO need to be fixed
        inp_t = torch.cat([vis_feat_t, prev_act_t, state_t], dim=1)

        out_act_t = self.action_regressor(inp_t)
        return out_act_t
        
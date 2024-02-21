import torch
import torch.nn as nn

from lightning import LightningModule
# model
from module.encoder import ResNetEncoder
from module.action_regressor import LSTMRegressor


class BaseNetwork(nn.Module):
    """
    Network which passes the input image through CNN and concatenates
    actions, state vector with CNN's output and passes that through LSTM.
    """
    def __init__(self, args):
        super().__init__()
        if args.vis_encoder.type == "resnet50":
            self.vis_encoder = ResNetEncoder()
            self.vis_fc = nn.Sequential(
                nn.Linear(self.vis_encoder.output_shape[0], args.vis_encoder.dframe),
                nn.ReLU(True),
            )
        elif args.vis_encoder.type == "resnet50_clip":
            # TODO: define clip encoder
            pass

        self.action_regressor = LSTMRegressor(
            args.dframe+args.dsensory+args.dinst, 
            args.temporal_network.hidden_size, 
            16,
            args.temporal_network.num_layers, 
            args.dropout
        )
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
        

class AlfredBaseline(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = BaseNetwork(args)
        self.mse = nn.MSELoss()

        # auto-logged by W&B
        self.save_hyperparameters()
    
    def forward(self, batch, prev_context=None):
        return self.model(batch, prev_context)

    def training_step(self, batch, _):
        action_output, _ = self.model(batch)
        loss = self.mse(action_output, torch.cat((batch['actions'], batch['camera_actions']), dim=2))

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, _):
        action_output, _ = self.model(batch)
        loss = self.mse(action_output, torch.cat((batch['actions'], batch['camera_actions']), dim=2))

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

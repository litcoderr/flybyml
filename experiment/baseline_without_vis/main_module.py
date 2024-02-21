import os
import wandb
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from lightning import LightningModule
from matplotlib import pyplot as plt

# model
from module.action_regressor import LSTMRegressor


class BaseWithoutVisNetwork(nn.Module):
    """
    Network without visual observations, which contains only LSTM.
    """
    def __init__(self, args):
        super().__init__()

        self.action_regressor = LSTMRegressor(
            args.dsensory+args.dinst, 
            args.temporal_network.hidden_size, 
            10,
            args.temporal_network.num_layers, 
            args.dropout
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, prev_context=None):
        # [b, seq_len, 8]
        inp_t = torch.cat([batch['sensory_observations'], batch['instructions']], dim=2)
        # [b, seq_len, 10]
        out_act_t, context = self.action_regressor(inp_t, prev_context)

        return self.sigmoid(out_act_t), context


class AlfredBaselineWithoutVis(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = BaseWithoutVisNetwork(args.model)
        self.mse = nn.MSELoss()

        # auto-logged by W&B
        self.save_hyperparameters()
    
    def forward(self, batch, prev_context=None):
        return self.model(batch, prev_context)

    def training_step(self, batch, _):
        action_output, _ = self.model(batch)
        loss = self.mse(action_output, batch['actions'])

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_id):
        action_output, _ = self.model(batch)
        loss = self.mse(action_output, batch['actions'])
        self.log("val_loss", loss)

        # save and upload plot
        b_size, seq_len, _ = tuple(action_output.shape)
        action_output = action_output.cpu().numpy()
        gt = batch['actions'].cpu().numpy()

        result_root =  Path(os.path.dirname(__file__)) / "logs" / self.args.run / "result"
        os.makedirs(result_root, exist_ok=True)

        image_log = {}
        keys = ['elevator', 'aileron', 'rudder', 'thrust', 'gear', 'flaps', 'trim', 'brake', 'speed_brake', 'reverse_thrust']
        for batch_idx in range(b_size):
            for key_idx, key in enumerate(keys):
                title = f'{batch_id}_{batch_idx}_{key}'

                plt.plot(np.arange(0, seq_len, 1), action_output[batch_idx, :, key_idx])
                plt.plot(np.arange(0, seq_len, 1), gt[batch_idx, :, key_idx])
                plt.legend(['output', 'gt'])
                plt.savefig(result_root/f'{title}.png')
                plt.close()

                image_log[title] = wandb.Image(str(result_root/f'{title}.png'))
        wandb.log(image_log)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

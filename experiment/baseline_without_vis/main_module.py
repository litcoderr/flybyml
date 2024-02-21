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
        self.model = BaseWithoutVisNetwork(args)
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
    
    def validation_step(self, batch, _):
        action_output, _ = self.model(batch)
        loss = self.mse(action_output, batch['actions'])

        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        action_output, _ = self.forward(batch) # [1, seq_len, 16]
        gt = batch['actions']

        result_root = Path(os.path.dirname(__file__)) / "logs" / self.args.run / "result"
        os.makedirs(result_root, exist_ok=True)

        log = {}
        keys = ['elevator', 'aileron', 'rudder', 'thrust', 'gear', 'flaps', 'trim', 'brake', 'speed_brake', 'reverse_thrust']
        for key_idx in range(gt.shape[2]):
            title = f'{batch_idx}_{keys[key_idx]}'
            plt.title(title)
            plt.plot(np.arange(0, gt.shape[1], 1), action_output.cpu().numpy()[0, :, key_idx])
            plt.plot(np.arange(0, gt.shape[1], 1), gt.cpu().numpy()[0, :, key_idx])
            plt.legend(['output', 'gt'])
            plt.savefig(result_root/f'{title}.png')
            plt.close()
            log[title] = wandb.Image(result_root/f'{title}.png')
        self.log(log)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

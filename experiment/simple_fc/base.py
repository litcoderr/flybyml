import os
import wandb
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from lightning import LightningModule
from matplotlib import pyplot as plt


class FCBase(nn.Module):
    """
    Simple Fully-Connected Network, 
    without visual observations but previous actions added.
    """
    def __init__(self, args):
        super().__init__()

        self.fc1 = nn.Linear(args.dsensory+args.dinst+args.daction, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        # [b, 18]
        x = torch.cat([batch['sensory_observations'], batch['instructions'], batch['prev_actions']], dim=1)
        # [b, 64]
        x = self.fc1(x)
        x = self.relu(x)
        # [b, 32]
        x = self.fc2(x)
        x = self.relu(x)
        # [b, 10]
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class FCBaseline(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = FCBase(args.model)
        self.mse = nn.MSELoss()

        # auto-logged by W&B
        self.save_hyperparameters()
    
    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch):
        action_output = self.model(batch)
        loss = self.mse(action_output, batch['actions'])

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_id):
        action_output = self.model(batch)
        loss = self.mse(action_output, batch['actions'])
        self.log("val_loss", loss)

        # save and upload plot
        b_size, _ = tuple(action_output.shape)
        action_output = action_output.cpu().numpy()
        gt = batch['actions'].cpu().numpy()

        result_root =  Path(os.path.dirname(__file__)) / "logs" / self.args.run / "result"
        os.makedirs(result_root, exist_ok=True)

        if self.current_epoch % self.args.train.plot_every_epochs == 0:
            image_log = {}
            keys = ['elevator', 'aileron', 'rudder', 'thrust', 'gear', 'flaps', 'trim', 'brake', 'speed_brake', 'reverse_thrust']
            for key_idx, key in enumerate(keys):
                title = f'{batch_id}_{key}'

                plt.plot(np.arange(0, b_size), action_output[:, key_idx])
                plt.plot(np.arange(0, b_size), gt[:, key_idx])
                plt.legend(['output', 'gt'])
                plt.xlabel('random batch (not sequential time step)')
                plt.ylabel('action')
                plt.savefig(result_root/f'{title}.png')
                plt.close()

                image_log[title] = wandb.Image(str(result_root/f'{title}.png'))
            wandb.log(image_log)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

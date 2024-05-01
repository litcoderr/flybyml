import os
import wandb
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

from pathlib import Path
from lightning import LightningModule
from matplotlib import pyplot as plt

class MLPV2(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.fc1 = nn.Linear(args.dsensory, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 16)
        self.bn5 = nn.BatchNorm1d(16)
        self.fc6 = nn.Linear(16, 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor):
        """
        Predict actions (elev, ail, thrust) based on objective/error of states (pitch, roll, spd, vert_spd).

        Args:
            x (torch.Tensor): state obj and err (pitch, roll, spd, vert_spd, e_pitch, e_roll, e_spd, e_vert_spd) for batch

        Returns:
            torch.Tensor: predicted actions (elev, ail, thr) for batch
        """
        # [b, 16]
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # [b, 32]
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # [b, 64]
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # [b, 32]
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        # [b, 16]
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu(x)
        # [b, 3]
        x = self.fc6(x)
        x = self.sigmoid(x)
        return x
    
class MLPModuleV2(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = MLPV2(args.model)
        self.mse = nn.MSELoss()

        # auto-logged by W&B
        self.save_hyperparameters()
    
    def forward(self, batch):
        return self.model(batch[0])

    def training_step(self, batch):
        action_output = self.model(batch[0])
        loss = self.mse(action_output, batch[1])

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_id):
        action_output = self.model(batch[0])
        loss = self.mse(action_output, batch[1])
        self.log("val_loss", loss)

        # save and upload plot
        b_size, _ = tuple(action_output.shape)
        action_output = action_output.cpu().numpy()
        gt = batch[1].cpu().numpy()

        result_root =  Path(os.path.dirname(__file__)) / "logs" / self.args.run / "result"
        os.makedirs(result_root, exist_ok=True)

        if self.current_epoch % self.args.train.plot_every_epochs == 0:
            image_log = {}
            keys = ['elevator', 'aileron', 'thrust']
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

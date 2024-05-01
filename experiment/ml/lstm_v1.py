import os
import wandb
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

from pathlib import Path
from lightning import LightningModule
from matplotlib import pyplot as plt

from module.action_regressor import LSTMRegressor

class LSTMV1(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.lstmrgr = LSTMRegressor(args.dsensory, args.hidden_size, 3, args.num_layers, args.dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor, prev_context=None):
        """
        Predict actions (elev, ail, thrust) based on objective/error of states (pitch, roll, spd, vert_spd).

        Args:
            x: [pitch, roll, spd, vert_spd, e_pitch, e_roll, e_spd, e_vert_spd] for sequential batch
            prev_context: (h_0, c_0)
        Returns:
            torch.Tensor: predicted actions (elev, ail, thr) for batch
        """
        
        # [b, seq_len, 8] -> [b, seq_len, 3]
        out, context = self.lstmrgr(x, prev_context)
        
        return self.sigmoid(out), context


class LSTMModuleV1(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = LSTMV1(args.model)
        self.mse = nn.MSELoss()

        # auto-logged by W&B
        self.save_hyperparameters()
    
    def forward(self, batch, prev_context=None):
        return self.model(batch[0], prev_context)

    def training_step(self, batch, _):
        action_output, _ = self.model(batch[0])
        loss = self.mse(action_output, batch[1])

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_id):
        action_output, _ = self.model(batch[0])
        loss = self.mse(action_output, batch[1])
        self.log("val_loss", loss)

        # save and upload plot
        b_size, seq_len, _ = tuple(action_output.shape)
        action_output = action_output.cpu().numpy()
        gt = batch[1].cpu().numpy()

        result_root =  Path(os.path.dirname(__file__)) / "logs" / self.args.run / "result"
        os.makedirs(result_root, exist_ok=True)

        if self.current_epoch % self.args.train.plot_every_epochs == 0:
            image_log = {}
            keys = ['elevator', 'aileron', 'thrust']
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

import torch
import torch.nn as nn

from lightning import LightningModule
# model
from model.baseline.base import BaseNetwork
from model.baseline.base_without_vis import BaseWithoutVisNetwork

models = {
    "base": BaseNetwork,
    "without_vis": BaseWithoutVisNetwork,
}

class AlfredBaseline(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = models[args.type](args)
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
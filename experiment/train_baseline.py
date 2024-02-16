import argparse
import torch
import torch.nn as nn

from pathlib import Path
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from models.base import BaseNetwork
from dataset.baseline import BaselineDataModule


class AlfredBaseline(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = BaseNetwork(args)
        self.mse = nn.MSELoss()

    def training_step(self, batch, _):
        action_output = self.model(batch)
        loss = self.mse(action_output, torch.cat((batch['actions'], batch['camera_actions']), dim=2))

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, _):
        action_output = self.model(batch)
        loss = self.mse(action_output, torch.cat((batch['actions'], batch['camera_actions']), dim=2))

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main(args):
    logger = TensorBoardLogger("logs", name="baseline")
    datamodule = BaselineDataModule(root=Path(args.dataset_root), batch_size=args.bsize)
    model = AlfredBaseline(args)

    trainer = Trainer(max_epochs=args.epochs, devices=args.gpus, logger=logger)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlfredBaseline Model")
    # module choice
    parser.add_argument("--vis_encoder", type=str, default="resnet50", help="Visual encoder choice")
    # embedding dimension
    parser.add_argument("--dframe", type=int, default=512, help="Image feature vector size")
    parser.add_argument("--dsensory", type=int, default=4, help="Output action vector size")
    parser.add_argument("--dinst", type=int, default=4, help="State vector (Objective) size")
    # lstm hyper parameters
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for the LSTM")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the LSTM")
    # training
    parser.add_argument("--epochs", type=int, default=100000, help="Number of training epochs")
    parser.add_argument("--bsize", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--dataset_root", type=str, default="/data/flybyml_dataset_v1", help="root directory of flybyml dataset")

    args = parser.parse_args()
    main(args)
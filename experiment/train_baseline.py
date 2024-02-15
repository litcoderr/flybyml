import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl

from pathlib import Path

from models.base import BaseNetwork
from dataset.baseline import BaselineDataModule


class AlfredBaseline(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = BaseNetwork(args)

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        loss = nn.MSELoss()(output, target)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main(args):
    datamodule = BaselineDataModule(root=Path(args.dataset_root))
    model = AlfredBaseline(args)
    trainer = pl.Trainer(max_epochs=args.epochs, devices=args.gpus)
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
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--dataset_root", type=str, default="/data/flybyml_dataset_v1", help="root directory of flybyml dataset")

    args = parser.parse_args()
    main(args)
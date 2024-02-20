import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

from lightning import Trainer
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from experiment.baseline.main_module import AlfredBaseline
from dataset.baseline import BaselineDataModule

def main(args):
    pl.seed_everything(42)
    datamodule = BaselineDataModule(root=Path(f"/data/{args.dataset.name}"), batch_size=args.train.batch_size)
    model = AlfredBaseline(args.model)
    
    logger = WandbLogger(project=args.project, name=args.run)
    logger.watch(model)
    
    ckpt_path = os.path.join(os.path.dirname(__file__), args.project, "logs", args.run)
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=args.train.save_epochs,
        monitor="val_loss", 
        dirpath=ckpt_path, 
        filename="{epoch:04d}-{val_loss:.3f}"
    )

    trainer = Trainer(max_epochs=args.train.num_epochs, devices=args.train.gpus, logger=logger, callbacks=[checkpoint_callback], deterministic=True)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    conf = OmegaConf.load(sys.argv[1])
    conf.merge_with_cli()

    main(conf)
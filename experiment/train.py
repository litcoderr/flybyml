import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

from lightning import Trainer
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from experiment.baseline.main_module import AlfredBaseline
from experiment.baseline_without_vis.main_module import AlfredBaselineWithoutVis
from dataset.baseline import BaselineDataModule

cur_dir = Path(os.path.dirname(__file__)) 

DATA_MODULE = {
    'baseline': BaselineDataModule,
    'baseline_without_vis': BaselineDataModule
}

PL_MODULE = {
    'baseline': AlfredBaseline,
    'baseline_without_vis': AlfredBaselineWithoutVis
}


def main(args):
    pl.seed_everything(42)
    datamodule = DATA_MODULE[args.project](args)
    model = PL_MODULE[args.project](args)
    
    logger = WandbLogger(project=args.project, name=args.run, entity='flybyml')
    logger.watch(model)
    
    ckpt_path = cur_dir / args.project / "logs" / args.run
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=args.train.save_epochs,
        monitor="val_loss", 
        dirpath=ckpt_path, 
        filename="{epoch:04d}-{val_loss:.3f}"
    )

    trainer = Trainer(max_epochs=args.train.num_epochs,
                      devices=args.train.gpus,
                      logger=logger,
                      callbacks=[checkpoint_callback],
                      deterministic=True,
                      check_val_every_n_epoch=args.train.check_val_every_n_epoch)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    config_name = sys.argv[1]
    conf = OmegaConf.load(cur_dir / "config"/ f"{config_name}.yaml")
    conf.merge_with_cli()

    main(conf)

import os
import sys
import lightning.pytorch as pl
from pathlib import Path
from omegaconf import OmegaConf
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from dataset.baseline import BaselineDataModule
from dataset.simple_fc import FCBaselineDataModule
from dataset.ml_subprob import MPCDataModule
from experiment.baseline.baseline import AlfredBaseline
from experiment.baseline.without_vis import AlfredBaselineWithoutVis
from experiment.baseline.teacher_force import AlfredBaselineTeacherForce
from experiment.simple_fc.base import FCBaseline
from experiment.simple_fc.batch_normalize import FCBaselineBatchNormalize
from experiment.ml.mlp_v1 import MLPModuleV1
from experiment.rl.ddpg_v1 import DDPGModuleV1

cur_dir = Path(os.path.dirname(__file__)) 

DATA_MODULE = {
    'baseline': {
        'baseline': BaselineDataModule,
        'without_vis': BaselineDataModule,
        'teacher_force': BaselineDataModule,
    },
    'simple_fc': {
        'base': FCBaselineDataModule,
        'batch_normalize': FCBaselineDataModule,
    },
    'ml': {
        'mlp_v1': MPCDataModule,
    }
}

PL_MODULE = {
    'baseline': {
        'baseline': AlfredBaseline,
        'without_vis': AlfredBaselineWithoutVis,
        'teacher_force': AlfredBaselineTeacherForce,
    },
    'simple_fc': {
        'base': FCBaseline,
        'batch_normalize': FCBaselineBatchNormalize,
    },
    'ml': {
        'mlp_v1': MLPModuleV1
    }
}

RL_MODULE = {
    'ddpg_v1': DDPGModuleV1
}

def main(args):
    if args.project == 'rl':
        # run custom training function for RL
        rl = RL_MODULE[args.run](args)
        rl.train()
    else:
        pl.seed_everything(42)
        datamodule = DATA_MODULE[args.project][args.run](args)
        model = PL_MODULE[args.project][args.run](args)
        
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
                        deterministic=True)
        trainer.fit(model, datamodule)


if __name__ == "__main__":
    config_name = sys.argv[1]
    conf = OmegaConf.load(cur_dir / "config"/ f"{config_name}.yaml")
    conf.merge_with_cli()

    main(conf)

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule


class DatasetType:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MPCDataset(Dataset):
    """
    MPC Dataset is for MPC (Model Predictive Control) task.
    Model predicts actions (elev, ail, thrust) based on objective/error of states (pitch, roll, spd, vert_spd).
    
    Referenced dataset for LSTM PID > https://colab.research.google.com/github/APMonitor/pds/blob/main/LSTM_Automation.ipynb
    
    Dataset:
        x: [pitch, roll, spd, vert_spd, e_pitch, e_roll, e_spd, e_vert_spd]
        y: [elev, ail, thr]
    """
    def __init__(self, root: Path, dataset_type: DatasetType):
        self.root = root
        self.dataset_type = dataset_type

        split_path = self.root / "split" / f"{self.dataset_type}.json"
        with open(split_path, "r") as f:
            session_ids = json.load(f)
    
        sensory_observations = []   # pitch, roll, spd, vert_spd
        actions = []                # elev, ail, thr
        
        for session_id in session_ids:            
            with open(self.root / "data" / f"{session_id}.json", "r") as f:
                data = json.load(f)
            
            for datum in data:
                sensory_observations.append(torch.tensor([
                    *datum['state']['attitude'][:2], 
                    datum['state']['speed'], 
                    datum['state']['vertical_speed']
                ]))
                actions.append(torch.tensor([
                    # normalize all control values ranging from 0 to 1
                    (datum['control']['elevator'] + 1) / 2, 
                    (datum['control']['aileron'] + 1) / 2,
                    datum['control']['thrust'],
                ]))
        
        total = len(sensory_observations)
        err = [     
            sensory_observations[idx] - sensory_observations[idx - 1] for idx in range(1, total) # e_pitch, e_roll, e_spd, e_vert_spd
        ]
        
        sensory_observations.pop(0)
        total -= 1
        
        self.x = [torch.cat((sensory_observations[idx], err[idx])) for idx in range(total)]
        self.y = actions
        self.length = total

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class MPCDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.train = MPCDataset(Path(args.dataset.root), DatasetType.TRAIN)
        self.val = MPCDataset(Path(args.dataset.root), DatasetType.VAL)
        self.test = MPCDataset(Path(args.dataset.root), DatasetType.TEST)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.train.batch_size, shuffle=True, num_workers=self.args.train.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.args.test.batch_size, num_workers=self.args.test.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.args.test.batch_size, num_workers=self.args.test.num_workers)

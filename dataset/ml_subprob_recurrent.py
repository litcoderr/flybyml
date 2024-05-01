import torch
import math
import json
import random

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule


class DatasetType:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class MPCRecurrentDataset(Dataset):
    """
    MPC Recurrent Dataset is for MPC (Model Predictive Control) task.
    Model predicts actions (elev, ail, thrust) based on objective/error of 'sequential' states (pitch, roll, spd, vert_spd) in batch.
    
    Referenced dataset for LSTM PID > https://colab.research.google.com/github/APMonitor/pds/blob/main/LSTM_Automation.ipynb
    
    Dataset:
        x: [pitch, roll, spd, vert_spd, e_pitch, e_roll, e_spd, e_vert_spd]
        y: [elev, ail, thr]
    """
    def __init__(self, root: Path, dataset_type: DatasetType, seq_length=31):
        self.root = root
        self.dataset_type = dataset_type

        split_path = self.root / "split" / f"{self.dataset_type}.json"
        with open(split_path, "r") as f:
            self.session_ids = json.load(f)
            
        all_x = []
        all_y = []
        
        for session_id in self.session_ids:            
            with open(self.root / "data" / f"{session_id}.json", "r") as f:
                data = json.load(f)
                
            total_length = len(data)
            start_idx = random.randint(1, total_length-seq_length)
            
            obs = []   # pitch, roll, spd, vert_spd
            act = []   # elev, ail, thr
            
            for datum in data[start_idx:start_idx+seq_length]:
                obs.append(torch.tensor([
                    *datum['state']['attitude'][:2], 
                    datum['state']['speed'], 
                    datum['state']['vertical_speed']
                ]))
                act.append(torch.tensor([
                    # normalize all control values ranging from 0 to 1
                    (datum['control']['elevator'] + 1) / 2, 
                    (datum['control']['aileron'] + 1) / 2,
                    datum['control']['thrust'],
                ]))
                
            err = [     
                obs[idx] - obs[idx - 1] for idx in range(1, seq_length) # e_pitch, e_roll, e_spd, e_vert_spd
            ]
            obs.pop(0)
            x = [torch.cat((obs[idx], err[idx])) for idx in range(seq_length-1)]

            all_x.append(torch.stack(x))
            all_y.append(torch.stack(act[:-1]))
            
        assert len(all_x), 40
        assert all_x[0].shape, (seq_length-1, 8)
        assert all_y[0].shape, (seq_length-1, 3)
            
        self.x = all_x
        self.y = all_y
        
    def __len__(self):
        return len(self.session_ids)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MPCRecurrentDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.train = MPCRecurrentDataset(Path(args.dataset.root), DatasetType.TRAIN)
        self.val = MPCRecurrentDataset(Path(args.dataset.root), DatasetType.VAL)
        self.test = MPCRecurrentDataset(Path(args.dataset.root), DatasetType.TEST)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.train.batch_size, shuffle=True, num_workers=self.args.train.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.args.test.batch_size, num_workers=self.args.test.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.args.test.batch_size, num_workers=self.args.test.num_workers)

import math
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule


class DatasetType:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class FCBaselineDataset(Dataset):
    def __init__(self, root: Path, dataset_type: DatasetType):
        self.root = root
        self.dataset_type = dataset_type

        # load split
        split_path = self.root / "split" / f"{self.dataset_type}.json"
        with open(split_path, "r") as f:
            # session_ids: ['session_id_0', 'session_id_1, ...]
            session_ids = json.load(f)
    
        # load all data once
        sensory_observations = []
        instructions = []
        actions = []
        prev_actions = []
        
        for session_id in session_ids:
            # read target runway data
            with open(self.root / "meta" / f"{session_id}.json", "r") as f:
                meta = json.load(f)
                tgt_position = torch.tensor(meta['target_rwy']['position']) # runway [lat, lon, alt]
                tgt_heading = meta['target_rwy']['attitude'][2] # runway heading
            
            # read flight data
            with open(self.root / "data" / f"{session_id}.json", "r") as f:
                data = json.load(f)
            
            for idx, datum in enumerate(data[1:]):
                # construct instructions
                relative_position = torch.tensor(datum['state']['position']) - tgt_position
                relative_heading = datum['state']['attitude'][2] - tgt_heading
                if relative_heading > 180:
                    relative_heading = - (360 - relative_heading)
                elif relative_heading < -180:
                    relative_heading += 360
                relative_heading = torch.tensor([math.radians(relative_heading)])
                instruction = torch.concat((relative_position, relative_heading))
                instructions.append(instruction)

                # construct observations
                sensory_observations.append(torch.tensor([*datum['state']['attitude'][:2], datum['state']['speed'], datum['state']['vertical_speed']]))

                # construct actions
                # normalize all values ranging from 0 to 1
                actions.append(torch.tensor([
                    (datum['control']['elevator'] + 1) / 2,
                    (datum['control']['aileron'] + 1) / 2,
                    (datum['control']['rudder'] + 1) / 2,
                    datum['control']['thrust'],
                    datum['control']['gear'],
                    datum['control']['flaps'],
                    (datum['control']['trim'] + 1) / 2,
                    datum['control']['brake'],
                    datum['control']['speed_brake'],
                    datum['control']['reverse_thrust'] * -1,
                ]))

                prev_idx = idx - 1
                prev_actions.append(torch.tensor([
                    (data[prev_idx]['control']['elevator'] + 1) / 2,
                    (data[prev_idx]['control']['aileron'] + 1) / 2,
                    (data[prev_idx]['control']['rudder'] + 1) / 2,
                    data[prev_idx]['control']['thrust'],
                    data[prev_idx]['control']['gear'],
                    data[prev_idx]['control']['flaps'],
                    (data[prev_idx]['control']['trim'] + 1) / 2,
                    data[prev_idx]['control']['brake'],
                    data[prev_idx]['control']['speed_brake'],
                    data[prev_idx]['control']['reverse_thrust'] * -1,
                ]))
        
        self.data = {
            'sensory_observations': sensory_observations,
            'instructions': instructions,
            'actions': actions,
            'prev_actions': prev_actions
        }
        self.length = len(self.data['sensory_observations'])

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {
            'sensory_observations': self.data['sensory_observations'][idx],
            'instructions': self.data['instructions'][idx],
            'actions': self.data['actions'][idx],
            'prev_actions': self.data['prev_actions'][idx]
        }


class FCBaselineDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.train = FCBaselineDataset(Path(args.dataset.root), DatasetType.TRAIN)
        self.val = FCBaselineDataset(Path(args.dataset.root), DatasetType.VAL)
        self.test = FCBaselineDataset(Path(args.dataset.root), DatasetType.TEST)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.train.batch_size, shuffle=True, num_workers=self.args.train.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.args.test.batch_size, num_workers=self.args.test.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.args.test.batch_size, num_workers=self.args.test.num_workers)

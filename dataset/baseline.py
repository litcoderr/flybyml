import torch
import os
import math
import json
import random

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, CenterCrop
from lightning import LightningDataModule


class DatasetType:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def generate_split(root: Path):
    percentage = [0.8, 0.1, 0.1]

    # collect session_ids
    session_ids = [p.split(".json")[0] for p in os.listdir(root / "data")]
    # random shuffle session_ids
    random.shuffle(session_ids)

    train_idx = math.ceil(len(session_ids)*percentage[0])
    val_idx = train_idx + math.ceil(len(session_ids)*percentage[1])

    train_ids = session_ids[:train_idx]
    val_ids = session_ids[train_idx:val_idx]
    test_ids = session_ids[val_idx:]

    os.makedirs(root / "split", exist_ok=True)    
    with open(root / "split" / f"{DatasetType.TRAIN}.json", "w") as f:
        json.dump(train_ids, f)

    with open(root / "split" / f"{DatasetType.VAL}.json", "w") as f:
        json.dump(val_ids, f)

    with open(root / "split" / f"{DatasetType.TEST}.json", "w") as f:
        json.dump(test_ids, f)


class BaselineDataset(Dataset):
    def __init__(self, root: Path, dataset_type: DatasetType, seq_length=30):
        self.root = root
        self.dataset_type = dataset_type

        # load split
        split_path = self.root / "split" / f"{self.dataset_type}.json"
        with open(split_path, "r") as f:
            # self.split ex) ['session_id_0', 'session_id_1, ...]
            self.split = json.load(f)
        
        self.seq_length = seq_length

    def __len__(self):
        return len(self.split)
    
    def __getitem__(self, idx):
        # read target runway data
        with open(self.root / "meta" / f"{self.split[idx]}.json", "r") as f:
            meta = json.load(f)
            tgt_position = torch.tensor(meta['target_rwy']['position']) # runway [lat, lon, alt]
            tgt_heading = meta['target_rwy']['attitude'][2] # runway heading
        
        # read flight data
        with open(self.root / "data" / f"{self.split[idx]}.json", "r") as f:
            data = json.load(f)
        
        # sample starting index
        total_length = len(data)
        start_idx = random.randint(0, total_length-self.seq_length)

        sensory_observations = []
        instructions = []
        actions = []
        camera_actions = []
        for datum in data[start_idx:start_idx+self.seq_length]:
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

            # construct camera
            camera_actions.append(torch.tensor(datum['control']['camera']))

        # # load image
        # to_tensor = ToTensor()
        # resize = Resize(size=256)
        # center_crop = CenterCrop(size=224)

        # img_root = self.root / "image" / self.split[idx]
        # visual_observations = []
        # img_names = os.listdir(img_root)
        # img_names.sort()
        # for img_name in img_names[start_idx:start_idx+self.seq_length]:
        #     img = to_tensor(Image.open(img_root / img_name))
        #     resized_img = resize(img)
        #     cropped_img = center_crop(resized_img)
        #     visual_observations.append(cropped_img)
        # visual_observations = torch.stack(visual_observations, dim=0) # [seq_length, 3, 224, 224]

        sensory_observations = torch.stack(sensory_observations, dim=0)
        instructions = torch.stack(instructions, dim=0)
        actions = torch.stack(actions, dim=0)
        camera_actions = torch.stack(camera_actions, dim=0)

        return {
            # 'visual_observations': visual_observations,
            'sensory_observations': sensory_observations,
            'instructions': instructions,
            'actions': actions,
            'camera_actions': camera_actions
        }


class BaselineDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.train = BaselineDataset(Path(args.dataset.root), DatasetType.TRAIN)
        self.val = BaselineDataset(Path(args.dataset.root), DatasetType.VAL)
        self.test = BaselineDataset(Path(args.dataset.root), DatasetType.TEST)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.train.batch_size, shuffle=True, num_workers=self.args.train.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.args.test.batch_size, num_workers=self.args.test.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.args.test.batch_size, num_workers=self.args.test.num_workers)

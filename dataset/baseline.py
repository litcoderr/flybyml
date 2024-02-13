from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import os
import math
import json
import random
import pytorch_lightning as pl

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, CenterCrop



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
    def __init__(self, root: str, dataset_type: DatasetType):
        self.root = root
        self.dataset_type = dataset_type

        # load split
        split_path = self.root / "split" / f"{self.dataset_type}.json"
        with open(split_path, "r") as f:
            # self.split ex) ['session_id_0', 'session_id_1, ...]
            self.split = json.load(f)
        
        self.max_length = 0
        for session_id in self.split:
            with open(self.root / "data" / f"{session_id}.json", "r") as f:
                data = json.load(f)
            self.max_length = max(self.max_length, len(data))

    def __len__(self):
        return len(self.split)
    
    def __getitem__(self, idx):
        # load image
        to_tensor = ToTensor()
        resize = Resize(size=256)
        center_crop = CenterCrop(size=224)

        img_root = self.root / "image" / self.split[idx]
        imgs = []
        for img_name in os.listdir(img_root):
            img = to_tensor(Image.open(img_root / img_name))
            resized_img = resize(img)
            cropped_img = center_crop(resized_img)
            imgs.append(cropped_img)
        imgs = torch.stack(imgs, dim=0)

        # read target runway data
        with open(self.root / "meta" / f"{self.split[idx]}.json", "r") as f:
            meta = json.load(f)
            tgt_position = torch.tensor(meta['target_rwy']['position']) # runway [lat, lon, alt]
            tgt_heading = meta['target_rwy']['attitude'][2] # runway heading

        # read flight data
        with open(self.root / "data" / f"{self.split[idx]}.json", "r") as f:
            data = json.load(f)

        observations = []
        instructions = []
        actions = []
        camera = []
        for datum in data:
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
            observations.append(torch.tensor([*datum['state']['attitude'][:2], datum['state']['speed'], datum['state']['vertical_speed']]))

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
            camera.append(torch.tensor(datum['control']['camera']))
        
        observations = torch.stack(observations, dim=0)
        instructions = torch.stack(instructions, dim=0)
        actions = torch.stack(actions, dim=0)
        camera = torch.stack(camera, dim=0)
        # pad if needed
        if len(data) < self.max_length:
            pad_length = self.max_length - len(data)
            observations = torch.cat((observations, torch.zeros((pad_length, observations.shape[1]))), dim=0)
            instructions = torch.cat((instructions, torch.zeros((pad_length, instructions.shape[1]))), dim=0)
            actions = torch.cat((actions, torch.zeros((pad_length, actions.shape[1]))), dim=0)
            camera = torch.cat((camera, torch.zeros((pad_length, camera.shape[1]))), dim=0)

        return {
            'observation': observations,
            'instruction': instructions,
            'action': actions,
            'camera': camera,
            'length': torch.tensor([len(data)])
        }


class BaselineDataModule(pl.LightningDataModule):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.batch_size = 2

        self.train = BaselineDataset(self.root, DatasetType.TRAIN)
        self.val = BaselineDataset(self.root, DatasetType.VAL)
        self.test = BaselineDataset(self.root, DatasetType.TEST)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


if __name__ == "__main__":
    root_path = Path("/data/flybyml_dataset_v1")
    #generate_split(root_path)

    train_datamodule = BaselineDataModule(root_path)

    train_loader = train_datamodule.train_dataloader()
    
    for batch in train_loader:
        # TODO need collate function for padding varying input lengths
        pass

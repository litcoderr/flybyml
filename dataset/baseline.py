import os
import math
import json
import random
from torch.utils.data import Dataset
from pathlib import Path


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

        # get split data
        split_path = self.root / "split" / f"{self.dataset_type}.json"
        with open(split_path, "r") as f:
            self.split = json.load(f)

    def __len__(self):
        # TODO
        return 0
    
    def __getitem__(self, idx):
        # TODO
        return None


if __name__ == "__main__":
    root_path = Path("D:\\dataset\\flybyml_dataset")
    #generate_split(root_path)

    train_dataset = BaselineDataset(
        root = root_path,
        dataset_type = DatasetType.TRAIN
    )

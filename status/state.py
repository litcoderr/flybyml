import torch
from .pos import Position
from .att import Attitude


class State:
    """
    Generates state tensor
    """
    def __init__(self, pos: Position, att: Attitude, spd: float, gear: float):
        self.pos = pos
        self.att = att
        self.spd = spd
        self.gear = gear
    
    def __str__(self):
        return f"{self.pos} {self.att} [Spd] {self.spd} [Gear] {self.gear}"
    
    def toTensor(self, device='cuda'):
        """
        [alt, pitch, roll, yaw, spd]
        """
        return torch.tensor([self.pos.alt, self.att.pitch, self.att.roll, self.att.yaw, self.spd]).to(device)

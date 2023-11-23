import torch
from .pos import Position
from .att import Attitude


class PlaneState:
    """
    Generates state tensor
    """
    def __init__(self, pos: Position, att: Attitude, spd: float, vert_spd:float, gear: float):
        self.pos = pos
        self.att = att
        self.spd = spd # m/s
        self.vert_spd = vert_spd # m/s
        self.gear = gear
    
    def __str__(self):
        return f"{self.pos} {self.att} [Spd] {self.spd} [VSpd] {self.vert_spd} [Gear] {self.gear}"

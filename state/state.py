import torch
from .pos import Position
from .att import Attitude


class PlaneState:
    """
    Generates state tensor
    """
    def __init__(self, pos: Position = Position(0,0,0), att: Attitude = Attitude(0,0,0), spd: float = 0, vert_spd:float = 0, gear: float = 0):
        self.pos = pos
        self.att = att
        self.spd = spd # m/s
        self.vert_spd = vert_spd # m/s
        self.gear = gear
    
    def __str__(self):
        return f"{self.pos} {self.att} [Spd] {self.spd} [VSpd] {self.vert_spd} [Gear] {self.gear}"

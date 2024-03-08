from .pos import Position
from .att import Attitude


class PlaneState:
    """
    Generates state tensor
    """
    def __init__(self, pos: Position = Position(0,0,0), att: Attitude = Attitude(0,0,0), spd: float = 0, vert_spd:float = 0):
        self.pos = pos
        self.att = att
        self.spd = spd # m/s
        self.vert_spd = vert_spd # m/s
    
    def __str__(self):
        return f"{self.pos} {self.att} [Spd] {self.spd} [VSpd] {self.vert_spd}"
    
    def serialize(self):
        return {
            "position": [self.pos.lat, self.pos.lon, self.pos.alt],
            "attitude": [self.att.pitch, self.att.roll, self.att.yaw],
            "speed": self.spd,
            "vertical_speed": self.vert_spd
        }
    
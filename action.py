from typing import Tuple

import xpc
from pos import Position
from att import Attitude

def setPos(pos: Position, att: Attitude, gear: float):
    with xpc.XPlaneConnect() as client:
        client.sendPOSI([pos.lat, pos.lon, pos.alt, att.pitch, att.roll, att.yaw, gear], 0)

def getPos() -> Tuple[Position, Attitude, float]:
    with xpc.XPlaneConnect() as client:
        Lat, Lon, Alt, Pitch, Roll, Yaw, Gear = client.getPOSI(0)

        pos = Position(Lat, Lon, Alt)
        att = Attitude(Pitch, Roll, Yaw)
    
    return pos, att, Gear

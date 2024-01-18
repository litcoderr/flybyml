from typing import List

class Camera:
    def __init__(self, x, y, z, heading, pitch, roll):
        self.x = x
        self.y = y
        self.z = z
        self.heading = heading
        self.pitch = pitch
        self.roll = roll
    
    def __str__(self) -> str:
        return f"x[{self.x}] y[{self.y}] z[{self.z}] heading[{self.heading}] pitch[{self.pitch}] roll[{self.roll}]"

class Controls:
    def __init__(self, elev, ail, rud, thr, gear, flaps, trim, brake, reverse, camera: Camera):
        """
        lat: [-1, 1]
        lon: [-1, 1]
        rud: [-1, 1]
        thr: [0, 1]
        gear: [0 / 1]
        flaps: [0, 1]
        trim: [-1, 1]
        brake: [0, 1]
        reverse: [-1 / 0] -1: reverse thrust fully deployed
        """
        self.elev = elev
        self.ail = ail
        self.rud = rud
        self.thr = thr
        self.gear = gear
        self.flaps = flaps
        self.trim = trim
        self.brake = brake
        self.reverse = reverse
        self.camera = camera
    
    def to_api_compatible(self) -> List[float]:
        return [float(self.elev), float(self.ail), float(self.rud), float(self.thr), float(self.gear), float(self.flaps)]

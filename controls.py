from typing import List

class Controls:
    def __init__(self, elev, ail, rud, thr, gear, flaps, trim):
        """
        lat: [-1, 1]
        lon: [-1, 1]
        rud: [-1, 1]
        thr: [0, 1]
        gear: [0 / 1]
        flaps: [0, 1]
        trim: [-1, 1]
        """
        self.elev = elev
        self.ail = ail
        self.rud = rud
        self.thr = thr
        self.gear = gear
        self.flaps = flaps
        self.trim = trim
    
    def to_api_compatible(self) -> List[float]:
        return [float(self.elev), float(self.ail), float(self.rud), float(self.thr), float(self.gear), float(self.flaps)]

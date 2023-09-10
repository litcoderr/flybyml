class Attitude:
    def __init__(self, pitch: float, roll: float, yaw: float):
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw

    def __str__(self):
        return f"[Att] pitch: {self.pitch:.1f} | roll: {self.roll:.1f} | yaw: {self.yaw:.1f}"

class CessnaInit(Attitude):
    def __init__(self, heading: float):
        super().__init__(5, 0, heading)
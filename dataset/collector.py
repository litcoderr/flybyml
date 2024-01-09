from aircraft import Aircraft
from aircraft.b738 import B738
from state.pos import Position


class Config:
    aircraft: Aircraft = B738
    init_pos: Position = Position(lat=0, lon=0, alt=0)
    heading: float = 0 # degrees
    speed: float = 0 # m/s

if __name__ == "__main__":
    # TODO 1. randomize configuration

    # TODO 2. apply configuration and launch session

    # TODO 3. record until touchdown at designated runway
    pass
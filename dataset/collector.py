from aircraft import Aircraft
from aircraft.b738 import B738
from state.pos import Position
from state.state import PlaneState
from controls import Controls
from agent import AgentInterface


class Config:
    def __init__(self):
        self.aircraft: Aircraft = B738
        self.init_pos: Position = Position(lat=0, lon=0, alt=0)
        self.init_heading: float = 0 # degrees
        self.init_speed: float = 0 # m/s

class HumanAgent(AgentInterface):
    def __init__(self, config: Config):
        super().__init__(config.aircraft)
        self.config = config
    
    def sample_action(self, state: PlaneState) -> Controls:
        # TODO
        pass

if __name__ == "__main__":
    # TODO 1. randomize configuration

    # TODO 2. apply configuration and launch session

    # TODO 3. record until touchdown at designated runway
    pass
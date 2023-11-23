from state.plane import PlaneState
from aircraft import Aircraft
from controls import Controls

class AgentInterface:
    """
    Agent Interface
    """
    def __init__(self, aircraft: Aircraft):
        self.aircraft = aircraft

    def sample_action(self, state: PlaneState) -> Controls:
        raise NotImplementedError()

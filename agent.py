from state.plane import PlaneState
from aircraft import Aircraft

class AgentInterface:
    """
    Agent Interface
    """
    def __init__(self, aircraft: Aircraft):
        self.aircraft = aircraft
        pass

    def sample_action(self, state: PlaneState):
        pass

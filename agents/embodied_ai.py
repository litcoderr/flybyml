from agents import AgentInterface
from flybyml.aircraft import Aircraft
from state.state import PlaneState
from controls import Controls

from PIL.Image import Image

class MLAgentBase(AgentInterface):
    def __init__(self, aircraft: Aircraft, success_distance: float):
        super().__init__(aircraft)
        self.dist_threshold_to_stop = success_distance
        self.control_buffer = Controls( # maintains previous control input
            elev = 0,
            ail = 0,
            rud = 0,
            thr = 0.8,
            gear = 1,
            flaps = 0,
            trim = 0
        )

    def is_goal_reached(self, state: PlaneState) -> bool:
        # TODO: define stop condition
        pass

'''
Clip Encoder + Reinforcement Learning (PPO)
'''
class ClipAgent(MLAgentBase):
    def __init__(self, aircraft, success_distance):
        super().__init__(aircraft, success_distance)
    
    def sample_action(self, state: PlaneState, vision: Image) -> Controls:
        if self.is_goal_reached(state):
            return None
        controls = Controls(
            # TODO: action
        )
        self.control_buffer = controls
        return self.control_buffer

'''
ResNet-50 Conv 
'''
class AlfredAgent(MLAgentBase):
    def __init__(self, aircraft, success_distance):
        super().__init__(aircraft, success_distance)
    
    def sample_action(self, state: PlaneState, vision: Image) -> Controls:
        if self.is_goal_reached(state):
            return None
        controls = Controls(
            # TODO: action
        )
        self.control_buffer = controls
        return self.control_buffer
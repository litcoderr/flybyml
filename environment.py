from typing import Tuple

import time

from agent import AgentInterface
from api import API
from state.state import PlaneState
from controls import Controls


class XplaneEnvironment:
    def __init__(self, agent: AgentInterface):
        self.agent = agent

        self.api = API()  # xpilot controller
    
    def reset(self, lat, lon, alt, heading, spd) -> PlaneState:
        """
        lat: degree
        lon: degree
        heading: degree
        alt: m
        spd: m/s
        """
        self.api.set_init_state(self.agent.aircraft, lat, lon, alt, heading, spd)
        state = self.getState()

        return state
    
    def step(self, **kwargs) -> Tuple[PlaneState, Controls, float]:
        current_state = self.getState()
        controls = self.agent.sample_action(current_state)
        self.api.send_ctrl(controls)
        return current_state, controls, time.time()

    def getState(self) -> PlaneState:
        while True:
            try:
                pos, att = self.api.get_posi_att()
                spd = self.api.get_indicated_airspeed()
                vert_spd = self.api.get_vertical_speed()
                state = PlaneState(pos, att, spd, vert_spd)
                break
            except:
                time.sleep(0.1)
        return state

import time
import torch

from agent import AgentInterface
from api import API
from state.state import PlaneState
from state.pos import AirportPosition


class XplaneEnvironment:
    def __init__(self, agent: AgentInterface, airport: AirportPosition, frame_interval: float):
        self.agent = agent
        self.airport = airport
        self.frame_interval = frame_interval

        self.api = API()  # xpilot controller
    
    def reset(self, heading, alt, spd) -> PlaneState:
        """
        heading: degree
        alt: m
        spd: m/s
        """
        # TODO should change to set position by absolute position
        self.api.set_posi_by_airport(self.agent.aircraft, self.airport, alt, heading, spd)
        state = self.getState()
        self.api.pause()

        return state
    
    def step(self, prev_state: PlaneState) -> PlaneState:
        controls = self.agent.sample_action(prev_state)

        self.api.resume()
        self.api.send_ctrl(controls)
        time.sleep(self.frame_interval)
        self.api.pause()
        return self.getState()

    def getState(self) -> PlaneState:
        while True:
            try:
                pos, att = self.api.get_posi()
                spd = self.api.get_indicated_airspeed()
                vert_spd = self.api.get_vertical_speed()
                state = PlaneState(pos, att, spd, vert_spd)
                break
            except:
                time.sleep(0.1)
        return state

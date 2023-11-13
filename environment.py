import time
import torch

from agent import AgentInterface
from api import API
from state.plane import PlaneState
from state.pos import AirportPosition


class XplaneEnvironment:
    def __init__(self, agent: AgentInterface, airport: AirportPosition, frame_interval: float):
        self.agent = agent
        self.airport = airport
        self.frame_interval = frame_interval

        self.api = API()  # xpilot controller
    
    def reset(self, heading, alt, spd):
        """
        heading: degree
        alt: m
        spd: m/s
        """
        self.api.set_posi(self.agent.aircraft, self.airport, alt, heading, spd)
        state = self.getState()
        self.api.pause()

        return state
    
    def step(self, action: torch.Tensor):
        self.api.resume()
        # TODO 1. input action
        time.sleep(self.frame_interval)
        self.api.pause()
        return self.getState()

    def getState(self) -> PlaneState:
        while True:
            try:
                pos, att, gear = self.api.get_posi()
                spd = self.api.get_indicated_airspeed()
                state = PlaneState(pos, att, spd, gear)
                break
            except:
                time.sleep(0.1)
        return state

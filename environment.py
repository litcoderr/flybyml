import time
import torch

from xp import XP
from util import ft_to_me, kts_to_mps
from status.state import State


class XplaneEnvironment:
    def __init__(self, aircraft, airport, frame_interval):
        self.aircraft = aircraft
        self.airport = airport
        self.frame_interval = frame_interval

        self.xp = XP()  # xpilot controller
    
    def reset(self, heading, alt, spd):
        self.xp.set_posi(self.aircraft, self.airport, alt, heading, kts_to_mps(spd))
        state = self.getState()
        self.xp.pause()

        return state
    
    def step(self, action: torch.Tensor):
        self.xp.resume()
        # 1. input action
        time.sleep(self.frame_interval)
        self.xp.pause()
        return self.getState()

    def getState(self) -> State:
        while True:
            try:
                pos, att, gear = self.xp.get_posi()
                spd = self.xp.get_indicated_airspeed()
                state = State(pos, att, spd, gear)
                break
            except:
                time.sleep(0.1)
        return state

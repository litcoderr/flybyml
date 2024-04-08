from typing import Tuple, Optional

import time

from aircraft.b738 import B738
from util import sample_weather, kts_to_mps
from agents import AgentInterface
from api import API
from state.state import PlaneState
from controls import Controls
from weather import Weather


class XplaneEnvironment:
    def __init__(self, agent: AgentInterface):
        self.agent = agent

        self.api = API()  # xpilot controller
    
    def reset(self, lat, lon, alt, heading, spd, zulu_time, 
              weather: Optional[Weather]=None, controls: Optional[Controls]=None, pause=False) -> Tuple[PlaneState, Controls]:
        """
        lat: degree
        lon: degree
        heading: degree
        alt: m
        spd: m/s
        zulu_time: GMT time. seconds since midnight
        weather: Weather object
        controls: Controls object
        """
        self.api.resume()
        if weather == None:
            weather = sample_weather()

        self.api.set_init_state(B738(), lat, lon, alt, heading, spd)
        # set time / weather
        while True:
            try:
                self.api.set_zulu_time(zulu_time)
                self.api.set_weather(weather)
                break
            except Exception as e:
                print(e)
                time.sleep(0.1)
        # init controls / state
        if controls:
            self.api.send_ctrl(controls)
            init_ctrl = controls
        else:
            init_ctrl = self.api.init_ctrl() 
        while True:
            # wait for simulator to load
            init_state = self.get_state()
            if init_state.spd > kts_to_mps(200):
                break
        
        if pause:
            self.api.pause()

        return init_state, init_ctrl
    
    def step(self, **kwargs) -> Tuple[PlaneState, Controls, float]:
        while True:
            try:
                current_state = self.get_state()
                controls = self.agent.sample_action(current_state, **kwargs)
                self.api.send_ctrl(controls)
                break
            except:
                time.sleep(0.1)
        return current_state, controls, time.time()

    def rl_step(self, action: Controls, step_interval=0.1) -> PlaneState:
        """
        take a step in this environment with given action(Controls)

        Args:
            action: Controls
            step_interval: interval (in seconds) between prev state and next state

        Return:
            next PlaneState
        """
        while True:
            try:
                self.api.resume()
                self.api.send_ctrl(action)
                break
            except:
                self.api.pause()
                time.sleep(0.05)
        time.sleep(step_interval)
        next_state = self.get_state()
        self.api.pause()
        return next_state

    def get_state(self) -> PlaneState:
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

from typing import Optional

import random

from aircraft import Aircraft
from aircraft.b738 import B738
from state.pos import Position
from state.state import PlaneState
from controls import Controls
from agent import AgentInterface
from api import API
from environment import XplaneEnvironment
from weather import Weather, ChangeMode, CloudBaseMsl, CloudCoverage


class Config:
    aircraft: Aircraft = B738()
    init_pos: Position = Position(lat=37.5326, lon=127.024612, alt=1000) 
    init_heading: float = 0 # degrees
    init_speed: float = 128 # m/s
    init_zulu_time: float = 0 # GMT time. seconds since midnight
    weather: Optional[Weather] = None

class HumanAgent(AgentInterface):
    def __init__(self, aircraft: Aircraft):
        super().__init__(aircraft)
        self.api: Optional[API] = None
    
    def set_api(self, api: API):
        self.api = api
    
    def sample_action(self, _: PlaneState, **kwargs) -> Controls:
        if self.api is None:
            raise Exception("Use set_api(api) to set api")
        controls = self.api.get_ctrl()
        return controls

def sample_zulu_time() -> float:
    """
    returns GMT time. seconds since midnight
    """
    seconds_per_day = 86400
    # TODO return seconds_per_day * random.random()
    return 200 * 60

def sample_weather() -> Weather:
    """
    returns sampled Weather object
    """
    weather = Weather(
        change_mode = ChangeMode(0),
        cloud_base_msl = CloudBaseMsl([500*0.3, 0, 0]),
        cloud_coverage = CloudCoverage([10, 0, 0])
    )
    # TODO sample weather properties by implementing sample method to weather
    return weather

if __name__ == "__main__":
    # set up human agent and environment
    human = HumanAgent(Config.aircraft)
    env = XplaneEnvironment(agent = human)
    human.set_api(env.api)

    # TODO randomize configuration for every run
    Config.init_zulu_time = sample_zulu_time()
    Config.weather = sample_weather()

    state = env.reset(
        lat = Config.init_pos.lat,
        lon = Config.init_pos.lon,
        alt = Config.init_pos.alt,
        heading = Config.init_heading,
        spd = Config.init_speed,
        zulu_time = Config.init_zulu_time,
        weather = Config.weather
    )

    while True:
        state, controls, abs_time = env.step()

from typing import Optional

import random
import numpy as np

from util import ft_to_me, haversine_distance_and_bearing
from aircraft import Aircraft
from aircraft.b738 import B738
from state.pos import Position
from state.state import PlaneState
from controls import Controls
from agent import AgentInterface
from airport import sample_tgt_rwy_and_position
from api import API
from environment import XplaneEnvironment
from weather import Weather, ChangeMode, \
    CloudBaseMsl, CloudTopMsl, CloudCoverage, CloudType, \
    Precipitation, RunwayWetness, Temperature, \
    WindMsl, WindDirection, WindSpeed, WindTurbulence, WindShearDirection, WindShearMaxSpeed


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
    return seconds_per_day * random.random()

def sample_weather(apt_elev: float) -> Weather:
    """
    apt_elev: airport elevation in meters
    returns sampled Weather object
    """
    # sample cloud
    cloud_base_msl = apt_elev + random.uniform(ft_to_me(1000), ft_to_me(4000))
    cloud_top_msl = random.uniform(cloud_base_msl + ft_to_me(1000), cloud_base_msl + ft_to_me(4000))
    cloud_coverage = random.uniform(0, 1)
    cloud_type = np.random.choice(np.arange(0, 4), p=[0.3, 0.3, 0.3, 0.1])

    # TODO sample precipitation and wind

    weather = Weather(
        change_mode = ChangeMode(random.randint(0, 6)),

        cloud_base_msl = CloudBaseMsl([cloud_base_msl, 0, 0]),
        cloud_top_msl = CloudTopMsl([cloud_top_msl, 0, 0]),
        cloud_coverage = CloudCoverage([cloud_coverage, 0, 0]),
        cloud_type = CloudType([cloud_type, 0, 0]),

        precipitation = Precipitation(0),
        runway_wetness = RunwayWetness(0),
        temperature = Temperature(-30),

        wind_msl = WindMsl([1000*0.3, 2100*0.3,0,0,0,0,0,0,0,0,0,0,0]),
        wind_direction = WindDirection([100,120,0,0,0,0,0,0,0,0,0,0,0]),
        wind_speed = WindSpeed([5,10,0,0,0,0,0,0,0,0,0,0,0]),
        wind_turbulence = WindTurbulence([0.5,0.5,0,0,0,0,0,0,0,0,0,0,0]),
        wind_shear_direction = WindShearDirection([100,120,0,0,0,0,0,0,0,0,0,0,0]),
        wind_shear_max_speed = WindShearMaxSpeed([5,10,0,0,0,0,0,0,0,0,0,0,0])
    )
    return weather

if __name__ == "__main__":
    # set up human agent and environment
    human = HumanAgent(Config.aircraft)
    env = XplaneEnvironment(agent = human)
    human.set_api(env.api)

    while True:
        # randomize configuration
        target_rwy, init_lat, init_lon = sample_tgt_rwy_and_position()
        Config.init_pos.lat = init_lat
        Config.init_pos.lon = init_lon
        Config.init_pos.alt = random.uniform(target_rwy.elev+ft_to_me(3000), target_rwy.elev+ft_to_me(5000))
        Config.init_zulu_time = sample_zulu_time()
        Config.weather = sample_weather(target_rwy.elev)

        state = env.reset(
            lat = Config.init_pos.lat,
            lon = Config.init_pos.lon,
            alt = Config.init_pos.alt,
            heading = Config.init_heading,
            spd = Config.init_speed,
            zulu_time = Config.init_zulu_time,
            weather = Config.weather
        )

        prev_state: Optional[PlaneState] = None
        while True:
            state, controls, abs_time = env.step()
            if prev_state is not None:
                dist, _ = haversine_distance_and_bearing(prev_state.pos.lat, prev_state.pos.lon, 0,
                                                         state.pos.lat, state.pos.lon, 0)
                if dist < 0.5:
                    break
            prev_state = state
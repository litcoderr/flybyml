from typing import Optional

import os
import time
import json
import random
import numpy as np
from queue import Queue
from pathlib import Path
import pygetwindow as pw
import pyautogui
import uuid

from util import ft_to_me, haversine_distance_and_bearing, kts_to_mps
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
from dataset.collector.atc import ATC
from dataset.collector.gui import StarterGui


class Config:
    aircraft: Aircraft = B738()
    init_pos: Position = Position(lat=37.5326, lon=127.024612, alt=1000) 
    init_heading: float = 0 # degrees
    init_speed: float = 128 # m/s
    init_zulu_time: float = 0 # GMT time. seconds since midnight
    weather: Optional[Weather] = None


def save_meta_data(root_dir, name):
    meta_path = Path(root_dir) / f"{name}.json"
    meta = {
        "init_zulu_time": Config.init_zulu_time,
        "weather": Config.weather.serialize()
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def take_screen_shot(root_dir, name):
    img_path = Path(root_dir) / f"{name}.jpg"
    window = pw.getWindowsWithTitle("X-System")
    x, y, width, height = window[0].left, window[0].top, window[0].width, window[0].height
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    # TODO maybe resize to make image smaller
    screenshot.save(img_path)


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
    cloud_top_msl = random.uniform(cloud_base_msl + ft_to_me(5000), cloud_base_msl + ft_to_me(8000))
    cloud_coverage = random.uniform(0, 1)
    cloud_type = float(np.random.choice(np.arange(0, 4), p=[0.3, 0.3, 0.3, 0.1]))

    # sample precipitation
    precipitation = random.uniform(0, 1)
    temperature = random.uniform(-50, 50)
    if precipitation < 0.1:
        runway_wetness = 0
    else:
        if temperature < 0:
            runway_wetness = 6 * precipitation - random.uniform(-1, 0)
        else:
            runway_wetness = 7 + 6 * precipitation - random.uniform(-1, 0)

    wind_msl = random.uniform(apt_elev + ft_to_me(50), cloud_top_msl)
    wind_direction = random.uniform(0, 360)
    wind_speed = random.uniform(0, kts_to_mps(16))
    wind_turbulence = float(np.random.choice(np.arange(0, 1, 0.2), p=[0.3,0.4,0.2,0.05,0.05]))
    wind_shear_direction = random.uniform(wind_direction-10, wind_direction+10)
    if wind_shear_direction < 0:
        wind_shear_direction += 360
    wind_shear_direction %= 360
    wind_shear_max_speed = float(np.random.choice(np.arange(0,20,2), p=[0.5,0.2,0.2,0.03,0.02,0.01,0.01,0.01,0.01,0.01]))

    weather = Weather(
        change_mode = ChangeMode(random.randint(0, 6)),

        cloud_base_msl = CloudBaseMsl([cloud_base_msl, 0, 0]),
        cloud_top_msl = CloudTopMsl([cloud_top_msl, 0, 0]),
        cloud_coverage = CloudCoverage([cloud_coverage, 0, 0]),
        cloud_type = CloudType([cloud_type, 0, 0]),

        precipitation = Precipitation(precipitation),
        runway_wetness = RunwayWetness(runway_wetness),
        temperature = Temperature(temperature),

        wind_msl = WindMsl([wind_msl, wind_msl + ft_to_me(1100),0,0,0,0,0,0,0,0,0,0,0]),
        wind_direction = WindDirection([wind_direction,wind_direction,0,0,0,0,0,0,0,0,0,0,0]),
        wind_speed = WindSpeed([wind_speed,wind_speed,0,0,0,0,0,0,0,0,0,0,0]),
        wind_turbulence = WindTurbulence([wind_turbulence,wind_turbulence,0,0,0,0,0,0,0,0,0,0,0]),
        wind_shear_direction = WindShearDirection([wind_shear_direction,wind_shear_direction,0,0,0,0,0,0,0,0,0,0,0]),
        wind_shear_max_speed = WindShearMaxSpeed([wind_shear_max_speed,wind_shear_max_speed,0,0,0,0,0,0,0,0,0,0,0])
    )
    return weather

if __name__ == "__main__":
    DATASET_ROOT = Path('D:\\dataset\\flybyml_dataset')
    IMG_ROOT = DATASET_ROOT / "image"
    METADATA_ROOT = DATASET_ROOT / "meta"
    CONTROL_ROOT = DATASET_ROOT / "control"
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(IMG_ROOT, exist_ok=True)
    os.makedirs(METADATA_ROOT, exist_ok=True)
    os.makedirs(CONTROL_ROOT, exist_ok=True)

    # set up human agent and environment
    human = HumanAgent(Config.aircraft)
    env = XplaneEnvironment(agent = human)
    human.set_api(env.api)

    while True:
        session_id = str(uuid.uuid4())
        img_dir = IMG_ROOT / session_id
        os.makedirs(img_dir, exist_ok=True)

        # randomize configuration
        target_rwy, init_lat, init_lon = sample_tgt_rwy_and_position()
        Config.init_pos.lat = init_lat
        Config.init_pos.lon = init_lon
        Config.init_pos.alt = random.uniform(target_rwy.elev+ft_to_me(4000), target_rwy.elev+ft_to_me(5000))
        Config.init_zulu_time = sample_zulu_time()
        Config.weather = sample_weather(target_rwy.elev)
        save_meta_data(METADATA_ROOT, session_id)

        state = env.reset(
            lat = Config.init_pos.lat,
            lon = Config.init_pos.lon,
            alt = Config.init_pos.alt,
            heading = Config.init_heading,
            spd = Config.init_speed,
            zulu_time = Config.init_zulu_time,
            weather = Config.weather
        )
        env.api.pause()

        # launch tkinter gui app that shows runway information.
        starter_gui = StarterGui(env.api, target_rwy)
        starter_gui.mainloop()
        window = pw.getWindowsWithTitle("X-System")
        window[0].activate()

        # initialize atc
        queue = Queue()
        atc = ATC(queue, target_rwy)
        atc.start()

        # run simulation until end of session
        prev_state: Optional[PlaneState] = None
        step_id = 0
        while True:
            state, controls, abs_time = env.step()
            take_screen_shot(img_dir, str(step_id).zfill(5))
            # TODO write data

            # send atc plane's state
            queue.put({"timestamp": time.time(), "is_running": True, "state": state})

            # stopping session if plane is not moving
            if prev_state is not None:
                dist, _ = haversine_distance_and_bearing(prev_state.pos.lat, prev_state.pos.lon, 0,
                                                         state.pos.lat, state.pos.lon, 0)
                if dist < 0.5:
                    break
            prev_state = state
            step_id += 1

        queue.put({"timestamp": time.time(), "is_running": False})
        atc.join()

from typing import Optional

import os
import sys
import time
import random
import numpy as np
import pygetwindow as pw
import pyautogui
from PIL.Image import Image
from pathlib import Path
from omegaconf import OmegaConf

from controls import Controls, Camera
from util import ft_to_me
from state.pos import Position
from airport import sample_tgt_rwy_and_position, Runway
from environment import XplaneEnvironment
from weather import Weather, ChangeMode, \
    CloudBaseMsl, CloudTopMsl, CloudCoverage, CloudType, \
    Precipitation, RunwayWetness, Temperature, \
    WindMsl, WindDirection, WindSpeed, WindTurbulence, WindShearDirection, WindShearMaxSpeed
from dataset.collector.gui import StarterGui

from agents.embodied_ai import AlfredBaselineTeacherForceAgent


class Config:
    init_pos: Position = Position(lat=37.5326, lon=127.024612, alt=1000) 
    init_heading: float = 0 # degrees
    init_speed: float = 128 # m/s
    init_zulu_time: float = 0 # GMT time. seconds since midnight
    weather: Optional[Weather] = None
    tgt_rwy: Optional[Runway] = None


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
    cloud_base_msl = apt_elev + random.uniform(ft_to_me(1500), ft_to_me(4000))
    cloud_top_msl = random.uniform(cloud_base_msl + ft_to_me(5000), cloud_base_msl + ft_to_me(8000))
    cloud_coverage = random.uniform(0, 1)
    cloud_type = float(np.random.choice(np.arange(0, 4), p=[0.4, 0.4, 0.2, 0.0]))

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
    wind_speed = float(np.random.choice(np.arange(0, 10, 2), p=[0.4, 0.4, 0.2, 0, 0]))
    wind_turbulence = float(np.random.choice(np.arange(0, 1, 0.2), p=[0.6, 0.35, 0.05, 0, 0]))
    wind_shear_direction = random.uniform(wind_direction-5, wind_direction+5)
    if wind_shear_direction < 0:
        wind_shear_direction += 360
    wind_shear_direction %= 360
    wind_shear_max_speed = float(np.random.choice(np.arange(0,5,1), p=[0.5, 0.3, 0.15, 0.05, 0]))

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


def get_screen() -> Image:
    window = pw.getWindowsWithTitle("X-System")
    x, y, width, height = window[0].left, window[0].top, window[0].width, window[0].height
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return screenshot


AGENT = {
    'baseline': {
        'teacher_force': AlfredBaselineTeacherForceAgent
    }
}

if __name__ == "__main__":
    cur_dir = Path(os.path.dirname(__file__)) 
    config_name = sys.argv[1]
    conf = OmegaConf.load(cur_dir / "config"/ f"{config_name}.yaml")
    conf.merge_with_cli()
    
    # set up human agent and environment
    agent = AGENT[conf.project][conf.run](conf)
    env = XplaneEnvironment(agent = agent)

    while True:
        # randomize configuration
        target_rwy, init_lat, init_lon = sample_tgt_rwy_and_position()
        Config.init_pos.lat = init_lat
        Config.init_pos.lon = init_lon
        Config.init_pos.alt = random.uniform(target_rwy.elev+ft_to_me(4000), target_rwy.elev+ft_to_me(5000))
        Config.init_zulu_time = sample_zulu_time()
        Config.weather = sample_weather(target_rwy.elev)
        Config.tgt_rwy = target_rwy

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

        # run simulation until end of session
        start_time = time.time()
        step_id = 0
        prev_controls = Controls(*[0 for _ in range(10)], camera=Camera(*[0 for _ in range(6)]))
        while True:
            # get state and control inputs
            screen = get_screen()
            state, controls, abs_time = env.step(
                screen=screen,
                tgt_rwy=Config.tgt_rwy,
                prev_actions=prev_controls
            )
            rel_time = abs_time - start_time
            prev_controls = controls

            step_id += 1

from typing import Optional

import sys
from omegaconf import OmegaConf

import math
import torch
import time
import random
import numpy as np
import pygetwindow as pw
import pyautogui
from PIL.Image import Image
from torchvision.transforms import ToTensor, Resize, CenterCrop

from util import ft_to_me
from aircraft import Aircraft
from aircraft.b738 import B738
from state.pos import Position
from state.state import PlaneState
from controls import Controls, Camera
from agents import AgentInterface
from airport import sample_tgt_rwy_and_position, Runway
from api import API
from environment import XplaneEnvironment
from weather import Weather, ChangeMode, \
    CloudBaseMsl, CloudTopMsl, CloudCoverage, CloudType, \
    Precipitation, RunwayWetness, Temperature, \
    WindMsl, WindDirection, WindSpeed, WindTurbulence, WindShearDirection, WindShearMaxSpeed
from dataset.collector.gui import StarterGui
from experiment.baseline.main_module import AlfredBaseline


class Config:
    aircraft: Aircraft = B738()
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


class MLAgent(AgentInterface):
    def __init__(self, args, aircraft: Aircraft):
        super().__init__(aircraft)
        self.api: Optional[API] = None
        self.model = AlfredBaseline.load_from_checkpoint('C:\\Users\\litco\\Desktop\\project\\flybyml\\checkpoint\\epoch=7041-step=35210.ckpt', args=args.model).to('cuda')
        self.context = None
    
    def set_api(self, api: API):
        self.api = api
    
    def sample_action(self, state: PlaneState, **kwargs) -> Controls:
        if self.api is None:
            raise Exception("Use set_api(api) to set api")
        # 1. Construct Model Input

        # set target
        tgt_rwy = Config.tgt_rwy.serialize()
        tgt_position = torch.tensor(tgt_rwy['position'])
        tgt_heading = tgt_rwy['attitude'][2]

        # state and image input
        state = state.serialize()

        # 1-1. construct instruction
        relative_position = torch.tensor(state['position']) - tgt_position
        relative_heading = state['attitude'][2] - tgt_heading
        if relative_heading > 180:
            relative_heading = - (360 - relative_heading)
        elif relative_heading < -180:
            relative_heading += 360
        relative_heading = torch.tensor([math.radians(relative_heading)])
        instruction = torch.concat((relative_position, relative_heading))
        instruction = instruction.reshape(1, 1, *instruction.shape)

        # 1-2. construct sensory observation
        sensory_observation = torch.tensor([*state['attitude'][:2], state['speed'], state['vertical_speed']])
        sensory_observation = sensory_observation.reshape(1, 1, *sensory_observation.shape)

        # 1-3. construct visual observation
        to_tensor = ToTensor()
        resize = Resize(size=256)
        center_crop = CenterCrop(size=224)

        visual_observation = to_tensor(kwargs['screen'])
        visual_observation = resize(visual_observation)
        visual_observation = center_crop(visual_observation)
        visual_observation = visual_observation.reshape(1, 1, *visual_observation.shape)

        # 2. infer
        with torch.no_grad():
            output, context = self.model({
                'visual_observations': visual_observation.to('cuda'),
                'sensory_observations': sensory_observation.to('cuda'),
                'instructions': instruction.to('cuda'),
            }, self.context)
            self.context = context
            output = output[0][0].to('cpu')
        
        # 3. construct action
        elevator = float(2 * output[0] - 1)
        aileron = float(2 * output[1] - 1)
        rudder = float(2 * output[2] - 1)
        thrust = float(output[3])
        gear = float(output[4])
        flaps = float(output[5])
        trim = float(2 * output[6] - 1)
        brake = float(output[7])
        spd_brake = float(output[8])
        reverser = float(-1 * output[9])

        controls = Controls(elev=elevator,
                            ail=aileron,
                            rud=rudder,
                            thr=thrust,
                            gear=gear,
                            flaps=flaps,
                            trim=trim,
                            brake=brake,
                            spd_brake=spd_brake,
                            reverse=reverser,
                            camera=Camera(*[float(camera_input) for camera_input in output[10:]]))
        return controls


if __name__ == "__main__":
    conf = OmegaConf.load(sys.argv[1])
    conf.merge_with_cli()
    
    # set up human agent and environment
    agent = MLAgent(conf, Config.aircraft)
    env = XplaneEnvironment(agent = agent)
    agent.set_api(env.api)

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
        buffer = []
        start_time = time.time()
        step_id = 0
        while True:
            # get state and control inputs
            screen = get_screen()
            state, controls, abs_time = env.step(screen=screen)
            rel_time = abs_time - start_time

            step_id += 1

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
from util import ft_to_me, sample_weather
from state.pos import Position
from airport import sample_tgt_rwy_and_position, Runway
from environment import XplaneEnvironment
from weather import Weather
from dataset.collector.gui import StarterGui

from agents.embodied_ai import AlfredBaselineTeacherForceAgent, FcBnAgent


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


def get_screen() -> Image:
    window = pw.getWindowsWithTitle("X-System")
    x, y, width, height = window[0].left, window[0].top, window[0].width, window[0].height
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return screenshot


AGENT = {
    'baseline': {
        'teacher_force': AlfredBaselineTeacherForceAgent
    },
    'simple_fc': {
        'batch_normalize': FcBnAgent
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

        state, _ = env.reset(
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

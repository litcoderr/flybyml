from typing import Optional

import os
import time
import json
import random
import numpy as np
import pygetwindow as pw
import pyautogui
import uuid
from queue import Queue
from pathlib import Path
from PIL.Image import Image
from enum import Enum

from util import ft_to_me, haversine_distance_and_bearing, sample_weather
from aircraft import Aircraft
from aircraft.b738 import B738
from state.pos import Position
from state.state import PlaneState
from controls import Controls
from agents import AgentInterface
from airport import sample_tgt_rwy_and_position, Runway
from api import API
from environment import XplaneEnvironment
from weather import Weather
from dataset.collector.atc import ATC
from dataset.collector.gui import StarterGui


class SessionState(Enum):
    Approaching = 1
    NearRunway = 2
    Abort = 3


class Config:
    aircraft: Aircraft = B738()
    init_pos: Position = Position(lat=37.5326, lon=127.024612, alt=1000) 
    init_heading: float = 0 # degrees
    init_speed: float = 128 # m/s
    init_zulu_time: float = 0 # GMT time. seconds since midnight
    weather: Optional[Weather] = None
    tgt_rwy: Optional[Runway] = None


def save_meta_data(root_dir, name):
    meta_path = Path(root_dir) / f"{name}.json"
    meta = {
        "init_zulu_time": Config.init_zulu_time,
        "weather": Config.weather.serialize(),
        "target_rwy": Config.tgt_rwy.serialize()
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def get_screen() -> Image:
    window = pw.getWindowsWithTitle("X-System")
    x, y, width, height = window[0].left, window[0].top, window[0].width, window[0].height
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return screenshot


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


if __name__ == "__main__":
    DATASET_ROOT = Path('D:\\dataset\\flybyml_dataset_v2')
    IMG_ROOT = DATASET_ROOT / "image"
    METADATA_ROOT = DATASET_ROOT / "meta"
    DATA_ROOT = DATASET_ROOT / "data"
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(IMG_ROOT, exist_ok=True)
    os.makedirs(METADATA_ROOT, exist_ok=True)
    os.makedirs(DATA_ROOT, exist_ok=True)

    # set up human agent and environment
    human = HumanAgent(Config.aircraft)
    env = XplaneEnvironment(agent = human)
    human.set_api(env.api)

    while True:
        session_id = str(uuid.uuid4())
        session_state = SessionState.Approaching
        print(f"starting {session_id}")
        img_dir = IMG_ROOT / session_id
        os.makedirs(img_dir, exist_ok=True)

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

        # initialize atc
        queue = Queue()
        atc = ATC(queue, target_rwy)
        atc.start()

        # run simulation until end of session
        prev_state: PlaneState = env.getState()
        buffer = []
        start_time = time.time()
        step_id = 0
        while True:
            # save previous image
            img_path = img_dir / f"{str(step_id).zfill(5)}.jpg"

            # get state and control inputs
            screen = get_screen()
            state, controls, abs_time = env.step()
            screen.save(img_path)

            # save data to buffer
            buffer.append({
                'state': state.serialize(),
                'control': controls.serialize(),
                'rel_time': abs_time - start_time
            })

            # send atc plane's state
            queue.put({"timestamp": time.time(), "is_running": True, "state": state})

            # stopping session if plane is not moving
            if prev_state is not None:
                dist, _ = haversine_distance_and_bearing(prev_state.pos.lat, prev_state.pos.lon, 0,
                                                         state.pos.lat, state.pos.lon, 0)
                if dist < 0.5:
                    break

                runway_dist, _ = haversine_distance_and_bearing(target_rwy.lat, target_rwy.lon, 0, state.pos.lat, state.pos.lon, 0)
                if session_state == SessionState.Approaching and runway_dist < 50:
                    session_state = SessionState.NearRunway
                if session_state == SessionState.NearRunway and runway_dist > 2000:
                    session_state = SessionState.Abort
                    break

            prev_state = state
            step_id += 1

        # save buffer
        data_path = DATA_ROOT / f"{session_id}.json"
        with open(data_path, "w") as f:
            json.dump(buffer, f)
        save_meta_data(METADATA_ROOT, session_id)
        
        print(f"saved {session_id}")

        queue.put({"timestamp": time.time(), "is_running": False})
        atc.join()

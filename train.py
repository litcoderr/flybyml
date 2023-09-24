from typing import Tuple

import random
from importify import Serializable

from environment import XplaneEnvironment
from aircraft import B738
from status.pos import Gimpo
from objective import Hold
from agent import Agent


class Config(Serializable):
    def __init__(self):
        super(Config, self).__init__()
        self.device = 'cuda'
        self.seed = 0

        self.max_step = 600  # max number of step for each episode
        self.frame_interval = 0.1  # time interval between timesteps in seconds


if __name__ == "__main__":
    config = Config()

    # initialize seed
    random.seed(config.seed)

    def sample_initial_state() -> Tuple[float, float, float]:
        """
        randomly sample heading, altitude, speed
        """
        heading = float(random.randrange(0, 359))
        altitude = float(random.randrange(1000, 3000))
        speed = float(random.randrange(230, 300))
        return heading, altitude, speed
    
    def sample_target_state(init_heading, init_alt, init_spd) -> Tuple[float, float, float]:
        heading_buffer = 30
        alt_buffer = 30
        spd_buffer = 20

        tgt_heading = float(random.randrange(init_heading-heading_buffer, init_heading+heading_buffer))
        if tgt_heading >= 360:
            tgt_heading -= 360
        elif tgt_heading < 0:
            tgt_heading += 360
        tgt_alt = float(random.randrange(init_alt-alt_buffer, init_alt+alt_buffer))
        tgt_spd = float(random.randrange(init_spd-spd_buffer, init_spd+spd_buffer))

        return tgt_heading, tgt_alt, tgt_spd

    env = XplaneEnvironment(aircraft=B738(), airport=Gimpo(), frame_interval=config.frame_interval)
    agent = Agent(device=config.device)

    # train
    while True:
        # initial state for every episode
        init_heading, init_alt, init_spd = sample_initial_state()
        # initial target for every episode
        tgt_heading, tgt_alt, tgt_spd = sample_target_state(init_heading, init_alt, init_spd)

        objective = Hold(tgt_heading, tgt_alt, tgt_spd)
        state = env.reset(
            heading = init_heading,
            alt = init_alt,
            spd = init_spd
        ).toTensor(config.device)  # reset and get initial state

        for timestep in range(config.max_step):
            # TODO sample action
            sampled_aciton = {}
            state = env.step(sampled_aciton).toTensor(config.device)
            print(state)

"""
Reinforcement Learning of flybyml agent
using Proximal Policy Optimization

목표: 수평을 맞춰라!
"""

import random

from tqdm import tqdm

from controls import Controls
from util import ft_to_me, kts_to_mps
from environment import XplaneEnvironment


FIXED_THR_VAL = 0.8


class PPOModuleV1:
    def __init__(self, args):
        self.args = args
        self.env = XplaneEnvironment(agent=None)
    
    def reset_env(self):
        """
        Reset simulator and
        pass random control inputs
        for random starting state initialization
        """
        state, control = self.env.reset(lat=0, lon=0,
                       alt=ft_to_me(20000),
                       heading=0, spd=kts_to_mps(300),
                       zulu_time=40000,
                       pause=True)

        rand_elev = 2 * random.random() -1
        ail_elev = 2 * random.random() -1
        control = Controls(elev=rand_elev, ail=ail_elev, thr=FIXED_THR_VAL)
        for _ in range(10):
            prev_state = state
            state = self.env.rl_step(control, self.args.step_interval)
        return state, prev_state
    
    def train(self):
        state, prev_state = self.reset_env()
        for epoch in tqdm(range(self.args.train.epoch), desc='epoch'):
            for local_step in tqdm(range(self.args.train.steps_per_epoch), desc='epoch'):
                pass

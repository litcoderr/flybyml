"""
Reinforcement Learning of flybyml agent
using Proximal Policy Optimization

목표: 수평을 맞춰라!
"""
from typing import Tuple, Optional

import torch
import random
import torch.nn as nn
import numpy as np

from torch import Tensor
from tqdm import tqdm
from torch.distributions.normal import Normal

from controls import Controls
from environment import XplaneEnvironment
from state.state import PlaneState
from util import ft_to_me, kts_to_mps


FIXED_THR_VAL = 0.8


def construct_observation(state: PlaneState, prev_state: PlaneState, objective, device):
    """
    construct observation based on current state and prev states

    Return:
        observation: Tensor [e_pitch, e_roll, e_spd, traj_pitch, traj_roll, traj_spd, d_roll, d_pitch, d_spd, i_roll, i_pitch, i_spd]
            e_: stands for error
            traj_: stands for trajectory error
            d_: stands for delta
            i_: stands for intensty [0-1]
    """
    # every value is normalized
    error = torch.tensor([(state.att.pitch - objective['pitch']) / 180,
                          (state.att.roll - objective['roll']) / 180])
    delta = torch.tensor([(state.att.pitch - prev_state.att.pitch) / 180,
                          (state.att.roll - prev_state.att.roll) / 180])
    return torch.cat([error, delta]).to(device)


def log_prob_from_dist(dist: Normal, act: Tensor) -> Tensor:
    return dist.log_prob(act).sum(axis=-1)


def act_to_control(act: np.array) -> Controls:
    """
    act: [elev, aileron]
    """
    return Controls(
        elev=act[0],
        ail=act[1],
        thr=FIXED_THR_VAL
    )


class Actor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mu = nn.Sequential(
            nn.Linear(args.model.obs_dim, 32), nn.ReLU(), \
            nn.Linear(32, 64), nn.ReLU(), \
            nn.Linear(64, 128), nn.ReLU(), \
            nn.Linear(128, 64), nn.ReLU(), \
            nn.Linear(64, 32), nn.ReLU(), \
            nn.Linear(32, args.model.act_dim), nn.Tanh()
        )
        self.log_std = torch.nn.Parameter(torch.as_tensor(
            -0.5 * np.ones(args.model.act_dim, dtype=np.float32)
        ))
    
    def get_distribution(self, obs):
        mu = self.mu(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)
    
    def forward(self, obs: Tensor, act: Optional[Tensor] = None):
        """
        Return:
            dist: action distribution given observation 
            log_prob: log probability of action regarding distribution
        """
        dist = self.get_distribution(obs)
        log_prob = None
        if act is not None:
            log_prob = log_prob_from_dist(dist, act)
        return dist, log_prob


class Critic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.critic = nn.Sequential(
            nn.Linear(args.model.obs_dim, 32), nn.ReLU(), \
            nn.Linear(32, 64), nn.ReLU(), \
            nn.Linear(64, 128), nn.ReLU(), \
            nn.Linear(128, 64), nn.ReLU(), \
            nn.Linear(64, 32), nn.ReLU(), \
            nn.Linear(32, 1)
        )

    def forward(self, obs):
        return torch.squeeze(self.critic(obs), -1)


class ActorCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.pi = Actor(args)
        self.v = Critic(args)
    
    def step(self, obs: Tensor):
        with torch.no_grad():
            dist = self.pi.get_distribution(obs)
            act = dist.sample()
            log_prob = log_prob_from_dist(dist, act)
            value = self.v(obs)
        return act.cpu().numpy(), value.cpu().numpy(), log_prob.cpu().numpy()
    

class PPOModuleV1:
    def __init__(self, args):
        self.args = args
        self.env = XplaneEnvironment(agent=None)
        self.obj = {
            'pitch': 0,
            'roll': 0
        }
        self.model = ActorCritic(args).to(self.args.device)
    
    def reset_env(self) -> Tuple[PlaneState, PlaneState]:
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
        ep_ret = 0  # episode return (cumulative value)
        ep_len = 0  # current episode's step
        for epoch in tqdm(range(self.args.train.epoch), desc='epoch'):
            for local_step in tqdm(range(self.args.train.steps_per_epoch), desc='epoch'):
                obs = construct_observation(state, prev_state, self.obj, self.args.device)
                
                # sample action, value, log probability of action
                action, value, log_prob = self.model.step(obs)

                # take env step using sampled action
                next_state = self.env.rl_step(act_to_control(action), self.args.step_interval)

                # TODO calculate reward
                pass

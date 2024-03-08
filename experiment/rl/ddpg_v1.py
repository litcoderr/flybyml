import time
"""
Training Objective:
    maximi
"""

import random
import torch
import torch.nn as nn

from copy import deepcopy
from torch import Tensor

from controls import Controls
from state.state import PlaneState
from util import ft_to_me, kts_to_mps
from environment import XplaneEnvironment


# environment constraint constants
MIN_ALT = ft_to_me(15000)
MAX_ALT = ft_to_me(25000)
MIN_HEADING = 0
MAX_HEADING = 360
MIN_SPD = kts_to_mps(160)
MAX_SPD = kts_to_mps(300)
MIN_PITCH = -25
MAX_PITCH = 25
MIN_ROLL = -40
MAX_ROLL = 40

# normalizing constant
PITCH_NORM = 180
ROLL_NORM = 180
SPD_NORM = kts_to_mps(300)


def sample_state():
    """
    Return:
        alt: sampled altitude
        heading: sampled heading
        spd: sampled spd
    """
    alt = random.uniform(MIN_ALT, MAX_ALT)
    heading = random.uniform(MIN_HEADING, MAX_HEADING)
    spd = random.uniform(MIN_SPD+kts_to_mps(30), MAX_SPD-kts_to_mps(30))
    return alt, heading, spd


def is_boundary(state: PlaneState):
    return state.pos.alt >= MIN_ALT and state.pos.alt <= MAX_ALT


def clip(v, min_v, max_v):
    if v < min_v:
        return min_v
    elif v > max_v:
        return max_v
    else:
        return v


class PiV1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mlp = nn.Sequential(
            nn.Linear(self.args.model.obs_dim, 32), nn.ReLU(), \
            nn.Linear(32, 64), nn.ReLU(), \
            nn.Linear(64, 128), nn.ReLU(), \
            nn.Linear(128, 64), nn.ReLU(), \
            nn.Linear(64, 32), nn.ReLU(), \
            nn.Linear(32, self.args.model.act_dim), nn.Tanh()
        )
    
    def forward(self, obs: Tensor) -> Tensor:
        """
        Args:
            obs: observation Tensor [b, 9] [e_pitch, e_roll, e_spd, d_roll, d_pitch, d_spd, i_roll, i_pitch, i_spd]
        Return:
            action tensor (delta of control values) [b, 3] [d_elev, d_ail, d_thrust]
                every delta control's range is [-1, 1]
        """
        return self.mlp(obs)


class QV1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mlp = nn.Sequential(
            nn.Linear(self.args.model.obs_dim + self.args.model.act_dim, 32), nn.ReLU(), \
            nn.Linear(32, 64), nn.ReLU(), \
            nn.Linear(64, 128), nn.ReLU(), \
            nn.Linear(128, 64), nn.ReLU(), \
            nn.Linear(64, 32), nn.ReLU(), \
            nn.Linear(32, 1)
        )
    
    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """
        Args:
            obs: observation Tensor [e_pitch, e_roll, e_spd, d_roll, d_pitch, d_spd, i_roll, i_pitch, i_spd]
            action: action Tensor [d_elev, d_ail, d_thrust]
        Return:
            value [b]
        """
        obs_action = torch.cat([obs, action], dim=-1) 
        return torch.squeeze(self.mlp(obs_action), dim=-1)


class ActorCriticModelV1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.q = QV1(args).to(self.args.device)
        self.pi = PiV1(args).to(self.args.device)

    def act(self, obs: Tensor) -> Tensor:
        """
        Sample action without gradient

        Args:
            obs: observation Tensor [e_pitch, e_roll, e_spd, d_roll, d_pitch, d_spd, i_roll, i_pitch, i_spd]
        Return:
            action tensor (delta value of controls) [d_elev, d_ail, d_thrust]
        """
        with torch.no_grad():
            return self.pi(obs)


class DDPGModuleV1:
    def __init__(self, args):
        self.args = args

        # initialize xplane environment
        # agent is not necessary, since we will be feeding in input through rl_step function
        self.env = XplaneEnvironment(agent=None)

        # initialize model
        self.policy = ActorCriticModelV1(args)
        self.target = deepcopy(self.policy)
        self.policy = self.policy.to(args.device)
        self.target = self.target.to(args.device)
    
    def get_observation(self, cur_state: PlaneState, prev_state: PlaneState, objective) -> Tensor:
        """
        construct observation based on current state and prev states

        Return:
            observation: Tensor [e_pitch, e_roll, e_spd, d_roll, d_pitch, d_spd, i_roll, i_pitch, i_spd]
                e_: stands for error
                d_: stands for delta
                i_: stands for intensty [0-1]
        """
        # every value is normalized
        error = torch.tensor([(cur_state.att.pitch - objective['pitch'][0])/PITCH_NORM,
                              (cur_state.att.roll - objective['roll'][0])/ROLL_NORM,
                              (cur_state.spd - objective['spd'][0])/SPD_NORM])
        delta = torch.tensor([(cur_state.att.pitch - prev_state.att.pitch)/PITCH_NORM,
                              (cur_state.att.roll - prev_state.att.roll)/ROLL_NORM,
                              (cur_state.spd - prev_state.spd)/SPD_NORM])
        intensity = torch.tensor([objective['pitch'][1], objective['roll'][1], objective['spd'][1]])
        return torch.cat([error, delta, intensity]).to(self.args.device)
    
    def update_control(self, act: Tensor):
        """
        Update self.control based on action_t
        Args:
            act: action tensor [d_elev, d_ail, d_thrust] range: [-1, 1]
        """
        act = act.detach()
        if act.get_device() != -1:
            act = act.to('cpu')
        act = act.numpy()

        self.control.elev += act[0]
        self.control.ail += act[1]
        self.control.thr += (act[2] + 1) / 2

        # clip control values
        self.control.elev = clip(self.control.elev, -1, 1)
        self.control.ail = clip(self.control.ail, -1, 1)
        self.control.thr = clip(self.control.thr, 0, 1)
    
    def train(self):
        for step in range(self.args.train.total_steps):
            #  reset environment with newly sampled weather etc.
            if step % self.args.train.reset_period == 0 or not is_boundary(state):
                alt, heading, spd = sample_state()
                self.env.api.resume()
                state, self.control = self.env.reset(0, 0, alt, heading, spd, 40000)
                self.env.api.pause()
                prev_state = state

                # construct objective
                # add buffer when sampling objective so that the agent has space to explore
                # key -> Tuple[objective value, intensity]
                objective = {
                    'pitch': (random.uniform(MIN_PITCH+5, MAX_PITCH-5), random.random()),
                    'roll': (random.uniform(MIN_ROLL+10, MAX_ROLL-10), random.random()),
                    'spd': (random.uniform(MIN_SPD+kts_to_mps(30), MAX_SPD-kts_to_mps(30)), random.random()),
                }
            obs_t = self.get_observation(state, prev_state, objective)

            # take a step in the environment
            if step >= self.args.train.start_steps:
                # sample action from actor-critic network (non-deterministic in training step for exploration)
                act_t = self.policy.act(obs_t)
                act_t += self.args.train.act_noise * torch.normal(0, 1, size=act_t.size()).to(self.args.device)
                # update control based on action_t
                self.update_control(act_t)
                # take step with updated control
                next_state = self.env.rl_step(self.control, self.args.step_interval)
            else:
                # TODO randomly sample action for better exploration
                pass
            

            prev_state = state
            state = next_state

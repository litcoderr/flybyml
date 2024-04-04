"""
Reinforcement Learning of flybyml agent
using Proximal Policy Optimization

목표: 수평을 맞춰라!
"""
from typing import Tuple, Optional

import torch
import scipy
import random
import torch.nn as nn
import numpy as np

from torch import Tensor
from tqdm import tqdm
from torch.distributions.normal import Normal
from mpi4py import MPI

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
    # TODO delta should be better defined and normalized. As of now, delta's value is dependant on timestep
    error = torch.tensor([(state.att.pitch - objective['pitch']) / 180,
                          (state.att.roll - objective['roll']) / 180])
    delta = torch.tensor([(state.att.pitch - prev_state.att.pitch) / 180,
                          (state.att.roll - prev_state.att.roll) / 180])
    return torch.cat([error, delta]).to(device)


def construct_reward(obs: Tensor) -> float:
    intensity = 0.3  # higher the intensity, lower the value
    rew = (-1/intensity) * torch.abs(obs[:2]) + 1
    rew = torch.sum(rew) / 2
    return rew.item()


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

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    MPI.COMM_WORLD.Allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_sum(x):
    return mpi_op(x, MPI.SUM)


def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std


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


class PPOBuffer:
    def __init__(self, args):
        self.device = args.device
        self.obs_buf = np.zeros((args.train.steps_per_epoch, args.model.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((args.train.steps_per_epoch, args.model.act_dim), dtype=np.float32)
        self.adv_buf = np.zeros((args.train.steps_per_epoch), dtype=np.float32)
        self.rew_buf = np.zeros((args.train.steps_per_epoch), dtype=np.float32)
        self.ret_buf = np.zeros((args.train.steps_per_epoch), dtype=np.float32)
        self.val_buf = np.zeros((args.train.steps_per_epoch), dtype=np.float32)
        self.logp_buf = np.zeros((args.train.steps_per_epoch), dtype=np.float32)

        self.gamma = args.train.gamma
        self.lam = args.train.lam

        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = args.train.steps_per_epoch
    
    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, val):
        """
        End of trajectory.
        Calculate GAE-Lambda advantage and Return for this trajectory
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], val)
        vals = np.append(self.val_buf[path_slice], val)

        # calculate GAE-Lambda advantage
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # calculate return
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Get all data at the end of epoch
        """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        # advantage normalization
        # TODO delete redundant mpi statistics code
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(
            obs = self.obs_buf,
            act = self.act_buf,
            ret = self.ret_buf,
            adv = self.adv_buf,
            logp = self.logp_buf
        )
        return {
            k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in data.items()
        }


class PPOModuleV1:
    def __init__(self, args):
        self.args = args
        self.env = XplaneEnvironment(agent=None)
        self.obj = {
            'pitch': 0,
            'roll': 0
        }
        self.model = ActorCritic(args).to(self.args.device)
        self.buf = PPOBuffer(args)
    
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
    
    def compute_loss_pi(self, data) -> Tensor:
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        pi_dist, logp = self.model.pi(obs, act)

        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.args.train.clip_ratio, 1+self.args.train.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # extra useful information
        approx_kl = (logp_old - logp).mean().item()
        ent = pi_dist.entropy().mean().item()
        clipped = ratio.gt(1+self.args.train.clip_ratio) | ratio.lt(1-self.args.train.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info
    
    def compute_loss_v(self, data) -> Tensor:
        obs, ret = data['obs'], data['ret']
        return ((self.model.v(obs) - ret)**2).mean()

    def update(self):
        data = self.buf.get()

        pi_loss_old, pi_info = self.compute_loss_pi(data)
        pi_loss_old = pi_loss_old.item()
        v_loss_old = self.compute_loss_v(data).item()

        for _ in range(self.args.train.pi_iters):
            # TODO we do not need MPI since we cannot run multiple simulations at the same time
            pass
        
        for _ in range(self.args.train.v_iters):
            pass

        # TODO

    def train(self):
        state, prev_state = self.reset_env()
        ep_ret = 0  # episode return (cumulative value)
        ep_len = 0  # current episode's step
        for epoch in tqdm(range(self.args.train.epoch), desc='epoch'):
            for local_step in tqdm(range(self.args.train.steps_per_epoch), desc='local step'):
                obs = construct_observation(state, prev_state, self.obj, self.args.device)
                
                # sample action, value, log probability of action
                action, value, log_prob = self.model.step(obs)

                # take env step using sampled action
                next_state = self.env.rl_step(act_to_control(action), self.args.step_interval)
                next_obs = construct_observation(next_state, state, self.obj, 'cpu')

                # calculate reward by taking this action
                rew = construct_reward(next_obs)
                ep_ret += rew
                ep_len += 1

                # TODO save to buffer for training
                self.buf.store(obs.cpu().numpy(), action, rew, value, log_prob)

                # update state and prev_state
                prev_state = state
                state = next_state

                episode_ended = ep_len == self.args.train.max_ep_len
                epoch_ended = local_step == self.args.train.steps_per_epoch -1
                if episode_ended or epoch_ended:
                    obs = construct_observation(state, prev_state, self.obj, self.args.device)
                    _, value, _ = self.model.step(obs)
                    self.buf.finish_path(value)

                    # TODO log episodic return only if episode ended

                    ep_ret = 0
                    ep_len = 0
                    state, prev_state = self.reset_env()

            # finished an epoch
            self.update()

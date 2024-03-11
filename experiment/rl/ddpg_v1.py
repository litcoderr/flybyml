"""
Training Objective:
    maximi
"""

import os
import random
import torch
import wandb
import torch.nn as nn
import numpy as np

from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from torch import Tensor
from torch.distributions.normal import Normal
from torch.optim import Adam

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

# frequency constant
# Tupe[min_freq, max_freq]  (unit corresponds to its key's unit)
# freq = change of unit per sec
FREQ = {
    'pitch': (2, 20),
    'roll': (2, 20),
    'spd': (kts_to_mps(5), kts_to_mps(10))
}


def sample_state():
    """
    Return:
        alt: sampled altitude
        heading: sampled heading
        spd: sampled spd
    """
    alt = random.uniform(MIN_ALT, MAX_ALT)
    heading = random.uniform(MIN_HEADING, MAX_HEADING)
    spd = random.uniform(kts_to_mps(210), MAX_SPD-kts_to_mps(30))
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
            obs: observation Tensor [b, 9] [e_pitch, e_roll, e_spd, traj_pitch, traj_roll, traj_spd, d_roll, d_pitch, d_spd, i_roll, i_pitch, i_spd]
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
            obs: observation Tensor [e_pitch, e_roll, e_spd, traj_pitch, traj_roll, traj_spd, d_roll, d_pitch, d_spd, i_roll, i_pitch, i_spd]
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
            obs: observation Tensor [e_pitch, e_roll, e_spd, traj_pitch, traj_roll, traj_spd, d_roll, d_pitch, d_spd, i_roll, i_pitch, i_spd]
        Return:
            action tensor (delta value of controls) [d_elev, d_ail, d_thrust]
        """
        with torch.no_grad():
            return self.pi(obs)


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.batch_size = args.train.batch_size
        self.max_size = args.train.replay_size
        self.size = 0
        self.ptr = 0

        self.obs = np.zeros((self.max_size, args.model.obs_dim), dtype=np.float32)
        self.act = np.zeros((self.max_size, args.model.act_dim), dtype=np.float32)
        self.rew = np.zeros(self.max_size, dtype=np.float32)
        self.next_obs = np.zeros((self.max_size, args.model.obs_dim), dtype=np.float32)
        self.done = np.zeros(self.max_size, dtype=np.float32)

    def store(self, obs: Tensor, act: Tensor, rew: Tensor, next_obs: Tensor, done: bool):
        if not obs.is_cpu:
            obs = obs.to('cpu')
        if not act.is_cpu:
            act = act.to('cpu')
        if not rew.is_cpu:
            rew = rew.to('cpu')
        if not next_obs.is_cpu:
            next_obs = next_obs.to('cpu')
        self.obs[self.ptr] = obs.detach().numpy()
        self.act[self.ptr] = act.detach().numpy()
        self.rew[self.ptr] = rew.detach().numpy()
        self.next_obs[self.ptr] = next_obs.detach().numpy()
        self.done[self.ptr] = done
        self.ptr += 1        
        if self.ptr >= self.max_size:
            self.ptr = 0
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        return dict(
            obs = torch.as_tensor(self.obs[idxs], dtype=torch.float32).to(self.args.device),
            act = torch.as_tensor(self.act[idxs], dtype=torch.float32).to(self.args.device),
            rew = torch.as_tensor(self.rew[idxs], dtype=torch.float32).to(self.args.device),
            next_obs = torch.as_tensor(self.next_obs[idxs], dtype=torch.float32).to(self.args.device),
            done = torch.as_tensor(self.done[idxs], dtype=torch.float32).to(self.args.device)
        )


class DDPGModuleV1:
    def __init__(self, args):
        self.args = args

        # initialize xplane environment
        # agent is not necessary, since we will be feeding in input through rl_step function
        self.env = XplaneEnvironment(agent=None)

        # initialize model
        self.policy = ActorCriticModelV1(args)
        self.target = deepcopy(self.policy)
        for p in self.target.parameters():
            p.requires_grad = False
        self.policy = self.policy.to(args.device)
        self.target = self.target.to(args.device)

        # initialize optimizer
        self.pi_optim = Adam(self.policy.pi.parameters(), lr=args.train.pi_lr)
        self.q_optim = Adam(self.policy.q.parameters(), lr=args.train.q_lr)

        # replay buffer
        self.buf = ReplayBuffer(args)

        # initialize logger
        wandb.init(project=args.project, name=args.run, entity="flybyml")
        wandb.config = args
        wandb.watch(self.policy)

        # configure model checkpoint save root
        self.ckpt_root = Path(os.path.dirname(__file__)) / "../" / args.project / "logs" / args.run
        os.makedirs(self.ckpt_root, exist_ok=True)
    
    def construct_observation(self, step: int, cur_state: PlaneState, prev_state: PlaneState, objective) -> Tensor:
        """
        construct observation based on current state and prev states

        Return:
            observation: Tensor [e_pitch, e_roll, e_spd, traj_pitch, traj_roll, traj_spd, d_roll, d_pitch, d_spd, i_roll, i_pitch, i_spd]
                e_: stands for error
                traj_: stands for trajectory error
                d_: stands for delta
                i_: stands for intensty [0-1]
        """
        traj = self.get_trajectory(step, objective)

        # every value is normalized
        error = torch.tensor([(cur_state.att.pitch - objective['pitch'][0])/PITCH_NORM,
                              (cur_state.att.roll - objective['roll'][0])/ROLL_NORM,
                              (cur_state.spd - objective['spd'][0])/SPD_NORM])
        traj_error = torch.tensor([(cur_state.att.pitch - traj['pitch'])/PITCH_NORM,
                              (cur_state.att.roll - traj['roll'])/ROLL_NORM,
                              (cur_state.spd - traj['spd'])/SPD_NORM])
        delta = torch.tensor([(cur_state.att.pitch - prev_state.att.pitch)/PITCH_NORM,
                              (cur_state.att.roll - prev_state.att.roll)/ROLL_NORM,
                              (cur_state.spd - prev_state.spd)/SPD_NORM])
        intensity = torch.tensor([objective['pitch'][1], objective['roll'][1], objective['spd'][1]])
        return torch.cat([error, traj_error, delta, intensity]).to(self.args.device)
    
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

        self.control.elev += 2 * act[0]
        self.control.ail += 2 * act[1]
        self.control.thr += act[2]

        # clip control values
        self.control.elev = clip(self.control.elev, -1, 1)
        self.control.ail = clip(self.control.ail, -1, 1)
        self.control.thr = clip(self.control.thr, 0, 1)

    def get_trajectory(self, step, objective):
        """
        Args:
            step: global step
            objective
        """
        rel_step = step - self.init_step
        init_error = {
            'pitch': objective['pitch'][0] - self.init_state.att.pitch,
            'roll': objective['roll'][0] - self.init_state.att.roll,
            'spd': objective['spd'][0] - self.init_state.spd
        }
        trajectory = {}
        for key, bias in init_error.items():
            slope = (FREQ[key][1] - FREQ[key][0]) * objective[key][1] + FREQ[key][0]  # (max_freq - min_freq) * intensity + min_freq
            if bias > 0:
                slope *= -1
            rel_trajectory = slope * self.args.step_interval * rel_step + bias
            if slope * rel_trajectory >= 0:
                trajectory[key] = objective[key][0]
            else:
                trajectory[key] = rel_trajectory + objective[key][0]
        return trajectory
    
    def construct_reward(self, obs) -> float:
        """
        Args:
            observation: Tensor [e_pitch, e_roll, e_spd, traj_pitch, traj_roll, traj_spd, d_roll, d_pitch, d_spd, i_roll, i_pitch, i_spd]
        """
        # reward distribution
        # TODO decrease scale(sigma) as timestep increases, to give mroe intense reward towards the end of episode
        scale = self.args.train.rew_scale
        normal = Normal(torch.tensor([0, 0, 0]).to(self.args.device), torch.tensor([scale, scale, scale]).to(self.args.device))
        rewards = torch.exp(normal.log_prob(obs[3:6]))
        return torch.sum(rewards)
    
    def update(self):
        batch = self.buf.sample_batch()

        # 1. optimize q policy network
        self.q_optim.zero_grad()
        
        q = self.policy.q(batch['obs'], batch['act'])
        with torch.no_grad():
            q_pi_targ = self.target.q(batch['next_obs'], self.target.pi(batch['next_obs']))
            bellman_backup = batch['rew'] + self.args.train.d_factor * (1-batch['done']) * q_pi_targ
        q_loss = ((q-bellman_backup)**2).mean()
        q_loss.backward()
        self.q_optim.step()
        
        # 2. optimize pi policy network
        for p in self.policy.q.parameters():
            p.requires_grad = False

        self.pi_optim.zero_grad()
        q_pi = self.policy.q(batch['obs'], self.policy.pi(batch['obs']))
        pi_loss = -q_pi.mean()
        pi_loss.backward()
        self.pi_optim.step()

        for p in self.policy.q.parameters():
            p.requires_grad = True

        # 3. update target network
        with torch.no_grad():
            for policy_p, target_p in zip(self.policy.parameters(), self.target.parameters()):
                target_p.data.mul_(self.args.train.polyak)
                target_p.data.add_((1-self.args.train.polyak) * policy_p.data)
        
        # return log
        return {
            'q_value': q.mean().item(),
            'q_loss': q_loss.item(),
            'pi_loss': pi_loss.item()
        }


    def test(self, global_step):
        test_return = 0
        with torch.no_grad():
            alt, heading, spd = sample_state()
            state, self.control = self.env.reset(0, 0, alt, heading, spd, 40000, pause=True)
            prev_state = state
            self.init_step = 0
            self.init_state = state

            # construct objective
            # add buffer when sampling objective so that the agent has space to explore
            # key -> Tuple[objective value, intensity]
            objective = {
                'pitch': (random.uniform(MIN_PITCH+5, MAX_PITCH-5), random.random()),
                'roll': (random.uniform(MIN_ROLL+10, MAX_ROLL-10), random.random()),
                'spd': (random.uniform(MIN_SPD+kts_to_mps(30), MAX_SPD-kts_to_mps(30)), random.random()),
            }

            for step in tqdm(range(self.args.train.test_n_steps)):
                obs = self.construct_observation(step, state, prev_state, objective)
                act = self.policy.act(obs)
                self.update_control(act)
                next_state = self.env.rl_step(self.control, self.args.step_interval)

                rew = self.construct_reward(obs)
                test_return += rew

                prev_state = state
                state = next_state
        # check if we should save the model or not
        save = True
        if len(os.listdir(self.ckpt_root)) > 0:
            original_name = os.listdir(self.ckpt_root)[0]
            original_return = float(original_name.split(".")[0].split("reward=")[1])
            if original_return >= test_return:
                save = False
            if save:
                os.remove(self.ckpt_root / original_name)
        if save:
            torch.save(self.policy.state_dict(), self.ckpt_root / f"reward={test_return}.ckpt")
        
        # log
        wandb.log({
            'test_return': test_return
        }, step=global_step)
    
    def train(self):
        done = True
        for step in tqdm(range(self.args.train.total_steps)):
            #  reset environment with newly sampled weather etc.
            if done:
                alt, heading, spd = sample_state()
                state, self.control = self.env.reset(0, 0, alt, heading, spd, 40000, pause=True)
                prev_state = state
                self.init_step = step
                self.init_state = state

                # construct objective
                # add buffer when sampling objective so that the agent has space to explore
                # key -> Tuple[objective value, intensity]
                objective = {
                    'pitch': (random.uniform(MIN_PITCH+5, MAX_PITCH-5), random.random()),
                    'roll': (random.uniform(MIN_ROLL+10, MAX_ROLL-10), random.random()),
                    'spd': (random.uniform(MIN_SPD+kts_to_mps(30), MAX_SPD-kts_to_mps(30)), random.random()),
                }
            obs = self.construct_observation(step, state, prev_state, objective)

            # take a step in the environment
            if step >= self.args.train.start_steps:
                # sample action from actor-critic network (non-deterministic in training step for exploration)
                act = self.policy.act(obs)
                act += self.args.train.act_noise * torch.normal(0, 1, size=act.size()).to(self.args.device)
            else:
                # randomly sample action for better exploration
                act = 2 * torch.rand(3) - 1
            # update control based on action_t
            self.update_control(act)
            # take step with updated control
            next_state = self.env.rl_step(self.control, self.args.step_interval)
            next_obs = self.construct_observation(step+1, next_state, state, objective)

            # handle end of episode
            done = not is_boundary(next_state) or (step - self.init_step) >= self.args.train.max_ep_len

            # calculate reward
            rew = self.construct_reward(obs)

            # store in buffer
            self.buf.store(obs, act, rew, next_obs, done)

            # update
            if step >= self.args.train.update_after and step % self.args.train.update_every == 0:
                for _ in range(self.args.train.update_every):
                    logs = self.update()
                    wandb.log(logs, step=step)
            
            if step % self.args.train.test_every == 0:
                self.test(global_step=step)
                done = True

            prev_state = state
            state = next_state

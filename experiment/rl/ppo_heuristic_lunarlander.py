"""
Reinforcement Learning of flybyml agent
using Proximal Policy Optimization

목표: 수평을 맞춰라!
"""
from typing import Tuple, Optional

import gymnasium as gym
import os
import torch
import wandb
import scipy
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from gymnasium.wrappers.time_limit import TimeLimit
from pathlib import Path
from tqdm import tqdm
from torch import Tensor
from torch.optim import Adam
from torch.distributions.normal import Normal
from IPython.display import clear_output

FIXED_THR_VAL = 0.8

def log_prob_from_dist(dist: Normal, act: Tensor) -> Tensor:
    return dist.log_prob(act).sum(axis=-1)

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


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
            act = torch.clamp(act, min=-1.0, max=1.0) # action should always be [-1, 1]
            log_prob = log_prob_from_dist(dist, act)
            value = self.v(obs)
        return act.cpu().numpy(), value.cpu().numpy(), log_prob.cpu().numpy()

    def infer(self, obs: Tensor):
        with torch.no_grad():
            dist = self.pi.get_distribution(obs)
        return dist.loc.cpu().numpy()


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
        adv_mean, adv_std = np.mean(self.adv_buf),  np.std(self.adv_buf)
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


class Logger:
    def __init__(self):
        self.epoch_dict = dict()
        self.log_dict = dict()
    
    def add(self, **kwargs):
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)
    
    def flush(self):
        wandb.log(self.log_dict)
        self.epoch_dict = dict()
        self.log_dict = dict()
    
    def log(self, key, with_min_max = False, average_only = False):
        mean = np.mean(self.epoch_dict[key])

        if average_only:
            self.log_dict[key] = mean
        else:
            std = np.std(self.epoch_dict[key])
            self.log_dict['Mean_'+key] = mean
            self.log_dict['Std_'+key] = std
        
        if with_min_max:
            self.log_dict['Min_'+key] = np.min(self.epoch_dict[key])
            self.log_dict['Max_'+key] = np.max(self.epoch_dict[key])


class PPOModuleHeuristicLunar:
    def __init__(self, args, train=True, ckpt_path=None):
        self.args = args
        env = gym.make('LunarLander-v2', continuous=True, enable_wind=False, render_mode='rgb_array')
        self.env = TimeLimit(env, max_episode_steps=self.args.train.steps_per_epoch)

        self.model = ActorCritic(args)
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))
        self.model = self.model.to(self.args.device)

        if train:
            self.buf = PPOBuffer(args)
            self.pi_optim = Adam(self.model.pi.parameters(), lr=args.train.pi_lr)
            self.v_optim = Adam(self.model.v.parameters(), lr=args.train.v_lr)

            # init wandb logger
            #wandb.init(project=args.project, name=args.run, config=dict(args), entity="flybyml")
            wandb.init(project=args.project, name=args.run, config=dict(args))
            wandb.watch(self.model)
            self.logger = Logger()
            # configure model checkpoint save root
            self.ckpt_root = Path(os.path.dirname(__file__)) / "../" / args.project / "logs" / args.run
            os.makedirs(self.ckpt_root, exist_ok=True)

    
    def reset_env(self) -> Tuple[torch.Tensor, dict]:
        """
        Reset simulator
        
        Returns:
            observation: ndarray
        """
        obs, _ = self.env.reset()
        return torch.from_numpy(obs).to(self.args.device)


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

        # optimize pi
        for pi_update_idx in range(self.args.train.pi_iters):
            self.pi_optim.zero_grad()
            pi_loss, pi_info = self.compute_loss_pi(data)
            if pi_info['kl'] > 1.5 * self.args.train.target_kl:
                # Early stopping due to reaching max kl
                break
            pi_loss.backward()
            self.pi_optim.step()
        
        # optimize v
        for _ in range(self.args.train.v_iters):
            self.v_optim.zero_grad()
            v_loss = self.compute_loss_v(data)
            v_loss.backward()
            self.v_optim.step()

        self.logger.add(
            StopIter = pi_update_idx,
            LossPi = pi_loss_old,
            LossV = v_loss_old,
            KL = pi_info['kl'],
            Entropy = pi_info['ent'],
            ClipFrac = pi_info['cf'],
            DeltaLossPi = (pi_loss.item() - pi_loss_old),
            DeltaLossV = (v_loss.item() - v_loss_old)
        )

    def test(self):
        obs = self.reset_env()
        plt.ion()
        _, ax = plt.subplots()
        while True:
            action = self.model.infer(obs)
            obs, _, term, trunc, _ = self.env.step(action)
            obs = torch.from_numpy(obs).to(self.args.device)

            img = self.env.render()
            clear_output(wait=True)
            ax.imshow(img)
            plt.axis('off')
            plt.pause(0.01)

            if term or trunc:
                break
        plt.ioff()
        plt.show()


    def train(self):
        obs = self.reset_env()
        ep_ret = 0  # episode return (cumulative value)
        ep_len = 0  # current episode's step
        # epoch
        for _ in tqdm(range(self.args.train.epoch), desc='epoch'):
            # local step
            for local_step in tqdm(range(self.args.train.steps_per_epoch), desc='localstep'):
                # sample action, value, log probability of action
                action, value, log_prob = self.model.step(obs)

                # take one timestep by applying sampled action
                obs, rew, term, trunc, _ = self.env.step(action)
                obs = torch.from_numpy(obs).to(self.args.device)

                ep_ret += rew
                ep_len += 1

                # save to buffer for training
                self.buf.store(obs.cpu().numpy(), action, rew, value, log_prob)
                self.logger.add(Vals=value)

                if term or trunc:
                    _, value, _ = self.model.step(obs)
                    self.buf.finish_path(value)

                    self.logger.add(EpRet=ep_ret)

                    ep_ret = 0
                    ep_len = 0
                    obs = self.reset_env()

            # finished an epoch
            self.update()

            save = True
            current_return = np.mean(self.logger.epoch_dict['EpRet'])
            if len(os.listdir(self.ckpt_root)) > 0:
                original_name = os.listdir(self.ckpt_root)[0]
                past_return = float(original_name.split(".")[0].split("EpRet=")[1])
                save = current_return > past_return
                if save:
                    os.remove(self.ckpt_root / original_name)
            if save:
                torch.save(self.model.state_dict(), self.ckpt_root / f"EpRet={current_return}.ckpt")

            # log
            self.logger.log('Vals', with_min_max=True)
            self.logger.log('EpRet', with_min_max=True)
            self.logger.log('StopIter', average_only=True)
            self.logger.log('LossPi', average_only=True)
            self.logger.log('LossV', average_only=True)
            self.logger.log('KL', average_only=True)
            self.logger.log('Entropy', average_only=True)
            self.logger.log('ClipFrac', average_only=True)
            self.logger.log('DeltaLossPi', average_only=True)
            self.logger.log('DeltaLossV', average_only=True)
            self.logger.flush()

if __name__ == "__main__":
    conf = OmegaConf.load("/Users/jameschee/project/flybyml/experiment/config/rl_ppo_lunarlander_heuristic.yaml")
    conf.merge_with_cli()

    model = PPOModuleHeuristicLunar(conf, train=False)
    model.test()

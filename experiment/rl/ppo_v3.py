"""
Reinforcement Learning of flybyml agent
using Proximal Policy Optimization

목표: 수평을 맞춰라!
"""
from typing import Tuple, List

import os
from omegaconf import OmegaConf
import torch
import wandb
import random
import numpy as np
import cv2

from pathlib import Path
from tqdm import tqdm
from torch import Tensor
from torch.optim import Adam
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from controls import Controls
from environment import XplaneEnvironment
from state.state import PlaneState
from module.s3d.s3dg import S3D
from util import ft_to_me, kts_to_mps
from experiment.rl.ppo_v1 import construct_reward, act_to_control, discount_cumsum, \
                                    ActorCritic, Logger


FIXED_THR_VAL = 0.8

def preprocess_video(ep_frames: List[np.array]) -> Tensor:
    frames = []
    toTensor = ToTensor()
    for frame in ep_frames:
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        frame = toTensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(frame)

    return torch.stack(frames).transpose(1, 0).unsqueeze(0)

def video_to_tensor(video_path: str) -> Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("not opened")

    frames = []
    toTensor = ToTensor()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        frame = toTensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = torch.tensor(frame, dtype=torch.float32) / 255.0
        frames.append(frame)
    if len(frames) % 2 == 1:
        frames = frames[:-1]
        
    cap.release()

    return torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0)

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
    
    def store(self, obs, act, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        # self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, rew, val):
        """
        End of trajectory.
        Calculate GAE-Lambda advantage and Return for this trajectory
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        self.rew_buf[self.path_start_idx, self.path_start_idx+len(rew)] = rew
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


class PPOModuleV3:
    """
    Dense version of PPO V2 (PPO with RoboCLIP):
        "dense" reward generation by utilizing expert demonstrations (IL)
    """
    def __init__(self, args, train=True, ckpt_path=None):
        self.args = args
        self.env = XplaneEnvironment(agent=None)
        self.obj = {
            'pitch': 0,
            'roll': 0
        }
        self.model = ActorCritic(args)
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))
        self.model = self.model.to(self.args.device)
        self.buf = PPOBuffer(args)

        self.pi_optim = Adam(self.model.pi.parameters(), lr=args.train.pi_lr)
        self.v_optim = Adam(self.model.v.parameters(), lr=args.train.v_lr)

        # load pretrained s3d module
        self.s3d = S3D(args.s3d.vocab, 512)
        self.s3d.load_state_dict(torch.load(args.s3d.pretrained))
        self.s3d = self.s3d.eval()
        self.s3d = self.s3d.to('cuda')

        if args.s3d.demotype == "video":
            demo = video_to_tensor(args.s3d.demo).to('cuda')
            with torch.no_grad():
                self.demo_emb = self.s3d(demo)["video_embedding"].to('cpu')

        if train:
            # init wandb logger
            wandb.init(project=args.project, name=args.run, config=dict(args), entity="flybyml")
            wandb.watch(self.model)
            self.logger = Logger()
            # configure model checkpoint save root
            self.ckpt_root = Path(os.path.dirname(__file__)) / "../" / args.project / "logs" / args.run
            os.makedirs(self.ckpt_root, exist_ok=True)

    
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

    def construct_observation(self, state: PlaneState, prev_state: PlaneState, objective, device):
        """
        construct observation based on current state and prev states
        """
        # every value is normalized
        error = torch.tensor([(state.att.pitch - objective['pitch']) / 180,
                            (state.att.roll - objective['roll']) / 180])
        delta = torch.tensor([(state.att.pitch - prev_state.att.pitch) * self.args.step_interval / 180,
                            (state.att.roll - prev_state.att.roll) * self.args.step_interval / 180])
        return torch.cat([error, delta]).to(device)

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

    def construct_s3d_reward(self, ep_frames: List[np.array], window_size: int = 32) -> float:
        """
        compute similarity between demo (text/video) and current episode video
        """
        rew = []
        video = preprocess_video(ep_frames)
        with torch.no_grad():
            for start_idx in range(len(ep_frames)-window_size+1):
                window = video[:, :, start_idx:start_idx+window_size, :, :].to('cuda')
                window_emb = self.s3d(window)["video_embedding"].to('cpu')
                rew.append(F.cosine_similarity(self.demo_emb, window_emb, dim=-1).item())

        return np.array(rew)

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
        state, prev_state = self.reset_env()
        while True:
            obs = self.construct_observation(state, prev_state, self.obj, self.args.device)
            action = self.model.infer(obs)

            prev_state = state
            state = self.env.rl_step(act_to_control(action), self.args.step_interval)


    def train(self):
        state, prev_state = self.reset_env()
        ep_ret = 0      # episode return for comparison (cumulative value)
        ep_len = 0  # current episode's step
        ep_frames = []
        for epoch in tqdm(range(self.args.train.epoch), desc='epoch'):
            for local_step in tqdm(range(self.args.train.steps_per_epoch), desc='local step'):
                obs = self.construct_observation(state, prev_state, self.obj, self.args.device)
                
                # sample action, value, log probability of action
                action, value, log_prob = self.model.step(obs)

                # take env step using sampled action
                next_state = self.env.rl_step(act_to_control(action), self.args.step_interval)
                next_obs = self.construct_observation(next_state, state, self.obj, 'cpu')
                                
                rew = construct_reward(next_obs)
                ep_ret += rew
                ep_len += 1
                ep_frames.append(self.env.render())

                self.buf.store(obs.cpu().numpy(), action, value, log_prob)
                self.logger.add(Vals=value)

                # update state and prev_state
                prev_state = state
                state = next_state

                episode_ended = ep_len == self.args.train.max_ep_len
                epoch_ended = local_step == self.args.train.steps_per_epoch -1
                if episode_ended or epoch_ended:
                    # compute windowed s3d reward once
                    s3d_rew = self.construct_s3d_reward(ep_frames)
                    obs = self.construct_observation(state, prev_state, self.obj, self.args.device)
                    _, value, _ = self.model.step(obs)
                    self.buf.finish_path(s3d_rew, value)

                    self.logger.add(EpRet=ep_ret)
                    self.logger.add(EpRetS3D=np.sum(s3d_rew))

                    ep_ret = 0
                    ep_len = 0
                    ep_frames = []
                    state, prev_state = self.reset_env()

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
            self.logger.log('EpRetS3D', with_min_max=True)
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
    conf = OmegaConf.load("C:/Users/lee/Desktop/ml/flybyml/experiment/config/rl_ppo_v1.yaml")
    conf.merge_with_cli()

    ckpt_path = "C:/Users/lee/Desktop/ml/flybyml/experiment/rl/logs/EpRet=191.80759536772968.ckpt"
    model = PPOModuleV3(conf, train=False, ckpt_path=ckpt_path)
    model.test()
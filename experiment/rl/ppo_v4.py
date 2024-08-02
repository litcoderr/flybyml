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
import imageio.v3 as iio
from PIL import Image

from pathlib import Path
from tqdm import tqdm
from torch import Tensor
from torch.optim import Adam
import torch.nn.functional as F
import clip
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from controls import Controls
from environment import XplaneEnvironment
from state.state import PlaneState
from util import ft_to_me, kts_to_mps
from experiment.rl.ppo_v1 import construct_reward, act_to_control, ActorCritic, Logger
from experiment.rl.ppo_v3 import PPOBuffer


DEMO_PATH = Path(os.path.dirname(__file__)) / "../../" / "data" / "demonstration"
FIXED_THR_VAL = 0.8


def cosine_similarity(vec1: np.array, vec2: np.array):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
def extract_frames_from_video(video_path: str) -> List[Image.Image]:
    frames = []
    cap = iio.imread(video_path)
    for frame in cap:
        frames.append(Image.fromarray(frame))
    return frames


class PPOModuleV4:
    """
    Novel reward function used.
        1. extract CLIP features from every demo 
        2. calculate dtw of current trajectory with the most similar one
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

        # load CLiP
        self.clip, self.preprocess = clip.load("ViT-B/32", device=self.args.device)
        # extract features of demo videos in advance
        self.demo_feats = []
        demo_names = os.listdir(str(DEMO_PATH))
        for demo_name in demo_names:
            frames = extract_frames_from_video(str(DEMO_PATH / demo_name))
            self.demo_feats.append([self.extract_feat(frame).squeeze(0) for frame in frames])

        if train:
            # init wandb logger
            wandb.init(project=args.project, name=args.run, config=dict(args), entity="flybyml")
            wandb.watch(self.model)
            self.logger = Logger()
            # configure model checkpoint save root
            self.ckpt_root = Path(os.path.dirname(__file__)) / "../" / args.project / "logs" / args.run
            os.makedirs(self.ckpt_root, exist_ok=True)

    def extract_feat(self, frame: Image.Image):
        with torch.no_grad():
            image = self.preprocess(frame).unsqueeze(0).to(self.args.device)
            return self.clip.encode_image(image).cpu().numpy()
        
    def construct_clip_reward(self, ep_frames: List[np.array], window_size: int = 32) -> np.array:
        """
        Compute dtw distance between current episode video and most similar demo
        based on clip features
        """
        rew = []
        ep_feats = [self.extract_feat(Image.fromarray(frame)).squeeze(0) for frame in ep_frames]
        
        for start_idx in range(len(ep_frames)-window_size+1):
            # find the most similar demo
            frame_sim = [cosine_similarity(ep_feats[start_idx], feats[0]) for feats in self.demo_feats]
            idx = np.argmax(frame_sim)
            
            # calculate distance
            dtw_dist, _ = fastdtw(ep_feats[start_idx:start_idx+window_size], self.demo_feats[idx], dist=euclidean)
            rew.append(1.0 / (1.0 + dtw_dist))

        return np.array(rew)

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
                    # compute windowed clip reward once
                    clip_rew = self.construct_clip_reward(ep_frames)
                    obs = self.construct_observation(state, prev_state, self.obj, self.args.device)
                    _, value, _ = self.model.step(obs)
                    self.buf.finish_path(clip_rew, value)

                    self.logger.add(EpRet=ep_ret)
                    self.logger.add(EpRetCLIP=np.sum(clip_rew))

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
            self.logger.log('EpRetCLIP', with_min_max=True)
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
    conf = OmegaConf.load("C:/Users/lee/Desktop/ml/flybyml/experiment/config/rl_ppo_v4.yaml")
    conf.merge_with_cli()

    ckpt_path = conf.ckpt_path
    model = PPOModuleV4(conf, train=False, ckpt_path=ckpt_path)
    model.test()
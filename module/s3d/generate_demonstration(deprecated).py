import os
import sys
import time
import random
from pathlib import Path
from omegaconf import OmegaConf
import pygetwindow as pw
import pyautogui
from mss import mss
import numpy as np
import cv2
from threading import Thread

from controls import Controls, Camera
from environment import XplaneEnvironment
from experiment.rl.ppo_v1 import PPOModuleV1, act_to_control
from util import ft_to_me, kts_to_mps

FIXED_THR_VAL = 0.8

# TODO: view problem. needs to be reviewed
class PPOAgent(PPOModuleV1):
    def __init__(self, args, ckpt_path=None):
        super().__init__(args, False, ckpt_path)

    def test(self, env, state):
        self.env = env
        prev_state = state
        
        while True:
            obs = self.construct_observation(state, prev_state, self.obj, self.args.device)
            action = self.model.infer(obs)

            prev_state = state
            state = self.env.rl_step(act_to_control(action), self.args.step_interval)

class VideoGenerator:
    def __init__(self, agent=None, duration=10):
        self.env = XplaneEnvironment(agent=None)
        self.agent = agent

        window = pw.getWindowsWithTitle("X-System")[0]
        self.monitor = {"top": window.top, "left": window.left, "width": window.width, "height": window.height}
        self.duration = duration

    def capture_frames(self):
        start_time = time.time()
        frames = []
        with mss() as sct:
            while time.time() - start_time < self.duration:
                frame = np.array(sct.grab(self.monitor), dtype=np.uint8)
                frames.append(frame)
                time.sleep(0.1)     # time step interval: same as ppo_v1
        return frames

    def generate_video(self, frames):
        cur_dir = Path(os.path.dirname(__file__))
        video_path = str(cur_dir / f'level_demonstration_{self.duration}s.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = len(frames) / self.duration   

        winsize = frames[0].shape
        video = cv2.VideoWriter(video_path, fourcc, fps, (winsize[1], winsize[0]))
        for fr in frames:
            video.write(fr[:, :, :3])
        video.release()

    def run(self):
        # set env
        rand_elev = 2 * random.random() -1
        control = Controls(elev=rand_elev, thr=FIXED_THR_VAL)
        state, ctrl = self.env.reset(37.5586545, 126.7944739, ft_to_me(20000), 0, kts_to_mps(300), 0, controls=control)
        
        # set agent
        if self.agent:
            simulation_thread = Thread(target=self.agent.test, args=(self.env, state), daemon=True)
            simulation_thread.start()
        
        # start capturing frames
        user_input = input("Start capturing video? (y/n): ").strip().lower()
        if user_input == 'y':            
            frames = self.capture_frames()
            self.generate_video(frames)

if __name__ == "__main__":
    # python generate_video.py [agent type] [video duration (sec)]
    if sys.argv[1] == 'ppo':
        conf = OmegaConf.load("C:/Users/lee/Desktop/ml/flybyml/experiment/config/rl_ppo_v1.yaml")
        ckpt_path = "C:/Users/lee/Desktop/ml/flybyml/experiment/rl/logs/EpRet=191.80759536772968.ckpt"
        
        agent = PPOAgent(conf, ckpt_path)
    else:
        agent = None
    video_generator = VideoGenerator(agent, int(sys.argv[2]))
    video_generator.run()
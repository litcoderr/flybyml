import os
import sys
import time
import random
from pathlib import Path
import pygetwindow as pw
import pyautogui
from mss import mss
import numpy as np
import cv2

from controls import Controls, Camera
from environment import XplaneEnvironment
from util import ft_to_me, kts_to_mps

FIXED_THR_VAL = 0.8

class VideoGenerator:
    def __init__(self, duration=10, demon=False):
        # set env
        env = XplaneEnvironment(agent=None)

        rand_elev = 2 * random.random() -1
        control = Controls(elev=rand_elev, thr=FIXED_THR_VAL, camera=Camera(0,0,-15,360,-32,-2))
        _, control = env.reset(37.5586545, 126.7944739, ft_to_me(1000), 0, kts_to_mps(300), 0, controls=control)

        # detect window
        window = pw.getWindowsWithTitle("X-System")[0]
        self.monitor = {"top": window.top, "left": window.left, "width": window.width, "height": window.height}
        self.duration = duration
        self.demon = demon

    def capture_frames(self):
        start_time = time.time()
        frames = []
        # use mss()
        # pyautogui is too slow to execute at 0.1sec time interval
        with mss() as sct:
            while time.time() - start_time < self.duration:
                frame = np.array(sct.grab(self.monitor), dtype=np.uint8)
                frames.append(frame)
                time.sleep(0.1)     # time step interval: same as ppo_v1
        return frames

    def generate_video(self, frames):    
        cur_dir = Path(os.path.dirname(__file__))
        sub_dir = "demonstration" if self.demon else "failure"

        video_path = str(cur_dir / sub_dir / f'level_off_{self.duration}s.mp4')
        os.makedirs(cur_dir / sub_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = len(frames) / self.duration   

        winsize = frames[0].shape
        video = cv2.VideoWriter(video_path, fourcc, fps, (winsize[1], winsize[0]))
        for fr in frames:
            video.write(fr[:, :, :3])
        video.release()

    def run(self):
        # start capturing frames
        user_input = input("Start capturing video? (y/n): ").strip().lower()
        if user_input == 'y':
            time.sleep(1)            
            frames = self.capture_frames()
            self.generate_video(frames)

if __name__ == "__main__":
    duration = int(sys.argv[1])
    demon = bool(sys.argv[2])

    video_generator = VideoGenerator(duration, demon)
    video_generator.run()
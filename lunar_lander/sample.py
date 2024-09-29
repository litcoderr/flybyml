import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from IPython import display


env = gym.make("LunarLander-v3", render_mode="rgb_array")

def heuristic(env, state):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        state (list): The state. Attributes:
        s[0] is the horizontal coordinate
        s[1] is the vertical coordinate
        s[2] is the horizontal speed
        s[3] is the vertical speed
        s[4] is the angle
        s[5] is the angular speed
        s[6] 1 if first leg has contact, else 0
        s[7] 1 if second leg has contact, else 0
    Returns:
        act: The heuristic to be fed into the step function to determine the next step and reward.

    """

    angle_targ = state[0] * 0.5 + state[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        state[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - state[4]) * 0.5 - (state[5]) * 1.0
    hover_todo = (hover_targ - state[1]) * 0.5 - (state[3]) * 0.5

    if state[6] or state[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(state[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if env.unwrapped.continuous:
        # Action is two floats [main engine, left-right engines].
        # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
        # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a

seed = 42

total_rew = 0
steps = 0
state, info = env.reset(seed=seed)

plt.ion()

while True:
    act = heuristic(env, state)
    state, rew, term, trunc, info = env.step(act)
    total_rew += rew

    plt.imshow(env.render())
    plt.pause(0.00001)
    display.display(plt.gcf())
    display.clear_output(wait=True)

    if steps % 20 == 0 or term or trunc:
        print("observations:", " ".join(f"{x:+0.2f}" for x in state))
        print(f"step {steps} total reward: {total_rew:+0.2f}")
    steps += 1

    if term or trunc:
        break
    
    plt.show()


env.close()
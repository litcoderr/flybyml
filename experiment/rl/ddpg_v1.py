import random
import torch
import torch.nn as nn

from torch import Tensor

from state.state import PlaneState
from util import ft_to_me, kts_to_mps
from environment import XplaneEnvironment


# environment constraint constants
MIN_ALT = ft_to_me(15000)
MAX_ALT = ft_to_me(25000)
MIN_HEADING = 0
MAX_HEADING = 360
MIN_SPD = kts_to_mps(180)
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
    return:
        alt: sampled altitude
        heading: sampled heading
        spd: sampled spd
    """
    alt = random.randint(MIN_ALT, MAX_ALT)
    heading = random.randint(MIN_HEADING, MAX_HEADING)
    spd = random.randint(MIN_SPD, MAX_SPD)
    return alt, heading, spd


def is_boundary(state: PlaneState):
    return state.pos.alt >= MIN_ALT and state.pos.alt <= MAX_ALT and \
           state.att.yaw >= MIN_HEADING and state.att.yaw <= MAX_HEADING and \
           state.spd >= MIN_SPD and state.spd <= MAX_SPD and \
           state.att.pitch >= MIN_PITCH and state.att.pitch <= MAX_PITCH and \
           state.att.roll >= MIN_ROLL and state.att.roll <= MAX_ROLL


class ActorCriticModelV1(nn.Module):
    def __init__(self):
        pass


class DDPGModuleV1:
    def __init__(self, args):
        self.args = args
        random.seed(self.args.seed)

        # initialize xplane environment
        # agent is not necessary, since we will be feeding in input through rl_step function
        self.env = XplaneEnvironment(agent=None)
    
    def get_observation(self, cur_state: PlaneState, prev_state: PlaneState, target) -> Tensor:
        """
        construct observation based on current state and prev states

        return:
            observation: Tensor [e_pitch, e_roll, e_spd, d_roll, d_pitch, d_spd, i_roll, i_pitch, i_spd]
                e_: stands for error
                d_: stands for delta
                i_: stands for intensty [0-1]
        """
        # TODO implement constructing observation
        # every value is normalized
        error = torch.tensor([(cur_state.att.pitch - target['pitch'][0])/PITCH_NORM,
                              cur_state.att.roll - target['roll'][0]/ROLL_NORM,
                              cur_state.spd - target['spd'][0]/SPD_NORM])
        delta = torch.tensor([(cur_state.att.pitch - prev_state.att.pitch)/PITCH_NORM,
                              cur_state.att.roll - prev_state.att.roll/ROLL_NORM,
                              cur_state.spd - prev_state.spd/SPD_NORM])
        intensity = torch.tensor([target['pitch'][1], target['roll'][1], target['spd'][1]])
        return torch.cat([error, delta, intensity]).to(self.args.device)
    
    def train(self):
        for step in range(self.args.train.total_steps):
            #  reset environment with newly sampled weather etc.
            if step % self.args.train.reset_period == 0:
                alt, heading, spd = sample_state()
                state = self.env.reset(0, 0, alt, heading, spd, 0, pause=True)
                prev_state = state

                # construct target            
                # add buffer when sampling target so that the agent has space to explore
                # key -> Tuple[target value, intensity]
                target = {
                    'pitch': (random.randint(MIN_PITCH+5, MAX_PITCH-5), random.random()),
                    'roll': (random.randint(MIN_ROLL+10, MAX_ROLL-10), random.random()),
                    'spd': (random.randint(MIN_SPD+kts_to_mps(30), MAX_SPD-kts_to_mps(30)), random.random()),
                }
            obs_t = self.get_observation(state, prev_state, target)
            print('test')

            if step > self.args.train.start_steps:
                # TODO sample action from actor-critic network
                pass
            else:
                # TODO randomly sample action for better exploration
                pass

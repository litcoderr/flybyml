import math
from environment import XplaneEnvironment
from agent import AgentInterface
from state.plane import PlaneState
from controls import Controls
from aircraft.c172sp import C172SP
from aircraft.b738 import B738
from state.pos import Gimpo

class Bug:
    pitch = 5
    roll = -10
    spd = 60

class HeuristicAgent(AgentInterface):
    def __init__(self):
        super().__init__(aircraft=C172SP())

        self.control_buffer = Controls(
            elev = 0,
            ail = 0,
            rud = 0,
            thr = 0.8,
            gear = 1,
            flaps = 0
        )
        self.state_buffer = None
    
    def calc_elev(self, state: PlaneState):
        MAX_PITCH_RATE = 1 # degree / sec
        PITCH_DIFF_THRESH = 5 # degree
        CONTROL_JERKNESS = 0.05
        
        def calc_target_pitchrate(pitch_diff):
            if pitch_diff < -PITCH_DIFF_THRESH:
                return MAX_PITCH_RATE
            elif pitch_diff > PITCH_DIFF_THRESH:
                return -MAX_PITCH_RATE
            else:
                return (-MAX_PITCH_RATE / PITCH_DIFF_THRESH) * pitch_diff
        
        def calc_elev_delta(pitch_rate_diff):
            if pitch_rate_diff < 0:
                return CONTROL_JERKNESS * math.pow(pitch_rate_diff, 2)
            else:
                return -1 * CONTROL_JERKNESS * math.pow(pitch_rate_diff, 2)
        
        pitch_diff = state.att.pitch - Bug.pitch
        pitch_rate = state.att.pitch - self.state_buffer.att.pitch
        target_pitchrate = calc_target_pitchrate(pitch_diff)
        pitch_rate_diff = pitch_rate - target_pitchrate
        elev_delta = calc_elev_delta(pitch_rate_diff)

        target_position = self.control_buffer.elev + elev_delta
        # print(f"pitch: {state.att.pitch:.2f} elevator: {target_position:.2f}")
        # clip elevator position
        if target_position >= 1:
            return 1
        elif target_position <= -1:
            return -1
        else:
            return target_position

    def calc_ail(self, state: PlaneState):
        MAX_ROLL_RATE = 1 # degree / sec
        ROLL_DIFF_THRESH = 5 # degree
        CONTROL_JERKNESS = 0.05
        
        def calc_target_rollrate(roll_diff):
            if roll_diff < -ROLL_DIFF_THRESH:
                return MAX_ROLL_RATE
            elif roll_diff > ROLL_DIFF_THRESH:
                return -MAX_ROLL_RATE
            else:
                return (-MAX_ROLL_RATE / ROLL_DIFF_THRESH) * roll_diff
        
        def calc_ail_delta(roll_rate_diff):
            if roll_rate_diff < 0:
                return CONTROL_JERKNESS * math.pow(roll_rate_diff, 2)
            else:
                return -1 * CONTROL_JERKNESS * math.pow(roll_rate_diff, 2)
        
        roll_diff = state.att.roll - Bug.roll
        roll_rate = state.att.roll - self.state_buffer.att.roll
        target_rollrate = calc_target_rollrate(roll_diff)
        roll_rate_diff = roll_rate - target_rollrate
        roll_delta = calc_ail_delta(roll_rate_diff)

        target_position = self.control_buffer.ail + roll_delta
        #print(f"roll: {state.att.roll:.2f} aileron: {target_position:.2f}")
        # clip elevator position
        if target_position >= 1:
            return 1
        elif target_position <= -1:
            return -1
        else:
            return target_position
    
    def sample_action(self, state: PlaneState) -> Controls:
        if self.state_buffer is None:
            self.state_buffer = state
        else:
            controls = Controls(
                elev = self.calc_elev(state),
                ail = self.calc_ail(state),
                rud = 0,
                thr = 0.8,
                gear = 1,
                flaps = 0
            )
            self.state_buffer = state
            self.control_buffer = controls
        return self.control_buffer

airport = Gimpo()
agent = HeuristicAgent()
env = XplaneEnvironment(agent, airport, frame_interval=0.1)

state = env.reset(heading=320, alt=1000, spd=30)
count = 0
while True:
    if count > 100:
        Bug.spd = 90
    count += 1
    state = env.step(state)

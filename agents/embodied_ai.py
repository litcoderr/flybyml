import math
import torch

from agents import AgentInterface
from aircraft.interface import Aircraft
from aircraft.b738 import B738
from state.state import PlaneState
from controls import Controls, Camera
from experiment.baseline.teacher_force import AlfredBaselineTeacherForce
from experiment.simple_fc.batch_normalize import FCBaselineBatchNormalize


class MLAgentBase(AgentInterface):
    def __init__(self, aircraft: Aircraft, success_distance: float):
        super().__init__(aircraft)
        self.dist_threshold_to_stop = success_distance
        self.control_buffer = Controls( # maintains previous control input
            elev = 0,
            ail = 0,
            rud = 0,
            thr = 0.8,
            gear = 1,
            flaps = 0,
            trim = 0
        )

    def is_goal_reached(self, state: PlaneState) -> bool:
        # TODO: define stop condition
        pass

'''
Clip Encoder + Reinforcement Learning (PPO)
'''
class ClipAgent(MLAgentBase):
    def __init__(self, aircraft, success_distance):
        super().__init__(aircraft, success_distance)
    
    def sample_action(self, state: PlaneState, **kwargs) -> Controls:
        if self.is_goal_reached(state):
            return None
        controls = Controls(
            # TODO: action
        )
        self.control_buffer = controls
        return self.control_buffer

'''
ResNet-50 Conv 
'''
class AlfredAgent(MLAgentBase):
    def __init__(self, aircraft, success_distance):
        super().__init__(aircraft, success_distance)
    
    def sample_action(self, state: PlaneState, **kwargs) -> Controls:
        if self.is_goal_reached(state):
            return None
        controls = Controls(
            # TODO: action
        )
        self.control_buffer = controls
        return self.control_buffer


class AlfredBaselineTeacherForceAgent(AgentInterface):
    def __init__(self, args):
        super().__init__(B738())
        self.model = AlfredBaselineTeacherForce.load_from_checkpoint('C:\\Users\\litco\\Desktop\\project\\flybyml\\checkpoint\\epoch=26499-val_loss=0.000.ckpt', args=args, map_location=torch.device("cuda"))
        self.context = None
    
    def sample_action(self, state: PlaneState, **kwargs) -> Controls:
        # 1. Construct Model Input
        # set target
        tgt_rwy = kwargs['tgt_rwy'].serialize()
        tgt_position = torch.tensor(tgt_rwy['position'])
        tgt_heading = tgt_rwy['attitude'][2]

        # state and image input
        state = state.serialize()

        # 1-1. construct instruction
        relative_position = torch.tensor(state['position']) - tgt_position
        relative_heading = state['attitude'][2] - tgt_heading
        if relative_heading > 180:
            relative_heading = - (360 - relative_heading)
        elif relative_heading < -180:
            relative_heading += 360
        relative_heading = torch.tensor([math.radians(relative_heading)])
        instruction = torch.concat((relative_position, relative_heading))
        instruction = instruction.reshape(1, 1, *instruction.shape)

        # 1-2. construct sensory observation
        sensory_observation = torch.tensor([*state['attitude'][:2], state['speed'], state['vertical_speed']])
        sensory_observation = sensory_observation.reshape(1, 1, *sensory_observation.shape)

        # 1-3. construct prev action
        prev_action = torch.tensor([
            (kwargs['prev_actions'].elev + 1) / 2,
            (kwargs['prev_actions'].ail + 1) / 2,
            (kwargs['prev_actions'].rud + 1) / 2,
            kwargs['prev_actions'].thr,
            kwargs['prev_actions'].gear,
            kwargs['prev_actions'].flaps,
            (kwargs['prev_actions'].trim + 1) / 2,
            kwargs['prev_actions'].brake,
            kwargs['prev_actions'].spd_brake,
            kwargs['prev_actions'].reverse * -1,
        ]).reshape(1, 1, 10)

        # 2. infer
        with torch.no_grad():
            output, context = self.model({
                'sensory_observations': sensory_observation.to('cuda'),
                'instructions': instruction.to('cuda'),
                'prev_actions': prev_action.to('cuda')
            }, self.context)
            self.context = context
            output = output[0][0].to('cpu')
        
        # 3. construct action
        elevator = float(2 * output[0] - 1)
        aileron = float(2 * output[1] - 1)
        rudder = float(2 * output[2] - 1)
        thrust = float(output[3])
        gear = float(output[4])
        flaps = float(output[5])
        trim = float(2 * output[6] - 1)
        brake = float(output[7])
        spd_brake = float(output[8])
        reverser = float(-1 * output[9])

        controls = Controls(elev=elevator,
                            ail=aileron,
                            rud=0,
                            thr=thrust,
                            gear=gear,
                            flaps=flaps,
                            trim=trim,
                            brake=0,
                            spd_brake=0,
                            reverse=0,
                            camera=Camera(*[0 for _ in range(6)]))
        return controls


class FcBnAgent(AgentInterface):
    def __init__(self, args):
        super().__init__(B738())
        self.model = FCBaselineBatchNormalize.load_from_checkpoint('C:\\Users\\litco\\Desktop\\project\\flybyml\\checkpoint\\epoch=0499-val_loss=0.001.ckpt', args=args, map_location=torch.device("cuda"))
    
    def sample_action(self, state: PlaneState, **kwargs) -> Controls:
        # 1. Construct Model Input
        # set target
        tgt_rwy = kwargs['tgt_rwy'].serialize()
        tgt_position = torch.tensor(tgt_rwy['position'])
        tgt_heading = tgt_rwy['attitude'][2]

        # state and image input
        state = state.serialize()

        # 1-1. construct instruction
        relative_position = torch.tensor(state['position']) - tgt_position
        relative_heading = state['attitude'][2] - tgt_heading
        if relative_heading > 180:
            relative_heading = - (360 - relative_heading)
        elif relative_heading < -180:
            relative_heading += 360
        relative_heading = torch.tensor([math.radians(relative_heading)])
        instruction = torch.concat((relative_position, relative_heading))
        instruction = instruction.reshape(1, *instruction.shape)

        # 1-2. construct sensory observation
        sensory_observation = torch.tensor([*state['attitude'][:2], state['speed'], state['vertical_speed']])
        sensory_observation = sensory_observation.reshape(1, *sensory_observation.shape)

        # 1-3. construct prev action
        prev_action = torch.tensor([
            (kwargs['prev_actions'].elev + 1) / 2,
            (kwargs['prev_actions'].ail + 1) / 2,
            (kwargs['prev_actions'].rud + 1) / 2,
            kwargs['prev_actions'].thr,
            kwargs['prev_actions'].gear,
            kwargs['prev_actions'].flaps,
            (kwargs['prev_actions'].trim + 1) / 2,
            kwargs['prev_actions'].brake,
            kwargs['prev_actions'].spd_brake,
            kwargs['prev_actions'].reverse * -1,
        ]).reshape(1, 10)

        # 2. infer
        with torch.no_grad():
            output = self.model({
                'sensory_observations': sensory_observation.to('cuda'),
                'instructions': instruction.to('cuda'),
                'prev_actions': prev_action.to('cuda')
            })
            # [1, 10]
            output = output[0].to('cpu')
        
        # 3. construct action
        elevator = float(2 * output[0] - 1)
        aileron = float(2 * output[1] - 1)
        rudder = float(2 * output[2] - 1)
        thrust = float(output[3])
        gear = float(output[4])
        flaps = float(output[5])
        trim = float(2 * output[6] - 1)
        brake = float(output[7])
        spd_brake = float(output[8])
        reverser = float(-1 * output[9])

        controls = Controls(elev=elevator,
                            ail=aileron,
                            rud=0,
                            thr=thrust,
                            gear=gear,
                            flaps=flaps,
                            trim=trim,
                            brake=0,
                            spd_brake=0,
                            reverse=0,
                            camera=Camera(*[0 for _ in range(6)]))
        return controls
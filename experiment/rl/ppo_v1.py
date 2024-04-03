"""
Reinforcement Learning of flybyml agent
using Proximal Policy Optimization

목표: 수평을 맞춰라!
"""


from tqdm import tqdm


class PPOModuleV1:
    def __init__(self, args):
        self.args = args
    
    def train(self):
        for epoch in tqdm(range(self.args.train.epoch), desc='epoch'):
            for local_step in tqdm(range(self.args.train.steps_per_epoch), desc='epoch'):
                pass

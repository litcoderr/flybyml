from importify import Serializable

from environment import XplaneEnvironment
from aircraft import B738
from status.pos import Gimpo

class Config(Serializable):
    def __init__(self):
        super(Config, self).__init__()
        self.max_step = 600
        self.frame_interval = 0.1


if __name__ == "__main__":
    config = Config()

    env = XplaneEnvironment(aircraft=B738(), airport=Gimpo(), frame_interval=config.frame_interval)

    # train
    while True:
        # initial state
        # TODO should randomize initial state
        state = env.reset(
            heading=300,
            alt = 5000,
            spd = 230
        ).toTensor('cuda')  # reset and get initial state

        for timestep in range(config.max_step):
            # TODO sample action
            sampled_aciton = {}
            state = env.step(sampled_aciton).toTensor('cuda')
            print(state)

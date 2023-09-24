import torch


class Objective:
    def __init__(self):
        pass

    def compute_reward(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class Hold(Objective):
    def __init__(self, tgt_heading, tgt_alt, tgt_spd):
        super(Hold, self).__init__()
        self.tgt_heading = tgt_heading
        self.tgt_alt = tgt_alt
        self.tgt_spd = tgt_spd

    def compute_reward(self, state: torch.Tensor) -> torch.Tensor:
        # TODO implement reward
        pass

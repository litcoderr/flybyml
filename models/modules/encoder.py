import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet50

class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False
        for module in self.backbone.modules():
            if "BatchNorm" in type(module).__name__:
                module.momentum = 0.0
        self.backbone.eval()

        self.output_shape = (2048,)

    def forward(self, rgb_observations: torch.Tensor) -> torch.Tensor:
        """
        rgb_observations: [batch, seq_len, channel, width, height]
        """
        shape = rgb_observations.shape

        rgb_observations = rgb_observations.reshape(-1, *shape[2:])
        rgb_observations = torch.stack([self.normalize(rgb) for rgb in rgb_observations])
        rgb_x = self.backbone(rgb_observations).float()

        return rgb_x
    
class ResNetCLIPEncoder(nn.Module):
    # TODO: implement following EmbClip paper
    pass

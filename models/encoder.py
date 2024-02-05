import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet50

class ResNetEncoder(nn.Module):
    '''
    img input in resnet-50: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
    
    Accepts PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects. 
    The images are resized to resize_size=[232], followed by a central crop of crop_size=[224].
    '''
    def __init__(self):
        super.__init__()
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

    def forward(self, rgb_observations: torch.Tensor) -> torch.Tensor:
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        rgb_observations = rgb_observations.permute(0, 3, 1, 2)
        rgb_observations = (rgb_observations.float() / 255.0)
        rgb_observations = torch.stack([self.normalize(rgb) for rgb in rgb_observations])
        rgb_x = self.backbone(rgb_observations).float()

        return rgb_x
    
class ResNetCLIPEncoder(nn.Module):
    # TODO: implement following EmbClip paper
    pass
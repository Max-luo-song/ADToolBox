import logging
from typing import Tuple
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.cotracker_blocks import BasicEncoder
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class CoTrackerCNNEncoder(nn.Module):
    def __init__(self, resolution: Tuple[int, int], output_dim: int, stride: int, pretrained: bool, freeze_mode: str):
        super().__init__()
        self.resolution = resolution
        self.stride = stride
        self.output_dim = output_dim
        assert freeze_mode in ["all", "none"], f"Freezing mode {freeze_mode} not supported"

        logger.info(f"Loading CoTracker CNN Encoder")
        logger.info(f"Freezing mode: {freeze_mode}")

        self.backbone = BasicEncoder(output_dim=output_dim, stride=stride)
        if pretrained:
            cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
            backbone_state_dict = cotracker.fnet.state_dict()
            self.backbone.load_state_dict(backbone_state_dict, strict=True)
            del cotracker

        if freeze_mode == "all":
            for param in self.backbone.parameters():
                param.requires_grad = False

        H, W = self.resolution
        assert H % self.stride == 0 and W % self.stride == 0, f"Image size {H}x{W} must be divisible by stride {self.stride}"

        self.transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        assert (H, W) == self.resolution, f"Image size {H}x{W} must be {self.resolution}"

        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.transform(x)
        outputs = self.backbone(x)

        return rearrange(outputs, "(b t) c h w -> b t c h w", b=B, t=T)
    
    @property
    def embedding_dim(self):
        return self.output_dim

    def set_image_size(self, image_size: Tuple[int, int]):
        self.resolution = image_size

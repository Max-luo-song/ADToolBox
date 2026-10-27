from omegaconf import DictConfig
import torch.nn as nn
from .conv_updater import ConvUpdater
from .efficient_updateformer import EfficientUpdateFormer

def load_point_updater(config: DictConfig) -> nn.Module:
    name = config.name
    kwargs = {k: v for k, v in config.items() if k != "name"}
    if name == "conv_updater":
        return ConvUpdater(**kwargs)
    elif name == "efficient_updateformer":
        return EfficientUpdateFormer(**kwargs)
    else:
        raise ValueError(f"Unknown point updater: {name}")


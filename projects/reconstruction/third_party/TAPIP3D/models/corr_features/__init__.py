# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

from box import Box
import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Callable, List, Protocol

from .knn_feature_4d_optimized import KNNCorrFeature4D_Optimized

class CorrFeatureProtocol(Protocol):
    def prepare(self, *, feats: torch.Tensor, pcds: torch.Tensor, queries: torch.Tensor, projector: Callable[[torch.Tensor], torch.Tensor]) -> Box:
        ...
    def __call__(self, ctx: Box, curr_coords: torch.Tensor) -> torch.Tensor:
        ...

def load_corr_processor(config: DictConfig, **kwargs) -> CorrFeatureProtocol:
    name = config.name
    kwargs.update({k: v for k, v in config.items() if k != "name"}) # type: ignore

    if name == "knn_corr_4d" or name == "knn_corr_4d_optimized":
        return KNNCorrFeature4D_Optimized(**kwargs)
    else:
        raise ValueError(f"Unknown corr feature: {name}")

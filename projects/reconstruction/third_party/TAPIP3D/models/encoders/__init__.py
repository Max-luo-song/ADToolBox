from typing import Tuple
import torch
import torch.nn as nn
from omegaconf import DictConfig
from third_party.TAPIP3D.models.encoders.cotracker_cnn import CoTrackerCNNEncoder
from typing import Protocol

class EncoderProtocol(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def embedding_dim(self) -> int:
        pass

def load_encoder(resolution: Tuple[int, int], config: DictConfig) -> EncoderProtocol:
    name = config.name
    kwargs = {k: v for k, v in config.items() if k != "name"}
    if name == "cotracker_cnn":
        return CoTrackerCNNEncoder(resolution, **kwargs) # type: ignore
    else:
        raise ValueError(f"Backbone {name} not supported")

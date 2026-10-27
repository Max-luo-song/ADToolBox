# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

from typing import List, Optional, Tuple
import torch
from dataclasses import dataclass

@dataclass
class TrainData:
    coords: List[torch.Tensor] # (B, T, N, 3)
    visibs: Optional[torch.Tensor] # (B, T, N)
    iter_idx: int
    frame_range: Tuple[int, int]
    confs: Optional[torch.Tensor] = None # (B, T, N)

    def __post_init__(self):
        if isinstance(self.coords, torch.Tensor):
            self.coords = [self.coords]
        assert not isinstance(self.coords, torch.Tensor) and isinstance(self.coords, list), "coords must be a list of tensors"
        assert self.frame_range[1] - self.frame_range[0] == self.coords[0].shape[1], \
            f"Frame range {self.frame_range} does not match the number of frames {self.coords[0].shape[1]}"

    def detach(self, clone: bool):
        if not clone:
            return TrainData(
                coords=[coords_.detach() for coords_ in self.coords],
                visibs=self.visibs.detach() if self.visibs is not None else None,
                confs=self.confs.detach() if self.confs is not None else None,
                iter_idx=self.iter_idx,
                frame_range=self.frame_range,
            )
        else:
            return TrainData(
                coords=[coords_.detach().clone() for coords_ in self.coords],
                visibs=self.visibs.detach().clone() if self.visibs is not None else None,
                confs=self.confs.detach().clone() if self.confs is not None else None,
                iter_idx=self.iter_idx,
                frame_range=self.frame_range,
            )

    def to(self, device: str):
        return TrainData(
            coords=[coords_.to(device) for coords_ in self.coords],
            visibs=self.visibs.to(device) if self.visibs is not None else None,
            confs=self.confs.to(device) if self.confs is not None else None,
            iter_idx=self.iter_idx,
            frame_range=self.frame_range,
        )

    def time_slice(self, *, rel_start: int, rel_end: int):
        assert rel_start >= 0 and rel_end <= self.coords[0].shape[1] and rel_start < rel_end, "the range of start and end is out of bounds"
        return TrainData(
            coords=[coords_[:, rel_start:rel_end] for coords_ in self.coords],
            visibs=self.visibs[:, rel_start:rel_end] if self.visibs is not None else None,
            confs=self.confs[:, rel_start:rel_end] if self.confs is not None else None,
            iter_idx=self.iter_idx,
            frame_range=(self.frame_range[0] + rel_start, self.frame_range[0] + rel_end),
        )

    def query_slice(self, s: slice):
        return TrainData(
            coords=[coords_[:, :, s] for coords_ in self.coords],
            visibs=self.visibs[:, :, s] if self.visibs is not None else None,
            confs=self.confs[:, :, s] if self.confs is not None else None,
            iter_idx=self.iter_idx,
            frame_range=self.frame_range,
        )
@dataclass
class Prediction:
    coords: torch.Tensor # (B, T, N, 3)
    visibs: torch.Tensor # (B, T, N)
    confs: Optional[torch.Tensor] = None # (B, T, N)

    def __post_init__(self):
        assert not self.coords.requires_grad and not self.visibs.requires_grad

    def to(self, device: str):
        return Prediction(
            coords=self.coords.to(device),
            visibs=self.visibs.to(device),
            confs=self.confs.to(device) if self.confs is not None else None,
        )

    def time_slice(self, start: int, end: int):
        assert start >= 0 and end <= self.coords.shape[1] and start < end, "the range of start and end is out of bounds"
        return Prediction(
            coords=self.coords[:, start:end],
            visibs=self.visibs[:, start:end],
            confs=self.confs[:, start:end] if self.confs is not None else None,
        )

    def query_slice(self, s: slice):
        return Prediction(
            coords=self.coords[:, :, s],
            visibs=self.visibs[:, :, s],
            confs=self.confs[:, :, s] if self.confs is not None else None,
        )

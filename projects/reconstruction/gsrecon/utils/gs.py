# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
from gsplat.rendering import rasterization
from tqdm import tqdm
import os
from IPython import embed
from torch.utils.data import Dataset, DataLoader
import random
import open3d as o3d
from PIL import Image
from torchvision import transforms as TF


palette_10 = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (70, 130, 180),
}

# 0:Road
# 1:Building
# 2:Vegetation
# 3:Vehicle
# 4:Person
# 5:Cyclist
# 6:Traffic Sign
# 7:truck
# 8:Sidewalk
# 9:Sky

def concat_list(list_1, list_2):
    if list_2[0].shape == torch.Size([0]):
        return list_1
    concated_list = []
    for i in range(len(list_1)):
        item = torch.concat((list_1[i],list_2[i]),dim=0)
        concated_list.append(item)
    return concated_list


def get_masked_gs(point_map, gs_map, mask, idx=None):
    # point_map: B,S,H,W,3
    # gs_map: B,S,H,W,C
    # mask:, B,S,H,W
    if idx is not None:
        point_map = point_map[:,idx,...]
        gs_map = gs_map[:,idx,...]
        mask = mask[:,idx,...]
    world_points = point_map[mask].reshape(-1, 3)
    rgbs = gs_map[...,:3][mask].reshape(-1, 3)
    opacity = gs_map[...,:3:4][mask].reshape(-1)
    scales = gs_map[...,4:7][mask].reshape(-1, 3)
    rotation = gs_map[...,7:11][mask].reshape(-1, 4)
    return world_points, rgbs, opacity, scales, rotation


def get_split_gs(gs_map, mask):
    rgbs = gs_map[..., :3][mask].reshape(-1, 3)
    opacity = gs_map[..., 3:4][mask].reshape(-1)
    scales = gs_map[..., 4:7][mask].reshape(-1, 3)
    rotation = gs_map[..., 7:11][mask].reshape(-1, 4)
    return rgbs, opacity, scales, rotation


def gs_dict(points, rgbs, opacity, scales, rotation):
    gs_dict = {}
    gs_dict['means'] = points
    gs_dict['quats'] = rotation
    if opacity.dim() == 1:
        opacity = opacity[...,None]
    gs_dict['opacities'] = opacity
    gs_dict['scales'] = scales
    gs_dict['features_dc'] = rgbs
    return gs_dict

def get_gs_items(gs_dict):
    points = gs_dict['means'] 
    rotation = gs_dict['quats'] 
    opacity = gs_dict['opacities'][...,0] 
    scales = gs_dict['scales'] 
    rgbs = gs_dict['features_dc'] 
    return points, rgbs, opacity, scales, rotation


import numpy as np

import numpy as np

def downsample_3dgs(points, rgbs, opacity, scales, rotation, num_points=200000):
    N = points.shape[0]
    if num_points >= N:
        return points, rgbs, opacity, scales, rotation

    # Compute importance weights: opacity * volume
    volume = scales.prod(dim=1)                    # (N,)
    weights = opacity * volume                     # (N,)
    weights = weights / weights.sum()              # Normalize to sum to 1

    # Sample indices with probability proportional to weights
    indices = torch.multinomial(weights, num_points, replacement=False)  # (num_points,)

    return (
        points[indices],
        rgbs[indices],
        opacity[indices],
        scales[indices],
        rotation[indices]
    )


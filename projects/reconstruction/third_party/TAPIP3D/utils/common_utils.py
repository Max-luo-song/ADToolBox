import pickle
from typing import Dict, Optional, Sequence
from pathlib import Path
import json
import torch
import numpy as np        
import logging
from rich.logging import RichHandler
from einops import repeat, rearrange
from third_party.TAPIP3D.datasets.datatypes import SliceData
from third_party.cotracker.model_utils import bilinear_sampler
from dataclasses import fields, is_dataclass
from functools import wraps
from box import Box

def setup_logger():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(markup=True)]
    )

def convert_tensor(x, allow_cast: bool):
    if not isinstance(x, torch.Tensor):
        return x
    if x.dtype not in [torch.float16, torch.bfloat16]:
        return x
    if allow_cast:
        return x.to(dtype=torch.float32)
    else:
        raise TypeError(f"Tensor of dtype {x.dtype} is not allowed. Expected float32.")
    return x

def convert_tensor_struct(data, allow_cast: bool):
    if isinstance(data, torch.Tensor):
        return convert_tensor(data, allow_cast)
    elif isinstance(data, list):
        return [convert_tensor_struct(item, allow_cast) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_tensor_struct(item, allow_cast) for item in data)
    elif isinstance(data, Box):
        return Box({key: process_input(value) for key, value in data.items()})
    elif isinstance(data, dict):
        return {key: convert_tensor_struct(value, allow_cast) for key, value in data.items()}
    elif is_dataclass(data):
        return type(data)(**{field.name: convert_tensor_struct(getattr(data, field.name), allow_cast) for field in fields(data)})
    else:
        return data

def ensure_float32(allow_cast: bool = True):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            new_args = tuple(convert_tensor_struct(arg, allow_cast) for arg in args)
            new_kwargs = {key: convert_tensor_struct(value, allow_cast) for key, value in kwargs.items()}
            with torch.autocast(device_type='cuda', enabled=False):
                return convert_tensor_struct(func(*new_args, **new_kwargs), True)
        return wrapper
    return decorator

def cast_float32():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            new_args = tuple(convert_tensor_struct(arg, True) for arg in args)
            new_kwargs = {key: convert_tensor_struct(value, True) for key, value in kwargs.items()}
            return convert_tensor_struct(func(*new_args, **new_kwargs), True)
        return wrapper
    return decorator

@ensure_float32(allow_cast=False)
def batch_unproject(depth: torch.Tensor, intrinsic: torch.Tensor, extrinsic: torch.Tensor, channel_last: bool = False) -> torch.Tensor:
    if depth.ndim == 4:
        pcds = batch_unproject(
            depth.view(-1, *depth.shape[2:]),
            intrinsic.view(-1, *intrinsic.shape[2:]),
            extrinsic.view(-1, *extrinsic.shape[2:]),
            channel_last
        )
        return pcds.reshape(depth.shape[:2] + pcds.shape[1:])
    assert depth.ndim == 3, "depth must be a 3D array"
    t, h, w = depth.shape
    assert intrinsic.shape == (t, 3, 3), f"intrinsic must be a 3D array of shape (t, 3, 3), got {intrinsic.shape}"
    assert extrinsic.shape == (t, 4, 4), f"extrinsic must be a 3D array of shape (t, 4, 4), got {extrinsic.shape}"

    # Generate 3D point cloud in world coordinates
    v, u = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing='ij')
    uv_homogeneous = torch.stack((u, v, torch.ones_like(u)), dim=-1).float()
    K_inv = torch.linalg.inv(intrinsic)
    camera_coords = torch.einsum("nij, xyj -> nxyi", K_inv, uv_homogeneous)
    camera_coords = camera_coords * depth[..., None]
    camera_coords = torch.cat((camera_coords, torch.ones_like(camera_coords[..., :1])), dim=-1)

    inv_extrinsics = torch.linalg.inv(extrinsic)
    world_coordinates = torch.einsum("nij, nxyj -> nxyi", inv_extrinsics, camera_coords)
    
    if channel_last:
        return world_coordinates[..., :3]
    else:
        return world_coordinates[..., :3].permute(0, 3, 1, 2)

@ensure_float32(allow_cast=False)
def batch_unproject_uv_space(depth: torch.Tensor, intrinsic: torch.Tensor, channel_last: bool = False, eps: float = 1e-5) -> torch.Tensor:
    if depth.ndim == 4:
        pcds = batch_unproject_uv_space(
            depth.view(-1, *depth.shape[2:]),
            intrinsic.view(-1, *intrinsic.shape[2:]),
            channel_last
        )
        return pcds.reshape(depth.shape[:2] + pcds.shape[1:])
    assert depth.ndim == 3, "depth must be a 3D array"
    t, h, w = depth.shape
    assert intrinsic.shape == (t, 3, 3), f"intrinsic must be a 3D array of shape (t, 3, 3), got {intrinsic.shape}"

    v, u = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing='ij')
    logd = torch.log(torch.clamp(depth, min=eps))
    focals = torch.sqrt(intrinsic[..., 0, 0] * intrinsic[..., 1, 1])

    u_v_logd = torch.stack((u[None].expand(t, -1, -1), v[None].expand(t, -1, -1), focals[:, None, None] * logd), dim=-1)
    
    if channel_last:
        return u_v_logd
    else:
        return u_v_logd.permute(0, 3, 1, 2)

@ensure_float32(allow_cast=False)
def batch_project(
    pts3d: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    pts3d_homogeneous = torch.cat((pts3d, torch.ones_like(pts3d[..., :1])), dim=-1) # (n, 4)

    pts_camera_homogeneous = torch.einsum("...ij, ...j -> ...i", extrinsics, pts3d_homogeneous) # (n, 4)
    pts_camera = pts_camera_homogeneous[..., :3] / pts_camera_homogeneous[..., 3:] # (n, 3)

    pts_image_homogeneous = torch.einsum("...ij, ...j -> ...i", intrinsics, pts_camera) # (n, 3)
    pts_image = pts_image_homogeneous[..., :2] / pts_image_homogeneous[..., 2:] # (n, 2)

    return pts_image

def project_queries(queries_3d: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    B, N, _ = queries_3d.shape
    T = intrinsics.shape[1]
    assert extrinsics.shape == (B, T, 4, 4), f"extrinsics should have shape (B, T, 4, 4), got {extrinsics.shape}"
    assert intrinsics.shape == (B, T, 3, 3), f"intrinsics should have shape (B, T, 3, 3), got {intrinsics.shape}"
    assert queries_3d.shape == (B, N, 4), f"queries_3d should have shape (B, N, 4), got {queries_3d.shape}"

    query_times = queries_3d[..., 0].to(torch.int64) # (B, N)
    batch_indices = repeat(torch.arange(B, device=queries_3d.device), "b -> b n", n=N)
    intrinsics = intrinsics[batch_indices, query_times] # (B, N, 3, 3)
    extrinsics = extrinsics[batch_indices, query_times] # (B, N, 4, 4)
    queries_2d = batch_project(queries_3d[..., 1:].reshape(B * N, 3), intrinsics.reshape(B * N, 3, 3), extrinsics.reshape(B * N, 4, 4)).reshape(B, N, 2)
    return torch.cat((query_times[..., None], queries_2d), dim=-1)

def unproject_queries(queries_2d: torch.Tensor, depths: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    B, N, _ = queries_2d.shape
    T = intrinsics.shape[1]
    H, W = depths.shape[2:]
    assert extrinsics.shape == (B, T, 4, 4), f"extrinsics should have shape (B, T, 4, 4), got {extrinsics.shape}"
    assert intrinsics.shape == (B, T, 3, 3), f"intrinsics should have shape (B, T, 3, 3), got {intrinsics.shape}"
    assert queries_2d.shape == (B, N, 3), f"queries_2d should have shape (B, N, 3), got {queries_2d.shape}"
    assert depths.shape == (B, T, H, W), f"depths should have shape (B, T, H, W), got {depths.shape}"

    query_times = queries_2d[..., 0].to(torch.int64) # (B, N)
    pcds = batch_unproject(depths, intrinsics, extrinsics) # (B, T, C, H, W)
    queries_3d = rearrange(bilinear_sampler(rearrange(pcds, "b t c h w -> b c t h w"), rearrange(queries_2d, "b n d -> b 1 1 n d")), "b c 1 1 n -> b n c")
    return torch.cat((query_times[..., None], queries_3d), dim=-1)

@ensure_float32(allow_cast=False)
def apply_homo_transform(coords: torch.Tensor, transform: torch.Tensor, pad_ones: bool = True):
    if pad_ones:
        coords_homo = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)
    else:
        coords_homo = coords
    coords_transformed_homo = torch.einsum("...ij,...j->...i", transform, coords_homo)
    coords_transformed = coords_transformed_homo[..., :-1] / coords_transformed_homo[..., -1:]
    return coords_transformed
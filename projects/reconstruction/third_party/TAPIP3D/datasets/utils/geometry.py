import numpy as np
import torch
from einops import repeat

def batch_unproject_np(depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray, is_distance: bool = False) -> np.ndarray:
    assert depth.ndim == 3, "depth must be a 2D array"
    t, h, w = depth.shape
    assert intrinsic.shape == (t, 3, 3), f"intrinsic must be a 3D array of shape (t, 3, 3), got {intrinsic.shape}"
    assert extrinsic.shape == (t, 4, 4), f"extrinsic must be a 3D array of shape (t, 4, 4), got {extrinsic.shape}"

    # Generate 3D point cloud in world coordinates
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    uv_homogeneous = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    K_inv = np.linalg.inv(intrinsic).astype(np.float32)
    camera_coords = np.einsum("nij, xyj -> nxyi", K_inv, uv_homogeneous)
    if is_distance:
        camera_coords = camera_coords / np.sqrt((camera_coords ** 2).sum(-1))[..., None]
    camera_coords = camera_coords * depth[..., None]
    camera_coords = np.concatenate((camera_coords, np.ones_like(camera_coords[..., :1])), axis=-1)

    inv_extrinsics = np.linalg.inv(extrinsic)
    world_coordinates = np.einsum("nij, nxyj -> nxyi", inv_extrinsics, camera_coords)
    return world_coordinates[..., :3]

def batch_distance_to_depth(distances: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    h, w = distances.shape[-2:]
    v, u = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    uv_homogeneous = torch.stack((u, v, torch.ones_like(u)), dim=-1).to(distances.device).to(intrinsics.dtype)
    if len(distances.shape) == 3: # t h w
        uv_homogeneous = repeat(uv_homogeneous, 'h w c -> t h w c', t=distances.shape[0])
    elif len(distances.shape) == 4: # b t h w
        uv_homogeneous = repeat(uv_homogeneous, 'h w c -> b t h w c', b=distances.shape[0], t=distances.shape[1])
    else:
        raise ValueError(f"distances must be a 3D or 4D tensor, got {len(distances.shape)}D tensor")

    K_inv = torch.linalg.inv(intrinsics)
    camera_coords = torch.einsum("...ij, ...xyj -> ...xyi", K_inv, uv_homogeneous)
    depth = camera_coords[..., -1] * distances / torch.sqrt((camera_coords ** 2).sum(-1))

    return depth

def batch_distance_to_depth_np(distances: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    h, w = distances.shape[-2:]
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    uv_homogeneous = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    if len(distances.shape) == 3: # t h w
        uv_homogeneous = repeat(uv_homogeneous, 'h w c -> t h w c', t=distances.shape[0])
    elif len(distances.shape) == 4: # b t h w
        uv_homogeneous = repeat(uv_homogeneous, 'h w c -> b t h w c', b=distances.shape[0], t=distances.shape[1])
    else:
        raise ValueError(f"distances must be a 3D or 4D tensor, got {len(distances.shape)}D tensor")

    K_inv = np.linalg.inv(intrinsics)
    camera_coords = np.einsum("...ij, ...xyj -> ...xyi", K_inv, uv_homogeneous)
    depth = camera_coords[..., -1] * distances / np.sqrt((camera_coords ** 2).sum(-1))

    return depth


def batch_depth_to_distance(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    h, w = depth.shape[-2:]
    v, u = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    uv_homogeneous = torch.stack((u, v, torch.ones_like(u)), dim=-1).to(depth.device).to(intrinsics.dtype)
    if len(depth.shape) == 3: # t h w
        uv_homogeneous = repeat(uv_homogeneous, 'h w c -> t h w c', t=depth.shape[0])
    elif len(depth.shape) == 4: # b t h w
        uv_homogeneous = repeat(uv_homogeneous, 'h w c -> b t h w c', b=depth.shape[0], t=depth.shape[1])
    else:
        raise ValueError(f"depth must be a 3D or 4D tensor, got {len(depth.shape)}D tensor")

    K_inv = torch.linalg.inv(intrinsics)
    camera_coords = torch.einsum("...ij, ...xyj -> ...xyi", K_inv, uv_homogeneous)
    distances = depth * torch.sqrt((camera_coords ** 2).sum(-1))

    return distances
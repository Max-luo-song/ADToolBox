# Modified by ADToolbox contributors, 2026: standalone reconstruction/edit/export extensions.
# Miscellaneous utility functions for exporting point clouds.
import importlib
import logging
import os

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import torch.distributed as dist

logger = logging.getLogger()

def load_gaussians_from_3dgs_ply(path):
    point_cloud = o3d.t.io.read_point_cloud(path)
    point = point_cloud.point
    means = torch.from_numpy(point.positions.numpy())
    f_dc = torch.cat([torch.from_numpy(point[f'f_dc_{index}'].numpy()) for index in range(3)], dim=-1)
    opacity = torch.from_numpy(point['opacity'].numpy())
    scales = torch.cat([torch.from_numpy(point[f'scale_{index}'].numpy()) for index in range(3)], dim=-1)
    quaternions = torch.cat([torch.from_numpy(point[f'rot_{index}'].numpy()) for index in range(4)], dim=-1)
    return {
        'means': means,
        'colors': torch.clamp(0.5 + 0.28209479177387814 * f_dc, 0.0, 1.0),
        'opacities': torch.sigmoid(opacity).reshape(-1),
        'scales': torch.exp(scales),
        'quaternions': F.normalize(quaternions, dim=-1),
    }

def import_str(string: str):
    """ Import a python module given string paths

    Args:
        string (str): The given paths

    Returns:
        Any: Imported python module / object
    """
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)

def export_points_to_ply(
    positions: torch.tensor,
    colors: torch.tensor,
    save_path: str,
    normalize: bool = False,
    ):
    # normalize points
    if normalize:
        aabb_min = positions.min(0)[0]
        aabb_max = positions.max(0)[0]
        positions = (positions - aabb_min) / (aabb_max - aabb_min)
    if isinstance(colors, torch.Tensor):
        positions = positions.cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()
    
    # clamp colors
    colors = np.clip(colors, a_min=0., a_max=1.)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)

def export_gaussians_to_ply(model, path, name='point_cloud.ply', aabb=None):
    model.eval()
    filename = os.path.join(path, name)
    map_to_tensors = {}
    
    with torch.no_grad():
        positions = model.means
        if aabb is not None:
            aabb = aabb.to(positions.device)
            aabb_min, aabb_max = aabb[:3], aabb[3:]
            aabb_center = (aabb_min + aabb_max) / 2
            aabb_sacle_max = (aabb_max - aabb_min).max() / 2 * 1.1
            vis_mask = torch.logical_and(positions >= aabb_min, positions < aabb_max).all(-1)
        else:
            aabb_center = positions.mean(0)
            aabb_sacle_max = (positions - aabb_center).abs().max() * 1.1
            vis_mask = torch.ones_like(positions[:, 0], dtype=torch.bool)
            
        positions = ((positions[vis_mask] - aabb_center) / aabb_sacle_max).cpu().numpy()
        map_to_tensors["positions"] = o3d.core.Tensor(positions, o3d.core.float32)
        map_to_tensors["normals"] = o3d.core.Tensor(np.zeros_like(positions), o3d.core.float32)

        colors = model.colors[vis_mask].data.cpu().numpy()
        map_to_tensors["colors"] = (colors * 255).astype(np.uint8)
        for i in range(colors.shape[1]):
            map_to_tensors[f"f_dc_{i}"] = colors[:, i : i + 1]

        shs = model.shs_rest[vis_mask].data.cpu().numpy()
        if model.config.sh_degree > 0:
            shs = shs.reshape((colors.shape[0], -1, 1))
            for i in range(shs.shape[-1]):
                map_to_tensors[f"f_rest_{i}"] = shs[:, i]

        map_to_tensors["opacity"] = model.opacities[vis_mask].data.cpu().numpy()

        scales = model.scales[vis_mask].data.cpu().unsqueeze(-1).numpy()
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i]

        quats = model.quats[vis_mask].data.cpu().unsqueeze(-1).numpy()

        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i]

    pcd = o3d.t.geometry.PointCloud(map_to_tensors)
    o3d.t.io.write_point_cloud(str(filename), pcd)
    
    logger.info(f"Exported point cloud to {filename}, containing {vis_mask.sum().item()} points.")


def export_gaussians_to_3dgs_ply(storage, path, name, opacity_threshold=0.005,
                                  max_points=500000, voxel_size=0.002,
                                  normalization_center=None, normalization_scale=None):
    """Write post-activation GSRecon Gaussians in the standard 3DGS PLY format."""
    def as_numpy(value):
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

    positions = as_numpy(storage['positions']).reshape(-1, 3)
    colors = as_numpy(storage['colors']).reshape(-1, 3)
    opacity = as_numpy(storage['opacity']).reshape(-1)
    scales = as_numpy(storage['scales']).reshape(-1, 3)
    rotations = as_numpy(storage['rotation']).reshape(-1, 4)
    dynamic_score = as_numpy(storage['dynamic_score']).reshape(-1) if 'dynamic_score' in storage else None
    timestamp = as_numpy(storage['timestamp']).reshape(-1) if 'timestamp' in storage else None

    valid = opacity >= opacity_threshold
    positions, colors, opacity = positions[valid], colors[valid], opacity[valid]
    scales, rotations = scales[valid], rotations[valid]
    if dynamic_score is not None:
        dynamic_score = dynamic_score[valid]
    if timestamp is not None:
        timestamp = timestamp[valid]

    if voxel_size and len(positions):
        importance = opacity * scales.prod(axis=1)
        order = np.argsort(-importance)
        voxels = np.floor(positions[order] / voxel_size).astype(np.int64)
        _, first = np.unique(voxels, axis=0, return_index=True)
        keep = order[first]
        positions, colors, opacity = positions[keep], colors[keep], opacity[keep]
        scales, rotations = scales[keep], rotations[keep]
        if dynamic_score is not None:
            dynamic_score = dynamic_score[keep]
        if timestamp is not None:
            timestamp = timestamp[keep]

    if max_points is not None and len(positions) > max_points:
        weights = opacity * scales.prod(axis=1)
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(weights), 1 / len(weights))
        keep = np.random.choice(len(positions), max_points, replace=False, p=weights)
        positions, colors, opacity = positions[keep], colors[keep], opacity[keep]
        scales, rotations = scales[keep], rotations[keep]
        if dynamic_score is not None:
            dynamic_score = dynamic_score[keep]
        if timestamp is not None:
            timestamp = timestamp[keep]

    if not len(positions):
        return
    if normalization_center is None:
        normalization_center = positions.mean(axis=0)
        normalization_scale = np.abs(positions - normalization_center).max() * 1.1
    normalization_scale = max(float(normalization_scale), 1e-8)
    positions = (positions - normalization_center) / normalization_scale
    scales = scales / normalization_scale

    # Gaussian Splatting viewers expect pre-activation values, not RGB/alpha/scale directly.
    f_dc = (np.clip(colors, 0, 1) - 0.5) / 0.2820948
    opacity = np.clip(opacity, 1e-6, 1 - 1e-6)
    opacity = np.log(opacity / (1 - opacity)).reshape(-1, 1)
    scales = np.log(np.clip(scales, 1e-10, None))
    fields = {
        'positions': o3d.core.Tensor(positions.astype(np.float32)),
        'normals': o3d.core.Tensor(np.zeros_like(positions, dtype=np.float32)),
        'opacity': o3d.core.Tensor(opacity.astype(np.float32)),
    }
    for index in range(3):
        fields[f'f_dc_{index}'] = o3d.core.Tensor(f_dc[:, index:index + 1].astype(np.float32))
        fields[f'scale_{index}'] = o3d.core.Tensor(scales[:, index:index + 1].astype(np.float32))
    for index in range(4):
        fields[f'rot_{index}'] = o3d.core.Tensor(rotations[:, index:index + 1].astype(np.float32))
    if dynamic_score is not None:
        fields['dynamic_score'] = o3d.core.Tensor(dynamic_score.reshape(-1, 1).astype(np.float32))
    if timestamp is not None:
        fields['timestamp'] = o3d.core.Tensor(timestamp.reshape(-1, 1).astype(np.float32))
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, name)
    o3d.t.io.write_point_cloud(filename, o3d.t.geometry.PointCloud(fields))
    print(f'Exported {len(positions)} Gaussians to {filename}')

def is_enabled() -> bool:
    """
    Returns:
        True if distributed training is enabled
    """
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    """
    Returns:
        The rank of the current process within the global process group.
    """
    return dist.get_rank() if is_enabled() else 0


def get_world_size():
    return dist.get_world_size() if is_enabled() else 1


def is_main_process() -> bool:
    """
    Returns:
        True if the current process is the main one.
    """
    return get_global_rank() == 0

# Modified by ADToolbox contributors, 2026: standalone reconstruction/edit/export extensions.
"""Camera-local offsets applied to GSRecon's predicted ego trajectory."""

import json

import numpy as np


def load_ego_trajectory_offsets(path):
    with open(path, encoding='utf-8') as handle:
        payload = json.load(handle)
    if payload.get('coordinate_system') != 'camera_local_forward_left_up':
        raise ValueError('ego offsets require coordinate_system=camera_local_forward_left_up')
    offsets = payload.get('offsets')
    if not isinstance(offsets, list) or not offsets:
        raise ValueError('ego offsets require a non-empty offsets list')
    if any(not isinstance(item, list) or len(item) != 3 or any(type(value) not in (int, float) for value in item) for item in offsets):
        raise ValueError('each ego offset must be [forward, left, up] numbers')
    result = np.asarray(offsets, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('ego offsets must be finite')
    return result


def apply_ego_trajectory_offsets(original_poses, offsets):
    poses = np.asarray(original_poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError('original ego poses must be [N, 4, 4]')
    if len(offsets) not in (1, len(poses)):
        raise ValueError(f'ego offsets need one vector or {len(poses)} frame vectors, got {len(offsets)}')
    offsets = np.broadcast_to(offsets, (len(poses), 3))
    shifted = poses.copy()
    shifted[:, 0, 3] += offsets[:, 1]
    shifted[:, 1, 3] += offsets[:, 2]
    shifted[:, 2, 3] -= offsets[:, 0]
    return shifted

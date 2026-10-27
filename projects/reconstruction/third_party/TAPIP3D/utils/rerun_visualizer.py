# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

from typing import Optional, Tuple, Union

import torch
import rerun
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
from utils.common_utils import batch_unproject, batch_project, batch_unproject_uv_space
from pathlib import Path
import uuid

def setup_visualizer(
    app_name: str = "Data Visualization", 
    web_port: int = 9091,
    ws_port: int = 9878,
    server_memory_limit: str = "25%",
    open_browser=False,
    serve=True,
) -> None:
    rerun.init(app_name, spawn=False, recording_id=uuid.uuid4())
    if serve:
        rerun.serve(open_browser=open_browser, web_port=web_port, ws_port=ws_port, 
                    default_blueprint=None, recording=None, server_memory_limit=server_memory_limit)

def save_recording(
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rerun.save(str(path))

def destroy() -> None:
    rerun.disconnect()

def to_torch(*args) -> Tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in args)

def to_np(*args) -> Tuple[np.ndarray, ...]:
    return tuple(x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in args)

def log_video(
    entity_name: str,
    rgb: Union[torch.Tensor, np.ndarray],
    intrinsics: Union[torch.Tensor, np.ndarray],
    extrinsics: Union[torch.Tensor, np.ndarray],
    depth: Union[torch.Tensor, np.ndarray],
    uv_space: bool = False,
) -> None:
    rgb, intrinsics, extrinsics, depth = to_np(rgb, intrinsics, extrinsics, depth)
    depth_torch, intrinsics_torch, extrinsics_torch = to_torch(depth, intrinsics, extrinsics)
    
    if uv_space:
        pcds_torch = batch_unproject_uv_space(depth_torch, intrinsics_torch)
    else:
        pcds_torch = batch_unproject(depth_torch, intrinsics_torch, extrinsics_torch)
    pcd = np.ascontiguousarray(pcds_torch.numpy())

    num_frames, _channels, height, width = rgb.shape
    assert _channels == 3, "RGB should have 3 channels"
    
    world_from_cam0 = np.linalg.inv(extrinsics[0])
    translation0 = world_from_cam0[:3, 3]
    rotation0 = R.from_matrix(world_from_cam0[:3, :3])
    rerun.log(f"/", rerun.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rerun.log(f"/{entity_name}", rerun.Transform3D(translation=translation0, rotation=rerun.Quaternion(xyzw=rotation0.as_quat())), static=True)
    for i in range(num_frames):
        rerun.set_time_sequence("frameid", i)
        rerun.log(f"/{entity_name}/world/pcd", rerun.Points3D(pcd[i].transpose(1, 2, 0).reshape(-1, 3), colors=rgb[i].transpose(1, 2, 0).reshape(-1, 3)))
        world_from_cam = np.linalg.inv(extrinsics[i])
        translation = world_from_cam[:3, 3]
        rotation = R.from_matrix(world_from_cam[:3, :3])
        rerun.log(
            f"/{entity_name}/world/camera",
            rerun.Transform3D(translation=translation, rotation=rerun.Quaternion(xyzw=rotation.as_quat())),
        )
        rerun.log(
            f"/{entity_name}/world/camera/image",
            rerun.Pinhole(image_from_camera=intrinsics[i], resolution=[width, height], camera_xyz=rerun.ViewCoordinates.RDF),
        )
        rerun.log(
            f"/{entity_name}/world/camera/image",
            rerun.Image(rgb[i].transpose(1, 2, 0)),
        )

def log_trajectory(
    entity_name: str,
    track_name: str,
    intrinsics: Union[torch.Tensor, np.ndarray],
    extrinsics: Union[torch.Tensor, np.ndarray],
    trajs: Union[torch.Tensor, np.ndarray],
    visibs: Union[torch.Tensor, np.ndarray],
    valids: Union[torch.Tensor, np.ndarray],
    queries: Optional[Union[torch.Tensor, np.ndarray]] = None,
    track_len: int = 8,
    cmap_name: str = "rainbow",
) -> None:
    num_frames, num_points, _ = trajs.shape

    intrinsics_torch, extrinsics_torch, trajs_torch = to_torch(intrinsics, extrinsics, trajs)
    trajs_2d_torch = batch_project(
        trajs_torch.reshape(num_frames * num_points, 3),
        torch.repeat_interleave(intrinsics_torch, num_points, dim=0),
        torch.repeat_interleave(extrinsics_torch, num_points, dim=0)
    ).reshape(num_frames, num_points, 2)
    trajs_2d = trajs_2d_torch.cpu().numpy()
    intrinsics, extrinsics, trajs, visibs, valids = to_np(intrinsics, extrinsics, trajs, visibs, valids)

    if queries is not None:
        queries = to_np(queries)[0]

    assert visibs.dtype == np.bool_, "Visibility should be boolean"
    assert valids.dtype == np.bool_, "Validity should be boolean"

    cmap = matplotlib.colormaps[cmap_name]
    norm = matplotlib.colors.Normalize() # type: ignore

    colors = cmap(norm(trajs[0, :, 1]))

    for i in range(num_frames):
        rerun.set_time_sequence("frameid", i)

        colors_rgba = colors.copy() # alpha may not work..
        colors_rgba[~valids[i], -1] = 0.0
        # colors_rgba[~visibs[i], -1] *= 0.2
        colors_rgba[~visibs[i], :3] = colors_rgba[~visibs[i], :3] * 0.4 + 0.6 * np.array([0, 0, 0])

        rerun.log(f"/{entity_name}/world/{track_name}/points_3d", rerun.Points3D(trajs[i][valids[i]], colors=colors_rgba[valids[i]], radii=0.005))
        rerun.log(f"/{entity_name}/world/camera/image/{track_name}/points_2d", rerun.Points2D(trajs_2d[i][valids[i]], colors=colors_rgba[valids[i]], radii=1))

        if queries is not None:
            queries_at_i = queries[queries[..., 0] == i][..., 1:]
            rerun.log(f"/{entity_name}/world/{track_name}/queries_3d", rerun.Points3D(queries_at_i, colors=np.array([255, 255, 255], dtype=np.uint8), radii=0.01))

        if i >= 1:
            visible_track_mask = valids[i] & visibs[i] & valids[i-1] & visibs[i-1]
            valid_track_mask = valids[i] & valids[i-1]
            colors_rgba = colors_rgba.copy()
            colors_rgba[:, -1] = 0.5
            rerun.log(
                f"/{entity_name}/world/camera/image/{track_name}/tracks_2d_{i}",
                rerun.LineStrips2D(
                    trajs_2d[i-1:i+1, valid_track_mask].transpose(1, 0, 2),
                    colors=colors_rgba[valid_track_mask]
                )
            )
            rerun.log(
                f"/{entity_name}/world/{track_name}/tracks_3d_{i}",
                rerun.LineStrips3D(
                    trajs[i-1:i+1, valid_track_mask].transpose(1, 0, 2),
                    colors=colors_rgba[valid_track_mask]
                )
            )

        if i >= track_len and track_len > 0:
            rerun.log(
                f"/{entity_name}/world/camera/image/{track_name}/tracks_2d_{i-track_len}",
                rerun.Clear(recursive=True)
            )
            rerun.log(
                f"/{entity_name}/world/{track_name}/tracks_3d_{i-track_len}",
                rerun.Clear(recursive=True)
            )

from typing import Tuple
import torch
from pathlib import Path
from third_party.cotracker.model_utils import get_points_on_a_grid
import third_party.TAPIP3D.models as models
import av
import cv2
import numpy as np
from einops import repeat, rearrange

def get_grid_queries(grid_size: int, depths: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor):
    if len (depths.shape) == 3:
        return get_grid_queries(
            grid_size=grid_size,
            depths=depths.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            extrinsics=extrinsics.unsqueeze(0)
        ).squeeze(0)

    image_size = depths.shape[-2:]
    xy = get_points_on_a_grid(grid_size, image_size).to(intrinsics.device) # type: ignore
    ji = torch.round(xy).to(torch.int32)
    d = depths[:, 0][torch.arange(depths.shape[0])[:, None], ji[..., 1], ji[..., 0]]

    assert d.shape[0] == 1, "batch size must be 1"
    mask = d[0] > 0
    d = d[:, mask]
    xy = xy[:, mask]
    ji = ji[:, mask]

    inv_intrinsics0 = torch.linalg.inv(intrinsics[0, 0])
    inv_extrinsics0 = torch.linalg.inv(extrinsics[0, 0])

    xy_homo = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)
    xy_homo = torch.einsum('ij,bnj->bni', inv_intrinsics0, xy_homo)
    local_coords = xy_homo * d[..., None]
    local_coords_homo = torch.cat([local_coords, torch.ones_like(local_coords[..., :1])], dim=-1)
    world_coords = torch.einsum('ij,bnj->bni', inv_extrinsics0, local_coords_homo)
    world_coords = world_coords[..., :3]

    queries = torch.cat([torch.zeros_like(xy[:, :, :1]), world_coords], dim=-1).to(depths.device)  # type: ignore
    return queries

@torch.inference_mode()
def _inference_with_grid(
    *,
    model: torch.nn.Module,
    video: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    query_point: torch.Tensor,
    num_iters: int = 6,
    grid_size: int = 8,
    **kwargs,
):
    if grid_size != 0:
        additional_queries = get_grid_queries(grid_size, depths=depths, intrinsics=intrinsics, extrinsics=extrinsics)
        query_point = torch.cat([query_point, additional_queries], dim=1)
        N_supports = additional_queries.shape[1]
    else:
        N_supports = 0

    preds, train_data_list = model(
        rgb_obs=video,
        depth_obs=depths,
        num_iters=num_iters,
        query_point=query_point,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        mode="inference",
        **kwargs
    )
    N_total = preds.coords.shape[2]
    preds = preds.query_slice(slice(0, N_total - N_supports))
    return preds, train_data_list

def load_model(checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model, cfg = models.from_pretrained(checkpoint_path)
    if hasattr(model, "eval_mode"):
        model.set_eval_mode("raw")
    model.eval()

    return model

def read_video(video_path: str) -> np.ndarray:
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames)

def resize_depth_bilinear(depth: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    is_valid = (depth > 0).astype(np.float32)
    depth_resized = cv2.resize(depth, new_shape, interpolation=cv2.INTER_LINEAR)
    is_valid_resized = cv2.resize(is_valid, new_shape, interpolation=cv2.INTER_LINEAR)
    depth_resized = depth_resized / (is_valid_resized + 1e-6)
    depth_resized[is_valid_resized <= 1e-6] = 0.0
    return depth_resized

@torch.no_grad()
def inference(
    *,
    model: torch.nn.Module,
    video: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    query_point: torch.Tensor,
    num_iters: int = 6,
    grid_size: int = 8,
    bidrectional: bool = True,
    vis_threshold = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _depths = depths.clone()
    _depths = _depths[_depths > 0].reshape(-1)
    q25 = torch.kthvalue(_depths, int(0.25 * len(_depths))).values
    q75 = torch.kthvalue(_depths, int(0.75 * len(_depths))).values
    iqr = q75 - q25
    _depth_roi = torch.tensor(
        [1e-7, (q75 + 1.5 * iqr).item()], 
        dtype=torch.float32, 
        device=video.device
    )

    T, C, H, W = video.shape
    assert depths.shape == (T, H, W)
    N = query_point.shape[0]

    model.set_image_size((H, W))

    preds, _ = _inference_with_grid(
        model=model,
        video=video[None],
        depths=depths[None],
        intrinsics=intrinsics[None],
        extrinsics=extrinsics[None],
        query_point=query_point[None],
        num_iters=num_iters,
        depth_roi=_depth_roi,
        grid_size=grid_size
    )

    if bidrectional and not model.bidirectional and (query_point[..., 0] > 0).any():
        preds_backward, _ = _inference_with_grid(
            model=model,
            video=video[None].flip(dims=(1,)),
            depths=depths[None].flip(dims=(1,)),
            intrinsics=intrinsics[None].flip(dims=(1,)),
            extrinsics=extrinsics[None].flip(dims=(1,)),
            query_point=torch.cat([T - 1 - query_point[..., :1], query_point[..., 1:]], dim=-1)[None],
            num_iters=num_iters,
            depth_roi=_depth_roi,
            grid_size=grid_size,
        )
        preds.coords = torch.where(
            repeat(torch.arange(T, device=video.device), 't -> b t n 3', b=1, n=N) < repeat(query_point[..., 0][None], 'b n -> b t n 3', t=T, n=N),
            preds_backward.coords.flip(dims=(1,)),
            preds.coords
        )
        preds.visibs = torch.where(
            repeat(torch.arange(T, device=video.device), 't -> b t n', b=1, n=N) < repeat(query_point[..., 0][None], 'b n -> b t n', t=T, n=N),
            preds_backward.visibs.flip(dims=(1,)),
            preds.visibs
        )

    coords, visib_logits = preds.coords, preds.visibs
    visibs = torch.sigmoid(visib_logits) >= vis_threshold
    return coords.squeeze(), visibs.squeeze()
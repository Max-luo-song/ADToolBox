# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple, Union
from box import Box
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import logging

import numpy as np
import os
import weakref
from third_party.TAPIP3D.datasets.datatypes import SliceData
from third_party.TAPIP3D.training.datatypes import Prediction, TrainData
from third_party.TAPIP3D.utils.common_utils import batch_project, batch_unproject, ensure_float32, cast_float32, apply_homo_transform
from .utils.cotracker_utils import posenc, get_1d_sincos_pos_embed_from_grid

from .corr_features import load_corr_processor
from .point_updaters import load_point_updater
from .encoders import load_encoder
from functools import partial

from torch.profiler import record_function

logger = logging.getLogger(__name__)

# https://github.com/pytorch/pytorch/issues/61474
def nanmin(tensor, dim=None, keepdim=False):
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)
    return output

def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output

def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output

def smart_tensor_op(fn: Callable[[torch.Tensor], torch.Tensor], x: Any):
    if isinstance(x, torch.Tensor):
        return fn(x)
    elif isinstance(x, list):
        return [smart_tensor_op(fn, x_i) for x_i in x]
    else:
        raise ValueError(f"Unsupported type: {type(x)}")

def _dummy_like(x: torch.Tensor):
    logger.warning("Debugging: Using dummy tensor")
    ret = torch.zeros_like(x)
    if x.requires_grad:
        ret.requires_grad = True
    return ret

class PointTracker3D(nn.Module):
    bidirectional = False
    EPS = 1e-6

    def __init__(
        self,
        image_size: Tuple[int, int],
        seq_len: int,
        encoder: DictConfig,
        corr_feature: DictConfig,
        point_updater: DictConfig,
        corr_levels: int = 4,
        norm_mode: str = "none",
        norm_scale: float = 1.0,
        use_local_pos_input: bool = False,
        use_uv_input: bool = False,
        find_knn_in_normalized_space: bool = False,
        eval_mode: Literal["raw", "local", "align_first"] = "local",
        scale_aug: Optional[Tuple[float, float]] = None,
        center_rgb: bool = False,
        relative: bool = False,
    ):
        super().__init__()
        assert relative == False, "relative is left for compatibility. It should not be used."
        assert norm_mode.lower() in ["none", "anisotropy", "isotropic"], "norm_mode must be one of 'none', 'anisotropy', 'isotropic'"
        logger.info(f"Initializing PointTracker3D with encoder: {encoder.name}")

        self.image_size = image_size
        self.corr_levels = corr_levels
        self.encoder = load_encoder(config=encoder, resolution=image_size)
        self.corr_processor = load_corr_processor(corr_feature, feat_dim=self.encoder.embedding_dim, corr_levels=corr_levels, image_size=image_size, use_local_pos_input=use_local_pos_input)
        self.point_updater = load_point_updater(point_updater)
        self.norm_mode = norm_mode.lower()
        self.norm_scale = norm_scale
        self.use_local_pos_input = use_local_pos_input
        self.use_uv_input = use_uv_input
        self.find_knn_in_normalized_space = find_knn_in_normalized_space
        self.eval_mode = eval_mode
        self.scale_aug = scale_aug
        self.center_rgb = center_rgb

        self.seq_len = seq_len
        time_grid = torch.linspace(0, seq_len - 1, seq_len).reshape(1, seq_len, 1)
        self.register_buffer(
            "time_emb", get_1d_sincos_pos_embed_from_grid(self.point_updater.input_dim, time_grid[0])
        )
        logger.info(f"Image Feature Dim: {self.encoder.embedding_dim}")

    def set_image_size(self, image_size: Tuple[int, int]):
        self.image_size = image_size
        self.corr_processor.set_image_size(image_size)
        self.encoder.set_image_size(image_size)

    # Copied from: https://github.com/facebookresearch/co-tracker/blob/b00a83b66a2e72c7136bdc91e258988f42ebf6bc/cotracker/models/core/cotracker/cotracker3_online.py#L145
    def interpolate_time_embed(self, x, t):
        previous_dtype = x.dtype
        T = self.time_emb.shape[1]

        if t == T:
            return self.time_emb

        time_emb = self.time_emb.float()
        time_emb = F.interpolate(
            time_emb.permute(0, 2, 1), size=t, mode="linear"
        ).permute(0, 2, 1)
        return time_emb.to(previous_dtype)

    @ensure_float32(allow_cast=False)
    def _project(
        self, 
        points: torch.Tensor, 
        intrinsics: torch.Tensor, 
        extrinsics: torch.Tensor,
        mean_coords: Optional[torch.Tensor] = None,
        std_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        Project 3D points to 2D pixel coordinates
        One should not assume this transform is invertible.

        Args:
            points: (B, N, 4), (t, x, y, z)
            intrinsics: (B, T, 3, 3)
            extrinsics: (B, T, 4, 4)
        Returns:
            coords_2d: (B, N, 3), the last dimension is (x, y, d). Note that x corresponds to width.
        '''
        
        B, N, _ = points.shape
        time_index = points[..., 0].long() # B, N
        coords = points[..., 1:]
        batch_index = torch.arange(B, device=points.device)[:, None].expand(-1, N)
        intrinsics = intrinsics[batch_index, time_index]  # B, N, 3, 3
        extrinsics = extrinsics[batch_index, time_index]  # B, N, 4, 4

        if self.norm_mode in ["anisotropy", "isotropic"]:
            coords = (coords / self.norm_scale * std_coords[:, None, :]) + mean_coords[:, None, :] # type: ignore
        elif self.norm_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

        coords_local = apply_homo_transform(coords, extrinsics)
        coords_pixel = torch.einsum("bnij,bnj->bni", intrinsics, coords_local)
        coords_pixel = coords_pixel[..., :2] / torch.clamp(coords_pixel[..., 2:3], min=self.EPS)
        
        clamped_x = torch.clamp(coords_pixel[..., 0], min=-self.image_size[1] * 2, max=self.image_size[1] * 2)
        clamped_y = torch.clamp(coords_pixel[..., 1], min=-self.image_size[0] * 2, max=self.image_size[0] * 2)
        
        result = torch.cat([clamped_x[..., None], clamped_y[..., None], coords_local[..., 2:]], dim=-1)
        return result

    def _forward_window_iter(
        self,
        coords: torch.Tensor,
        visibs: torch.Tensor,
        projector: Callable[[torch.Tensor], torch.Tensor],
        corr_ctx: Box,
        track_mask: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        B, T = coords.shape[:2]
        N = coords.shape[2]
        
        # Fetch correlation features
        with record_function("corr_processor_forward"):
            corr_embs = self.corr_processor(
                ctx=corr_ctx,
                curr_coords=coords,
            )

        # Update coordinates
        updater_input: Any = [visibs[..., None], corr_embs]

        rel_coords_forward = coords[:, :-1] - coords[:, 1:]
        rel_coords_backward = coords[:, 1:] - coords[:, :-1]
        rel_coords_forward = torch.nn.functional.pad(
            rel_coords_forward, (0, 0, 0, 0, 0, 1)
        )
        rel_coords_backward = torch.nn.functional.pad(
            rel_coords_backward, (0, 0, 0, 0, 1, 0)
        )
        rel_pos_emb_input = posenc(
            torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
            min_deg=-2,
            max_deg=14,
        )
        updater_input.append(rel_pos_emb_input)

        if self.use_local_pos_input or self.use_uv_input:
            coords_with_time = torch.cat([
                repeat(torch.arange(T, device=coords.device, dtype=coords.dtype), "t -> b t n 1", b=B, n=N),
                coords
            ], dim=-1)
            pixel_coords = rearrange(projector(rearrange(coords_with_time, "b t n c -> b (t n) c")), "b (t n) c -> b t n c", t=T)[..., :2]
            normalized_pixel_coords = pixel_coords / torch.tensor([self.image_size[1] - 1, self.image_size[0] - 1], device=pixel_coords.device)
            pixel_pos_emb_input = posenc(normalized_pixel_coords, min_deg=-2, max_deg=14)
            updater_input.append(pixel_pos_emb_input)

        updater_input = torch.cat(updater_input, dim=-1) # (B, T, N, D)

        assert self.point_updater.input_dim == updater_input.shape[-1], "Input dimension of point_updater must match the last dimension of updater_input." \
                                                                        f"{self.point_updater.input_dim} != {updater_input.shape[-1]}"
        updater_input = updater_input + self.interpolate_time_embed(updater_input, T)[:, :, None, :]
        
        if "_SVD" in self.point_updater.__class__.__name__:
            assert track_mask.all(), "Track mask must be all True for SVD updater"
            out = self.point_updater(coords, updater_input, mask=None)
            internal_deltas = []
        elif "_Refine" in self.point_updater.__class__.__name__:
            assert track_mask.all(), "Track mask must be all True for refine updater"
            out, internal_deltas = self.point_updater(coords, updater_input, mask=None)
        else:
            out = self.point_updater(updater_input, mask=repeat(track_mask, "b n -> (b t) n", t=T))
            internal_deltas = []
        delta_coords = out[..., :3]
        delta_visibs = out[..., 3]

        internal_coords = [internal_delta + coords for internal_delta in internal_deltas]

        visibs = visibs + delta_visibs
        coords = coords + delta_coords
        assert coords.dtype == torch.float32 and all(internal_coord.dtype == torch.float32 for internal_coord in internal_coords), \
            "coords should be float32. we could use bfloat16 for delta, but not for coords"
        return internal_coords + [coords], visibs

    def _forward_window(
        self,
        *,
        coords_init: torch.Tensor,
        visibs_init: torch.Tensor,
        num_iters: int,
        corr_ctx: Any,
        projector: Callable[[torch.Tensor], torch.Tensor],
        track_mask: torch.Tensor,
        check_ref: bool = False,
    ) -> Generator[Union[TrainData, Prediction], None, None]:
        """
        This function performs the actual forward pass. 
        All the coordinates in this function are in the normalized space.
        """
        B, T = coords_init.shape[:2]
        N = coords_init.shape[2]
        
        # Mostly following cotracker3's approach
        coords = coords_init.clone()
        visibs = visibs_init.clone()

        for it in range(num_iters):
            weak_coords = weakref.ref(coords)
            weak_visibs = weakref.ref(visibs)
            
            coords = coords.detach().clone()
            visibs = visibs.detach().clone()
            
            if check_ref:
                assert weak_coords() is None and weak_visibs() is None, "coords or visibs is not immediately deleted. there might be a bug in the code"
            
            coords_list, visibs = self._forward_window_iter(
                coords=coords,
                visibs=visibs,
                corr_ctx=corr_ctx,
                projector=projector,
                track_mask=track_mask,
            )
            coords = coords_list[-1]

            yield TrainData(
                coords=coords_list,
                visibs=visibs,
                iter_idx=it,
                frame_range=(0, T),
            )
            
            del coords_list
            
        yield Prediction(
            coords=coords.detach().clone(),
            visibs=visibs.detach().clone(),
        )

    def _wrapped_forward_window(        
        self,
        *,
        feats: torch.Tensor,
        depths: torch.Tensor,
        num_iters: int,
        queries: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        window_start: int,
        window_end: int,
        coords_init: torch.Tensor,
        visibs_init: torch.Tensor,
        track_mask: torch.Tensor,
        mode: Literal["training", "inference"],
        shared_corr_ctx: Any,
        check_ref: bool = False,
        depth_roi: Optional[Tuple[float, float]] = None,
    ) -> Generator[Union[TrainData, Prediction], None, None]:
        """
        Same as _forward_window, but in the original coordinate space
        """
        B, T = feats.shape[:2]
        N = queries.shape[1]

        # To avoid confusion, move all variables depending on the coordinate system to a separate namespace
        original_ctx = Box(
            depths=depths,
            queries=queries,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            coords_init=coords_init,
        )
        del depths, queries, intrinsics, extrinsics, coords_init

        normalized_ctx = Box()

        original_ctx.query_coords = original_ctx.queries[..., 1:] # (B, N, 3)
        original_ctx.query_coords_homo = torch.cat(
            [original_ctx.query_coords, original_ctx.query_coords.new_ones(B, N, 1)], dim=-1
        ) # (B, N, 4)

        query_frames = original_ctx.queries[..., 0].long()
        original_ctx.extrinsics_at_query_frames = original_ctx.extrinsics[
            repeat(torch.arange(B), 'b -> b n', n=N),
            query_frames,
        ] # B, N, 4, 4

        with torch.autocast(device_type="cuda", enabled=False):

            original_ctx.inv_extrinsics = torch.linalg.inv(original_ctx.extrinsics)
            original_ctx.camera_locs = original_ctx.inv_extrinsics[..., :3, 3]

            # local -> normalized
            original_ctx.pcds = batch_unproject(original_ctx.depths, original_ctx.intrinsics, original_ctx.extrinsics)
            if self.norm_mode != "none":
                _original_pcds = original_ctx.pcds[:, window_start:window_end].clone()
                _original_pcds[(original_ctx.depths[:, window_start:window_end] == 0)[:, :, None, :, :].expand(-1, -1, 3, -1, -1)] = torch.nan
                if depth_roi is not None:
                    assert B == 1
                    depth_roi = depth_roi.reshape(2)
                    _original_pcds.masked_fill_(original_ctx.depths[:, window_start:window_end, None] > depth_roi[1], torch.nan)
                    _original_pcds.masked_fill_(original_ctx.depths[:, window_start:window_end, None] < depth_roi[0], torch.nan)
                if self.norm_mode == "anisotropy":
                    original_ctx.mean_coords = torch.nanmean(_original_pcds, dim=(1, 3, 4), keepdim=True) # (B, T, C, H, W) -> (B, 1, C, 1, 1)
                    original_ctx.std_coords = nanstd(_original_pcds, dim=(1, 3, 4), keepdim=True) # (B, T, C, H, W) -> (B, 1, C, 1, 1)
                elif self.norm_mode == "isotropic":
                    original_ctx.mean_coords = torch.nanmean(_original_pcds, dim=(1, 3, 4), keepdim=True) # (B, T, C, H, W) -> (B, 1, C, 1, 1)
                    original_ctx.std_coords = nanstd(_original_pcds - original_ctx.mean_coords, dim=(1, 2, 3, 4), keepdim=True).expand(-1, -1, 3, -1, -1) # (B, T, C, H, W) -> (B, 1, 3, 1, 1)
                else:
                    raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

                if self.scale_aug is not None and mode == "training":
                    assert self.training, "scale_aug is only supported in training mode"
                    scale_factor = np.random.uniform(self.scale_aug[0], self.scale_aug[1])
                    original_ctx.std_coords = original_ctx.std_coords / scale_factor
                
                normalized_ctx.pcds = (original_ctx.pcds - original_ctx.mean_coords) / original_ctx.std_coords * self.norm_scale

                original_ctx.mean_coords = original_ctx.mean_coords.reshape(-1, 3)
                original_ctx.std_coords = original_ctx.std_coords.reshape(-1, 3)

                normalized_ctx.queries = original_ctx.queries.clone()
                normalized_ctx.queries[..., 1:] = (normalized_ctx.queries[..., 1:] - original_ctx.mean_coords[:, None, :]) / original_ctx.std_coords[:, None, :] * self.norm_scale
                normalized_ctx.coords_init = (original_ctx.coords_init - original_ctx.mean_coords[:, None, None, :]) / original_ctx.std_coords[:, None, None, :] * self.norm_scale
                normalized_ctx.camera_locs = (original_ctx.camera_locs - original_ctx.mean_coords[:, None, :]) / original_ctx.std_coords[:, None, :]

                normalized_ctx.projector = partial(
                    self._project,
                    intrinsics=original_ctx.intrinsics,
                    extrinsics=original_ctx.extrinsics,
                    mean_coords=original_ctx.mean_coords,
                    std_coords=original_ctx.std_coords,
                )
            else:
                normalized_ctx.pcds = original_ctx.pcds
                normalized_ctx.queries = original_ctx.queries
                normalized_ctx.coords_init = original_ctx.coords_init
                normalized_ctx.projector = partial(
                    self._project,
                    intrinsics=original_ctx.intrinsics,
                    extrinsics=original_ctx.extrinsics,
                )
                
            with record_function("prepare_window"):
                if "gridpool" in str (type(self.corr_processor)).lower():
                    assert B == 1, "Gridpool only supports batch size 1"
                    assert self.norm_mode == "isotropic", "Gridpool only supports isotropic normalization"
                    corr_ctx = self.corr_processor.prepare_window(  # type: ignore
                        feats=feats,
                        camera_locs=normalized_ctx.camera_locs,
                        pcds=normalized_ctx.pcds,
                        queries=normalized_ctx.queries,
                        projector=normalized_ctx.projector,
                        shared_ctx=shared_corr_ctx,
                        # this only works for batch size 1
                        normalizer=lambda x: (x - original_ctx.mean_coords[0, None, :]) / original_ctx.std_coords[0, None, :] * self.norm_scale,
                    )
                else:
                    corr_ctx = self.corr_processor.prepare_window(  # type: ignore
                        feats=feats,
                        camera_locs=normalized_ctx.camera_locs,
                        pcds=normalized_ctx.pcds,
                        queries=normalized_ctx.queries,
                        projector=normalized_ctx.projector,
                        shared_ctx=shared_corr_ctx,
                    )

            corr_ctx = corr_ctx.time_slice(start=window_start, end=window_end)

        for output in self._forward_window(
            coords_init=normalized_ctx.coords_init,
            visibs_init=visibs_init,
            num_iters=num_iters,
            corr_ctx=corr_ctx,
            projector=lambda x: normalized_ctx.projector(torch.cat([x[..., :1] + window_start, x[..., 1:]], dim=-1)),
            track_mask=track_mask,
            check_ref=check_ref,
        ):
            with torch.autocast(device_type="cuda", enabled=False):
                # first denormalize the coords
                if self.norm_mode != "none":
                    assert self.norm_mode in ["anisotropy", "isotropic"], "norm_mode must be one of 'anisotropy', 'isotropic'"
                    if isinstance(output, TrainData):
                        output.coords = [
                            (coords_ / self.norm_scale * original_ctx.std_coords[:, None, None, :]) + original_ctx.mean_coords[:, None, None, :]
                            for coords_ in output.coords
                        ]
                    else:
                        output.coords = (output.coords / self.norm_scale * original_ctx.std_coords[:, None, None, :]) + original_ctx.mean_coords[:, None, None, :]
                
                if isinstance(output, TrainData):
                    output.frame_range = (window_start, window_end)

            yield output

            if isinstance(output, TrainData):
                weak_output = weakref.ref(output) 
                del output

                if check_ref:
                    assert weak_output() is None, "output is not be immediately deleted. there might be a bug in the code"

    def encode_rgbs(
        self,
        rgb_obs: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if self.center_rgb:
            rgb_obs = rgb_obs * 2. - 1.
        if not torch.is_grad_enabled():
            if chunk_size is None:
                chunk_size = self.seq_len
            feats = []
            for i in range(0, rgb_obs.shape[1], chunk_size):
                feats.append(self.encoder(rgb_obs[:, i:i+chunk_size]))
            return torch.cat(feats, dim=1)
        else:
            return self.encoder(rgb_obs)

    def streaming_forward(
        self,
        *,
        rgb_obs: torch.Tensor,
        depth_obs: torch.Tensor,
        num_iters: int,
        query_point: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        mode: Literal["training", "inference"],
        flags: List[str] = [],
        image_feats: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        check_ref: bool = False,
        transformed: bool = False,
        depth_roi: Optional[Tuple[float, float]] = None,
    ) -> Generator[Union[TrainData, Prediction], None, None]:
        '''
        Save VRAM by destructing the computation graph after each TrainData object is yielded.
        '''
        assert mode in ["training", "inference"], "unknown mode"
        
        if mode == "training":
            assert "local" in flags, "data should be transformed to local space before sending to training"
        elif self.eval_mode != "raw" and not transformed:
            with torch.autocast(device_type="cuda", enabled=False):
                B, T = rgb_obs.shape[:2]
                N = query_point.shape[1]

                _query_coords = query_point[..., 1:]
                _query_frames = query_point[..., 0].long()

                if self.eval_mode == "local":
                    _inv_extrinsics = torch.linalg.inv(extrinsics) # (B, T, 4, 4)
                    _extrinsics_at_query_frames = extrinsics[
                        repeat(torch.arange(B), 'b -> b n', n=N),
                        _query_frames,
                    ] # B, N, 4, 4
                    _query_coords_transformed = apply_homo_transform(_query_coords, transform=_extrinsics_at_query_frames)

                    query_point = torch.cat(
                        [_query_frames[..., None].to(_query_coords_transformed.dtype), _query_coords_transformed], dim=-1
                    )
                    extrinsics = repeat(
                        torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device), 
                        "i j -> b t i j", 
                        b=B, t=T
                    )
                else:
                    raise ValueError(f"Unknown eval_mode: {self.eval_mode}")

            for output in self.streaming_forward(
                rgb_obs=rgb_obs,
                depth_obs=depth_obs,
                num_iters=num_iters,
                query_point=query_point,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                mode=mode,
                flags=flags,
                image_feats=image_feats,
                chunk_size=chunk_size,
                check_ref=check_ref,
                depth_roi=depth_roi,
                transformed=True,
            ):
                with torch.autocast(device_type="cuda", enabled=False):
                    if isinstance(output, TrainData):
                        output.coords = [apply_homo_transform(coords_, transform=_inv_extrinsics[:, output.frame_range[0]:output.frame_range[1], None, :, :]) for coords_ in output.coords]
                    else:
                        assert isinstance(output, Prediction)
                        output.coords = apply_homo_transform(output.coords, transform=_inv_extrinsics[:, :, None, :, :])
                yield output
                del output
            return

        assert self.image_size == rgb_obs.shape[-2:]
        
        pad = 0
        assert self.seq_len % 2 == 0, "Sequence length must be even for online tracking"

        if rgb_obs.shape[1] % (self.seq_len // 2) != 0:
            pad = self.seq_len // 2 - rgb_obs.shape[1] % (self.seq_len // 2)
            rgb_obs = torch.cat([rgb_obs, rgb_obs[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1)
            depth_obs = torch.cat([depth_obs, depth_obs[:, -1:].expand(-1, pad, -1, -1)], dim=1)
            intrinsics = torch.cat([intrinsics, intrinsics[:, -1:].expand(-1, pad, -1, -1)], dim=1)
            extrinsics = torch.cat([extrinsics, extrinsics[:, -1:].expand(-1, pad, -1, -1)], dim=1)
            if image_feats is not None:
                image_feats = torch.cat([image_feats, image_feats[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1)
        
        B, T = rgb_obs.shape[:2]
        N = query_point.shape[1]

        # Prepare image features
        assert rgb_obs.min() >= -1e-6 and rgb_obs.max() <= 1 + 1e-6, "RGB inputs should be in the range of [0, 1]"
        if image_feats is None:
            image_feats = self.encode_rgbs(rgb_obs, chunk_size=chunk_size)
        
        image_feats = image_feats.to(dtype=torch.float32)

        query_coords = query_point[..., 1:]
        query_frames = query_point[..., 0].long()

        pred = Prediction(
            coords=repeat(query_coords, "b n c -> b t n c", t=T).clone(),
            visibs=torch.zeros(B, T, N, device=query_point.device, dtype=query_point.dtype),
        )
        original_pcds = batch_unproject(depth_obs, intrinsics, extrinsics)

        with record_function("prepare_shared_global"):
            if "gridpool" in str (type(self.corr_processor)).lower():
                focals = torch.sqrt(intrinsics[..., 0, 0] * intrinsics[..., 1, 1])
                shared_corr_ctx = self.corr_processor.prepare_shared(
                    pcds=original_pcds,
                    feats=image_feats,
                    depths=depth_obs,
                    queries=query_point,
                    focals=focals,
                )
            else:
                shared_corr_ctx = self.corr_processor.prepare_shared(
                    pcds=original_pcds,
                    feats=image_feats,
                    queries=query_point,
                )

        for window_end in range(self.seq_len, T + 1, self.seq_len // 2):
            window_start = window_end - self.seq_len

            coords_init = pred.coords[:, window_start : window_start + self.seq_len // 2] # (B, W/2, N, 3)
            visibs_init = pred.visibs[:, window_start : window_start + self.seq_len // 2] # (B, W/2, N)
            coords_init = torch.cat([coords_init, repeat(coords_init[:, -1], "b n c -> b w n c", w=self.seq_len // 2)], dim=1)
            visibs_init = torch.cat([visibs_init, repeat(visibs_init[:, -1], "b n -> b w n", w=self.seq_len // 2)], dim=1)
            
            to_copy = query_frames < window_end - self.seq_len // 2 # (B, N)

            coords_init = torch.where(
                repeat(to_copy, "b n -> b w n c", w=self.seq_len, c=3), 
                coords_init, 
                repeat(query_coords, "b n c -> b w n c", w=self.seq_len)
            ).clone()
            visibs_init = torch.where(
                repeat(to_copy, "b n -> b w n", w=self.seq_len), 
                visibs_init, 
                torch.zeros_like(visibs_init)
            ).clone()
            
            track_mask = query_frames < window_end # (B, N)
            if B > 1:
                raise NotImplementedError("Currently the efficient implementation only works for batch size 1")
                for output in self._wrapped_forward_window(
                    feats=image_feats,
                    depths=depth_obs,
                    num_iters=num_iters,
                    queries=query_point,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    window_start=window_start,
                    window_end=window_end,
                    coords_init=coords_init,
                    visibs_init=visibs_init,
                    track_mask=track_mask,
                    shared_corr_ctx=shared_corr_ctx,
                    check_ref=check_ref,
                    depth_roi=depth_roi,
                    mode=mode
                ):
                    if isinstance(output, TrainData):
                        if output.frame_range[1] > T - pad:
                            assert output.frame_range[1] == T
                            output = output.time_slice(rel_start=0, rel_end=output.frame_range[1] - output.frame_range[0] - pad)
                        yield output
                    else:
                        assert isinstance(output, Prediction)
                        pred.coords[:, window_start:window_end] = output.coords.clone()
                        pred.visibs[:, window_start:window_end] = output.visibs.clone()

                    del output
            else:
                # B == 1
                if track_mask[0].any():
                    mask = track_mask[0]
                    if os.environ.get("SAVE_ATTNMAP"):
                        mask = mask.new_ones(mask.shape)
                    for output in self._wrapped_forward_window(
                        feats=image_feats,
                        depths=depth_obs,
                        num_iters=num_iters,
                        queries=query_point[:, mask, :],
                        intrinsics=intrinsics,
                        extrinsics=extrinsics,
                        window_start=window_start,
                        window_end=window_end,
                        coords_init=coords_init[:, :, mask],
                        visibs_init=visibs_init[:, :, mask],
                        track_mask=torch.ones_like(track_mask, dtype=torch.bool)[:, mask],
                        shared_corr_ctx=shared_corr_ctx.select_queries(mask),
                        check_ref=check_ref,
                        depth_roi=depth_roi,
                        mode=mode,
                    ):
                        if isinstance(output, TrainData):
                            coords = [coords_.new_zeros(B, self.seq_len, N, 3) for coords_ in output.coords]
                            visibs = output.visibs.new_zeros(B, self.seq_len, N) # type: ignore
                            visibs[:, :, mask] = output.visibs # type: ignore
                            for i, coords_ in enumerate(output.coords):
                                coords[i][:, :, mask] = coords_

                            output.coords = coords
                            output.visibs = visibs

                            if output.frame_range[1] > T - pad:
                                assert output.frame_range[1] == T
                                output = output.time_slice(rel_start=0, rel_end=output.frame_range[1] - output.frame_range[0] - pad)

                            del coords, visibs, coords_

                            yield output
                        else:
                            assert isinstance(output, Prediction)
                            pred.coords[:, window_start:window_end, mask] = output.coords.clone()
                            pred.visibs[:, window_start:window_end, mask] = output.visibs.clone()

                        del output
        
        pred.coords = pred.coords[:, :T - pad]
        pred.visibs = pred.visibs[:, :T - pad]
        yield pred

    # Note: DDP does support using a dataclass as a return type, but other return types like Generator may cause wrong gradients to be computed.
    @cast_float32()
    def forward(
        self,
        *,
        rgb_obs: torch.Tensor,
        depth_obs: torch.Tensor,
        num_iters: int,
        query_point: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        mode: Literal["training", "inference"],
        flags: List[str] = [],
        depth_roi: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Prediction, List[TrainData]]:
        pred = None
        train_data = []
        for output in self.streaming_forward(
            rgb_obs=rgb_obs,
            depth_obs=depth_obs,
            num_iters=num_iters,
            query_point=query_point,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            depth_roi=depth_roi,
            flags=flags,
            mode=mode,
        ):
            if isinstance(output, Prediction):
                assert pred is None, "Internal error"
                pred = output
            else:
                train_data.append(output)
        assert pred is not None, "Internal error"
        return pred, train_data

    def set_eval_mode(self, eval_mode: Literal["raw", "local"]):
        assert eval_mode in ["raw", "local"], "Unknown eval mode"
        self.eval_mode = eval_mode

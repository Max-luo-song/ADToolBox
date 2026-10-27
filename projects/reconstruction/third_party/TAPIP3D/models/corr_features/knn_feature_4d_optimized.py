# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

from dataclasses import dataclass, fields
from box import Box
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
import logging
from einops import rearrange, repeat
import torch.nn as nn

from third_party.TAPIP3D.utils.common_utils import batch_project, ensure_float32, cast_float32
from ..utils.layers import build_mlp
from ..utils.cotracker_utils import posenc, bilinear_sampler, get_support_points
from third_party.TAPIP3D.models.utils.cotracker_blocks import Attention, CrossAttnBlock, AttnBlock
logger = logging.getLogger(__name__)

import third_party.pointops2.functions.pointops as pointops

def _get_index_offset_for_knnquery(batch_indices, dtype=torch.long):
    """A helper function to get index offset from ordered
    batch indices.

    Args:
        batch_indices: a tensor of shape (B,), where each element
            denotes the batch index.
    Returns:
        a tensor of shape (max_batch_index + 1,) where each element
            denotes the cumsum of number of elements belonging to the batch
    """
    assert (batch_indices[1:] >= batch_indices[:-1]).all()
    max_batch_index = batch_indices.max()

    offset = torch.zeros(
        (max_batch_index + 1, ), dtype=dtype, device=batch_indices.device
    )
    inds, counts = torch.unique_consecutive(batch_indices, return_counts=True)
    offset[inds] = counts
    existence = offset > 0

    offset = torch.cumsum(offset, dim=-1)

    return offset, existence

@dataclass
class KNNQueryInput:
    context_coords: torch.Tensor
    query_coords: torch.Tensor
    context_batch_offsets: torch.Tensor
    query_batch_offsets: torch.Tensor

    def __post_init__(self):
        assert self.context_coords.dtype == torch.float32
        assert self.query_coords.dtype == torch.float32
        assert self.context_batch_offsets.dtype == torch.int32
        assert self.query_batch_offsets.dtype == torch.int32
        assert self.context_batch_offsets.shape[0] == self.query_batch_offsets.shape[0]
        assert len (self.context_coords.shape) == 2 and len (self.query_coords.shape) == 2
        assert len (self.context_batch_offsets.shape) == 1 and len (self.query_batch_offsets.shape) == 1
    
    def contiguous(self) -> 'KNNQueryInput':
        return KNNQueryInput(
            context_coords=self.context_coords.contiguous(),
            query_coords=self.query_coords.contiguous(),
            context_batch_offsets=self.context_batch_offsets.contiguous(),
            query_batch_offsets=self.query_batch_offsets.contiguous(),
        )
    
class NeighborTransformer(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_ratio: float):
        super().__init__()
        self.dim = dim
        self.output_token_1 = nn.Parameter(torch.randn(1, dim))
        self.output_token_2 = nn.Parameter(torch.randn(1, dim))
        self.xblock1_2 = CrossAttnBlock(dim, context_dim=dim, num_heads=num_heads, dim_head=head_dim, mlp_ratio=mlp_ratio)
        self.xblock2_1 = CrossAttnBlock(dim, context_dim=dim, num_heads=num_heads, dim_head=head_dim, mlp_ratio=mlp_ratio)
        self.aggr1 = Attention(dim, context_dim=dim, num_heads=num_heads, dim_head=head_dim)
        self.aggr2 = Attention(dim, context_dim=dim, num_heads=num_heads, dim_head=head_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert len (x.shape) == 3, "x should be of shape (B, N, D)"
        assert len (y.shape) == 3, "y should be of shape (B, N, D)"
        new_x = self.xblock1_2(x, y)
        new_y = self.xblock2_1(y, x)
        out1 = self.aggr1(repeat(self.output_token_1, 'n d -> b n d', b=x.shape[0]), context=new_x)
        out2 = self.aggr2(repeat(self.output_token_2, 'n d -> b n d', b=x.shape[0]), context=new_y)
        return out1 + out2

def smart_tensor_op(fn: Callable[[torch.Tensor], torch.Tensor], x: Any) -> Any:
    if torch.is_tensor(x):
        return fn(x)
    elif isinstance(x, list):
        return [smart_tensor_op(fn, item) for item in x]
    elif isinstance(x, tuple):
        return tuple(smart_tensor_op(fn, item) for item in x)
    elif isinstance(x, dict):
        return {k: smart_tensor_op(fn, v) for k, v in x.items()}
    else:
        return x

@dataclass
class CorrContext:
    projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    feat_pyramid: Optional[List[torch.Tensor]] = None
    pcd_pyramid: Optional[List[torch.Tensor]] = None
    depth_pyramid: Optional[List[torch.Tensor]] = None
    query_feat_pyramid: Optional[List[torch.Tensor]] = None
    query_support_offset_pyramid: Optional[List[torch.Tensor]] = None
    query_support_feat_pyramid: Optional[List[torch.Tensor]] = None
    query_support_ti_pyramid: Optional[List[torch.Tensor]] = None

    def verify_window(self) -> bool:
        for field in fields(self):
            if getattr(self, field.name) is None:
                return False
        return True

    def verify_shared(self) -> bool:
        return self.query_support_ti_pyramid is not None

    def time_slice(self, start: int, end: int) -> 'CorrContext':
        assert self.verify_window()
        return CorrContext(
            projector=lambda x: self.projector(torch.cat([x[..., :1] + start, x[..., 1:]], dim=-1)), # type: ignore
            feat_pyramid=smart_tensor_op(lambda x: x[:, start:end], self.feat_pyramid),
            pcd_pyramid=smart_tensor_op(lambda x: x[:, start:end], self.pcd_pyramid),
            depth_pyramid=smart_tensor_op(lambda x: x[:, start:end], self.depth_pyramid),
            query_feat_pyramid=self.query_feat_pyramid,
            query_support_offset_pyramid=self.query_support_offset_pyramid,
            query_support_feat_pyramid=self.query_support_feat_pyramid,
        )
    
    def select_queries(self, mask: Union[torch.Tensor, List[int], slice]) -> 'CorrContext':
        self_dict = self.__dict__.copy()
        if isinstance(mask, torch.Tensor):
            assert len(mask.shape) == 1
        if self.query_feat_pyramid is not None:
            self_dict["query_feat_pyramid"] = [x[:, mask] for x in self.query_feat_pyramid] # type: ignore
        if self.query_support_offset_pyramid is not None:
            self_dict["query_support_offset_pyramid"] = [x[:, mask] for x in self.query_support_offset_pyramid] # type: ignore
        if self.query_support_feat_pyramid is not None:
            self_dict["query_support_feat_pyramid"] = [x[:, mask] for x in self.query_support_feat_pyramid] # type: ignore
        if self.query_support_ti_pyramid is not None:
            self_dict["query_support_ti_pyramid"] = [x[:, mask] for x in self.query_support_ti_pyramid] # type: ignore
        return CorrContext(**self_dict)

    def copy(self) -> 'CorrContext':
        return CorrContext(**{k: v for k, v in self.__dict__.items() if v is not None})

class KNNCorrFeature4D_Optimized(nn.Module):
    def __init__(
        self, 
        image_size: Tuple[int, int], 
        corr_levels: int, 
        feat_dim: int, 
        k_neighbors: int,
        posenc_mlp_dim: int, 
        posenc_mlp_layers: int,
        transformer_dim: int,
        transformer_num_heads: int,
        transformer_head_dim: int,
        transformer_mlp_ratio: float,
        use_features: bool = False,
        use_absolute_pos: bool = False,
        use_local_pos_input: bool = False,
        share_weights_across_levels: bool = False,
        better_depth_downsample: bool = False,
        _clip_depth: Optional[float] = None
    ):
        super().__init__()

        self.image_size = image_size
        self.k_neighbors = k_neighbors
        self.use_local_pos_input = use_local_pos_input
        self.share_weights_across_levels = share_weights_across_levels
        self.corr_levels = corr_levels

        self.transformer_dim = transformer_dim
        self.feat_dim = feat_dim
        self.use_features = use_features
        self.use_absolute_pos = use_absolute_pos

        assert better_depth_downsample, "bilinear downsampling has been deprecated"
        self.better_depth_downsample = better_depth_downsample

        self.in_norm = nn.LayerNorm(feat_dim)

        self.posenc_mlps = nn.ModuleList([
            build_mlp(
                input_size=3 * 33 if not use_local_pos_input else 4 * 33,
                output_size=transformer_dim,
                hidden_size=posenc_mlp_dim,
                n_layers=posenc_mlp_layers,
                output_norm=nn.LayerNorm,
                last_act_layer=nn.GELU,
            )
            for _ in range(corr_levels if not share_weights_across_levels else 1)
        ])
        self.transformers = nn.ModuleList([
            NeighborTransformer(
                dim=transformer_dim,
                num_heads=transformer_num_heads,
                head_dim=transformer_head_dim,
                mlp_ratio=transformer_mlp_ratio,
            )
            for _ in range(corr_levels if not share_weights_across_levels else 1)
        ])
        self._clip_depth = _clip_depth
        if transformer_dim != feat_dim:
            self.feat_transform = nn.Linear(feat_dim, transformer_dim)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    @torch.no_grad()
    def multi_knnquery(self, k_neighbors: int, inputs: List[KNNQueryInput]) -> List[torch.Tensor]:
        context_offsets, query_offsets = [], []
        cum_context_offset, cum_query_offset = 0, 0
        
        for input in inputs:
            context_offsets.append(cum_context_offset)
            query_offsets.append(cum_query_offset)
            cum_context_offset += input.context_coords.shape[0]
            cum_query_offset += input.query_coords.shape[0]
        query_input = KNNQueryInput(
            context_coords=torch.cat([input.context_coords for input in inputs], dim=0),
            query_coords=torch.cat([input.query_coords for input in inputs], dim=0),
            context_batch_offsets=torch.cat([input.context_batch_offsets + context_offsets[i] for i, input in enumerate(inputs)], dim=0),
            query_batch_offsets=torch.cat([input.query_batch_offsets + query_offsets[i] for i, input in enumerate(inputs)], dim=0),
        )
        query_input = query_input.contiguous()
        knn_idx, knn_dist = pointops.knnquery(k_neighbors, query_input.context_coords, query_input.query_coords, query_input.context_batch_offsets, query_input.query_batch_offsets) # type: ignore
        
        knn_idxs = []
        for i in range(len(inputs)):
            knn_idxs.append(knn_idx[query_offsets[i]:query_offsets[i] + inputs[i].query_coords.shape[0]] - context_offsets[i])

        return knn_idxs

    def gather_neighbors(self, *, map_coords: torch.Tensor, map_feats: torch.Tensor, knn_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        neighbor_features = torch.gather(
            input=map_feats,       # (B*T, H*W, D)
            index=repeat(knn_idx, 'b n k -> b (n k) d', d=map_feats.shape[-1]),
            dim=1
        ) # (B*T, N*K, D)
        neighbor_coords = torch.gather(
            input=map_coords,       # (B*T, H*W, C)
            index=repeat(knn_idx, 'b n k -> b (n k) d', d=map_coords.shape[-1]),
            dim=1
        ) # (B*T, N*K, 3)
        neighbor_features = rearrange(neighbor_features, 'b (n k) c -> b n k c', n=knn_idx.shape[1])
        neighbor_coords = rearrange(neighbor_coords, 'b (n k) c -> b n k c', n=knn_idx.shape[1])
        return neighbor_features, neighbor_coords
    
    def knn_interp(self, *, query_coords: torch.Tensor, map_coords: torch.Tensor, map_feats: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        neighbor_features, neighbor_coords = self.gather_neighbors(map_coords=map_coords, map_feats=map_feats, knn_idx=knn_idx)

        dists_ab = torch.einsum('b n i, b n k i -> b n k i', query_coords, neighbor_coords).sum(dim=-1) # (B, N, K)
        dists_aa = (query_coords ** 2).sum(dim=-1, keepdim=True) # (B, N, 1)
        dists_bb = (neighbor_coords ** 2).sum(dim=-1) # (B, N, K)
        dists2 = dists_aa - 2 * dists_ab + dists_bb

        weights = dists2 / (dists2.sum(dim=-1, keepdim=True) + 1e-5)
        result = (weights[..., None] * neighbor_features).sum(dim=-2)
        return result

    # we may want to figure out a better way to downsample the pointcloud later
    def _build_pyramid(self, pcds: torch.Tensor, feats: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        B, T = pcds.shape[:2]

        # Downsample the pointcloud to the resolution of feature map
        pcds = rearrange(pcds, "b t c h w -> (b t) c h w")
        pcds_raw = pcds.clone()
        if self.better_depth_downsample:
            pcds = torch.nn.functional.interpolate(pcds, size=feats[0].shape[-2:], mode='nearest-exact')
        else:
            pcds = torch.nn.functional.interpolate(pcds, size=feats[0].shape[-2:], mode='bilinear')
        pcds = rearrange(pcds, "(b t) c h w -> b t c h w", b=B, t=T)

        pcds_pyramid, feats_pyramid = [pcds], [feats]

        for level in range(self.corr_levels - 1):
            last_pcd = rearrange(pcds_pyramid[-1], 'b t c h w -> (b t) c h w')
            if self.better_depth_downsample:
                pcds_pyramid.append(rearrange(torch.nn.functional.interpolate(pcds_raw, size=(last_pcd.shape[-2] // 2, last_pcd.shape[-1] // 2), mode='nearest-exact'), '(b t) c h w -> b t c h w', b=B, t=T))
            else:
                pcds_pyramid.append(rearrange(torch.nn.functional.avg_pool2d(last_pcd, kernel_size=2, stride=2), '(b t) c h w -> b t c h w', b=B, t=T))
            last_feat = rearrange(feats_pyramid[-1], 'b t c h w -> (b t) c h w')
            feats_pyramid.append(rearrange(torch.nn.functional.avg_pool2d(last_feat, kernel_size=2, stride=2), '(b t) c h w -> b t c h w', b=B, t=T)) # type: ignore

        return pcds_pyramid, feats_pyramid # type: ignore

    @torch.no_grad()
    def prepare_shared_support_ti_singlepass(self, ctx: CorrContext, pcd_pyramid: List[torch.Tensor], queries: torch.Tensor) -> CorrContext:
        B, T = pcd_pyramid[0].shape[:2]
        N = queries.shape[1]
        query_frames = queries[..., 0].long()
        query_indices = torch.argsort(query_frames, dim=-1)

        queries_sorted = queries[torch.arange(B)[:, None], query_indices]
        query_batch_indices = T * torch.arange(B, device=pcd_pyramid[0].device, dtype=torch.long)[:, None] + queries_sorted[..., 0].long()
        query_batch_offsets, existence = _get_index_offset_for_knnquery(query_batch_indices)
        num_exist = existence.long().sum().item()

        # filter out empty batches
        query_batch_offsets = torch.unique_consecutive(query_batch_offsets).to(torch.int32)
        if query_batch_offsets[0] == 0:
            query_batch_offsets = query_batch_offsets[1:]

        query_knn_coords = queries_sorted[..., 1:].reshape(-1, 3)
        knn_queries: List[KNNQueryInput] = []

        ctx.query_support_ti_pyramid = []
        for pcds_level in pcd_pyramid:

            context_batch_offsets = torch.arange(num_exist, device=pcd_pyramid[0].device, dtype=torch.int32) + 1
            hw = pcds_level.shape[-1] * pcds_level.shape[-2]
            context_batch_offsets = (context_batch_offsets * hw).reshape(-1)
            context_coords = rearrange(pcds_level, 'b t c h w -> (b t) (h w) c')[: existence.shape[0]][existence]
            context_coords = context_coords.reshape(-1, 3)
            assert context_coords.dtype == torch.float32
            assert query_knn_coords.dtype == torch.float32

            knn_queries.append(KNNQueryInput(
                context_coords=context_coords,
                query_coords=query_knn_coords,
                context_batch_offsets=context_batch_offsets,
                query_batch_offsets=query_batch_offsets,
            ))

        knn_idxs = self.multi_knnquery(self.k_neighbors, knn_queries)

        for i in range(len(knn_idxs)):
            knn_idx = knn_idxs[i]
            pcds_level = pcd_pyramid[i]
            hw = pcds_level.shape[-1] * pcds_level.shape[-2]
            knn_idx = knn_idx % hw
            assert hw >= self.k_neighbors
            support_ti_ = repeat(query_frames, "b n -> b n k 2", k=self.k_neighbors).to(torch.float32).contiguous()
            support_ti_[torch.arange(B)[:, None], query_indices, :, 1] = rearrange(knn_idx, "(b n) k -> b n k", b=B).to(support_ti_.dtype)
            ctx.query_support_ti_pyramid.append(support_ti_)
        return ctx
    
    def prepare_shared(self, pcds: torch.Tensor, feats: torch.Tensor, queries: torch.Tensor) -> CorrContext:
        ctx = CorrContext()
        pcd_pyramid, feat_pyramid = self._build_pyramid(pcds, feats) # type: ignore
        ctx = self.prepare_shared_support_ti_singlepass(ctx, pcd_pyramid, queries)
        assert ctx.verify_shared()
        return ctx
    
    @ensure_float32(allow_cast=False)
    def prepare_window(self, *, shared_ctx: CorrContext, feats: torch.Tensor, pcds: torch.Tensor, queries: torch.Tensor, camera_locs: torch.Tensor, projector: Callable[[torch.Tensor], torch.Tensor]) -> CorrContext:
        ctx = shared_ctx.copy()
        assert ctx.verify_shared()

        ctx.projector = projector

        B, T = feats.shape[:2]
        N = queries.shape[1]

        ctx.pcd_pyramid, ctx.feat_pyramid = self._build_pyramid(pcds, feats) # type: ignore
        assert ctx.feat_pyramid is not None and ctx.pcd_pyramid is not None and ctx.query_support_ti_pyramid is not None

        ctx.depth_pyramid = []
        ctx.query_feat_pyramid = []
        ctx.query_support_offset_pyramid = []
        ctx.query_support_feat_pyramid = []
        query_coords = queries[..., 1:] # (B, N, 3)

        query_times = queries[..., 0].long()

        query_coords_2d_raw = projector(queries)[..., :2]
        for i in range(len(ctx.feat_pyramid)):
            query_coords_2d = query_coords_2d_raw.clone()
            query_coords_2d[..., 0] *= ctx.feat_pyramid[i].shape[-1] / self.image_size[-1]
            query_coords_2d[..., 1] *= ctx.feat_pyramid[i].shape[-2] / self.image_size[-2]
            query_txy = torch.cat([
                queries[..., 0:1],
                query_coords_2d,
            ], dim=-1)

            pcds = ctx.pcd_pyramid[i] # b t c h w
            feats = ctx.feat_pyramid[i] # b t c h w

            h, w = pcds.shape[-2:]

            interpolated_feats = rearrange(bilinear_sampler(
                rearrange(feats, 'b t c h w -> b c t h w'), 
                rearrange(query_txy, 'b n c -> b n 1 1 c')
            ), 'b c n 1 1 -> b n c')
            
            projected_pcds = rearrange(ctx.projector(
                torch.cat(
                    [ 
                        repeat(torch.arange(T, device=pcds.device, dtype=pcds.dtype), 't -> b (t h w) 1', b=B, h=h, w=w), 
                        rearrange(pcds, 'b t c h w -> b (t h w) c', b=B),
                    ], 
                    dim=-1
                )
            ), 'b (t h w) c -> b t h w c', b=B, t=T, h=h, w=w)
            pcds_depth = projected_pcds[..., -1:]
            ctx.depth_pyramid.append(pcds_depth)

            # bilinear_sampler treats the coordinates as (x, y). 
            # So we need to flip the coordinates to (i, t)
            support_features = rearrange(bilinear_sampler(
                rearrange(feats, 'b t c h w -> b c t (h w)'), 
                rearrange(ctx.query_support_ti_pyramid[i].flip(-1), 'b n k c -> b (n k) 1 c'),
                mode="nearest"
            ), 'b c (n k) 1 -> b n k c', n=N)
            support_coords = rearrange(bilinear_sampler(
                rearrange(pcds, 'b t c h w -> b c t (h w)'), 
                rearrange(ctx.query_support_ti_pyramid[i].flip(-1), 'b n k c -> b (n k) 1 c'),
                mode="nearest"
            ), 'b c (n k) 1 -> b n k c', n=N)

            # add the query point itself as the first support point
            support_coords = torch.cat([
                rearrange(query_coords, 'b n c -> b n 1 c'),
                support_coords,
            ], dim=2)
            support_features = torch.cat([
                rearrange(interpolated_feats, 'b n c -> b n 1 c'),
                support_features,
            ], dim=2)

            if self.use_local_pos_input:
                # this only works when all queries are at t=0
                support_coords_ = rearrange(support_coords, 'b n k c -> b (n k) c')
                support_coords_ = torch.cat([
                    support_coords_.new_zeros(B, support_coords_.shape[1], 1),
                    support_coords_,
                ], dim=-1)
                projected_coords = ctx.projector(support_coords_)
                support_depths = rearrange(projected_coords[..., -1:], 'b (n k) 1 -> b n k 1', b=B, n=N)
                
                mean_depths = pcds_depth.mean(dim=(2, 3, 4))
                std_depths = pcds_depth.std(dim=(2, 3, 4))
                if self._clip_depth is not None:
                    std_depths = torch.clamp(std_depths, min=self._clip_depth)

                mean_depths_at_query_times = mean_depths[repeat(torch.arange(B), 'b -> b n', n=N), query_times]
                std_depths_at_query_times = std_depths[repeat(torch.arange(B), 'b -> b n', n=N), query_times]
                if self._clip_depth is not None:
                    std_depths_at_query_times = torch.clamp(std_depths_at_query_times, min=self._clip_depth)

                support_depths = (support_depths - mean_depths_at_query_times[:, :, None, None]) / std_depths_at_query_times[:, :, None, None]
                support_coords = torch.cat([support_coords, support_depths], dim=-1)

            support_offsets = support_coords - support_coords[:, :, :1]

            ctx.query_support_feat_pyramid.append(support_features)
            ctx.query_support_offset_pyramid.append(support_offsets)
            ctx.query_feat_pyramid.append(interpolated_feats)
        
        assert ctx.verify_window()
        return ctx
    
    @cast_float32()
    def forward_level(self, ctx: Box, curr_coords: torch.Tensor, level: int, curr_knn_idx: torch.Tensor) -> torch.Tensor:
        B, T, feature_dim, H, W = ctx.feat_pyramid[level].shape
        num_points = ctx.query_feat_pyramid[level].shape[-2]

        with torch.autocast("cuda", enabled=False):
            map_pcds = rearrange(ctx.pcd_pyramid[level], 'b t c h w -> (b t) (h w) c')
            map_feats = rearrange(ctx.feat_pyramid[level], 'b t c h w -> (b t) (h w) c')
            curr_coords = rearrange(curr_coords, 'b t n c -> (b t) n c')
            query_support_feats = ctx.query_support_feat_pyramid[level] # (B, num_points, k_neighbors, C)
            query_support_offsets = ctx.query_support_offset_pyramid[level] # (B, num_points, k_neighbors, 3 or 4)

            map_feats = self.in_norm(map_feats)
            query_support_feats = self.in_norm(query_support_feats)

            if self.use_local_pos_input:
                hw = map_pcds.shape[1]
                map_pcds_ = torch.cat([
                    repeat(torch.arange(T, device=map_pcds.device, dtype=map_pcds.dtype), 't -> b (t hw) 1', b=B, hw=hw), 
                    rearrange(map_pcds, '(b t) hw c -> b (t hw) c', b=B),
                ], dim=-1)
                projected_map_pcds = ctx.projector(map_pcds_)
                map_pcds_depth = projected_map_pcds[..., -1:]

                curr_coords_ = torch.cat([
                    repeat(torch.arange(T, device=curr_coords.device, dtype=curr_coords.dtype), 't -> b (t n) 1', b=B, n=num_points),
                    rearrange(curr_coords, '(b t) n c -> b (t n) c', b=B),
                ], dim=-1)
                projected_curr_coords = ctx.projector(curr_coords_)
                curr_coords_depth = projected_curr_coords[..., -1:]

                map_pcds_depth = rearrange(map_pcds_depth, 'b (t hw) 1 -> b t hw 1', b=B, t=T)
                curr_coords_depth = rearrange(curr_coords_depth, 'b (t n) 1 -> b t n 1', b=B, t=T)

                mean_map_pcds_depth = map_pcds_depth.mean(dim=-2, keepdim=True)
                std_map_pcds_depth = map_pcds_depth.std(dim=-2, keepdim=True)
                if self._clip_depth is not None:
                    std_map_pcds_depth = torch.clamp(std_map_pcds_depth, min=self._clip_depth)
                map_pcds_depth = (map_pcds_depth - mean_map_pcds_depth) / std_map_pcds_depth
                curr_coords_depth = (curr_coords_depth - mean_map_pcds_depth) / std_map_pcds_depth

                map_pcds_with_depth = torch.cat([map_pcds, rearrange(map_pcds_depth, 'b t hw 1 -> (b t) hw 1')], dim=-1)
                curr_coords_with_depth = torch.cat([curr_coords, rearrange(curr_coords_depth, 'b t n 1 -> (b t) n 1')], dim=-1)

                neighbor_features, neighbor_coords_with_depth = self.gather_neighbors(map_coords=map_pcds_with_depth, map_feats=map_feats, knn_idx=curr_knn_idx) # (B*T, num_points, k_neighbors, C)
                neighbor_offset = neighbor_coords_with_depth - rearrange(curr_coords_with_depth, 'b n c -> b n 1 c') # (B*T, num_points, k_neighbors, C)
            else:
                neighbor_features, neighbor_coords = self.gather_neighbors(map_coords=map_pcds, map_feats=map_feats, knn_idx=curr_knn_idx) # (B*T, num_points, k_neighbors, C)
                neighbor_offset = neighbor_coords - rearrange(curr_coords, 'b n c -> b n 1 c') # (B*T, num_points, k_neighbors, C)

        module_index = 0 if self.share_weights_across_levels else level
        posenc_mlp = self.posenc_mlps[module_index]
        transformer = self.transformers[module_index]

        query_support_posenc = posenc_mlp(posenc(query_support_offsets, min_deg=-2, max_deg=14))
        neighbor_posenc = posenc_mlp(posenc(neighbor_offset, min_deg=-2, max_deg=14))

        if self.transformer_dim != self.feat_dim:
            query_support_feats = self.feat_transform(query_support_feats)
            neighbor_features = self.feat_transform(neighbor_features)

        output = transformer(
            repeat(query_support_feats + query_support_posenc, 'b n k c -> (b t n) k c', t=T),
            rearrange(neighbor_features + neighbor_posenc, 'bt n k c -> (bt n) k c')
        )
        output = rearrange(output, '(b t n) 1 c -> b t n c', b=B, t=T, n=num_points) # (B, T, num_points, mlp_dim)

        if self.use_features:
            output = torch.cat([output, query_support_feats[:, None, :, 0, :].expand(-1, T, -1, -1)], dim=-1)

        if self.use_absolute_pos:
            curr_posenc = posenc(curr_coords, min_deg=-2, max_deg=14) # (B, T, num_points, posenc_dim)
            output = torch.cat([output, curr_posenc], dim=-1)

        return output

    @cast_float32()
    def forward(self, ctx: Box, curr_coords: torch.Tensor, max_points_per_chunk: Optional[int] = 384) -> torch.Tensor:
        num_points = curr_coords.shape[2]
        if not torch.is_grad_enabled() and max_points_per_chunk is not None:
            results = []
            for start in range(0, num_points, max_points_per_chunk):
                end = start + max_points_per_chunk
                sub_ctx = ctx.select_queries(slice(start, end))
                sub_coords = curr_coords[:, :, start:end]
                results.append(self.forward(sub_ctx, sub_coords, max_points_per_chunk=None))
            return torch.cat(results, dim=-2)

        curr_knn_idxs = []
        with torch.autocast("cuda", enabled=False):
            knn_inputs: List[KNNQueryInput] = []
            for i in range(self.corr_levels):
                map_pcds = rearrange(ctx.pcd_pyramid[i], 'b t c h w -> (b t) (h w) c')
                curr_coords_ = rearrange(curr_coords, 'b t n c -> (b t) n c')
                hw = map_pcds.shape[-2]
                assert hw >= self.k_neighbors

                knn_inputs.append(KNNQueryInput(
                    query_coords=curr_coords_.reshape(-1, 3),
                    context_coords=map_pcds.reshape(-1, 3),
                    query_batch_offsets=(torch.arange(curr_coords_.shape[0], device=curr_coords_.device, dtype=torch.int32) + 1) * curr_coords_.shape[1],
                    context_batch_offsets=(torch.arange(map_pcds.shape[0], device=map_pcds.device, dtype=torch.int32) + 1) * map_pcds.shape[1],
                ))

            knn_outputs = self.multi_knnquery(self.k_neighbors, knn_inputs)
            for i in range(self.corr_levels):
                hw = ctx.pcd_pyramid[i].shape[-2] * ctx.pcd_pyramid[i].shape[-1]
                curr_knn_idxs.append((knn_outputs[i] % hw).reshape(-1, num_points, self.k_neighbors).long())

        corr_embs: Any = []
        for i in range(self.corr_levels):
            corr_embs.append(
                self.forward_level(
                    ctx=ctx,
                    curr_coords=curr_coords,
                    level=i,
                    curr_knn_idx=curr_knn_idxs[i],
                ) # (B, T, N, C)
            )
        corr_embs = torch.cat(corr_embs, dim=-1)
        return corr_embs

    def set_image_size(self, image_size: Tuple[int, int]):
        self.image_size = image_size
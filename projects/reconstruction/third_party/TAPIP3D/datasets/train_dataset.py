from concurrent.futures import ThreadPoolExecutor
import os
import typing
import cv2
from einops import rearrange, repeat
from hydra import initialize_config_dir, compose
import numpy as np
import logging
import torch
import tap
import time
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Callable, Dict, Literal, Optional, Union, Tuple, List, Any, TypedDict
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch.utils.data import Dataset
from datasets.base_dataset import BaseDataset
from datasets.utils.crop_utils import CropArgs, get_crop_args
from utils.common_utils import setup_logger
from datasets.datatypes import RawSliceData, SliceData
from datasets.providers.base_provider import BaseDataProvider
from datasets.utils.random_utils import WeightedSampler, RandomMapping
import datasets.data_ops as data_ops
from utils.rerun_visualizer import setup_visualizer, log_trajectory, log_video, destroy
from line_profiler import profile

_ProviderInfo = TypedDict('_ProviderInfo', {'name': str, 'weight': Union[float, int], 'stride': int, 'config': DictConfig, 'override_anno': Optional[str]})

logger = logging.getLogger(__name__)

class TrainDataset(BaseDataset):
    def __init__(
        self,
        frames_per_sample: int,
        seed: int,
        epoch_len: int,
        resolution: Tuple[int, int],
        different_seeds_per_epoch: bool,
        traj_sampling: DictConfig,
        resize: DictConfig,
        providers_config: List[_ProviderInfo],
        pad_trajs: bool = True,
        transform: Optional[ListConfig] = None,
        num_threads: int = 16,
    ):
        self.resolution = resolution
        self.frames_per_sample = frames_per_sample
        self.traj_sampling_cfg = traj_sampling
        self.resize_cfg = resize
        self.seed = seed
        self.epoch_len = epoch_len
        if transform is not None:
            self.transform = data_ops.from_config(transform)
        else:
            self.transform = None
        self.pad_trajs = pad_trajs
        self.num_threads = num_threads
        self.providers_config = providers_config
        self.different_seeds_per_epoch = different_seeds_per_epoch
        self.data_providers: List[BaseDataProvider] = [
            BaseDataProvider.from_config(provider_info['config'], name=provider_info['name'], override_anno=provider_info.get('override_anno', None))
            for provider_info in providers_config
        ]
        
        self.random_mapping: Optional[RandomMapping] = None
        self.provider_sampler = WeightedSampler([provider_info['weight'] for provider_info in providers_config])
        self.sample_groups: List[list] = [[] for _ in range(len(providers_config))]
        self.group_samplers: List[Any] = [None for _ in range(len(providers_config))]
        self.seq_lens: List[List[int]] = []

        total_frames = 0
        total_seqs = 0
        
        for idx, provider_info in enumerate(providers_config):
            seq_lens = self.data_providers[idx].load_seq_lens()
            total_seqs += len(seq_lens)
            total_frames += sum(seq_lens)

            weights = []
            for seq_id, seq_len in enumerate(seq_lens):
                if isinstance(provider_info['stride'], ListConfig):
                    strides = list(provider_info['stride'])
                else:
                    strides = [provider_info['stride']]
                for stride in strides:
                    if seq_len < (frames_per_sample - 1) * stride:
                        logger.warning(f"Sequence {seq_id} of provider {provider_info['name']} is too short to be used for training")
                        continue
                    weights.append(seq_len - (frames_per_sample - 1) * stride)
                    self.sample_groups[idx].append((seq_id, stride))

            self.group_samplers[idx] = WeightedSampler(weights=weights)
            self.seq_lens.append(seq_lens)

        logger.info(f"Initialized dataset with {total_frames} frames and {total_seqs} sequences")
        logger.info(f"Epoch length is set to {epoch_len}")

    def set_epoch(self, epoch: int):
        if self.different_seeds_per_epoch:
            self.random_mapping = RandomMapping(f"global_seed:{self.seed}, epoch:{epoch}")
        else:
            self.random_mapping = RandomMapping(f"global_seed:{self.seed}, epoch:none")

    def _apply_homo_transform(self, coords: np.ndarray, transform: np.ndarray, homo_input: bool = False):
        if homo_input:
            coords_homo = coords
        else:
            coords_homo = np.concatenate([coords, np.ones_like(coords[..., :1])], axis=-1)
        coords_transformed_homo = np.einsum("...ij,...j->...i", transform, coords_homo)
        coords_transformed = coords_transformed_homo[..., :-1] / coords_transformed_homo[..., -1:]
        return coords_transformed

    def _postprocess_slice(self, sample_id: int, raw_slice_data: RawSliceData, rng: np.random.Generator) -> SliceData:
        if self.resize_cfg.mode == "aug_crop":
            crop_args = get_crop_args(
                target_resolution=self.resolution,
                src_resolution=(raw_slice_data.rgbs[0].shape[:2]),
                aug_crop=self.resize_cfg.aug_crop,
                rng=rng,
            )
        elif self.resize_cfg.mode == "simple_resize":
            crop_args = CropArgs(
                crop_start=(0, 0),
                crop_end=(raw_slice_data.rgbs[0].shape[:2]),
                src_resolution=(raw_slice_data.rgbs[0].shape[:2]),
                target_resolution=self.resolution,
            )
        elif self.resize_cfg.mode == "none":
            crop_args = None
            assert self.resolution == raw_slice_data.rgbs[0].shape[:2], \
                "When resize mode is none, the image passed from data provider must be of the same resolution as the dataset resolution"
        else:
            raise ValueError(f"Unknown resize mode: {self.resize_cfg.mode}")

        if crop_args is not None:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                gt_intrinsics = crop_args.update_intrinsics(raw_slice_data.gt_intrinsics)
                est_intrinsics = crop_args.update_intrinsics(raw_slice_data.est_intrinsics)
                rgbs = crop_args.crop_img(raw_slice_data.rgbs, interp_mode=cv2.INTER_LANCZOS4, executor=executor)
                gt_depths = crop_args.crop_img(raw_slice_data.gt_depths, interp_mode=cv2.INTER_NEAREST_EXACT, executor=executor)
                est_depths = crop_args.crop_img(raw_slice_data.est_depths, interp_mode=cv2.INTER_NEAREST_EXACT, executor=executor)
                if raw_slice_data.segmentation is not None:
                    segmentation = crop_args.crop_img(raw_slice_data.segmentation, interp_mode=cv2.INTER_NEAREST_EXACT, executor=executor)
                else:
                    segmentation = None
                assert rgbs[0].shape[:2] == self.resolution, "Internal Error"
        else:
            gt_intrinsics = raw_slice_data.gt_intrinsics
            est_intrinsics = raw_slice_data.est_intrinsics
            rgbs = raw_slice_data.rgbs
            gt_depths = raw_slice_data.gt_depths
            est_depths = raw_slice_data.est_depths
            segmentation = raw_slice_data.segmentation

        _num_trajs = raw_slice_data.gt_trajs_3d.shape[1]
        gt_traj_local = self._apply_homo_transform(raw_slice_data.gt_trajs_3d, repeat(raw_slice_data.gt_extrinsics, "t i j -> t n i j", n=_num_trajs))
        gt_traj_2d = self._apply_homo_transform(
            gt_traj_local, 
            repeat(gt_intrinsics, "t i j -> t n i j", n=_num_trajs), 
            homo_input=True
        )

        visibs = raw_slice_data.visibs.copy()
        visibs[gt_traj_2d[:, :, 0] > self.resolution[1] - 1] = False
        visibs[gt_traj_2d[:, :, 0] < 0] = False
        visibs[gt_traj_2d[:, :, 1] > self.resolution[0] - 1] = False
        visibs[gt_traj_2d[:, :, 1] < 0] = False

        center_coords = np.array([self.resolution[1] / 2, self.resolution[0] / 2], dtype=np.float32)
        dists = np.linalg.norm(gt_traj_2d - center_coords, axis=-1)

        valids = raw_slice_data.valids.copy()
        valids[dists > self.traj_sampling_cfg.outlier_threshold] = False

        visible_ratio = (visibs & valids).sum(axis=0) / visibs.shape[0]
        select_indices = np.where(visible_ratio >= self.traj_sampling_cfg.min_visible_ratio)[0]

        visibs = visibs[:, select_indices]
        valids = valids[:, select_indices]
        gt_traj_3d = raw_slice_data.gt_trajs_3d[:, select_indices]
        est_traj_3d = raw_slice_data.est_trajs_3d[:, select_indices]
        gt_traj_2d = gt_traj_2d[:, select_indices]
        gt_traj_local = gt_traj_local[:, select_indices]
        gt_queries = raw_slice_data.gt_query_point[select_indices] if raw_slice_data.gt_query_point is not None else None
        est_queries = raw_slice_data.est_query_point[select_indices] if raw_slice_data.est_query_point is not None else None

        visible_first_indices = visibs[0, :].nonzero()[0]
        visible_last_indices = visibs[-1, :].nonzero()[0]
        visible_middle_indices = visibs[visibs.shape[0] // 2, :].nonzero()[0]

        if self.traj_sampling_cfg.traj_mode == "visible_first":
            select_indices = visible_first_indices
        elif self.traj_sampling_cfg.traj_mode == "visible_last":
            select_indices = visible_last_indices
        elif self.traj_sampling_cfg.traj_mode == "visible_first+visible_last":
            select_indices = np.concatenate([visible_first_indices, visible_last_indices])
        elif self.traj_sampling_cfg.traj_mode == "visible_first+visible_middle":
            select_indices = np.concatenate([visible_first_indices, visible_middle_indices])
        elif self.traj_sampling_cfg.traj_mode == "visible_first+visible_middle+visible_last":
            select_indices = np.concatenate([visible_first_indices, visible_middle_indices, visible_last_indices])
        elif self.traj_sampling_cfg.traj_mode == "random":
            select_indices = np.arange(visibs.shape[1])
        else:
            raise ValueError(f"Unknown trajectory selection mode: {self.traj_sampling_cfg.traj_mode}")

        select_indices = select_indices[rng.permutation(len(select_indices))[: self.traj_sampling_cfg.num_traj]]

        gt_traj_3d = gt_traj_3d[:, select_indices]
        gt_traj_2d = gt_traj_2d[:, select_indices]
        gt_traj_local = gt_traj_local[:, select_indices]
        est_traj_3d = est_traj_3d[:, select_indices]
        visibs = visibs[:, select_indices]
        valids = valids[:, select_indices]
        gt_queries = gt_queries[select_indices] if gt_queries is not None else None
        est_queries = est_queries[select_indices] if est_queries is not None else None

        if self.traj_sampling_cfg.query_mode is not None:
            assert raw_slice_data.gt_query_point is None and raw_slice_data.est_query_point is None, "Query point must not be provided if query_mode is not None"
            if self.traj_sampling_cfg.query_mode == "random":
                query_times, gt_query_points, est_query_points = [], [], []
                for i in range(gt_traj_3d.shape[1]):
                    indices = np.where((visibs & valids)[:, i])[0]
                    assert len(indices) > 0, f"No visible points found for trajectory {i}. There must be a bug in the trajectory sampler"
                    query_time = rng.choice(indices)
                    query_times.append(query_time)
                    gt_query_points.append(gt_traj_3d[query_time, i])
                    est_query_points.append(est_traj_3d[query_time, i])

                if len(gt_query_points) > 0:
                    gt_query_point = np.stack(gt_query_points, axis=0)
                    est_query_point = np.stack(est_query_points, axis=0)
                    query_time = np.array(query_times, dtype=gt_query_point.dtype)[:, None]
                    gt_query_point = np.concatenate([query_time, gt_query_point], axis=-1)
                    est_query_point = np.concatenate([query_time, est_query_point], axis=-1)
                else:
                    gt_query_point = np.zeros((gt_traj_3d.shape[0], 0, 4))
                    est_query_point = np.zeros((est_traj_3d.shape[0], 0, 4))
            elif self.traj_sampling_cfg.query_mode == "cotracker":
                N_rand = gt_traj_3d.shape[1] // 4
                query_times, gt_query_points, est_query_points = [], [], []
                for i in range(gt_traj_3d.shape[1]):
                    indices = np.where((visibs & valids)[:, i])[0]
                    assert len(indices) > 0, f"No visible points found for trajectory {i}. There must be a bug in the trajectory sampler"
                    if i < N_rand:
                        query_time = rng.choice(indices)
                    else:
                        query_time = np.argmax((visibs & valids)[:, i])
                    query_times.append(query_time)
                    gt_query_points.append(gt_traj_3d[query_time, i])
                    est_query_points.append(est_traj_3d[query_time, i])

                if len(gt_query_points) > 0:
                    gt_query_point = np.stack(gt_query_points, axis=0)
                    est_query_point = np.stack(est_query_points, axis=0)
                    query_time = np.array(query_times, dtype=gt_query_point.dtype)[:, None]
                    gt_query_point = np.concatenate([query_time, gt_query_point], axis=-1)
                    est_query_point = np.concatenate([query_time, est_query_point], axis=-1)
                else:
                    gt_query_point = np.zeros((gt_traj_3d.shape[0], 0, 4), dtype=gt_traj_3d.dtype)
                    est_query_point = np.zeros((est_traj_3d.shape[0], 0, 4), dtype=est_traj_3d.dtype)

            elif self.traj_sampling_cfg.query_mode == "first":
                assert (visibs & valids)[0, :].all(), "First frame must be visible since we use it as query point"
                gt_query_point = np.concatenate([np.zeros_like(gt_traj_3d[0, :, :1]), gt_traj_3d[0, :]], axis=-1)
                est_query_point = np.concatenate([np.zeros_like(est_traj_3d[0, :, :1]), est_traj_3d[0, :]], axis=-1)
            else:
                raise ValueError(f"Unknown query mode: {self.traj_sampling_cfg.query_mode}")
        else:
            gt_query_point = gt_queries
            est_query_point = est_queries

        if self.traj_sampling_cfg.mask_negative_depth:
            valids[gt_traj_local[:, :, -1] <= 0] = False
        
        return SliceData(
            rgbs=rearrange(torch.tensor(rgbs.copy()), "t h w c -> t c h w").to(torch.float32) / 255.0,
            gt_depths=torch.tensor(gt_depths.copy()),
            est_depths=torch.tensor(est_depths.copy()),
            gt_trajs_3d=torch.tensor(gt_traj_3d.copy()),
            est_trajs_3d=torch.tensor(est_traj_3d.copy()),
            visibs=torch.tensor(visibs.copy()),
            valids=torch.tensor(valids.copy()),
            seq_name=raw_slice_data.seq_name,
            seq_id=torch.tensor(raw_slice_data.seq_id, dtype=torch.int32),
            segmentation=torch.tensor(segmentation.copy()) if segmentation is not None else None,
            gt_intrinsics=torch.tensor(gt_intrinsics.copy()),
            est_intrinsics=torch.tensor(est_intrinsics.copy()),
            gt_query_point=torch.tensor(gt_query_point.copy()),
            est_query_point=torch.tensor(est_query_point.copy()),
            orig_resolution=torch.tensor(raw_slice_data.orig_resolution.copy()),
            gt_extrinsics=torch.tensor(raw_slice_data.gt_extrinsics.copy()),
            est_extrinsics=torch.tensor(raw_slice_data.est_extrinsics.copy()),
            same_scale=raw_slice_data.same_scale,
            flags=raw_slice_data.flags.copy(),
            sample_id=torch.tensor(sample_id, dtype=torch.int32),
            est_depth_roi=torch.tensor(raw_slice_data.est_depth_roi, dtype=torch.float32) if raw_slice_data.est_depth_roi is not None else None,
            gt_depth_roi=torch.tensor(raw_slice_data.gt_depth_roi, dtype=torch.float32) if raw_slice_data.gt_depth_roi is not None else None,
        )

    def __len__(self) -> int:
        return self.epoch_len

    @profile
    def __getitem__(self, index: int) -> SliceData:
        assert self.random_mapping is not None, "set_epoch must be called before getting items"

    
        trial = 0
        slice_data = None

        while slice_data is None:
            provider_idx = self.provider_sampler.sample(self.random_mapping.random(f"select_provider:{index}:{trial}"))
            provider = self.data_providers[provider_idx]

            group_index = self.group_samplers[provider_idx].sample(self.random_mapping.random(f"select_group:{index}:{trial}"))
            seq_id, stride = self.sample_groups[provider_idx][group_index]

            start = self.random_mapping.randint(f"start:{index}:{trial}", 0, self.seq_lens[provider_idx][seq_id] - (self.frames_per_sample - 1) * stride)
            rng = np.random.default_rng(self.random_mapping.randint(f"rng_seed:{index}:{trial}", 0, 2**32))
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                try:
                    slice_data = provider.load_slice(
                        seq_id=seq_id,
                        start=start,
                        length=self.frames_per_sample,
                        stride=stride,
                        rng=rng,
                        executor=executor,
                    )
                except Exception as e:
                    slice_data = None
                    logger.warning(f"An error occurred in data provider: {self.providers_config[provider_idx]['name']}/{seq_id}: {e}")
                    import traceback; traceback.print_exc()

                if slice_data is not None:
                    if self.transform is not None:
                        slice_data = self.transform(slice_data, executor=executor, target_resolution=self.resolution, rng=rng)
                    slice_data = self._postprocess_slice(index, slice_data, rng=rng)
                    if slice_data.gt_trajs_3d.shape[1] == 0:
                        slice_data = None
                        logger.warning(f"No valid trajectories found in {self.providers_config[provider_idx]['name']}/{seq_id}")

                if slice_data is None:
                    logger.warning(f"An error occurred while loading a sample from {self.providers_config[provider_idx]['name']}/{seq_id}")

                trial += 1

        if self.pad_trajs and self.traj_sampling_cfg.num_traj is not None:
            if slice_data.gt_trajs_3d.shape[1] < self.traj_sampling_cfg.num_traj:
                if slice_data.gt_trajs_3d.shape[1] * 2 < self.traj_sampling_cfg.num_traj:
                    logger.warning(f"More than half of the trajectories in {self.providers_config[provider_idx]['name']}/{seq_id} are padded. Check the dataset configuration")
                slice_data.gt_trajs_3d = torch.cat([ # pad the trajectories with the last trajectory
                    slice_data.gt_trajs_3d, 
                    slice_data.gt_trajs_3d[:, -1:].repeat(1, self.traj_sampling_cfg.num_traj - slice_data.gt_trajs_3d.shape[1], 1)
                ], dim=1)
                slice_data.est_trajs_3d = torch.cat([ # pad the trajectories with the last trajectory
                    slice_data.est_trajs_3d, 
                    slice_data.est_trajs_3d[:, -1:].repeat(1, self.traj_sampling_cfg.num_traj - slice_data.est_trajs_3d.shape[1], 1)
                ], dim=1)
                slice_data.visibs = torch.cat([
                    slice_data.visibs,
                    slice_data.visibs[:, -1:].repeat(1, self.traj_sampling_cfg.num_traj - slice_data.visibs.shape[1])
                ], dim=1)
                slice_data.valids = torch.cat([
                    slice_data.valids,
                    slice_data.valids[:, -1:].repeat(1, self.traj_sampling_cfg.num_traj - slice_data.valids.shape[1])
                ], dim=1)
                slice_data.gt_query_point = torch.cat([
                    slice_data.gt_query_point,
                    slice_data.gt_query_point[-1:].repeat(self.traj_sampling_cfg.num_traj - slice_data.gt_query_point.shape[0], 1)
                ], dim=0)
                slice_data.est_query_point = torch.cat([
                    slice_data.est_query_point,
                    slice_data.est_query_point[-1:].repeat(self.traj_sampling_cfg.num_traj - slice_data.est_query_point.shape[0], 1)
                ], dim=0)

        return slice_data
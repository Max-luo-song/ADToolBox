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
import datasets.data_ops as data_ops
from line_profiler import profile

_ProviderInfo = TypedDict('_ProviderInfo', {'name': str, 'weight': Union[float, int], 'stride': int, 'config': DictConfig, 'override_anno': Optional[str]})

logger = logging.getLogger(__name__)

class EvalDataset(BaseDataset):
    def __init__(
        self,        
        resolution: Tuple[int, int],
        provider_config: _ProviderInfo,
        query_mode: Literal["first_visible", "pass_through"],
        # first_n_seqs: Optional[int] = None,
        sample_n_seqs: Optional[int] = None,
        num_threads: int = 16,
        seed: int = 42,
        transform: Optional[ListConfig] = None,
    ):
        self.resolution = resolution
        self.data_provider: BaseDataProvider = BaseDataProvider.from_config(provider_config['config'], name=provider_config['name'], override_anno=provider_config['override_anno'])

        self.seq_lens = self.data_provider.load_seq_lens()
        self.stride = provider_config['stride']
        self.query_mode = query_mode
        self.sample_n_seqs = sample_n_seqs
        self.seed = seed
        if transform is not None:
            self.transform = data_ops.from_config(transform)
        else:
            self.transform = None

        if sample_n_seqs is not None:
            rng = np.random.default_rng(seed)
            self.seq_ids = rng.choice(len(self.seq_lens), sample_n_seqs, replace=False)
        else:
            self.seq_ids = np.arange(len(self.seq_lens))

        assert query_mode in ["first_visible", "pass_through"], f"Invalid query mode: {query_mode}"

        self.num_threads = num_threads

        logger.info(f"Initialized eval dataset with {sum(self.seq_lens[seq_id] for seq_id in self.seq_ids)} frames and {len(self.seq_lens)} sequences")

    def set_epoch(self, epoch: int):
        pass

    def _apply_homo_transform(self, coords: np.ndarray, transform: np.ndarray, homo_input: bool = False):
        if homo_input:
            coords_homo = coords
        else:
            coords_homo = np.concatenate([coords, np.ones_like(coords[..., :1])], axis=-1)
        coords_transformed_homo = np.einsum("...ij,...j->...i", transform, coords_homo)
        coords_transformed = coords_transformed_homo[..., :-1] / coords_transformed_homo[..., -1:]
        return coords_transformed

    def _postprocess_slice(self, sample_id: int, raw_slice_data: RawSliceData) -> SliceData:
        crop_args = CropArgs(
            crop_start=(0, 0),
            crop_end=(raw_slice_data.rgbs[0].shape[:2]),
            src_resolution=(raw_slice_data.rgbs[0].shape[:2]),
            target_resolution=self.resolution,
        )
        # import ipdb; ipdb.set_trace()

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
            
        gt_query_point = raw_slice_data.gt_query_point
        est_query_point = raw_slice_data.est_query_point
        gt_trajs_3d = raw_slice_data.gt_trajs_3d
        est_trajs_3d = raw_slice_data.est_trajs_3d
        visibs = raw_slice_data.visibs
        valids = raw_slice_data.valids

        if self.query_mode == "first_visible":
            assert gt_query_point is None, "You should not override query point provided by the eval dataset"

            to_query = (raw_slice_data.visibs & raw_slice_data.valids).any(0)

            gt_trajs_3d = gt_trajs_3d[:, to_query] # type: ignore
            est_trajs_3d = est_trajs_3d[:, to_query] # type: ignore
            visibs = visibs[:, to_query] # type: ignore
            valids = valids[:, to_query] # type: ignore

            query_frames = (visibs & valids).argmax(0)
            gt_query_coords = gt_trajs_3d[query_frames, np.arange(gt_trajs_3d.shape[1])]
            est_query_coords = est_trajs_3d[query_frames, np.arange(est_trajs_3d.shape[1])]

            gt_query_point = np.concatenate([query_frames[..., None].astype(np.float32), gt_query_coords], axis=-1)
            est_query_point = np.concatenate([query_frames[..., None].astype(np.float32), est_query_coords], axis=-1)
        else:
            assert gt_query_point is not None, "query_point should be provided by the eval dataset when query_mode is pass_through"
            assert est_query_point is not None, "query_point should be provided by the eval dataset when query_mode is pass_through"

        # est_depths = -torch.nn.functional.max_pool2d(-torch.from_numpy(est_depths), kernel_size=5, stride=1, padding=2)
        # est_depths = est_depths.numpy()

        return SliceData(
            rgbs=rearrange(torch.from_numpy(rgbs), "t h w c -> t c h w").to(torch.float32) / 255.0,
            gt_depths=torch.from_numpy(gt_depths),
            est_depths=torch.from_numpy(est_depths),
            gt_trajs_3d=torch.from_numpy(gt_trajs_3d),
            est_trajs_3d=torch.from_numpy(est_trajs_3d),
            visibs=torch.from_numpy(visibs),
            valids=torch.from_numpy(valids),
            seq_name=raw_slice_data.seq_name,
            seq_id=torch.tensor(raw_slice_data.seq_id, dtype=torch.int32),
            gt_intrinsics=torch.from_numpy(gt_intrinsics),
            gt_extrinsics=torch.from_numpy(raw_slice_data.gt_extrinsics),
            est_intrinsics=torch.from_numpy(est_intrinsics),
            est_extrinsics=torch.from_numpy(raw_slice_data.est_extrinsics),
            gt_query_point=torch.from_numpy(gt_query_point),
            est_query_point=torch.from_numpy(est_query_point),
            orig_resolution=torch.from_numpy(raw_slice_data.orig_resolution),
            same_scale=raw_slice_data.same_scale,
            flags=raw_slice_data.flags.copy(),
            segmentation=torch.from_numpy(segmentation.copy()) if segmentation is not None else None,
            sample_id=torch.tensor(sample_id, dtype=torch.int32),
            est_depth_roi=torch.tensor(raw_slice_data.est_depth_roi, dtype=torch.float32) if raw_slice_data.est_depth_roi is not None else None,
            gt_depth_roi=torch.tensor(raw_slice_data.gt_depth_roi, dtype=torch.float32) if raw_slice_data.gt_depth_roi is not None else None,
        )

    def __len__(self) -> int:
        return len(self.seq_ids)

    @profile
    def __getitem__(self, index: int) -> SliceData:
        # index = 32
        # print ("DEBUG!!!!!!!!!!!!!!!!!\n" * 10)
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            slice_data = self.data_provider.load_slice(
                seq_id=int(self.seq_ids[index]),
                start=0,
                length=self.seq_lens[self.seq_ids[index]],
                stride=self.stride,
                rng=np.random.default_rng(self.seed + index),
                executor=executor,
            )
            if self.transform is not None:
                slice_data = self.transform(slice_data, rng=np.random.default_rng(self.seed + index + 1), executor=executor)
            slice_data = self._postprocess_slice(index, slice_data)

        return slice_data
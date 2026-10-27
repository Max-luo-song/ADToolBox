from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
import os
from typing import Any, Dict, List, Optional, Set, Tuple
from einops import rearrange
import numpy as np
from omegaconf import DictConfig

from datasets.providers.base_provider import BaseDataProvider
from datasets.datatypes import RawSliceData
import cv2
import logging

from datasets.utils.geometry import batch_distance_to_depth_np
from datasets.utils.crop_utils import get_crop_args
from line_profiler import profile

import h5py

logger = logging.getLogger(__name__)

class CustomKubricDataProvider(BaseDataProvider):
    def __init__(self, cfg: DictConfig, override_anno: Optional[str] = None):
        super().__init__(cfg, override_anno)
        self.data_root: str = cfg.data_root
        self.h5_files = [x for x in os.listdir(self.data_root) if x.endswith(".h5")]
        self.h5_files.sort()
        self.dynamic_filter_threshold: float = cfg.get("dynamic_filter_threshold", 0.0)
        self.keep_static_prob: float = cfg.get("keep_static_prob", 1.0)
        logger.info(f"Loaded Kubric data from {self.data_root} with {len(self.h5_files)} sequences")

    @profile
    def _load_slice(
        self, 
        seq_id: int, 
        start: int, 
        length: int, 
        stride: int, 
        rng: np.random.Generator,
        executor: ThreadPoolExecutor,
    ) -> RawSliceData:
        with h5py.File(os.path.join(self.data_root, self.h5_files[seq_id]), "r") as h5_file:
            depth_range: np.ndarray = h5_file['depth_range'][:].astype(np.float32) # type: ignore
            traj_3d: np.ndarray = h5_file['traj_3d'][:, start:start+length*stride:stride, :].transpose(1, 0, 2).astype(np.float32) # type: ignore
            traj_2d: np.ndarray = h5_file['target_points'][:, start:start+length*stride:stride, :].transpose(1, 0, 2).astype(np.float32) # type: ignore
            visibs: np.ndarray = ~ h5_file['occluded'][:, start:start+length*stride:stride].transpose(1, 0) # type: ignore
            rgbs: np.ndarray = np.round((h5_file['video'][start:start+length*stride:stride] + 1) / 2. * 255).astype(np.uint8) # type: ignore
            depth_16bit: np.ndarray = h5_file['depth_16bit'][start:start+length*stride:stride] # type: ignore
            distances: np.ndarray = depth_range[0] + depth_16bit.astype(np.float32) * (depth_range[1] - depth_range[0]) / 65535.
            assert distances.shape[-1] == 1
            distances = distances[..., 0]
            intrinsics: np.ndarray = h5_file['intrinsics'][start:start+length*stride:stride].astype(np.float32) # type: ignore
            extrinsics: np.ndarray = h5_file['matrix_world'][start:start+length*stride:stride].astype(np.float32) # type: ignore
            segmentation: np.ndarray = h5_file['segmentations'][start:start+length*stride:stride].astype(np.int32) # type: ignore
            assert segmentation.shape[-1] == 1
            segmentation = segmentation[..., 0]
        # There are negative values in the intrinsics matrix.. We take the absolute value
        # To correct this effect, we need to modify the extrinsics matrix
        intrinsics = np.abs(intrinsics)
        intrinsics[:, 0, :] *= rgbs[0].shape[1]
        intrinsics[:, 1, :] *= rgbs[0].shape[0]
        extrinsics = np.linalg.inv(extrinsics)
        extrinsics = np.einsum('ij,bjk->bik', np.diag(np.array([1., -1., -1., 1.], dtype=np.float32)), extrinsics)

        visibs[traj_2d[:, :, 0] > rgbs[0].shape[1] - 1] = False
        visibs[traj_2d[:, :, 0] < 0] = False
        visibs[traj_2d[:, :, 1] > rgbs[0].shape[0] - 1] = False
        visibs[traj_2d[:, :, 1] < 0] = False

        if self.dynamic_filter_threshold > 0.0 and self.keep_static_prob < 1.0:
            max_xyz = np.max(np.abs(traj_3d), axis=0)
            min_xyz = np.min(np.abs(traj_3d), axis=0)
            diff_l2 = np.linalg.norm(max_xyz - min_xyz, axis=-1)
            dynamic_filter = diff_l2 > self.dynamic_filter_threshold
            keep_mask = rng.random(traj_3d.shape[1], dtype=np.float32) < self.keep_static_prob
            dynamic_filter[keep_mask] = True
            visibs = visibs[:, dynamic_filter]
            traj_3d = traj_3d[:, dynamic_filter]

        depths = batch_distance_to_depth_np(distances, intrinsics)

        return RawSliceData.create(
            seq_name=self.h5_files[seq_id],
            seq_id=seq_id,
            gt_trajs_3d=traj_3d,
            visibs=visibs,
            valids=np.ones_like(visibs, dtype=np.bool_),
            rgbs=rgbs,
            gt_depths=depths,
            gt_query_point=None,
            gt_intrinsics=intrinsics,
            gt_extrinsics=extrinsics,
            orig_resolution=np.array([rgbs[0].shape[0], rgbs[0].shape[1]], dtype=np.int32), # (h, w)
            segmentation=segmentation,
            copy_gt_to_est=True,
        )
    
    def load_seq_lens(self) -> List[int]:
        seq_lens = []
        for f in self.h5_files:
            with h5py.File(os.path.join(self.data_root, f), "r") as h5_file:
                seq_lens.append(h5_file['intrinsics'].shape[0])
        return seq_lens

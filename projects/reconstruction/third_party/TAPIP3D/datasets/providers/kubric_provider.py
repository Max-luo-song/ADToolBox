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

logger = logging.getLogger(__name__)

class KubricDataProvider(BaseDataProvider):
    def __init__(self, cfg: DictConfig, override_anno: Optional[str] = None):
        super().__init__(cfg, override_anno)
        self.data_root: str = cfg.data_root
        self.video_folders: List[str] = [os.path.join(self.data_root, f) for f in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, f))]
        self.video_folders.sort()
        self.dynamic_filter_threshold: float = cfg.get("dynamic_filter_threshold", 0.0)
        self.keep_static_prob: float = cfg.get("keep_static_prob", 1.0)
        logger.info(f"Loaded Kubric data from {self.data_root} with {len(self.video_folders)} sequences")

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
        annot_file = os.path.join(self.data_root, os.path.basename(self.video_folders[seq_id]), f"{os.path.basename(self.video_folders[seq_id])}.npy")
        annot = np.load(annot_file, allow_pickle=True).item()

        depth_range = annot['depth_range']

        traj_2d = annot['coords'].transpose(1, 0, 2)[start:start+length*stride:stride]
        traj_3d = annot['traj_3d'].transpose(1, 0, 2)[start:start+length*stride:stride]
        visibs = ~ annot['occluded'].transpose(1, 0)[start:start+length*stride:stride]

        rgb_list: List[Future[np.ndarray]] = []
        distance_list: List[Future[np.ndarray]] = []
        for frame_id in range(start, start+length*stride, stride):
            rgb_path = os.path.join(self.data_root, os.path.basename(self.video_folders[seq_id]), "frames", f"{frame_id:03d}.png")
            depth_path = os.path.join(self.data_root, os.path.basename(self.video_folders[seq_id]), "frames", f"{frame_id:03d}_depth.png")
            rgb_list.append(executor.submit(lambda rgb_path=rgb_path: cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)))
            distance_list.append(
                executor.submit(
                    lambda depth_range=depth_range, depth_path=depth_path: 
                    depth_range[0] + cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) * (depth_range[1] - depth_range[0]) / 65535., 
                )
            )

        # The depth annotation in kubric is actually the distance from the camera
        rgbs: Any = [f.result() for f in rgb_list]
        distances = np.stack([f.result() for f in distance_list])

        # There are negative values in the intrinsics matrix.. We take the absolute value
        # To correct this effect, we need to modify the extrinsics matrix
        intrinsics = np.abs(annot['intrinsics'][start:start+length*stride:stride])
        intrinsics[:, 0, :] *= rgbs[0].shape[1]
        intrinsics[:, 1, :] *= rgbs[0].shape[0]
        extrinsics = annot['matrix_world'][start:start+length*stride:stride]
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

        rgbs = np.stack(rgbs)
        depths = batch_distance_to_depth_np(distances, intrinsics)

        return RawSliceData.create(
            seq_name=self.video_folders[seq_id],
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
            copy_gt_to_est=True,
        )
    
    def load_seq_lens(self) -> List[int]:
        seq_lens = []
        for f in self.video_folders:
            annot_file = os.path.join(self.data_root, os.path.basename(f), f"{os.path.basename(f)}.npy")
            annot = np.load(annot_file, allow_pickle=True).item()
            seq_lens.append(len(annot['intrinsics']))
        return seq_lens

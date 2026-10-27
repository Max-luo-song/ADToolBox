from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any, List, Tuple
from einops import repeat
import numpy as np
from omegaconf import DictConfig
import torch

from datasets.providers.base_provider import BaseDataProvider
from datasets.datatypes import RawSliceData, SliceData
from datasets.utils.crop_utils import get_crop_args
from PIL import Image
import cv2
import logging
logger = logging.getLogger(__name__)

class PointOdysseyDataProvider(BaseDataProvider):
    def __init__(self, cfg: DictConfig, override_anno: Optional[str] = None):
        super().__init__(cfg, override_anno)
        self.split: str = cfg.split
        self.data_root: str = os.path.join(cfg.data_root, self.split)
        self.video_dirs: List[str] = sorted([os.path.join(self.data_root, d) for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))])
        logger.info(f"Loaded PointOdyssey data from {self.data_root} with {len(self.video_dirs)} videos")
        self.valid_depth_range: Tuple[float, float] = cfg.valid_depth_range

        self.dynamic_filter_threshold: float = cfg.dynamic_filter_threshold
        self.keep_static_prob: float = cfg.keep_static_prob

    def _apply_homo_transform(self, coords: np.ndarray, transform: np.ndarray, homo_input: bool = False):
        if homo_input:
            coords_homo = coords
        else:
            coords_homo = np.concatenate([coords, np.ones_like(coords[..., :1])], axis=-1)
        coords_transformed_homo = np.einsum("...ij,...j->...i", transform, coords_homo)
        coords_transformed = coords_transformed_homo[..., :-1] / coords_transformed_homo[..., -1:]
        return coords_transformed
        
    def _load_slice(
        self, 
        seq_id: int, 
        start: int, 
        length: int, 
        stride: int, 
        rng: np.random.Generator,
        executor: ThreadPoolExecutor
    ) -> RawSliceData:
        video_dir = self.video_dirs[seq_id]
        rgb_folder = os.path.join(video_dir, 'rgbs')
        depth_folder = os.path.join(video_dir, 'depths')
        mask_folder = os.path.join(video_dir, 'masks')
        annotation_path = os.path.join(video_dir, 'anno.npz')
        
        # Load annotations
        anns = np.load(annotation_path, mmap_mode='r')
        trajs_2d: np.ndarray = anns['trajs_2d'][start:start + length * stride:stride]
        trajs_3d: Any = anns['trajs_3d'][start:start + length * stride:stride]
        visibs = anns['visibs'][start:start + length * stride:stride]
        valids = anns['valids'][start:start + length * stride:stride]

        intrinsics = anns['intrinsics'][start:start + length * stride:stride]
        extrinsics = anns['extrinsics'][start:start + length * stride:stride].reshape(-1, 4, 4)

        # Load RGB, depth, and mask frames
        rgb_obs: Any = []
        depth_obs: Any = []
        mask_obs: Any = []
        for frame_idx in range(start, start + length * stride, stride):
            rgb_path = os.path.join(rgb_folder, f'rgb_{frame_idx:05d}.jpg')
            depth_path = os.path.join(depth_folder, f'depth_{frame_idx:05d}.png')
            mask_path = os.path.join(mask_folder, f'mask_{frame_idx:05d}.png')

            assert os.path.exists(rgb_path), f"RGB image {rgb_path} does not exist"
            assert os.path.exists(depth_path), f"Depth image {depth_path} does not exist"
            assert os.path.exists(mask_path), f"Mask image {mask_path} does not exist"

            rgb_obs.append(executor.submit(lambda path: Image.open(path).convert('RGB'), rgb_path))
            depth_obs.append(
                executor.submit(
                    lambda path: cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0 * 1000.0, 
                    depth_path
                )
            )
            mask_obs.append(
                executor.submit(
                    lambda path: (
                        cv2.cvtColor(
                            cv2.imread(path, cv2.IMREAD_COLOR), 
                            cv2.COLOR_BGR2GRAY
                        ).astype(np.float32) / 255.0 > 0.001
                    ).astype(np.float32),
                    mask_path
                )
            )
        
        rgb_obs = np.stack([rgb.result() for rgb in rgb_obs]) # (t, h, w, 3)
        depth_obs = np.stack([depth.result() for depth in depth_obs]) # (t, h, w)
        mask_obs = np.stack([mask.result() for mask in mask_obs]) # (t, h, w)

        mask = mask_obs.astype(np.float32)
        mask[mask < self.valid_depth_range[0]] = 0.0
        mask[mask > self.valid_depth_range[1]] = 0.0
        depth_obs = depth_obs * mask

        local_trajs_3d = self._apply_homo_transform(trajs_3d, repeat(extrinsics, 't i j -> t n i j', n=trajs_3d.shape[1]), homo_input=False)
        valids[local_trajs_3d[:, :, -1] < self.valid_depth_range[0]] = False
        valids[local_trajs_3d[:, :, -1] > self.valid_depth_range[1]] = False

        visibs[trajs_2d[:, :, 0] > rgb_obs[0].shape[1] - 1] = False
        visibs[trajs_2d[:, :, 0] < 0] = False
        visibs[trajs_2d[:, :, 1] > rgb_obs[0].shape[0] - 1] = False
        visibs[trajs_2d[:, :, 1] < 0] = False

        assert rgb_obs.dtype == np.uint8

        if self.dynamic_filter_threshold > 0.0:
            max_xyz = np.max(np.abs(trajs_3d), axis=0)
            min_xyz = np.min(np.abs(trajs_3d), axis=0)
            diff_l2 = np.linalg.norm(max_xyz - min_xyz, axis=-1)
            dynamic_filter = diff_l2 > self.dynamic_filter_threshold
            keep_mask = rng.random(trajs_3d.shape[1], dtype=np.float32) < self.keep_static_prob
            dynamic_filter[keep_mask] = True
            trajs_2d = trajs_2d[:, dynamic_filter]
            trajs_3d = trajs_3d[:, dynamic_filter]
            visibs = visibs[:, dynamic_filter]
            valids = valids[:, dynamic_filter]

        return RawSliceData(
            rgbs=rgb_obs,
            seq_name=video_dir,
            seq_id=seq_id,
            trajs_3d=trajs_3d,
            visibs=visibs,
            valids=valids,
            depths=depth_obs,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            orig_resolution=np.array([rgb_obs[0].shape[0], rgb_obs[0].shape[1]], dtype=np.int32), # (h, w)
        )

    def load_seq_lens(self) -> List[int]:
        seq_lens = []
        for video_dir in self.video_dirs:
            info = np.load(os.path.join(video_dir, 'info.npz'))
            num_frames, num_points, _ = info['trajs_3d']
            seq_lens.append(num_frames)
        return seq_lens

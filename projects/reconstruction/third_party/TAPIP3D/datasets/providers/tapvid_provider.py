from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import os
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import tqdm
from einops import repeat, rearrange

from datasets.providers.base_provider import BaseDataProvider
from datasets.datatypes import RawSliceData, SliceData
from PIL import Image
import cv2
import pickle
import logging

from datasets.utils.dataclass_utils import load_dataclass
from dataclasses import dataclass

from datasets.utils.crop_utils import get_crop_args
from evaluation.tapvid3d_splits import MINIVAL_FILES, FULL_EVAL_FILES
import mediapy as media

logger = logging.getLogger(__name__)

def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return media.resize_video(video, output_size)

def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.
    Given a set of frames and tracks with no query points, use the first
    visible point in each track as the query.
    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3]
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1]
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1]
    """
    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)

    return {
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.

    Given a set of frames and tracks with no query points, sample queries
    strided every query_stride frames, ignoring points that are not visible
    at the selected frames.

    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
      query_stride: When sampling query points, search for un-occluded points
        every query_stride frames and convert each one into a query.

    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3].  The video
          has floats scaled to the range [-1, 1].
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1].
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1].
        trackgroup: Index of the original track that each query point was
          sampled from.  This is useful for visualization.
    """
    tracks = []
    occs = []
    queries = []
    trackgroups = []
    total = 0
    trackgroup = np.arange(target_occluded.shape[0])
    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0:1]),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
        trackgroups.append(trackgroup[mask])
        total += np.array(np.sum(target_occluded[:, i] == 0))

    return {
        "video": frames[np.newaxis, ...],
        "query_points": np.concatenate(queries, axis=0)[np.newaxis, ...],
        "target_points": np.concatenate(tracks, axis=0)[np.newaxis, ...],
        "occluded": np.concatenate(occs, axis=0)[np.newaxis, ...],
        "trackgroup": np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }

class TAPVidProvider(BaseDataProvider):
    def __init__(self, cfg: DictConfig, override_anno: Optional[str] = None):
        super().__init__(cfg, override_anno)
        self.data_path: str = cfg.data_path
        self.query_mode: str = cfg.query_mode
        assert self.query_mode in ["first", "strided"], "query_mode must be either 'first' or 'strided'"
        with open(self.data_path, "rb") as f:
            self.data = pickle.load(f)
        if isinstance(self.data, dict):
            self.keys = list(self.data.keys())
        else:
            self.keys = list(range(len(self.data)))
        self.keys.sort()
        logger.info(f"Loaded TAP-Vid: {self.data_path} with {len(self.data)} sequences")

    def _load_slice(
        self, 
        seq_id: int, 
        start: int, 
        length: int, 
        stride: int,
        rng: np.random.Generator,
        executor: ThreadPoolExecutor,
    ) -> RawSliceData:
        item = self.data[self.keys[seq_id]]

        if self.query_mode == "first":
            sample = sample_queries_first(item['occluded'], item['points'], item['video'])
        elif self.query_mode == "strided":
            sample = sample_queries_strided(item['occluded'], item['points'], item['video'])
        assert sample["target_points"].shape[1] == sample["query_points"].shape[1]
        
        occluded = rearrange(sample['occluded'][0], 'n t -> t n')
        visibs = ~ occluded
        video = sample['video'][0]
        T, H, W, C = video.shape

        query_frames = sample['query_points'][0][..., 0]
        query_2d = sample['query_points'][0][..., 1:][..., ::-1] * np.array([W - 1, H - 1], dtype=np.float32)
        trajs_2d = rearrange(sample['target_points'][0], 'n t c -> t n c') * np.array([W - 1, H - 1], dtype=np.float32)
        
        dummy_depths = np.ones_like(video[..., 0], dtype=np.float32)

        dummy_intrs = np.array([[W, 0.0, W//2],
                                [0.0, W, H//2],
                                [0.0, 0.0, 1.0]], 
                                dtype=np.float32
                              )[None].repeat(T, axis=0)
        dummy_extrs = np.eye(4, dtype=np.float32)[None].repeat(T, axis=0)

        dummy_invintr = np.linalg.inv(dummy_intrs[0])
        trajs_2d_homo = np.concatenate([trajs_2d, np.ones_like(trajs_2d[:, :, 0:1])], axis=-1)
        query_2d_homo = np.concatenate([query_2d, np.ones_like(query_2d[:, 0:1])], axis=-1)
        trajs_3d = np.einsum('ij, tnj -> tni', dummy_invintr, trajs_2d_homo) 
        queries_3d = np.einsum('ij, nj -> ni', dummy_invintr, query_2d_homo)
        queries = np.concatenate([query_frames[..., None], queries_3d], axis=-1)

        valids = visibs.copy()
        if self.query_mode == "first":
            # Following the evaluation protocol of TAP-Vid
            mask = np.arange(T)[..., None] >= query_frames[None]
            valids = np.logical_and(valids, mask)
            visibs = visibs & valids


        return RawSliceData.create(
            seq_name=str(self.keys[seq_id]),
            seq_id=seq_id,
            gt_trajs_3d=trajs_3d.astype(np.float32),
            gt_query_point=queries.astype(np.float32),
            visibs=visibs,
            valids=valids,  # davis does not provide annotation for invisible points
            rgbs=video,
            gt_depths=dummy_depths,
            gt_intrinsics=dummy_intrs,
            gt_extrinsics=dummy_extrs,
            orig_resolution=np.array([H, W], dtype=np.int32), # (h, w)
            copy_gt_to_est=True,
        )
    
    def load_seq_lens(self) -> List[int]:
        seq_lens = []
        for key in self.keys:
            seq_lens.append(self.data[key]['video'].shape[0])
        return seq_lens

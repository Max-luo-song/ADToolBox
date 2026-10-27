from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import tqdm
from einops import repeat

from datasets.providers.base_provider import BaseDataProvider
from datasets.datatypes import RawSliceData, SliceData
from PIL import Image
import cv2
import logging
import json
from glob import glob
from itertools import product
import imageio.v3 as iio

from datasets.utils.dataclass_utils import load_dataclass
from dataclasses import dataclass

from datasets.utils.crop_utils import get_crop_args
from datasets.utils.colmap import get_colmap_camera_params
from evaluation.tapvid3d_splits import MINIVAL_FILES, FULL_EVAL_FILES

logger = logging.getLogger(__name__)

def load_data_dict(data_dir, train_names, val_names):
    train_imgs = np.array(
        [iio.imread(os.path.join(data_dir, "rgb/1x", f"{name}.png")) for name in train_names]
    )
    val_covisibles = np.array(
        [
            iio.imread(
                os.path.join(
                    data_dir, "flow3d_preprocessed/covisible/1x/val/", f"{name}.png"
                )
            )
            for name in tqdm(val_names, desc="Loading val covisibles")
        ]
    )
    train_depths = np.array(
        [
            np.load(os.path.join(data_dir, "depth/1x", f"{name}.npy"))[..., 0]
            for name in train_names
        ]
    )
    train_Ks, train_w2cs = get_colmap_camera_params(
        os.path.join(data_dir, "flow3d_preprocessed/colmap/sparse/"),
        [name + ".png" for name in train_names],
    )
    train_Ks = train_Ks[:, :3, :3]
    scale = np.load(os.path.join(data_dir, "flow3d_preprocessed/colmap/scale.npy")).item()
    train_c2ws = np.linalg.inv(train_w2cs)
    train_c2ws[:, :3, -1] *= scale
    train_w2cs = np.linalg.inv(train_c2ws)
    keypoint_paths = sorted(glob(os.path.join(data_dir, "keypoint/2x/train/0_*.json"))) # type: ignore
    keypoints_2d = []
    for keypoint_path in keypoint_paths:
        with open(keypoint_path) as f:
            keypoints_2d.append(json.load(f))
    keypoints_2d = np.array(keypoints_2d)
    keypoints_2d[..., :2] *= 2.0
    time_ids = np.array(
        [int(os.path.basename(p).split("_")[1].split(".")[0]) for p in keypoint_paths]
    )
    time_pairs = np.array(list(product(time_ids, repeat=2)))
    index_pairs = np.array(list(product(range(len(time_ids)), repeat=2)))
    keypoints_3d = []
    for i, kps_2d in zip(time_ids, keypoints_2d):
        K = train_Ks[i]
        w2c = train_w2cs[i]
        depth = train_depths[i]
        is_kp_visible = kps_2d[:, 2] == 1
        is_depth_valid = (
            cv2.remap(
                (depth != 0).astype(np.float32),
                kps_2d[None, :, :2].astype(np.float32),
                None,  # type: ignore
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )[0]
            == 1
        )
        kp_depths = cv2.remap(
            depth,  # type: ignore
            kps_2d[None, :, :2].astype(np.float32),
            None,  # type: ignore
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )[0]
        kps_3d = (
            np.einsum(
                "ij,pj->pi",
                np.linalg.inv(K),
                np.pad(kps_2d[:, :2], ((0, 0), (0, 1)), constant_values=1),
            )
            * kp_depths[:, None]
        )
        kps_3d = np.einsum(
            "ij,pj->pi",
            np.linalg.inv(w2c)[:3],
            np.pad(kps_3d, ((0, 0), (0, 1)), constant_values=1),
        )
        kps_3d = np.concatenate(
            [kps_3d, (is_kp_visible & is_depth_valid)[:, None]], axis=1
        )
        kps_3d[kps_3d[:, -1] != 1] = 0.0
        keypoints_3d.append(kps_3d)
    keypoints_3d = np.array(keypoints_3d)
    return {
        "train_imgs": train_imgs,
        "val_covisibles": val_covisibles,
        "train_depths": train_depths,
        "train_Ks": train_Ks,
        "train_w2cs": train_w2cs,
        "keypoints_2d": keypoints_2d,
        "keypoints_3d": keypoints_3d,
        "time_ids": time_ids,
        "time_pairs": time_pairs,
        "index_pairs": index_pairs,
    }


class IPhoneDataProvider(BaseDataProvider):
    def __init__(self, cfg: DictConfig, override_anno: Optional[str] = None):
        super().__init__(cfg, override_anno)
        self.data_root: str = cfg.data_root

        assert os.path.exists(self.data_root), "Data root must exist"

        self.seq_names = [name for name in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, name))]
        self.seq_names.sort()

        logger.info(f"Loaded iPhone from {self.data_root} with {len(self.seq_names)} sequences")
    
    def _load_slice(
        self, 
        seq_id: int, 
        start: int, 
        length: int, 
        stride: int,
        rng: np.random.Generator,
        executor: ThreadPoolExecutor,
    ) -> RawSliceData:
        train_info_path = os.path.join(self.data_root, self.seq_names[seq_id], 'splits', 'train.json')
        with open(train_info_path, 'r') as f:
            train_info = json.load(f)
        val_info_path = os.path.join(self.data_root, self.seq_names[seq_id], 'splits', 'val.json')
        with open(val_info_path, 'r') as f:
            val_info = json.load(f)

        data_dict = load_data_dict(
            os.path.join(self.data_root, self.seq_names[seq_id]),
            train_info['frame_names'],
            val_info['frame_names'],
        )
        video_len = len(data_dict['train_imgs'])
        num_keypoints = data_dict['keypoints_3d'].shape[1]
        trajs = np.zeros((video_len, num_keypoints, 3), dtype=np.float32)
        valids = np.zeros((video_len, num_keypoints), dtype=np.bool_)

        for t, keypoints in zip(data_dict['time_ids'], data_dict['keypoints_3d']):
            trajs[t, :, :] = keypoints[:, :3]
            valids[t, :] = keypoints[:, -1] == 1

        rgbs = data_dict['train_imgs'][start:start+length*stride:stride][..., :3]
        depths = data_dict['train_depths'][start:start+length*stride:stride]
        intrinsics = data_dict['train_Ks'][start:start+length*stride:stride]
        extrinsics = data_dict['train_w2cs'][start:start+length*stride:stride]
        trajs = trajs[start:start+length*stride:stride]
        valids = valids[start:start+length*stride:stride]

        query_frames = np.argmax(valids, axis=0)
        queries = trajs[query_frames, np.arange(num_keypoints)]
        query_points = np.concatenate([query_frames[..., None].astype(np.float32), queries], axis=-1)

        return RawSliceData.create(
            seq_name=self.seq_names[seq_id],
            seq_id=seq_id,
            gt_trajs_3d=trajs,
            gt_query_point=query_points,
            visibs=valids.copy(),
            valids=valids, 
            rgbs=rgbs,
            gt_depths=depths,
            gt_intrinsics=intrinsics.astype(np.float32),
            gt_extrinsics=extrinsics.astype(np.float32),
            orig_resolution=np.array([rgbs[0].shape[0], rgbs[0].shape[1]], dtype=np.int32), # (h, w)
            copy_gt_to_est=True,
        )
    
    def load_seq_lens(self) -> List[int]:
        seq_lens = []
        for seq_name in tqdm(self.seq_names, "Loading Sequence Lengths"):
            info_path = os.path.join(self.data_root, seq_name, 'splits', 'train.json')
            with open(info_path, 'r') as f:
                info = json.load(f)
                seq_lens.append(len (info['frame_names']))
        return seq_lens

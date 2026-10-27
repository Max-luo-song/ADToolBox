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

from datasets.utils.dataclass_utils import load_dataclass
from dataclasses import dataclass

from datasets.utils.crop_utils import get_crop_args
from evaluation.tapvid3d_splits import MINIVAL_FILES, FULL_EVAL_FILES

logger = logging.getLogger(__name__)

class LSFOdysseyProvider(BaseDataProvider):
    def __init__(self, cfg: DictConfig, override_anno: Optional[str] = None):
        super().__init__(cfg, override_anno)
        self.split: str = cfg.split
        self.data_root: str = os.path.join(cfg.data_root, self.split)
        self.only_first_n_seqs: Optional[int] = cfg.get("only_first_n_seqs", None)
        self.shuffle: bool = cfg.get("shuffle", False)

        self.seqs = []
        assert os.path.exists(self.data_root), f"{self.data_root} does not exist"
        for group in os.listdir(self.data_root):
            if not os.path.isdir(os.path.join(self.data_root, group)):
                continue
            for seq in os.listdir(os.path.join(self.data_root, group)):
                if os.path.isfile(os.path.join(self.data_root, group, seq, "rgb.mp4")):
                    self.seqs.append(f"{group}/{seq}")
        self.seqs.sort()

        if self.shuffle:
            rng = np.random.RandomState(42)
            rng.shuffle(self.seqs)

        if self.only_first_n_seqs is not None:
            self.seqs = self.seqs[:self.only_first_n_seqs]

        logger.info(f"Loaded LSFOdyssey from {self.data_root} with {len(self.seqs)} sequences")

    def read_mp4(self, name_path):
        vidcap = cv2.VideoCapture(name_path)
        frames = []
        while (vidcap.isOpened()):
            ret, frame = vidcap.read()
            if ret == False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        vidcap.release()
        return frames
    
    def _load_slice(
        self, 
        seq_id: int, 
        start: int, 
        length: int, 
        stride: int,
        rng: np.random.Generator,
        executor: ThreadPoolExecutor,
    ) -> RawSliceData:
        camera_info = np.load(os.path.join(self.data_root, self.seqs[seq_id], "intris.npz"))
        rgbs = self.read_mp4(os.path.join(self.data_root, self.seqs[seq_id], "rgb.mp4"))
        depths_npz = np.load(os.path.join(self.data_root, self.seqs[seq_id], "deps.npz"))
        tracks_npz = np.load(os.path.join(self.data_root, self.seqs[seq_id], "track.npz"))

        rgbs = np.stack(rgbs[start:start+length*stride:stride])
        depths = np.stack(depths_npz['deps'][start:start+length*stride:stride])[:, 0]
        intrinsics = camera_info['intris'][0, start:start+length*stride:stride]
        extrinsics = camera_info['extris'][0, start:start+length*stride:stride]

        tracks = tracks_npz['track_g']
        trajs_uv = tracks[..., 0:2]  # shape: (T, N, 2)
        trajs_z = tracks[..., 2:3]  # shape: (T, N, 1)
        visibs = tracks[..., 3].astype(np.bool_) # shape: (T, N)
        valids = tracks[..., 4].astype(np.bool_) # shape: (T, N)

        trajs_3d = np.concatenate([trajs_uv, np.ones_like(trajs_uv[..., :1])], axis=-1)
        trajs_3d = np.einsum('tij,tnj->tni', np.linalg.inv(intrinsics), trajs_3d)
        trajs_3d = trajs_3d * trajs_z
        trajs_3d = np.concatenate([trajs_3d, np.ones_like(trajs_3d[..., :1])], axis=-1)
        trajs_3d = np.einsum('tij,tnj->tni', np.linalg.inv(extrinsics), trajs_3d)
        trajs_3d = trajs_3d[..., :3]
        
        assert visibs[0].all()

        query_points = trajs_3d[0]
        query_points_t = np.zeros_like(query_points[..., :1])  # shape: (N, 1)
        query_points = np.concatenate([query_points_t, query_points], axis=-1)  # shape: (N, 4)

        return RawSliceData.create(
            seq_name=self.seqs[seq_id],
            seq_id=seq_id,
            gt_trajs_3d=trajs_3d.astype(np.float32),
            gt_query_point=query_points,
            visibs=visibs,
            valids=valids, 
            rgbs=rgbs,
            gt_depths=depths.astype(np.float32),
            gt_intrinsics=intrinsics,
            gt_extrinsics=extrinsics,
            orig_resolution=np.array([rgbs[0].shape[0], rgbs[0].shape[1]], dtype=np.int32), # (h, w)
            copy_gt_to_est=True,
        )
    
    def load_seq_lens(self) -> List[int]:
        seq_lens = []
        for seq in tqdm(self.seqs, "Loading Sequence Lengths"):
            npz_path = os.path.join(self.data_root, seq, "intris.npz")
            with open(npz_path, 'rb') as in_f:
                in_npz = np.load(in_f, allow_pickle=True)
                seq_lens.append(in_npz['intris'].shape[1])
        return seq_lens

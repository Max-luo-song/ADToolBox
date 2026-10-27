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
import re

from datasets.utils.geometry import batch_distance_to_depth
from datasets.utils.crop_utils import get_crop_args
from line_profiler import profile
import yaml
from einops import repeat

logger = logging.getLogger(__name__)

EXCLUDE_SEQS = set([
    "20200908-subject-05/20200908_151833/932122060861",
    "20201002-subject-08/20201002_112816/932122060857",
    "20200928-subject-07/20200928_154520/932122060857",
    "20201002-subject-08/20201002_112627/932122060857",
    "20200908-subject-05/20200908_143353/932122060857",
    "20200928-subject-07/20200928_154739/932122060857",
])

class DexYCBDataProvider(BaseDataProvider):
    def __init__(self, cfg: DictConfig, override_anno: Optional[str] = None):
        super().__init__(cfg, override_anno)
        self.data_root = cfg.data_root
        self.max_obj_points = cfg.max_obj_points
        self.max_hand_points = cfg.max_hand_points
        self.shuffle = cfg.get("shuffle", False)
        self._single_idx = cfg.get("single_idx", None)

        subjects = [name for name in os.listdir(self.data_root) if "subject-" in name]
        self.seq_paths = []
        self._serials = []
        for subject in subjects:
            for date in os.listdir(os.path.join(self.data_root, subject)):
                for serial in os.listdir(os.path.join(self.data_root, subject, date)):
                    if not os.path.isdir(os.path.join(self.data_root, subject, date, serial)):
                        continue
                    if f"{subject}/{date}/{serial}" in EXCLUDE_SEQS:
                        continue
                    self.seq_paths.append(f"{subject}/{date}/{serial}")
                    self._serials.append(serial)

        self.seq_paths.sort()

        if self.shuffle:
            rng = np.random.RandomState(42)
            rng.shuffle(self.seq_paths)

        if self._single_idx is not None:
            self.seq_paths = [self.seq_paths[self._single_idx]]

        self._serials = list(set(self._serials))
        self._serials.sort()

        self._h = 480
        self._w = 640
        self._intrinsics = []
        for s in self._serials:
            intr_file = os.path.join(self.data_root, "calibration", "intrinsics",
                                    "{}_{}x{}.yml".format(s, self._w, self._h))
            with open(intr_file, 'r') as f:
                intr = yaml.load(f, Loader=yaml.FullLoader)
            intr = intr['color']
            self._intrinsics.append(intr)

        def _count_images(path: str):
            return len ([name for name in os.listdir(os.path.join(self.data_root, path)) if re.fullmatch(r"color_\d+\.jpg$", name)])
        self.seq_lens = [_count_images(path) for path in self.seq_paths]
        logger.info(f"Loaded Dex-YCB data from {self.data_root} with {len(self.seq_paths)} sequences")

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
        rgb_futures = []
        depth_futures = []
        label_futures = []
        vertices_futures = []
        for i in range(start, start+length*stride, stride):
            rgb_futures.append(executor.submit(lambda idx: cv2.cvtColor(cv2.imread(os.path.join(self.data_root, self.seq_paths[seq_id], f"color_{idx:06d}.jpg")), cv2.COLOR_BGR2RGB), i))
            depth_futures.append(executor.submit(lambda idx: cv2.imread(os.path.join(self.data_root, self.seq_paths[seq_id], f"aligned_depth_to_color_{idx:06d}.png"), cv2.IMREAD_ANYDEPTH), i))
            label_futures.append(executor.submit(lambda idx: dict(np.load(os.path.join(self.data_root, self.seq_paths[seq_id], f"labels_{idx:06d}.npz"))), i))
            vertices_futures.append(executor.submit(lambda idx: np.load(os.path.join(self.data_root, self.seq_paths[seq_id], f"rendered_vertices_{idx:06d}.npz")), i))
        rgbs = np.stack([future.result() for future in rgb_futures])
        depths = np.stack([future.result() for future in depth_futures]).astype(np.float32) / 1000
        labels = [future.result() for future in label_futures]
        vertices = [future.result() for future in vertices_futures]
        vertices = {key: np.stack([v[key] for v in vertices]) for key in vertices[0].keys()}
    
        meta_path = os.path.join(self.data_root, *self.seq_paths[seq_id].split("/")[:-1], "meta.yml")
        with open(meta_path, 'r') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)

        ycb_ids = meta['ycb_ids']

        intrs_dict = self._intrinsics[self._serials.index(self.seq_paths[seq_id].split("/")[-1])].copy()
        intrinsics = np.array(
            [
                [intrs_dict['fx'], 0.0, intrs_dict['ppx']],
                [0.0, intrs_dict['fy'], intrs_dict['ppy']],
                [0.0, 0.0, 1.0]
            ],
            dtype=np.float32
        )[None].repeat(rgbs.shape[0], axis=0)
        extrinsics = np.eye(4, dtype=np.float32)[None].repeat(rgbs.shape[0], axis=0)
        segmentation = np.stack([labels[i]['seg'] for i in range(len(labels))]).astype(np.int32)

        poses = np.stack([labels[i]['pose_y'] for i in range(len(labels))])
        poses_std = np.std(poses, axis=0)

        n_objs = poses.shape[1]
        is_moving = poses_std.max(axis=(-1, -2)) > 1e-4
        moving_objs = np.arange(n_objs)[is_moving]
        moving_objs_ycb = [ycb_ids[obj_id] for obj_id in moving_objs]

        assert (vertices['groups'][:1] == vertices['groups'][:]).all()
        mask = np.zeros(vertices['visibs'].shape[1], dtype=np.bool_)
        for obj_id in moving_objs:
            mask[vertices['groups'][0] == obj_id] = True
        mask[vertices['groups'][0] >= 255] = True

        vertices = {key: vertices[key][:, mask] for key in vertices.keys()}

        trajs = vertices['vertices']
        visibs = vertices['visibs']
        valids = vertices['valids']

        trajs_2d = np.einsum("tij,tnj->tni", intrinsics, trajs)
        trajs_2d = trajs_2d[..., :2] / np.clip(trajs_2d[..., 2:3], 1e-4, np.inf)
        trajs_2d = np.round(trajs_2d).astype(np.int32)
        trajs_2d[..., 0] = np.clip(trajs_2d[..., 0], 0, rgbs[0].shape[1]-1)
        trajs_2d[..., 1] = np.clip(trajs_2d[..., 1], 0, rgbs[0].shape[0]-1)

        # maybe first dilate the depthmap a bit?
        sampled_depths = depths[np.arange(len(rgbs))[:, None], trajs_2d[..., 1], trajs_2d[..., 0]]

        # todo: check whether this is necessary
        # import ipdb; ipdb.set_trace()
        # visibs[sampled_depths > trajs[..., 2] + 0.03] = False

        visibs = visibs & valids

        visibs_ratio = visibs.sum(axis=0) / visibs.shape[0]
        mask = visibs_ratio >= 0.1
        trajs = trajs[:, mask]
        visibs = visibs[:, mask]
        valids = valids[:, mask]
        groups = vertices['groups'][:, mask]

        query_frames = np.argmax(visibs, axis=0).astype(np.int32)
        query_coords = trajs[query_frames, np.arange(trajs.shape[1])]
        query_points = np.concatenate([query_frames[..., None], query_coords], axis=-1).astype(np.float32)

        groups = groups[0]
        num_obj_points = np.sum(groups < 255)
        num_hand_points = np.sum(groups >= 255)

        obj_perm = rng.permutation(num_obj_points)[:self.max_obj_points]
        hand_perm = rng.permutation(num_hand_points)[:self.max_hand_points]
        
        obj_query_points = query_points[groups < 255][obj_perm]
        hand_query_points = query_points[groups >= 255][hand_perm]
        obj_trajs = trajs[:, groups < 255][:, obj_perm]
        hand_trajs = trajs[:, groups >= 255][:, hand_perm]
        obj_visibs = visibs[:, groups < 255][:, obj_perm]
        hand_visibs = visibs[:, groups >= 255][:, hand_perm]
        obj_valids = valids[:, groups < 255][:, obj_perm]
        hand_valids = valids[:, groups >= 255][:, hand_perm]

        query_points = np.concatenate([obj_query_points, hand_query_points], axis=0)
        trajs = np.concatenate([obj_trajs, hand_trajs], axis=1)
        visibs = np.concatenate([obj_visibs, hand_visibs], axis=1)
        valids = np.concatenate([obj_valids, hand_valids], axis=1)

        num_points = visibs.shape[1]
        if num_points == 0:
            print (f"{self.seq_paths[seq_id]}: No usable trajectories found")

        return RawSliceData.create(
            seq_name=self.seq_paths[seq_id],
            seq_id=seq_id,
            gt_trajs_3d=trajs,
            visibs=visibs,
            valids=valids,
            gt_query_point=query_points,
            rgbs=rgbs,
            gt_depths=depths,
            gt_intrinsics=intrinsics,
            gt_extrinsics=extrinsics,
            orig_resolution=np.array([rgbs[0].shape[0], rgbs[0].shape[1]], dtype=np.int32), # (h, w)
            segmentation=segmentation,
            copy_gt_to_est=True,
        )
    
    def load_seq_lens(self) -> List[int]:
        return self.seq_lens

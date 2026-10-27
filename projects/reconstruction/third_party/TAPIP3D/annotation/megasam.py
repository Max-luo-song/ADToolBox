from typing import Dict, Tuple, Optional
import tap
import os
import copy
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from annotation.utils import generate_annotations
from annotation.base_annotator import BaseAnnotator
from datasets.datatypes import RawSliceData
from functools import partial
import tempfile
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from pathlib import Path
import cv2
from pathlib import Path

DEFAULT_SCRIPT_PATH = Path(__file__).parent.parent / "third_party" / "megasam" / "inference.py"

def get_local_rank():
    if not dist.is_initialized():
        return None
    else:
        assert "LOCAL_RANK" in os.environ, "LOCAL_RANK is not set"
        return int(os.environ["LOCAL_RANK"])

class MegaSAMAnnotator(BaseAnnotator):
    def __init__(self, script_path: Path = DEFAULT_SCRIPT_PATH, use_gt_intrinsics: bool = False, num_workers: int = 8, depth_model: str = "dav2", resolution: int = 384 * 512):
        self.script_path = script_path
        self.use_gt_intrinsics = use_gt_intrinsics
        self.num_workers = num_workers
        self.depth_model = depth_model
        self.resolution = resolution
        assert self.depth_model in ["dav1", "dav2", "videoda", "moge"], "Invalid depth model"

    def process_video(self, rgbs: np.ndarray, gt_intrinsics: Optional[np.ndarray] = None, return_raw_depths: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with tempfile.TemporaryDirectory() as temp_dir:
            # first, write rgbs to jpg files
            for i in range(len(rgbs)):
                Image.fromarray(rgbs[i]).save(os.path.join(temp_dir, f"{i:08d}.jpg"))
            # then, run the script
            cmd = [
                "python",
                Path(self.script_path).resolve(),
                f"--input_dir={Path(temp_dir).resolve()}",
                f"--output_path={(Path(temp_dir) / 'output.npz').resolve()}",
                f"--depth_model={self.depth_model}",
                f"--resolution={self.resolution}",
            ]
            if self.use_gt_intrinsics:
                H, W = rgbs[0].shape[:2]
                gt_intrinsic0 = gt_intrinsics[0]
                fov_x = 2 * np.arctan(W / (2 * gt_intrinsic0[0][0]))
                fov_x = np.rad2deg(fov_x)
                cmd.append(f"--fov_x={fov_x}")

            env = os.environ.copy()
            if get_local_rank() is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(get_local_rank())
            subprocess.run(cmd, env=env)

            data = np.load(Path(temp_dir) / "output.npz")
            images, depths, intrinsic, cam_c2w = data["images"], data["depths"], data["intrinsic"], data["cam_c2w"]
            raw_depths = data["depths_raw"]

        extrinsics = np.linalg.inv(cam_c2w)
        orig_h, orig_w = images[0].shape[:2]
        target_h, target_w = rgbs[0].shape[:2]

        depth_futures = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for i in range(len(images)):
                depth_futures.append(executor.submit(lambda depth: cv2.resize(depth.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR), depths[i]))
            depths = np.stack([future.result() for future in depth_futures])

        intrinsic[0, :] = intrinsic[0, :] * target_w / orig_w
        intrinsic[1, :] = intrinsic[1, :] * target_h / orig_h

        if self.use_gt_intrinsics:
            intrinsics = gt_intrinsics.copy()
        else:
            intrinsics = np.repeat(np.expand_dims(intrinsic, axis=0), len(images), axis=0)
        
        if return_raw_depths:
            return raw_depths.astype(np.float32), intrinsics.astype(np.float32), extrinsics.astype(np.float32)
        else:
            return depths.astype(np.float32), intrinsics.astype(np.float32), extrinsics.astype(np.float32)

    @torch.inference_mode()
    def __call__(self, data: RawSliceData) -> Dict[str, np.ndarray]:
        # TODO:
        # 1. check for updates on github
        # 2. try other depth models like depthpro and moge
        # 3. try utilising gt intrinsics
        depths, intrinsics, extrinsics = self.process_video(data.rgbs, gt_intrinsics=data.gt_intrinsics if self.use_gt_intrinsics else None)

        return dict(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depths=depths,
        )

    @property
    def device(self):
        pass

    def to(self, device: str):
        pass

class Args(tap.Tap):
    output_path: str
    provider_cfg: str
    num_dataloader_threads: int = 8
    num_dataloader_workers: int = 4
    use_gt_intrinsics: bool = False
    mod10: Optional[int] = None
    script_path: Path = Path(__file__).parent.parent / "third_party" / "megasam" / "mega-sam" / "inference.py"

if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    args = Args().parse_args()
    annotator = MegaSAMAnnotator(args.script_path, use_gt_intrinsics=args.use_gt_intrinsics)

    generate_annotations(args.output_path, annotator, args.provider_cfg, num_dataloader_threads=args.num_dataloader_threads, num_dataloader_workers=args.num_dataloader_workers, mod10=args.mod10)

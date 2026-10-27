from typing import Dict
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
import third_party.depth_pro as depth_pro
from third_party.depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

class DepthProAnnotator(BaseAnnotator):
    def __init__(self, checkpoint: str, infer_intrinsics: bool = False):
        self.checkpoint = checkpoint
        self.infer_intrinsics = infer_intrinsics

        config = copy.deepcopy(DEFAULT_MONODEPTH_CONFIG_DICT)
        config.checkpoint_uri = self.checkpoint

        self.model, self.transform = depth_pro.create_model_and_transforms(config)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, data: RawSliceData) -> Dict[str, np.ndarray]:
        depths = []
        intrinsics = []

        for i in tqdm(range(len(data.rgbs)), position=1, desc="Generating depths"):
            image = data.rgbs[i]
            intrinsics = data.gt_intrinsics[i]
            if not self.infer_intrinsics:
                f_px = np.sqrt(intrinsics[0][0] * intrinsics[1][1])
            else:
                f_px = None

            image = self.transform(image).to(self.device)
            prediction = self.model.infer(image, f_px=f_px)
            depth = prediction["depth"].cpu().numpy()
            if self.infer_intrinsics:
                focallength_px = prediction["focallength_px"].item()
                intrinsic = np.array([[focallength_px, 0, 0], [0, focallength_px, 0], [0, 0, 1]])
                intrinsics.append(intrinsic)
            depths.append(depth)
            
        if self.infer_intrinsics:
            return dict(
                depths=np.stack(depths),
                intrinsics=np.stack(intrinsics),
            )
        else:
            return dict(
                depths=np.stack(depths),
            )

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to(self, device: str):
        self.model.to(device)

class Args(tap.Tap):
    output_path: str
    provider_cfg: str
    infer_intrinsics: int
    num_dataloader_threads: int = 8
    num_dataloader_workers: int = 4
    checkpoint: str = os.path.join(os.environ.get("CHECKPOINT_DIR"), "depth_pro.pt") # type: ignore

if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    args = Args().parse_args()
    annotator = DepthProAnnotator(checkpoint=args.checkpoint, infer_intrinsics=args.infer_intrinsics)

    generate_annotations(args.output_path, annotator, args.provider_cfg, num_dataloader_threads=args.num_dataloader_threads, num_dataloader_workers=args.num_dataloader_workers)

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
from third_party.unidepth.models.unidepthv2.unidepthv2 import UniDepthV2

class UniDepthAnnotator(BaseAnnotator):
    def __init__(self):
        name = "unidepth-v2-vitl14"

        self.model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}", revision="1d0d3c52f60b5164629d279bb9a7546458e6dcc4")
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, data: RawSliceData) -> Dict[str, np.ndarray]:
        depths = []
        intrinsics_list = []
        
        for i in tqdm(range(len(data.rgbs)), position=1, desc="Generating depths"):
            image = data.rgbs[i]
            intrinsics = data.gt_intrinsics[i]

            assert image.dtype == np.uint8, "Image must be uint8"

            image = torch.tensor(image).to(self.device)
            intrinsics = torch.tensor(intrinsics).to(self.device)
            predictions = self.model.infer(image.permute(2, 0, 1), intrinsics)
            depth = predictions["depth"].cpu()[0][0].numpy()
            intrinsics = predictions["intrinsics"].cpu()[0].numpy()

            depths.append(depth)
            intrinsics_list.append(intrinsics)
        
        return dict(
            depths=np.stack(depths),
            intrinsics=np.stack(intrinsics_list),
        )

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to(self, device: str):
        self.model.to(device)

class Args(tap.Tap):
    output_path: str
    provider_cfg: str
    num_dataloader_threads: int = 8
    num_dataloader_workers: int = 4
    checkpoint: str = os.path.join(os.environ.get("CHECKPOINT_DIR"), "depth_pro.pt") # type: ignore

if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    args = Args().parse_args()
    annotator = UniDepthAnnotator()

    generate_annotations(args.output_path, annotator, args.provider_cfg, num_dataloader_threads=args.num_dataloader_threads, num_dataloader_workers=args.num_dataloader_workers)

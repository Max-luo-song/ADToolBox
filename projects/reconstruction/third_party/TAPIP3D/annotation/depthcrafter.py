from typing import Dict, Optional
from einops import rearrange
import tap
import os
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from annotation.utils import generate_annotations
from annotation.base_annotator import BaseAnnotator
from datasets.datatypes import RawSliceData
from functools import partial
from third_party.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from third_party.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from torch.utils.data import TensorDataset

class ABModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.b = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32), requires_grad=True)
    def forward(self, x):
        return 1 / torch.clamp(self.a * x + self.b, min=1e-3)

class DepthCrafterAnnotator(BaseAnnotator):
    def __init__(
        self, 
        window_size: int = 110,
        overlap: int = 25,
        num_inference_steps: int = 5,
        guidance_scale: float = 1.0,
        min_res: Optional[int] = None, 
        max_res: Optional[int] = None,
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            "tencent/DepthCrafter",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self.pipe = DepthCrafterPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.min_res = min_res
        self.max_res = max_res

        self.window_size = window_size
        self.overlap = overlap
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

    @torch.inference_mode()
    def __call__(self, data: RawSliceData) -> Dict[str, np.ndarray]:
        rgbs = torch.from_numpy(data.rgbs).to(self.device).float() / 255.0
        est_depths = data.est_depths
        T, H, W, C = rgbs.shape

        if self.min_res is not None and max(H, W) < self.min_res:
            scale = self.min_res / max(H, W)
            model_H = round(H * scale / 64) * 64
            model_W = round(W * scale / 64) * 64
        elif self.max_res is not None and max(H, W) > self.max_res:
            scale = self.max_res / max(H, W)
            model_H = round(H * scale / 64) * 64
            model_W = round(W * scale / 64) * 64
        else:
            model_H = H
            model_W = W

        rgbs = torch.nn.functional.interpolate(
            rearrange(rgbs, "t h w c -> t c h w"),
            size=(model_H, model_W), 
            mode="bilinear"
        )

        res = self.pipe(
            rgbs,
            height=rgbs.shape[2],
            width=rgbs.shape[3],
            output_type="pt",
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            window_size=self.window_size,
            overlap=self.overlap,
            track_time=False,
        ).frames[0] # type: ignore

        res = torch.nn.functional.interpolate(
            res,
            size=(H, W),
            mode="nearest-exact",
        )
        res = rearrange(res, "t c h w -> t h w c")

        res = res.sum(-1) / res.shape[-1]
        res = (res - res.min()) / (res.max() - res.min())
        res = res.cpu().numpy()

        valid_mask = (est_depths > 1e-2) & (res > 1e-3)
        ref_depths_masked = est_depths[valid_mask].reshape((-1, 1)).astype(np.float32)
        pred_disp_masked = res[valid_mask].reshape((-1, 1)).astype(np.float32)

        x_t = torch.from_numpy(pred_disp_masked).to(self.device)
        y_t = torch.from_numpy(ref_depths_masked).to(self.device)

        lr = 1e-2
        T = 1000
        
        with torch.inference_mode(False):
            model_lm = ABModel().to(self.device)
            optim = torch.optim.Adam(model_lm.parameters(), lr=lr)
            for i in range(T):
                optim.zero_grad()
                loss = (torch.abs(model_lm(x_t.clone()) - y_t.clone()) / y_t.clone()).mean()
                loss.backward()
                optim.step()
                print(f"iter {i}, loss {loss.item()}")

        a = model_lm.a.item()
        b = model_lm.b.item()

        pred_disp = np.clip(res, a_min=1e-3, a_max=None)
        aligned_pred = a * pred_disp + b
        aligned_pred = np.clip(aligned_pred, a_min=1e-3, a_max=None) 
        depths_pred = 1. / aligned_pred

        return dict(
            depths=depths_pred,
        )

    @property
    def device(self):
        return self.pipe.device

    def to(self, device: str):
        self.pipe.to(device)

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
    annotator = DepthCrafterAnnotator(min_res=512, max_res=1024)

    generate_annotations(args.output_path, annotator, args.provider_cfg, num_dataloader_threads=args.num_dataloader_threads, num_dataloader_workers=args.num_dataloader_workers)
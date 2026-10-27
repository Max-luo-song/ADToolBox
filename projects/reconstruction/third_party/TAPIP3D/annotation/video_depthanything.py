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
from third_party.video_depth_anything.video_depth_anything.video_depth import VideoDepthAnything

from PIL import Image

import numpy as np
from typing import Optional, Tuple, List, Dict, Union

from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

degree = 1
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
ransac = RANSACRegressor(max_trials=1000)
linear_model = make_pipeline(poly_features, ransac)

# https://github.com/DepthAnything/PromptDA/issues/19
def scale_shift_align_depth(src, tar, mask, disp=False, fit_type="poly", min_tar_val=1e-4, min_output_disp=1e-2):
    '''
	src: HxW, monocular depth to be aligned
	tar: HxW, target depth map, typically either metric depth or SFM depth
	mask: HxW, boolean mask, used to select the valid region in the target depth map
	disp: Whether it is disparity. “depth_anything” and “midas” are disparity models and should be set to True; “depth_pro” and “metric3d_v2” are depth models and should be set to False.
	fit_type: Fitting method, options are ‘poly’ or ‘ransac’
    '''
    tar_val = tar[mask].astype(np.float32)
    src_val = src[mask].astype(np.float32)
    if disp:
        tar_val = np.clip(tar_val, min_tar_val, None)
        tar_val = 1 / tar_val

    if fit_type == "poly":
        a, b = np.polyfit(src_val, tar_val, deg=1)
    elif fit_type == "ransac":
        linear_model.fit(src_val[:, None], tar_val[:, None])
        a = linear_model.named_steps["ransacregressor"].estimator_.coef_
        b = linear_model.named_steps["ransacregressor"].estimator_.intercept_
        a, b = a.item(), b.item()
    else:
        a, b = 1, 0
    if a < 0:
        return False, None
    src_ = a * src + b
    if disp:
        src_ = 1 / np.clip(src_, min_output_disp, None)
    return True, src_

def distance_to_depth(coords: np.ndarray, distances: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    # coords: (..., 2), distances: (...), intrinsics: (..., 3, 3)
  
    coords_homogeneous = np.concatenate((coords, np.ones_like(coords[..., :1])), axis=-1).astype(np.float32)
    K_inv = np.linalg.inv(intrinsics)
    camera_coords = np.einsum("...ij, ...j -> ...i", K_inv, coords_homogeneous)
    depth = camera_coords[..., -1] * distances / np.sqrt((camera_coords ** 2).sum(-1))

    return depth

def MAD(x: np.ndarray):
    return np.median(np.abs(x - np.median(x)))

class VideoDepthAnythingAnnotator(BaseAnnotator):
    def __init__(self, model_type: str, checkpoint: str):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        self.model = VideoDepthAnything(**model_configs[model_type])
        self.model.load_state_dict(torch.load(checkpoint))
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, data: RawSliceData) -> Dict[str, np.ndarray]:
        assert data.annotated, "please pass a data provider with estimated depth. recovering scale and shift from gt is not fair!"
        
        depth_list, _ = self.model.infer_video_depth(data.rgbs, target_fps=None, device=str(self.device))

        orig_intrinsics = data.est_intrinsics # (t, 3, 3)
        orig_depths = data.est_depths # (t, h, w)
        
        T, H, W = orig_depths.shape
        
        # for simplicity, we just use the median intrinsics
        intrinsics = np.median(orig_intrinsics, axis=0)[None].repeat(T, axis=0)

        inverse_depths = np.stack(depth_list)
        valid_mask = orig_depths > 0 & (inverse_depths > 0)
        
        ok, depths = scale_shift_align_depth(
            src=inverse_depths, 
            tar=orig_depths, 
            mask=valid_mask, 
            disp=True, 
            fit_type="ransac",
            min_tar_val=1e-4,
            min_output_disp=1. / (orig_depths.max() * 2) # type: ignore
        )
        assert ok, "failed to align depth"
        assert np.isfinite(depths).all(), "depths are not finite" # type: ignore
        
        return dict(
            depths=depths,
            intrinsics=intrinsics,
        )

        # we have a group of depth pairs: src[i], dst[i]
        # want to solve for a scale and a shift so that: \sum |((scale * src[i] + shift) - dst[i]) / dst[i]| is minimized.

        # eps = 1e-6
        # shift_min = 

        # orig_depths_torch = torch.from_numpy(orig_depths).float().to(self.device)
        # depths_torch = torch.from_numpy(np.stack(depth_list)).float().to(self.device)
        # valid_mask = depths_torch > 0.
        # # here we need to be very careful about valid

        # coef_scale = torch.
        # coef_shift = 
        # b = np.ones(T, H*W)

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
    mod10: Optional[int] = None
    idx: Optional[int] = None
    checkpoint: str = os.path.join(os.environ.get("CHECKPOINT_DIR"), "video_depth_anything_vitl.pth") # type: ignore

if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    args = Args().parse_args()
    annotator = VideoDepthAnythingAnnotator(model_type="vitl", checkpoint=args.checkpoint)

    generate_annotations(args.output_path, annotator, args.provider_cfg, num_dataloader_threads=args.num_dataloader_threads, num_dataloader_workers=args.num_dataloader_workers, mod10=args.mod10, _idx=args.idx)

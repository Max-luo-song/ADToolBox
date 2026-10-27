# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

from concurrent.futures import ThreadPoolExecutor
import shlex
import tap
import torch
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime
from einops import repeat
from utils.common_utils import setup_logger
import logging
from annotation.megasam import MegaSAMAnnotator
import numpy as np
import cv2
from datasets.data_ops import _filter_one_depth

from utils.inference_utils import load_model, read_video, inference, get_grid_queries, resize_depth_bilinear

logger = logging.getLogger(__name__)

DEFAULT_QUERY_GRID_SIZE = 32

class Arguments(tap.Tap):
    input_path: str
    device: str = "cuda"
    num_iters: int = 6
    support_grid_size: int = 16
    num_threads: int = 8
    resolution_factor: int = 2
    vis_threshold: Optional[float] = 0.9
    checkpoint: Optional[str] = None
    output_dir: str = "outputs/inference"
    depth_model: str = "moge"

def prepare_inputs(input_path: str, inference_res: Tuple[int, int], support_grid_size: int, num_threads: int = 8, device: str = "cpu"):
    if not Path (input_path).is_file():
        raise ValueError(f"Input file not found: {input_path}")
    video, depths, intrinsics, extrinsics, query_point = None, None, None, None, None
    if input_path.endswith((".mp4", ".avi", ".mov", ".webm")):
        video = read_video(input_path)
    elif input_path.endswith(".npz"):
        data = np.load(input_path)
        video = data['video']
        assert video.ndim == 4 and video.shape[-1] == 3 and video.dtype == np.uint8, f"Invalid video shape or dtype: {video.shape}, {video.dtype}"
        depths = data.get('depths', None)
        intrinsics = data.get('intrinsics', None)
        extrinsics = data.get('extrinsics', None)
        query_point = data.get('query_point', None)
    else:
        raise ValueError(f"Unsupported input type: {input_path}. Supported formats are .mp4 and .npz.")
    
    if depths is None:
        logger.info(f"No depth provided, running MegaSAM to get depths")
        megasam = MegaSAMAnnotator(
            script_path=Path(__file__).parent / "third_party" / "megasam" / "inference.py",
            depth_model=args.depth_model,
            resolution=inference_res[0] * inference_res[1]
        )
        megasam.to(args.device)
        depths, intrinsics, extrinsics = megasam.process_video(video, gt_intrinsics=intrinsics, return_raw_depths=True)
        _original_res = video.shape[1:3]
    else:
        _original_res = depths.shape[1:3]

    if intrinsics is None:
        raise ValueError("Intrinsics must be provided if depth is provided")
    if extrinsics is None:
        logger.info(f"No extrinsics provided, using identity matrix for all frames")
        extrinsics = repeat(np.eye(4), "i j -> t i j", t=len(video))
    
    intrinsics[:, 0, :] *= (inference_res[1] - 1) / (_original_res[1] - 1)
    intrinsics[:, 1, :] *= (inference_res[0] - 1) / (_original_res[0] - 1)

    # resize & remove edges
    with ThreadPoolExecutor(num_threads) as executor:
        video_futures = [executor.submit(cv2.resize, rgb, (inference_res[1], inference_res[0]), interpolation=cv2.INTER_LINEAR) for rgb in video]
        depths_futures = [executor.submit(resize_depth_bilinear, depth, (inference_res[1], inference_res[0])) for depth in depths]
        
        video = np.stack([future.result() for future in video_futures])
        depths = np.stack([future.result() for future in depths_futures])

        depths_futures = [executor.submit(_filter_one_depth, depth, 0.08, 15, intrinsic) for depth, intrinsic in zip(depths, intrinsics)]
        depths = np.stack([future.result() for future in depths_futures])

    video = (torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0).to(device)
    depths = torch.from_numpy(depths).float().to(device)
    intrinsics = torch.from_numpy(intrinsics).float().to(device)
    extrinsics = torch.from_numpy(extrinsics).float().to(device)

    if query_point is None:
        support_grid_size = 0
        query_point = get_grid_queries(grid_size=DEFAULT_QUERY_GRID_SIZE, depths=depths, intrinsics=intrinsics, extrinsics=extrinsics)
        logger.info(f"No queries provided, using a grid at the first frame as queries")
    else:
        query_point = torch.from_numpy(query_point).float().to(device)

    return video, depths, intrinsics, extrinsics, query_point, support_grid_size

if __name__ == "__main__":
    setup_logger()
    args = Arguments().parse_args()

    output_dir = Path(args.output_dir) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint)
    model.to(args.device)

    inference_res = (int(model.image_size[0] * np.sqrt(args.resolution_factor)), int(model.image_size[1] * np.sqrt(args.resolution_factor)))
    model.set_image_size(inference_res)

    # Prepare inputs
    video, depths, intrinsics, extrinsics, query_point, support_grid_size = prepare_inputs(
        input_path=args.input_path, 
        inference_res=inference_res, 
        support_grid_size=args.support_grid_size,
        num_threads=args.num_threads,
        device=args.device
    )

    # Run inference
    with torch.autocast("cuda", dtype=torch.bfloat16):
        coords, visibs = inference(
            model=model,
            video=video,
            depths=depths,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            query_point=query_point,
            num_iters=args.num_iters,
            grid_size=support_grid_size,
        )
    
    # Save results
    video = video.cpu().numpy()
    depths = depths.cpu().numpy()
    intrinsics = intrinsics.cpu().numpy()
    extrinsics = extrinsics.cpu().numpy()
    coords = coords.cpu().numpy()
    visibs = visibs.cpu().numpy()
    query_point = query_point.cpu().numpy()

    npz_path = Path(output_dir / Path (args.input_path).name).with_suffix(".result.npz")
    npz_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(
        npz_path,
        video=video,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        coords=coords,
        visibs=visibs,
        query_points=query_point,
    )

    logger.info(f"Results saved to {npz_path.resolve()}.\nTo visualize them, run: `[bold yellow]python visualize.py {shlex.quote(str(npz_path.resolve()))}[/bold yellow]`")
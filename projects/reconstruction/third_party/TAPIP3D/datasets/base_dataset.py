from concurrent.futures import ThreadPoolExecutor
import os
import typing
import cv2
from einops import rearrange, repeat
from hydra import initialize_config_dir, compose
import numpy as np
import logging
import torch
import tap
import time
import av
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Callable, Dict, Literal, Optional, Union, Tuple, List, Any, TypedDict
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from datasets.utils.crop_utils import CropArgs, get_crop_args
from utils.common_utils import setup_logger
from datasets.datatypes import RawSliceData, SliceData
from datasets.providers.base_provider import BaseDataProvider
from datasets.utils.random_utils import WeightedSampler, RandomMapping
from utils.rerun_visualizer import setup_visualizer, log_trajectory, log_video, destroy, save_recording

CONFIG_DIR = Path(__file__).parent.parent / "configs" / "dataset"
logger = logging.getLogger(__name__)

class BaseDataset(Dataset):

    def set_epoch(self, epoch: int) -> None:
        pass
        
    @classmethod
    def from_config(cls, cfg: Union[DictConfig, str], **kwargs) -> 'BaseDataset':
        if isinstance(cfg, str):
            cfg = cls.load_config(cfg)
        assert all (key not in kwargs for key in cfg.keys()), "Overwriting dataset config is not allowed"
        kwargs.update(cfg)

        from datasets.train_dataset import TrainDataset
        from datasets.eval_dataset import EvalDataset
        from datasets.delta_wrapper import DeltaDatasetWrapper
        from datasets.delta_datasets.tapvid2d_dataset import TapVid2DDataset
        if cfg.type == "train":
            dataset_cls = TrainDataset
        elif cfg.type == "eval":
            dataset_cls = EvalDataset
        elif cfg.type == "delta_tapvid2d":
            dataset_cls = lambda **kwargs: DeltaDatasetWrapper(TapVid2DDataset(**kwargs))
        else:
            raise ValueError(f"Unknown dataset type: {cfg.type}")
        kwargs.pop("type")
        return dataset_cls(** kwargs) # type: ignore
    
    @classmethod
    def load_config(cls, cfg: str) -> DictConfig:
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.absolute())):
            cfg_dirs = cfg.split("/")
            cfg = compose(config_name=cfg) # type: ignore
            for key in cfg_dirs[:-1]:
                cfg = cfg[key] # type: ignore
        return typing.cast(DictConfig, cfg)
    
class Arguments(tap.Tap):
    task: str = "visualize"
    config: str = "dr_train"
    idx: Optional[int] = 0
    num_workers: int = 8
    batch_size: int = 1
    annot_mode: Literal["gt", "est"] = "gt"

if __name__ == "__main__":
    setup_logger()
    args = Arguments().parse_args()
    dataset = BaseDataset.from_config(args.config)
    if args.task == "speed_test":
        dataset.set_epoch(0)

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=SliceData.collate
        )
        
        start_time = time.time()
        num_batches = 0
        pbar = tqdm(dataloader, desc="Processing batches", unit="batch")
        for _ in pbar:
            num_batches += 1
            avg_time_per_batch = (time.time() - start_time) / num_batches
            pbar.set_postfix(avg_time=f"{avg_time_per_batch:.4f} seconds")
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_batch = total_time / num_batches
        
        print(f"Average time per batch: {avg_time_per_batch:.4f} seconds")
        print(f"Batches processed per second: {1/avg_time_per_batch:.2f}")
    elif args.task == "visualize":
        dataset.set_epoch(0)
        setup_visualizer(serve=False)

        assert args.idx is not None, "An index must be provided"
        item = dataset[args.idx].with_annot_mode(args.annot_mode)
        # item = item.time_slice(0, 12)

        log_video(
            entity_name=f"video_{args.idx}",
            rgb=item.rgbs,
            intrinsics=item.intrinsics,
            extrinsics=item.extrinsics,
            depth=item.depths,
            # uv_space=True,
        )

        log_trajectory(
            entity_name=f"video_{args.idx}",
            track_name="traj",
            intrinsics=item.intrinsics,
            extrinsics=item.extrinsics,
            trajs=item.trajs_3d,
            visibs=item.visibs,
            valids=item.valids,
            queries=item.query_point,
        )

        save_recording(Path("test.rrd"))
        destroy()

        video_path = Path("demo_rgb.mp4")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(video_path), mode="w")
        stream = container.add_stream('mpeg4', rate=5)
        stream.width = item.rgbs.shape[3]
        stream.height = item.rgbs.shape[2]
        stream.pix_fmt = 'yuv420p'
        
        for rgb in item.rgbs:
            rgb = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            packet = stream.encode(frame)
            container.mux(packet)
        container.close()

        if item.segmentation is not None:
            video_path = Path("demo_segmentation.mp4")
            video_path.parent.mkdir(parents=True, exist_ok=True)
            import numpy as np
            import random

            unique_ids = np.unique(item.segmentation)
            color_map = {id_: [random.randint(0, 255) for _ in range(3)] for id_ in unique_ids}

            container = av.open(str(video_path), mode="w")
            stream = container.add_stream('mpeg4', rate=5)
            stream.width = item.segmentation.shape[2]
            stream.height = item.segmentation.shape[1]
            stream.pix_fmt = 'yuv420p'
            
            for seg in item.segmentation:
                color_image = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
                for id_, color in color_map.items():
                    color_image[seg == id_] = color
                frame = av.VideoFrame.from_ndarray(color_image, format="rgb24")
                packet = stream.encode(frame)
                container.mux(packet)
            container.close()

    else:
        raise ValueError(f"Unknown task: {args.task}")
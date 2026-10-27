from hydra import initialize_config_dir, compose
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import typing
from typing import Any, Dict, Tuple, Union
import torch.nn as nn
import logging
from .point_tracker_3d import PointTracker3D

logger = logging.getLogger(__name__)

CONFIG_ROOT = Path(__file__).parent.parent / "configs" / "model"

def load_config(cfg: Union[DictConfig, str]) -> DictConfig:
    if isinstance(cfg, str):
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT.absolute())):
            cfg = compose(config_name=cfg)
    return typing.cast(DictConfig, cfg)

def from_config(cfg: Union[DictConfig, str], **kwargs) -> nn.Module:
    cfg = load_config(cfg)
    assert all (key not in kwargs for key in cfg.keys()), "Overwriting model config is not allowed"
    kwargs.update(cfg)
    name = kwargs.pop("name")
    if name == "point_tracker_3d_local" or name == "point_tracker_3d":
        return PointTracker3D(**kwargs)
    else:
        raise ValueError(f"Model {cfg.name} not supported")

def from_pretrained(ckpt_path: Union[str, Path]) -> Tuple[nn.Module, DictConfig]:
    assert Path(ckpt_path).exists(), f"Checkpoint {ckpt_path} does not exist"
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "cfg" in ckpt:
        cfg = ckpt["cfg"]
        model = from_config(cfg['model'], image_size=cfg['train_dataset']['resolution']) # checkpoint trained by our code
        model.load_state_dict(ckpt["weight"], strict=True)
    else:
        model, cfg = smart_load(ckpt) # checkpoint trained by others
    # cfg = ckpt_config
    # model = from_config(cfg['model'], image_size=cfg['train_dataset']['resolution']) # checkpoint trained by our code
    # model.load_state_dict(ckpt, strict=True)
    return  model  ,  cfg #

def smart_load(ckpt: Dict[str, Any]) -> Tuple[nn.Module, DictConfig]:
    if "headxz.2.bias" in ckpt:
        logger.info("SpaTracker checkpoint detected")
        cfg = OmegaConf.create({
            "model": {
                "image_size": [384, 512],
                "seq_len": 12,
                "online": True,
            }
        })
        model = SpaTracker(**cfg.model) # type: ignore
        model.model.load_state_dict(ckpt, strict=True)
        return model, cfg
    else:
        return model, cfg
        # raise ValueError("Unsupported checkpoint")

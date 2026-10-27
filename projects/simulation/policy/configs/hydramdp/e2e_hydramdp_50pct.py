"""HydraMDP data scaling: 50% training data."""

import os

SIMENGINE_ROOT = os.getenv("SIMENGINE_ROOT", os.path.abspath("."))

_base_ = ["../navformer/e2e_hydramdp.py"]

nav_filter_path_train = "configs/navsim_splits/navtrain_split/navtrain_50pct.yaml"

data = dict(train=dict(nav_filter_path=nav_filter_path_train))

load_from = os.path.join(
    SIMENGINE_ROOT,
    "data/policy_engine/ckpts/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth",
)

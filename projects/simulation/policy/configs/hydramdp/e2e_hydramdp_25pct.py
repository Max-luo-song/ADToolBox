"""HydraMDP data scaling: 25% training data."""

_base_ = ["./e2e_hydramdp_50pct.py"]

nav_filter_path_train = 'configs/navsim_splits/navtrain_split/navtrain_25pct.yaml'

data = dict(train=dict(nav_filter_path=nav_filter_path_train))

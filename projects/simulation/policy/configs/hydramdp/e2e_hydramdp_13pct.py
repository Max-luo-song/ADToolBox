"""HydraMDP data scaling: 13% training data."""

_base_ = ["./e2e_hydramdp_50pct.py"]

nav_filter_path_train = 'configs/navsim_splits/navtrain_split/navtrain_13pct.yaml'

data = dict(train=dict(nav_filter_path=nav_filter_path_train))

"""Post-training with SimEngine: original and BWM EP rollouts."""

# Training uses the supplied HydraMDP rare-case splits via data.train.finetune_yaml.
# Generate new splits when changing the base model/data; see README.md.

_base_ = ["./e2e_hydramdp_50pct_rlft_rare_rollout.py"]

synthetic_folder_names = [
    "/path/to/synthetic/rollouts/collision",
    "/path/to/synthetic/rollouts/ep_1pct",
    "/path/to/synthetic/rollouts/off_road",
    "/path/to/synthetic/rollouts/collision_augmented",
]

data = dict(train=dict(folder_name=synthetic_folder_names))

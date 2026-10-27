"""RL fine-tuning on rare synthetic replays, excluding real failures."""

# Training uses the supplied HydraMDP rare-case splits via data.train.finetune_yaml.
# Generate new splits when changing the base model/data; see README.md.

_base_ = ["./e2e_hydramdp_50pct_rlft_rare_rollout.py"]

data = dict(train=dict(customized_filter='v2', include_real_failures=False))

"""RL fine-tuning on rare real logs (reward shaping enabled; PG=0.01)."""

# Training uses the supplied HydraMDP rare-case splits via data.train.finetune_yaml.
# Generate new splits when changing the base model/data; see README.md.
# A matching release checkpoint has not yet been selected.

_base_ = ["./e2e_hydramdp_50pct_rlft_common_log.py"]

# Include rare failures and normal scenarios sampled from the same logs.
# normal_ratio=1 preserves the historical rare-log dataset default.
data = dict(train=dict(normal_only=False, normal_ratio=1))

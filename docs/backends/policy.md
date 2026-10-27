# Policy simulation and supervised fine-tuning profile

Install [WorldEngine](https://github.com/OpenDriveLab/WorldEngine) following its own environment, dataset, renderer and model instructions. Backend versions are not yet pinned by a successful clean GPU reproduction; treat this profile as an explicit launcher contract. Private experiment configurations and pre-existing datasets are not distributed.

## Closed-loop inference

Configure `simulation.root`, `simulation.simulation_python`, `simulation.policy_python` and `timeout`. TOOLBOX launches `projects/SimEngine/worldengine/runner/run_simulation.py` and `projects/AlgEngine/closed_loop/sim_test.py` as separate processes with their own environments. Optional `simulation.script` selects a trusted custom shell launcher instead. No hard-coded conda names or user home directories are required by the built-in launcher.

Provide a compatible policy config, checkpoint, dataset label and existing scenario/assets. Optional scenario/asset folders and frame limits are passed explicitly. Each built-in run uses a new output directory; fresh successful runner reports are required. Map annotations and evaluation support are not inferred from a Gaussian scene.

Custom reconstruction-to-simulation conversion is exposed as a separate helper under `extensions/scene_conversion`. It stages edited target Gaussians and camera metadata; it does not create a valid map or complete policy training annotations. A domain-specific scene builder is still required before policy_runner.

## ILFT

Configure `training.root`, `training.python` and `timeout`. Provide config, base checkpoint, valid `ann_file`, token file, empty work directory and epochs. The launcher invokes `projects/AlgEngine/scripts/train.py` with explicit settings. It requires a new checkpoint and records a training log. No reinforcement learning, automatic label generation or automatic post-training score improvement is claimed.

See [validation status](../validation.md) for what has actually run.

# Policy simulation and supervised fine-tuning guide

The simulation stack is bundled under `projects/simulation`:

- `engine/` — closed-loop Gaussian simulation (entry `engine/simengine/runner/run_simulation.py`)
- `policy/` — policy client and ILFT training (entries `policy/closed_loop/sim_test.py`,
  `policy/scripts/train.py`)
- `scripts/` — scene conversion chain: `stage_scene.py`, `build_dynamic_scene.py`,
  `export_policy_training_data.py`

Verify the tree and print the setup checklist:

```bash
python scripts/setup_project.py simulation
```

Prepare the simulation and policy Python environments, checkpoints and any map/metric data your
scenarios need. Private experiment configurations and pre-existing datasets are not distributed.

## Closed-loop inference

Configure `simulation.root` (normally `projects/simulation`), `simulation.simulation_python`,
`simulation.policy_python` and `timeout`. ADToolbox launches
`engine/simengine/runner/run_simulation.py` and `policy/closed_loop/sim_test.py` as separate
processes with their own environments. Optional `simulation.script` selects a trusted custom
shell launcher instead. No hard-coded conda names or user home directories are required by the
built-in launcher.

Provide a compatible policy config, checkpoint, dataset label and existing scenario/assets.
Optional scenario/asset folders and frame limits are passed explicitly. Each built-in run uses a
new output directory; fresh successful runner reports are required. Map annotations and
evaluation support are not inferred from a Gaussian scene.

## Converting an edited reconstruction into a simulation scene

`scripts/convert_scene.py` chains the two bundled helpers:

1. `stage_scene.py` stages edited target Gaussians, normalized camera metadata and timestamps
   from a reconstruction output directory.
2. `build_dynamic_scene.py` clusters the staged dynamic Gaussians, aligns coordinates and writes
   the scenario pickle plus renderer asset the simulation engine consumes.

```bash
python scripts/convert_scene.py \
  --source <reconstruction_output_dir> --staging runs/staging --output <sim_scene_root> \
  --instance-id 0 --scene-id my-scene-000 \
  --translation-scale <externally-calibrated-scale> \
  --python /path/to/scene-builder-env/bin/python
```

The converter is deliberately constrained: it handles edited target tracks and explicit timing.
It does not create a road map, collision-responsibility annotations or training labels.

## ILFT

Configure `training.root`, `training.python` and `timeout`. Provide config, base checkpoint,
valid `ann_file`, token file, empty work directory and epochs. The launcher invokes
`policy/scripts/train.py` with explicit settings. It requires a new checkpoint and records a
training log.

`projects/simulation/scripts/export_policy_training_data.py` converts a simulated OpenScene
rollout into the `training_infos.pkl` + `training_tokens.yaml` pair accepted as `ann_file` and
`token_file`. A mined failure case is not automatically a valid training example; review labels
before training. No reinforcement learning, automatic label generation or automatic post-training
score improvement is claimed.

See [validation status](../validation.md) for what has actually run.

# Simulation and policy training stack

Closed-loop Gaussian simulation and supervised policy fine-tuning, with ADToolbox
scene-conversion additions. Used by the `policy_runner`, `failure_case_filter` and
`policy_trainer` tools.

## Layout

- `engine/` — the simulation engine (`simengine` package). Closed-loop rollout entry:
  `engine/simengine/runner/run_simulation.py`.
- `policy/` — the policy engine. Closed-loop policy client: `policy/closed_loop/sim_test.py`;
  ILFT training entry: `policy/scripts/train.py`; model configs under `policy/configs/`.
- `scripts/` — the ADToolbox scene-conversion and export chain:
  - `stage_scene.py` — stage edited Gaussian target frames and normalized camera metadata
    from a reconstruction output directory.
  - `build_dynamic_scene.py` — cluster staged dynamic Gaussians, align coordinates and write
    the scenario pickle plus renderer asset the engine consumes.
  - `export_policy_training_data.py` — convert a simulated OpenScene rollout into the
    `training_infos.pkl` + `training_tokens.yaml` pair accepted by the trainer.
  - `closed_loop_test.sh`, `multigpu_closed_loop_test.sh` — engine-side launch helpers.
- `docs/` — engine documentation (installation, configuration, data organization).

`scripts/convert_scene.py` at the repository root chains the staging and building steps.

## Environment conventions

- `SIMREPO_ROOT` — this directory (the simulation tree root).
- `SIMENGINE_ROOT` — `engine/`.
- `POLICYENGINE_ROOT` — `policy/`.

The simulation and policy sides run in separate Python environments; the ADToolbox launcher
takes both interpreters from your runtime configuration (`simulation.simulation_python`,
`simulation.policy_python`).

## Setup

```bash
python scripts/setup_project.py simulation   # from the repository root
```

Then prepare the engine environments, checkpoints and map/metric data as described in
`docs/installation.md` and [the policy guide](../../docs/backends/policy.md). Checkpoints and
datasets are not distributed with this repository.

## Attribution

This tree is derived from an upstream Apache-2.0 closed-loop simulation and training framework
and retains its `LICENSE` (also under `licenses/simulation/` at the repository root). The
`policy/` subtree includes mmdetection3d-derived components with their own terms; review all
subcomponents before redistribution. See `THIRD_PARTY_NOTICES.md`. ADToolbox modifications are
marked in the affected source files.

"""Verify a bundled engine tree and print the remaining setup steps for it."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "reconstruction": [
        "projects/reconstruction/inference.py",
        "projects/reconstruction/gsrecon/models/scene_model.py",
        "projects/reconstruction/third_party/TAPIP3D/utils/inference_utils.py",
    ],
    "simulation": [
        "projects/simulation/engine/simengine/runner/run_simulation.py",
        "projects/simulation/policy/closed_loop/sim_test.py",
        "projects/simulation/policy/scripts/train.py",
        "projects/simulation/scripts/stage_scene.py",
        "projects/simulation/scripts/build_dynamic_scene.py",
        "projects/simulation/scripts/export_policy_training_data.py",
    ],
}

NEXT_STEPS = {
    "reconstruction": (
        "Create a dedicated Python environment (see projects/reconstruction/requirements.txt),\n"
        "download the model checkpoints described in projects/reconstruction/README.md,\n"
        "then set reconstruction.root / reconstruction.python / reconstruction.checkpoint\n"
        "in your runtime configuration. See docs/backends/reconstruction.md."
    ),
    "simulation": (
        "Create the simulation and policy Python environments (see docs/backends/policy.md),\n"
        "prepare checkpoints and map/metric data, then set simulation.root, simulation.simulation_python\n"
        "and simulation.policy_python in your runtime configuration."
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", choices=sorted(REQUIRED))
    args = parser.parse_args()
    missing = [path for path in REQUIRED[args.project] if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("Bundled engine is incomplete; missing:\n" + "\n".join(missing))
    print(f"Bundled {args.project} engine tree is complete.")
    print(NEXT_STEPS[args.project])


if __name__ == "__main__":
    main()

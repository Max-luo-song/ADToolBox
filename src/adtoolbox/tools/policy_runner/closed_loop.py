"""Launch simulation and policy clients in separately configured Python environments."""

import os
import signal
import subprocess
import time
from pathlib import Path

from ...core.errors import BackendError, DependencyError
from ...core.io import file_path, positive


def launch(root, settings, data, config_path, checkpoint_path, result_dir):
    if not settings.get("simulation_python") or not settings.get("policy_python"):
        raise DependencyError("Configure simulation.simulation_python and simulation.policy_python")
    sim_root = root / "engine"
    policy_root = root / "policy"
    simulation = file_path(str(sim_root / "simengine/runner/run_simulation.py"), "simulation entry")
    policy = file_path(str(policy_root / "closed_loop/sim_test.py"), "policy client entry")
    for directory in ("plan_traj", "frames", "merged_ann_files", "sim_output"):
        (result_dir / directory).mkdir(parents=True)
    scene_folder = data.get("scenario_data_folder_path") or str(
        root / "data/sim_engine/scenarios/original" / data["data_type"]
    )
    asset_folder = data.get("asset_folder_path") or str(root / "data/sim_engine/assets" / data["data_type"] / "assets")
    if not Path(scene_folder).is_dir() or not Path(asset_folder).is_dir():
        raise DependencyError("Simulation requires existing scenario and asset directories")
    args = {
        "debug_mode": "True",
        "debug_scene_name": "null",
        "data_file_folder_path": str(Path(scene_folder).resolve()),
        "asset_folder_path": str(Path(asset_folder).resolve()),
        "data_pkl_file_name": data.get("scenario_pkl_file_name", "all_scenarios.pkl"),
        "output_dir": str(result_dir / "sim_output"),
        "job_name": "adtoolbox",
        "use_planner_actions": "true",
        "ego_controller": "log_play_controller",
        "ego_policy": "env_input_policy",
        "ego_client": "navformer_client",
        "ego_navigation": "trajectory_navigation",
        "planner_data_path": str(result_dir / "plan_traj"),
        "planner_client_folder": str(result_dir / "frames"),
        "with_metric_manager": str(data.get("with_metric_manager", True)).lower(),
        "visualize_video": "false",
        "visualize_BEV": "false",
        "with_dense_reward_manager": "false",
        "distributed_mode": "SINGLE_NODE",
        "enable_resume": "false",
        "completed_scenarios_dir": str(result_dir / "completed_scenarios"),
    }
    if data.get("react_type", "NR") == "R":
        args.update(agent_policy="idm_policy", agent_navigation="idm_navigation")
    for key in ("num_scenarios", "max_frame_per_scene"):
        if key in data:
            args[key] = str(data[key])
    sim_command = [str(settings["simulation_python"]), str(simulation)] + [f"{k}={v}" for k, v in args.items()]
    policy_command = [
        str(settings["policy_python"]),
        str(policy),
        str(config_path),
        str(checkpoint_path),
        "--log-dir",
        str(result_dir / "sim_output"),
        "--cfg-options",
        f"sim.monitored_folder={result_dir / 'frames'}",
        f"sim.plan_save_path={result_dir / 'plan_traj'}",
        f"sim.merged_ann_save_dir={result_dir / 'merged_ann_files'}",
        "sim.clean_temp_files=True",
        "sim.clean_record_data=False",
        f"data_root={result_dir / 'sim_output/openscene_format'}/",
    ]
    env = os.environ.copy()
    env["SIMREPO_ROOT"] = str(root)
    env["PYTHONPATH"] = os.pathsep.join([str(sim_root), str(policy_root), env.get("PYTHONPATH") or str(root)])
    deadline = time.monotonic() + positive(settings.get("timeout", 86400), "simulation.timeout")
    processes = []
    try:
        with (
            (result_dir / "simulation.log").open("w") as sim_log,
            (result_dir / "policy.log").open("w") as policy_log,
        ):
            for command, cwd, log in [
                (sim_command, sim_root, sim_log),
                (policy_command, policy_root, policy_log),
            ]:
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=cwd,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except FileNotFoundError:
                    raise DependencyError("Configured backend Python executable does not exist") from None
                processes.append(process)
            while any(process.poll() is None for process in processes):
                if any(process.poll() not in (None, 0) for process in processes):
                    raise BackendError(f"Closed-loop process failed; inspect logs in {result_dir}")
                if time.monotonic() > deadline:
                    raise BackendError(f"Closed-loop run timed out; inspect logs in {result_dir}")
                time.sleep(0.2)
            if any(process.returncode != 0 for process in processes):
                raise BackendError(f"Closed-loop process failed; inspect logs in {result_dir}")
    finally:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass

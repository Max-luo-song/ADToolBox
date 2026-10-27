import os
import re
from datetime import datetime, timezone

from ...backends.process import invoke, root_settings
from ...core.errors import BackendError, InputError
from ...core.io import file_path, read_json


def _label(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise InputError("Model and dataset labels may only contain letters, numbers, underscores and hyphens")
    return value


def evaluate(data, config):
    root, settings = root_settings(config, "simulation", ["root"])
    cfg = file_path(data["config_path"], "config_path")
    checkpoint = file_path(data["checkpoint_path"], "checkpoint_path")
    model, dataset = _label(data["model_name"]), _label(data["data_type"])
    react = data.get("react_type", "NR")
    env = os.environ.copy()
    env["SIMREPO_ROOT"] = str(root)  # Engine-side wire protocol read by the bundled simulation engine.
    env["PYTHONPATH"] = env.get("PYTHONPATH") or str(root)
    for key, target in {
        "scenario_data_folder_path": "SCENARIO_DATA_FOLDER_PATH",
        "asset_folder_path": "SCENARIO_ASSET_FOLDER_PATH",
        "scenario_pkl_file_name": "SCENARIO_PKL_FILE_NAME",
        "num_scenarios": "NUM_SCENARIOS",
        "max_frame_per_scene": "MAX_FRAME_PER_SCENE",
        "with_metric_manager": "WITH_METRIC_MANAGER",
    }.items():
        if key in data:
            env[target] = str(data[key]).lower() if isinstance(data[key], bool) else str(data[key])
    report_dir = root / "experiments/closed_loop_exps" / model / f"{dataset}_{react}"
    # Ignore pre-existing reports: a successful process must generate fresh evidence.
    previous = (
        {p: p.stat().st_mtime_ns for p in report_dir.rglob("runner_report_*.json")} if report_dir.exists() else {}
    )
    if settings.get("script"):
        script = file_path(settings["script"], "simulation script")
        invoke(
            ["bash", str(script), str(cfg), str(checkpoint), model, dataset, react, dataset, "adtoolbox"],
            cwd=root,
            env=env,
            timeout=settings.get("timeout", 86400),
        )
    else:
        from .closed_loop import launch

        report_dir = root / "experiments/adtoolbox" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        launch(root, settings, data, cfg, checkpoint, report_dir)
    reports = [
        p for p in report_dir.rglob("runner_report_*.json") if p not in previous or p.stat().st_mtime_ns != previous[p]
    ]
    if not reports:
        raise BackendError("Simulation produced no fresh runner report")
    entries = []
    for path in reports:
        value = read_json(path)
        entries.extend(value if isinstance(value, list) else [value])
    if not entries or any(not isinstance(entry, dict) or entry.get("succeeded") is not True for entry in entries):
        raise BackendError("A runner report contains an unsuccessful scenario")
    return {
        "policy_result": "completed",
        "result_dir": str(report_dir),
        "reports": entries,
        "_artifacts": [{"kind": "runner_report", "path": str(p)} for p in reports],
        "_warnings": ["Metrics are available only when the backend configuration and scene annotations support them."],
    }

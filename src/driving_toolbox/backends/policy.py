import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from ..core.errors import BackendError, InputError
from ..core.io import file_path, read_json
from .process import invoke, root_settings


def _label(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise InputError(
            "Model and dataset labels may only contain letters, numbers, underscores and hyphens"
        )
    return value


def evaluate(data, config):
    root, settings = root_settings(config, "simulation", ["root"])
    cfg = file_path(data["config_path"], "config_path")
    checkpoint = file_path(data["checkpoint_path"], "checkpoint_path")
    model, dataset = _label(data["model_name"]), _label(data["data_type"])
    react = data.get("react_type", "NR")
    env = os.environ.copy()
    env["WORLDENGINE_ROOT"] = str(root)  # External wire protocol; public configuration stays functional.
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
        {p: p.stat().st_mtime_ns for p in report_dir.rglob("runner_report_*.json")}
        if report_dir.exists()
        else {}
    )
    if settings.get("script"):
        script = file_path(settings["script"], "simulation script")
        invoke(
            ["bash", str(script), str(cfg), str(checkpoint), model, dataset, react, dataset, "toolbox"],
            cwd=root,
            env=env,
            timeout=settings.get("timeout", 86400),
        )
    else:
        from .closed_loop import launch

        report_dir = root / "experiments/toolbox" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        launch(root, settings, data, cfg, checkpoint, report_dir)
    reports = [
        p
        for p in report_dir.rglob("runner_report_*.json")
        if p not in previous or p.stat().st_mtime_ns != previous[p]
    ]
    if not reports:
        raise BackendError("Simulation produced no fresh runner report")
    entries = []
    for path in reports:
        value = read_json(path)
        entries.extend(value if isinstance(value, list) else [value])
    if not entries or any(
        not isinstance(entry, dict) or entry.get("succeeded") is not True for entry in entries
    ):
        raise BackendError("A runner report contains an unsuccessful scenario")
    return {
        "policy_result": "completed",
        "result_dir": str(report_dir),
        "reports": entries,
        "_artifacts": [{"kind": "runner_report", "path": str(p)} for p in reports],
        "_warnings": [
            "Metrics are available only when the backend configuration and scene annotations support them."
        ],
    }


def train(data, config):
    root, settings = root_settings(config, "training", ["root", "python"])
    script = file_path(str(root / "projects/AlgEngine/scripts/train.py"), "training script")
    paths = {
        key: file_path(data[key], key) for key in ("config_path", "base_checkpoint", "ann_file", "token_file")
    }
    work = Path(data["work_dir"]).expanduser().resolve()
    if work.exists() and any(work.iterdir()):
        raise InputError("work_dir must be empty to avoid mixing checkpoints from earlier training")
    work.mkdir(parents=True, exist_ok=True)
    epochs = data.get("epochs", 1)
    command = [
        str(settings["python"]),
        str(script),
        str(paths["config_path"]),
        "--work-dir",
        str(work),
        "--no-validate",
        "--cfg-options",
        f"load_from={paths['base_checkpoint']}",
        f"total_epochs={epochs}",
        f"runner.max_epochs={epochs}",
        f"data.train.ann_file={paths['ann_file']}",
        f"data.train.nav_filter_path={paths['token_file']}",
        "data.train.finetune_yaml=" + json.dumps([str(paths["token_file"])]),
    ]
    env = os.environ.copy()
    env["WORLDENGINE_ROOT"] = str(root)
    env["PYTHONPATH"] = str(root / "projects/AlgEngine") + os.pathsep + env.get("PYTHONPATH", str(root))
    process = invoke(
        command, cwd=root, env=env, timeout=settings.get("timeout", 86400), log_path=work / "training.log"
    )
    (work / "training.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    checkpoints = sorted(work.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime_ns)
    if not checkpoints:
        raise BackendError("Training finished without producing an epoch checkpoint")
    return {
        "training_mode": "ilft",
        "checkpoint": str(checkpoints[-1]),
        "epochs": epochs,
        "work_dir": str(work),
        "_artifacts": [{"kind": "checkpoint", "path": str(checkpoints[-1])}],
    }

import json
import os
from pathlib import Path

from ...backends.process import invoke, root_settings
from ...core.errors import BackendError, InputError
from ...core.io import file_path


def train(data, config):
    root, settings = root_settings(config, "training", ["root", "python"])
    script = file_path(str(root / "policy/scripts/train.py"), "training script")
    paths = {key: file_path(data[key], key) for key in ("config_path", "base_checkpoint", "ann_file", "token_file")}
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
    env["SIMREPO_ROOT"] = str(root)
    env["PYTHONPATH"] = str(root / "policy") + os.pathsep + env.get("PYTHONPATH", str(root))
    process = invoke(command, cwd=root, env=env, timeout=settings.get("timeout", 86400), log_path=work / "training.log")
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

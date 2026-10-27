import json
import subprocess
from pathlib import Path

from ..core.errors import BackendError, DependencyError, InputError


def invoke(command, *, cwd=None, payload=None, timeout=3600, env=None, log_path=None):
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise InputError("Backend command must be a non-empty array of strings")
    if not 0 < float(timeout) < float("inf"):
        raise InputError("Backend timeout must be finite and positive")
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            input=json.dumps(payload, allow_nan=False) if payload is not None else None,
            capture_output=True,
            text=True,
            timeout=float(timeout),
            check=False,
        )
    except FileNotFoundError:
        raise DependencyError("Backend executable not found; run adtoolbox doctor") from None
    except subprocess.TimeoutExpired:
        if log_path is not None:
            Path(log_path).write_text("Backend timed out\n", encoding="utf-8")
        raise BackendError("Backend timed out") from None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        # Backend logs may contain credentials; detailed diagnostics belong in the user's runtime.
        raise BackendError(f"Backend exited with code {result.returncode}; inspect its local logs")
    return result


def configured(tool, data, config):
    settings = config.get("backends", {}).get(tool)
    if not settings or not settings.get("command"):
        return None
    result = invoke(settings["command"], cwd=settings.get("cwd"), payload=data, timeout=settings.get("timeout", 3600))
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise BackendError("Backend stdout must contain exactly one JSON object") from None
    if not isinstance(value, dict) or value.get("status") != "success" or not isinstance(value.get("result"), dict):
        raise BackendError("Backend must return status=success and an object result")
    return value["result"]


def root_settings(config, name, required):
    from ..core.projects import project_settings

    project_name = "simulation" if name == "training" else name
    settings = {**project_settings(project_name, config), **config.get(name, {})}
    missing = [key for key in required if not settings.get(key)]
    if missing:
        raise DependencyError(f"Configure {name}: {', '.join(missing)}; see docs/backends/README.md")
    root = Path(settings["root"]).expanduser().resolve()
    if not root.is_dir():
        raise DependencyError(f"{name}.root is not an installed backend directory")
    return root, settings

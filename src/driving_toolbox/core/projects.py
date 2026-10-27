"""Resolve included compute projects independently from the private application."""

import json
import os
import sys
from pathlib import Path

from ..backends.process import invoke
from .errors import BackendError, DependencyError


def project_settings(name, config):
    section = config.get(name, {})
    projects = (
        Path(
            config.get("projects_root")
            or os.environ.get("TOOLBOX_PROJECTS_DIR")
            or Path(__file__).resolve().parents[3] / "projects"
        )
        .expanduser()
        .resolve()
    )
    return {"root": str(projects / name), **section}


def run_project(name, payload, config, extra_env=None):
    settings = project_settings(name, config)
    root = Path(settings["root"]).expanduser().resolve()
    bridge = root / "bridge.py"
    if not bridge.is_file():
        raise DependencyError(
            f"Missing included project: {name}. Use the full repository checkout or configure {name}.root"
        )
    env = os.environ.copy()
    env.update(extra_env or {})
    result = invoke(
        [str(settings.get("python", sys.executable)), str(bridge)],
        cwd=root,
        payload=payload,
        env=env,
        timeout=settings.get("timeout", 1800),
    )
    try:
        response = json.loads(result.stdout)
    except ValueError:
        raise BackendError(f"{name} returned invalid JSON") from None
    if response.get("status") != "success":
        raise BackendError(f"{name} did not complete successfully")
    return response["result"]

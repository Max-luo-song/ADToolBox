"""Run the included editing project and retain its executable renderer config."""

import os
from pathlib import Path

from ...core.errors import InputError
from ...core.io import read_json
from ...core.projects import project_settings, run_project


def generate(data, config):
    if not data.get("output_path"):
        raise InputError("Natural-language editing requires output_path")
    settings = config.get("llm", {})
    env = {"ADTOOLBOX_EDITING_ROOT": project_settings("scene_editing", config)["root"]}
    for target, value in {
        "OPENAI_API_KEY": os.environ.get(settings.get("api_key_env", "ADTOOLBOX_API_KEY")),
        "OPENAI_BASE_URL": settings.get("base_url") or os.environ.get("ADTOOLBOX_BASE_URL"),
        "OPENAI_MODEL": settings.get("model") or os.environ.get("ADTOOLBOX_MODEL"),
    }.items():
        if value:
            env[target] = str(value)
    payload = {
        "edit_request": data["edit_plan"],
        "output_path": str(Path(data["output_path"]).expanduser().resolve()),
        "instance_info_path": data.get("instance_info_path"),
        "asset_path": data.get("asset_path"),
        "asset_scale": data.get("asset_scale"),
        "frame_count": data.get("frame_count", 40),
        "output_kind": data.get("output_kind", "rigid_config"),
    }
    result = run_project("scene_editing", payload, config, env)
    result["backend_config_path"] = result["saved_path"]
    result["edit_config"] = read_json(result["saved_path"])
    return result

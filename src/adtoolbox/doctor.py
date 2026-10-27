import importlib.util
import os
import shutil
from pathlib import Path


def diagnose(config):
    checks = {
        "core": "ready",
        "llm": all(
            os.environ.get(key) or config.get("llm", {}).get(field)
            for key, field in [("ADTOOLBOX_BASE_URL", "base_url"), ("ADTOOLBOX_MODEL", "model")]
        )
        and bool(os.environ.get(config.get("llm", {}).get("api_key_env", "ADTOOLBOX_API_KEY"))),
    }
    for module in ("mcp", "cv2", "ultralytics"):
        checks[module] = importlib.util.find_spec(module) is not None
    for binary in ("ffmpeg", "ffprobe"):
        checks[binary] = shutil.which(binary) is not None
    for name in ("reconstruction", "simulation", "training"):
        settings = config.get(name, {})
        checks[name] = {
            "configured": bool(settings),
            "root_exists": Path(settings["root"]).expanduser().is_dir() if settings.get("root") else False,
        }
    return {"checks": checks, "note": "Readiness is not proof of a successful GPU inference or training run."}

import json
import math
from pathlib import Path

from .errors import InputError


def read_json(path):
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return str(path)


def file_path(value, label):
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{label} must be a non-empty path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise InputError(f"{label} does not exist: {path}")
    return path


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise InputError(f"{label} must be a finite positive number")
    return value

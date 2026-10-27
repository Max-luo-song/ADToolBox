import shutil
import tempfile
from pathlib import Path

from ...backends.process import invoke
from ...core.errors import DependencyError
from ...core.io import read_json
from ...core.llm import ask_json


def check(data, config):
    issues = []
    video = Path(data["rendered_video"]).expanduser()
    for key in ("rendered_video", "scene_ply", "scene_config"):
        if data.get(key):
            path = Path(data[key]).expanduser()
            if not path.is_file() or path.stat().st_size == 0:
                issues.append(f"Missing or empty {key}")
    if data.get("scene_config") and Path(data["scene_config"]).expanduser().is_file():
        try:
            read_json(data["scene_config"])
        except ValueError:
            issues.append("Invalid edit configuration JSON")
    artifacts_valid = not issues
    visual = None
    if data.get("visual_check") and not issues:
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise DependencyError("Visual checks need ffmpeg and ffprobe on PATH")
        duration = float(
            invoke([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)]).stdout
        )
        count = min(data.get("frame_count", 8), 32)
        with tempfile.TemporaryDirectory(prefix="adtoolbox-frames-") as temporary:
            images = []
            for i in range(count):
                output = Path(temporary) / f"{i:03d}.jpg"
                invoke(
                    [
                        ffmpeg,
                        "-y",
                        "-loglevel",
                        "error",
                        "-ss",
                        str(duration * (i + 0.5) / count),
                        "-i",
                        str(video),
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale=640:-2",
                        str(output),
                    ]
                )
                images.append(output)
            schema = {
                "type": "object",
                "required": ["requirement_consistency", "visual_valid", "evidence", "issues"],
                "properties": {
                    "requirement_consistency": {"type": "boolean"},
                    "visual_valid": {"type": "boolean"},
                    "evidence": {"type": "array"},
                    "issues": {"type": "array"},
                },
            }
            visual = ask_json(
                "Inspect sampled rendered driving frames against the requirement. Cite frame numbers. "
                "You cannot certify physical validity, collision responsibility or unsampled continuity.",
                {"requirement": data["scene_requirement"]},
                schema,
                config,
                images,
            )
            issues.extend(str(x) for x in visual["issues"])
    passed = not issues and visual is not None and visual["requirement_consistency"] and visual["visual_valid"]
    return {
        "passed": bool(passed),
        "artifacts_valid": artifacts_valid,
        "visual_evaluation": visual,
        "issues": issues,
        "physics_verified": False,
        "geometry_verified": False,
        "_warnings": [
            "File checks alone do not establish requirement consistency. Physics and geometry remain unverified."
        ],
    }

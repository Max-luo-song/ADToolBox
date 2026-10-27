# Adapted from adagent contributors (2026), Apache-2.0. Modified for explicit timing and standalone use.
"""Stage one edited Gaussian instance for a separately installed simulation scene builder."""

from __future__ import annotations

import argparse
import math
import json
import shutil
from pathlib import Path

def _normalized_camera(scene_dir: Path, frame_count: int) -> tuple[dict, str]:
    """Use rendered ego poses and convert them into normalized PLY coordinates."""
    ply_path = scene_dir / "ply_camera_params.json"
    ply_camera = json.loads(ply_path.read_text(encoding="utf-8"))
    frames = ply_camera.get("frames") or []
    if len(frames) < frame_count:
        raise ValueError("fewer PLY camera poses than edited instance frames")
    render_path = scene_dir / "camera_params.json"
    if not render_path.is_file():
        ply_camera["frames"] = frames[:frame_count]
        return ply_camera, "ply_camera_params.json"

    render_camera = json.loads(render_path.read_text(encoding="utf-8"))
    render_frames = render_camera.get("frames") or []
    if len(render_frames) < frame_count:
        raise ValueError("fewer rendered camera poses than edited instance frames")
    transform = ply_camera.get("ply_world_transform") or {}
    center = transform.get("center")
    scale = float(transform.get("scale", 0.0))
    if not isinstance(center, list) or len(center) != 3 or any(not isinstance(value, (int, float)) for value in center) or scale <= 0:
        raise ValueError("ply camera parameters have invalid center/scale")
    normalized_frames = []
    for frame in render_frames[:frame_count]:
        pose = frame["world_to_camera_opencv"]
        if not isinstance(pose, list) or len(pose) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in pose):
            raise ValueError("rendered camera pose must be a 4x4 matrix")
        normalized = [list(row) for row in pose]
        for row in range(3):
            normalized[row][3] = (float(pose[row][3]) + sum(float(pose[row][axis]) * float(center[axis]) for axis in range(3))) / scale
        normalized_frames.append({
            "frame_index": int(frame["frame_index"]),
            "intrinsics": frame["intrinsics"],
            "world_to_camera_opencv": normalized,
        })
    ply_camera["frames"] = normalized_frames
    ply_camera["camera_source"] = "camera_params.json (rendered ego poses) normalized with ply_world_transform"
    return ply_camera, "camera_params.json"


def prepare(source: Path, destination: Path, instance_id: int = 0, source_fps: float | None = None) -> dict:
    source = source.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Destination must be empty")
    destination = destination.resolve()
    instance_dir = source / "001" / "target_object" / f"instance_{instance_id:04d}"
    targets = sorted(instance_dir.glob("target_*.ply"))
    if not targets:
        raise FileNotFoundError(f"no edited instance frames in {instance_dir}")
    indices = [int(path.stem.rsplit("_", 1)[-1]) for path in targets]
    if indices != list(range(len(targets))):
        raise ValueError("instance frames must be contiguous from frame zero")

    camera, camera_source = _normalized_camera(source / "001", len(targets))

    manifest_path = source / "point_clouds" / "dynamic_frames_001.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if [item["frame_index"] for item in manifest[: len(targets)]] != indices:
            raise ValueError("instance frames and dynamic timestamps do not match")
        manifest_source = "dynamic_frames_001.json"
    else:
        if source_fps is None or not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Missing frame manifest: provide explicit positive source_fps")
        manifest = [{"frame_index": index, "source_timestamp_seconds": index / source_fps,
                     "timestamp": index / source_fps} for index in indices]
        manifest_source = "explicit_source_fps"

    image_dir = destination / "001"
    point_dir = destination / "point_clouds"
    dynamic_dir = point_dir / "dynamic_frames_001"
    image_dir.mkdir(parents=True, exist_ok=True)
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    for index, target in enumerate(targets):
        shutil.copy2(source / "001" / f"view_{index}.png", image_dir / f"view_{index}.png")
        shutil.copy2(target, dynamic_dir / f"dynamic_{index:03d}.ply")
    shutil.copy2(source / "point_clouds" / "static_sky_001.ply", point_dir / "static_sky_001.ply")
    (image_dir / "ply_camera_params.json").write_text(json.dumps(camera, indent=2), encoding="utf-8")
    staged_manifest = [
        {"frame_index": index, "timestamp": manifest[index].get("source_timestamp_seconds", manifest[index]["timestamp"]),
         "file": f"dynamic_{index:03d}.ply"}
        for index in indices
    ]
    (point_dir / "dynamic_frames_001.json").write_text(
        json.dumps(staged_manifest, indent=2), encoding="utf-8"
    )
    return {"source": str(source), "destination": str(destination), "instance_id": instance_id,
            "frame_count": len(targets), "camera_source": camera_source,
            "manifest_source": manifest_source,
            "expert_ego_trajectory": camera_source == "camera_params.json"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, default=0)
    parser.add_argument("--source-fps", type=float)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination, args.instance_id, args.source_fps), indent=2))


if __name__ == "__main__":
    main()

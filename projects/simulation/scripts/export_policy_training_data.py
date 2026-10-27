#!/usr/bin/env python3
"""Convert a simulated OpenScene rollout into supervised fine-tuning data."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import yaml


def _load_infos(meta_dir: Path) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for path in sorted(meta_dir.glob("*.pkl")):
        with path.open("rb") as file:
            payload = pickle.load(file)
        if isinstance(payload, dict):
            payload = payload.get("infos", [payload])
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a list of frame dictionaries")
        infos.extend(payload)
    if not infos:
        raise FileNotFoundError(f"no metadata pickle files found under {meta_dir}")
    return infos


def _resolve_image_paths(infos: list[dict[str, Any]], sensor_root: Path) -> None:
    for info in infos:
        for camera in info.get("cams", {}).values():
            data_path = Path(str(camera["data_path"]))
            if not data_path.is_absolute():
                data_path = sensor_root / data_path
            data_path = data_path.resolve()
            if not data_path.is_file():
                raise FileNotFoundError(f"camera image does not exist: {data_path}")
            camera["data_path"] = str(data_path)


def _normalize_infos(infos: list[dict[str, Any]]) -> None:
    infos.sort(key=lambda item: (str(item.get("log_token", "")), int(item["frame_idx"])))
    for info in infos:
        token = str(info["token"])
        log_token = str(info.get("log_token") or info.get("scene_token") or "scene")
        info["token"] = token
        info["log_token"] = log_token
        info["scene_token"] = log_token
        info["log_name"] = str(info.get("log_name") or log_token)
        info["scene_name"] = str(info.get("scene_name") or info["log_name"])
        info.setdefault("sample_prev", None)
        info.setdefault("sample_next", None)
        info.setdefault("map_location", "synthetic")
        info.setdefault("lidar_path", None)


def _write_tokens(infos: list[dict[str, Any]], path: Path) -> None:
    tokens = sorted({str(info["token"]) for info in infos})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump({"tokens": tokens}, file, sort_keys=False)


def convert(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    openscene_dir = input_dir / "openscene_format"
    if openscene_dir.is_dir():
        input_dir = openscene_dir
    meta_dir = input_dir / "meta_datas"
    sensor_root = input_dir / "sensor_blobs"
    if not sensor_root.is_dir():
        raise FileNotFoundError(f"sensor_blobs directory does not exist: {sensor_root}")

    infos = _load_infos(meta_dir)
    _resolve_image_paths(infos, sensor_root)
    _normalize_infos(infos)

    output_dir.mkdir(parents=True, exist_ok=True)
    infos_path = output_dir / "training_infos.pkl"
    with infos_path.open("wb") as file:
        pickle.dump(infos, file, protocol=pickle.HIGHEST_PROTOCOL)

    tokens_path = output_dir / "training_tokens.yaml"
    _write_tokens(infos, tokens_path)
    return {
        "infos_path": str(infos_path),
        "tokens_path": str(tokens_path),
        "frames": len(infos),
        "scenes": len({str(info["log_token"]) for info in infos}),
        "cameras": len(infos[0].get("cams", {})),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = convert(args.input_dir.expanduser().resolve(), args.output_dir.expanduser().resolve())
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

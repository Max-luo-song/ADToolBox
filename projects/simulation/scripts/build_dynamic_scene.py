# Modified by ADToolbox contributors, 2026: standalone simulation and scene conversion support.
#!/usr/bin/env python3
"""Build a SimEngine scene with the dynamic-Gaussian lead car as an agent."""

from __future__ import annotations

import argparse
import json
import pickle
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from scene_conversion_common import (
    CAMERA_NAMES,
    CAMERA_TO_EGO_ROTATION,
    PACIFICA_REAR_AXLE_TO_CENTER,
    align_reconstruction,
    camera_positions_to_vehicle_centers,
    make_asset,
    make_frame,
    read_3dgs_ply,
    rotation_matrix_from_quaternion,
    rotation_matrix_to_quaternion,
)


FRONT_CAR_ID = "frontcar"
STRAIGHT_DRIVING_COMMAND = np.array([0, 1, 0, 0], dtype=np.int64)

# Positive yaw turns left in SimEngine forward-left-up coordinates.
CAMERA_YAWS_DEGREES = {
    "CAM_F0": 0.0,
    "CAM_L0": 45.0,
    "CAM_L1": 90.0,
    "CAM_L2": 135.0,
    "CAM_B0": 180.0,
    "CAM_R2": -135.0,
    "CAM_R1": -90.0,
    "CAM_R0": -45.0,
}


def make_surround_cameras(
    width: int, height: int, intrinsic: np.ndarray,
) -> dict[str, Any]:
    """Build a co-located eight-camera ring with 45-degree yaw spacing."""
    cameras = {}
    for name in CAMERA_NAMES:
        yaw = np.deg2rad(CAMERA_YAWS_DEGREES[name])
        yaw_rotation = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        sensor2ego = yaw_rotation @ CAMERA_TO_EGO_ROTATION
        cameras[name] = {
            "channel": name,
            "sensor2ego_rotation": rotation_matrix_to_quaternion(sensor2ego),
            "sensor2ego_translation": np.zeros(3),
            "intrinsic": intrinsic.copy(),
            "distortion": np.zeros(5),
            "height": height,
            "width": width,
        }
    return cameras


def make_surround_frame(
    token: str, scene_id: str, frame_idx: int, cameras: dict[str, Any],
    intrinsic: np.ndarray, ego2global: np.ndarray, frame_period: float,
) -> dict[str, Any]:
    """Build an OpenScene frame whose calibration matches the renderer ring."""
    frame = make_frame(
        token, scene_id, frame_idx, cameras, intrinsic, ego2global, frame_period
    )
    frame["driving_command"] = STRAIGHT_DRIVING_COMMAND.copy()
    for name, camera in cameras.items():
        frame["cams"][name]["sensor2lidar_rotation"] = (
            rotation_matrix_from_quaternion(camera["sensor2ego_rotation"])
        )
        frame["cams"][name]["sensor2lidar_translation"] = np.asarray(
            camera["sensor2ego_translation"], dtype=np.float64
        )
    return frame


def load_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        signature = file.read(24)
    if len(signature) != 24 or signature[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    return struct.unpack(">II", signature[16:24])


def numbered_images(source: Path) -> list[Path]:
    paths = list((source / "001").glob("view_*.png"))
    paths.sort(key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
    if not paths:
        raise FileNotFoundError(f"no view_*.png below {source / '001'}")
    indices = [int(path.stem.rsplit("_", 1)[-1]) for path in paths]
    if indices != list(range(len(paths))):
        raise ValueError("source image indices are not contiguous from zero")
    return paths


def load_camera_parameters(path: Path, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as file:
        params = json.load(file)
    if "OpenCV" not in params.get("coordinate_convention", ""):
        raise ValueError(f"{path} does not declare the OpenCV convention")
    frames = sorted(params["frames"], key=lambda frame: frame["frame_index"])
    if [frame["frame_index"] for frame in frames] != list(range(frame_count)):
        raise ValueError("camera parameter frames do not match source images")
    intrinsics = np.asarray([frame["intrinsics"] for frame in frames], dtype=np.float64)
    world_to_camera = np.asarray(
        [frame["world_to_camera_opencv"] for frame in frames], dtype=np.float64
    )
    if intrinsics.shape != (frame_count, 3, 3) or world_to_camera.shape != (frame_count, 4, 4):
        raise ValueError("unexpected camera parameter shape")
    camera_to_world = np.linalg.inv(world_to_camera)
    if not np.allclose(world_to_camera @ camera_to_world, np.eye(4), atol=2e-5):
        raise ValueError("invalid world-to-camera matrices")
    return intrinsics, camera_to_world


def load_dynamic_manifest(source: Path, frame_count: int) -> tuple[list[Path], np.ndarray]:
    manifest_path = source / "point_clouds" / "dynamic_frames_001.json"
    with manifest_path.open(encoding="utf-8") as file:
        records = sorted(json.load(file), key=lambda record: record["frame_index"])
    if [record["frame_index"] for record in records] != list(range(frame_count)):
        raise ValueError("dynamic Gaussian frames do not match source images")
    timestamps = np.asarray([record["timestamp"] for record in records], dtype=np.float64)
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("dynamic frame timestamps must be increasing")
    frame_dir = source / "point_clouds" / "dynamic_frames_001"
    paths = [frame_dir / record["file"] for record in records]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing dynamic frames: {missing}")
    return paths, timestamps


def largest_gaussian_component(
    gaussians: dict[str, np.ndarray], opacity_threshold: float, cluster_eps: float,
    cluster_min_samples: int,
) -> np.ndarray:
    opacity = 1.0 / (1.0 + np.exp(-np.clip(gaussians["opacities"], -30.0, 30.0)))
    visible_indices = np.flatnonzero(opacity >= opacity_threshold)
    if len(visible_indices) < cluster_min_samples:
        raise ValueError("too few visible dynamic Gaussians")
    labels = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples).fit_predict(
        gaussians["means"][visible_indices]
    )
    component_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if not len(component_labels):
        raise ValueError("DBSCAN found no dynamic Gaussian component")
    return visible_indices[labels == component_labels[np.argmax(counts)]]


def smooth_track(points: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return points.copy()
    if window % 2 == 0:
        raise ValueError("smoothing window must be odd")
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.median(padded[idx:idx + window], axis=0) for idx in range(len(points))])


def extract_front_car(
    paths: list[Path], source_poses: np.ndarray, translation_scale: float,
    opacity_threshold: float, cluster_eps: float, cluster_min_samples: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    world_rotation = CAMERA_TO_EGO_ROTATION @ source_poses[0, :3, :3].T
    source_origin = source_poses[0, :3, 3]
    frame_data: list[tuple[dict[str, np.ndarray], np.ndarray]] = []
    source_centers = []
    diagnostics = []

    for frame_idx, path in enumerate(paths):
        gaussians = read_3dgs_ply(path, max_points=10_000_000)
        component = largest_gaussian_component(
            gaussians, opacity_threshold, cluster_eps, cluster_min_samples
        )
        component_points = gaussians["means"][component]
        center = np.median(component_points, axis=0)
        source_centers.append(center)
        frame_data.append((gaussians, component))
        diagnostics.append({
            "frame_index": frame_idx,
            "source_file": path.name,
            "source_gaussian_count": int(len(gaussians["means"])),
            "front_car_gaussian_count": int(len(component)),
            "source_center": center.astype(float).tolist(),
        })

    source_centers_array = smooth_track(np.asarray(source_centers, dtype=np.float64))
    world_centers = (
        (source_centers_array - source_origin) @ world_rotation.T * translation_scale
    )
    derivatives = np.gradient(world_centers[:, :2], axis=0)
    headings = np.unwrap(np.arctan2(derivatives[:, 1], derivatives[:, 0]))

    reference_idx = int(np.argmax([len(component) for _, component in frame_data]))
    reference_gaussians, component = frame_data[reference_idx]
    reference = {key: value[component].copy() for key, value in reference_gaussians.items()}
    reference_world_center = world_centers[reference_idx]
    reference_heading = headings[reference_idx]
    cos_heading, sin_heading = np.cos(reference_heading), np.sin(reference_heading)
    object_to_world_rotation = np.array(
        [[cos_heading, -sin_heading, 0.0], [sin_heading, cos_heading, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    world_means = (
        (reference["means"].astype(np.float64) - source_origin) @ world_rotation.T
        * translation_scale
    )
    reference["means"] = (
        (world_means - reference_world_center) @ object_to_world_rotation
    ).astype(np.float32)
    source_rotations = np.asarray(
        [rotation_matrix_from_quaternion(quat) for quat in reference["quats"]]
    )
    world_rotations = np.einsum("ij,njk->nik", world_rotation, source_rotations)
    local_rotations = np.einsum("ij,njk->nik", object_to_world_rotation.T, world_rotations)
    reference["quats"] = np.asarray(
        [rotation_matrix_to_quaternion(rotation) for rotation in local_rotations], dtype=np.float32
    )

    local_low, local_high = np.quantile(reference["means"], [0.05, 0.95], axis=0)
    dimensions = local_high - local_low
    for item, center in zip(diagnostics, world_centers):
        item["simengine_center_metres"] = center.astype(float).tolist()
    return reference, world_centers, headings, dimensions, diagnostics


def gaussian_state_dict(
    gaussians: dict[str, np.ndarray], translation_scale: float,
) -> dict[str, torch.Tensor]:
    sh_c0 = 0.28209479177387814
    colors = np.clip(gaussians["f_dc"] * sh_c0 + 0.5, 1e-4, 1.0 - 1e-4)
    features_dc = np.log(colors / (1.0 - colors)).astype(np.float32)
    count = len(gaussians["means"])
    return {
        "gauss_params.means": torch.from_numpy(gaussians["means"].astype(np.float32)),
        "gauss_params.scales": torch.from_numpy(
            (gaussians["scales"] + np.log(translation_scale)).astype(np.float32)
        ),
        "gauss_params.quats": torch.from_numpy(gaussians["quats"].astype(np.float32)),
        "gauss_params.features_dc": torch.from_numpy(features_dc),
        "gauss_params.features_rest": torch.empty((count, 0, 3), dtype=torch.float32),
        "gauss_params.opacities": torch.from_numpy(gaussians["opacities"][:, None].astype(np.float32)),
    }


def add_rigid_front_car(
    asset_path: Path, gaussians: dict[str, np.ndarray], positions: np.ndarray,
    headings: np.ndarray, timestamps: np.ndarray, translation_scale: float,
) -> None:
    asset = torch.load(asset_path, map_location="cpu")
    quaternions = np.asarray([
        rotation_matrix_to_quaternion(np.array([
            [np.cos(heading), -np.sin(heading), 0.0],
            [np.sin(heading), np.cos(heading), 0.0],
            [0.0, 0.0, 1.0],
        ]))
        for heading in headings
    ], dtype=np.float32)
    state_dict = gaussian_state_dict(gaussians, translation_scale)
    state_dict["instance_trans"] = torch.from_numpy(positions.astype(np.float32))
    state_dict["instance_quats"] = torch.from_numpy(quaternions)
    asset[f"object_{FRONT_CAR_ID}"] = {
        "config": {
            "type": "RigidSubModel",
            "sh_degree": 0,
            "scale_dim": 3,
            "log_timestamps": torch.from_numpy((timestamps * 1e6).astype(np.float64)),
        },
        "state_dict": state_dict,
    }
    torch.save(asset, asset_path)


def make_track_state(
    positions: np.ndarray, headings: np.ndarray, frame_period: float,
    dimensions: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    length, width, height = dimensions
    count = len(positions)
    return {
        "position": positions.astype(np.float32),
        "heading": headings.astype(np.float32),
        "velocity": np.gradient(positions[:, :2], frame_period, axis=0).astype(np.float32),
        "angular_velocity": np.gradient(headings, frame_period).astype(np.float32),
        "valid": np.ones(count, dtype=np.int8),
        "length": np.full((count, 1), length, dtype=np.float32),
        "width": np.full((count, 1), width, dtype=np.float32),
        "height": np.full((count, 1), height, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/highway_new"))
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/sim_engine/custom/highway_new_dynamic_frontcar"),
    )
    parser.add_argument("--scene-id", default="highway_new_dynamic_frontcar-000")
    parser.add_argument("--translation-scale", type=float, default=180.0)
    parser.add_argument("--max-background-points", type=int, default=500_000)
    parser.add_argument("--opacity-threshold", type=float, default=0.5)
    parser.add_argument("--cluster-eps", type=float, default=0.004)
    parser.add_argument("--cluster-min-samples", type=int, default=8)
    args = parser.parse_args()
    if args.translation_scale <= 0 or args.max_background_points <= 0:
        raise ValueError("translation scale and point count must be positive")
    if len(args.scene_id) < 5 or args.scene_id[-4] != "-" or not args.scene_id[-3:].isdigit():
        raise ValueError("--scene-id must end in a reproduction suffix such as -000")

    image_paths = numbered_images(args.source)
    width, height = load_png_size(image_paths[0])
    if any(load_png_size(path) != (width, height) for path in image_paths):
        raise ValueError("source image sizes are inconsistent")
    intrinsics, source_poses = load_camera_parameters(
        args.source / "001" / "ply_camera_params.json", len(image_paths)
    )
    dynamic_paths, timestamps = load_dynamic_manifest(args.source, len(image_paths))
    frame_periods = np.diff(timestamps)
    if not np.allclose(frame_periods, np.median(frame_periods), rtol=0.0, atol=1e-9):
        raise ValueError("SimEngine input requires uniformly spaced dynamic frames")
    frame_period = float(np.median(frame_periods))

    background = read_3dgs_ply(
        args.source / "point_clouds" / "static_sky_001.ply", args.max_background_points
    )
    background, ego2globals = align_reconstruction(background, source_poses, args.translation_scale)
    car_gaussians, car_positions, car_headings, car_dimensions, diagnostics = extract_front_car(
        dynamic_paths, source_poses, args.translation_scale, args.opacity_threshold,
        args.cluster_eps, args.cluster_min_samples,
    )

    camera_positions = ego2globals[:, :3, 3].astype(np.float32)
    ego_headings = np.unwrap(
        np.arctan2(ego2globals[:, 1, 0], ego2globals[:, 0, 0])
    ).astype(np.float32)
    ego_positions = camera_positions_to_vehicle_centers(camera_positions, ego_headings)
    render_intrinsic = np.median(intrinsics, axis=0)
    cameras = make_surround_cameras(width, height, render_intrinsic)
    scene_id = args.scene_id
    frames = {
        f"camera-{idx:03d}": make_surround_frame(
            f"camera-{idx:03d}", scene_id, idx, cameras, intrinsics[idx],
            ego2globals[idx], frame_period,
        )
        for idx in range(len(image_paths))
    }

    ego_state = make_track_state(
        ego_positions, ego_headings, frame_period, (5.176, 2.297, 1.777)
    )
    car_state = make_track_state(
        car_positions, car_headings, frame_period,
        (float(car_dimensions[0]), float(car_dimensions[1]), float(car_dimensions[2])),
    )
    scene = {
        "id": scene_id,
        "name": scene_id,
        "token": scene_id,
        "dataset": "custom.highway_new.dynamic_gaussian",
        "map": "synthetic",
        "log_length": len(image_paths),
        "sample_rate": frame_period / 0.05,
        "base_timestamp": 0,
        "sdc_id": "ego",
        "cameras": cameras,
        "lidar": {
            "channel": "LIDAR_TOP",
            "sensor2ego_rotation": np.array([1.0, 0.0, 0.0, 0.0]),
            "sensor2ego_translation": np.zeros(3),
        },
        "object_track": {
            "ego": {
                "type": "VEHICLE",
                "metadata": {"type": "VEHICLE", "track_length": len(image_paths), "object_id": "ego"},
                "state": ego_state,
            },
            FRONT_CAR_ID: {
                "type": "VEHICLE",
                "metadata": {
                    "type": "VEHICLE",
                    "track_length": len(image_paths),
                    "object_id": FRONT_CAR_ID,
                    "asset_key": f"object_{FRONT_CAR_ID}",
                },
                "state": car_state,
            },
        },
        "dynamic_map_states": {},
        "map_features": {
            "synthetic_lane": {
                "type": "LANE_SURFACE_STREET",
                "polyline": ego_positions[:, :2].astype(np.float64),
                "roadblock_id": 0,
                "entry_lanes": [],
                "exit_lanes": [],
                "left_neighbor": [],
                "right_neighbor": [],
            }
        },
        "metadata": {
            "old_origin_in_current_coordinate": np.zeros(2),
            "digitaltwin_asset_id": scene_id[:-4],
            "digitaltwin_ego2globals": list(ego2globals),
            "openscene_data_infos_dict": frames,
            "rear_axle_to_center_dist": PACIFICA_REAR_AXLE_TO_CENTER,
            "source_translation_scale": args.translation_scale,
            "source_translation_unit": "GSRecon reconstruction unit",
            "dynamic_gaussian_agent_id": FRONT_CAR_ID,
            "dynamic_gaussian_reference_frame": int(np.argmax([
                item["front_car_gaussian_count"] for item in diagnostics
            ])),
            "dynamic_gaussian_extraction": "sigmoid(opacity) threshold + largest DBSCAN component",
            "background_source": "static_sky_001.ply (dynamic_001.ply deliberately excluded)",
            "camera_layout": "co-located 360-degree ring at 45-degree yaw intervals",
            "camera_yaws_degrees": CAMERA_YAWS_DEGREES.copy(),
        },
    }

    scenario_dir = args.output_root / "scenarios"
    asset_path = (
        args.output_root / "assets" / scene_id[:-4] / "background" / f"{scene_id[:-4]}.ckpt"
    )
    scenario_dir.mkdir(parents=True, exist_ok=True)
    make_asset(background, asset_path, args.translation_scale)
    add_rigid_front_car(
        asset_path, car_gaussians, car_positions, car_headings, timestamps,
        args.translation_scale,
    )
    scenario_path = scenario_dir / "all_scenarios.pkl"
    with scenario_path.open("wb") as file:
        pickle.dump({scene_id: scene}, file, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "scene_id": scene_id,
        "frame_count": len(image_paths),
        "frame_period_seconds": frame_period,
        "sample_rate": scene["sample_rate"],
        "translation_scale_metres_per_reconstruction_unit": args.translation_scale,
        "background_gaussian_count": len(background["means"]),
        "front_car_asset_gaussian_count": len(car_gaussians["means"]),
        "front_car_dimensions_metres": {
            "length": float(car_dimensions[0]),
            "width": float(car_dimensions[1]),
            "height": float(car_dimensions[2]),
        },
        "front_car_frames": diagnostics,
    }
    report_path = args.output_root / "front_car_extraction.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print(f"wrote {scenario_path}")
    print(f"wrote {asset_path}")
    print(f"wrote {report_path}")
    print(
        "front car: "
        f"{len(car_gaussians['means']):,} Gaussians, "
        f"L/W/H={car_dimensions[0]:.3f}/{car_dimensions[1]:.3f}/{car_dimensions[2]:.3f} m, "
        f"start={car_positions[0, :2]}, end={car_positions[-1, :2]}"
    )


if __name__ == "__main__":
    main()

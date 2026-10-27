# Modified by ADToolbox contributors, 2026: standalone simulation and scene conversion support.
#!/usr/bin/env python3
"""Build a minimal SimEngine input from native 3DGS PLY and camera poses.

The camera is used as the ego reference because the source has no independent
camera-to-vehicle calibration. Translation remains in reconstruction units
unless ``--translation-scale`` is supplied with an external metric alignment.
"""

from __future__ import annotations

import argparse
import json
import pickle
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch


CAMERA_NAMES = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0")

# Columns are OpenCV camera axes (right, down, forward) expressed in an ego
# frame whose axes are (forward, left, up).
CAMERA_TO_EGO_ROTATION = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)

# SimEngine renders from the Pacifica rear axle, whereas this source provides
# camera optical-center poses.  Keep the reconstruction poses as camera poses
# and offset the simulated vehicle center so its rear axle is the camera.
PACIFICA_REAR_AXLE_TO_CENTER = 3.089 - 1.67


PLY_SCALAR_DTYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def read_3dgs_ply(path: Path, max_points: int) -> dict[str, np.ndarray]:
    """Read a binary PLY containing native 3DGS attributes.

    PLY exporters are free to reorder properties, so parsing by record offset
    would silently corrupt scales, rotations, or SH coefficients.
    """
    with path.open("rb") as file:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            chunk = file.read(1)
            if not chunk:
                raise ValueError(f"{path} has no PLY end_header")
            header.extend(chunk)
        text = header.decode("ascii")
        if "format binary_little_endian 1.0" not in text:
            raise ValueError(f"{path} is not a little-endian binary PLY")
        lines = text.splitlines()
        count_line = next((line for line in lines if line.startswith("element vertex ")), None)
        if count_line is None:
            raise ValueError(f"{path} has no vertex count")
        count = int(count_line.split()[-1])
        properties = []
        in_vertices = False
        for line in lines:
            if line.startswith("element vertex "):
                in_vertices = True
            elif line.startswith("element "):
                in_vertices = False
            elif in_vertices and line.startswith("property "):
                fields = line.split()
                if len(fields) != 3 or fields[1] not in PLY_SCALAR_DTYPES:
                    raise ValueError(f"unsupported PLY property: {line}")
                properties.append((fields[2], PLY_SCALAR_DTYPES[fields[1]]))
        names = [name for name, _ in properties]
        required = {"x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
                    "f_dc_0", "f_dc_1", "f_dc_2", "opacity"}
        missing = sorted(required.difference(names))
        if missing:
            raise ValueError(f"{path} is not a native 3DGS PLY; missing {', '.join(missing)}")
        offset = file.tell()
    dtype = np.dtype(properties)
    if path.stat().st_size - offset < count * dtype.itemsize:
        raise ValueError(f"{path} is shorter than its header declares")
    points = np.memmap(path, mode="r", offset=offset, dtype=dtype, shape=(count,))
    indices = np.linspace(0, count - 1, min(count, max_points), dtype=np.int64)
    return {
        "means": np.asarray(points[["x", "y", "z"]][indices].tolist(), dtype=np.float32),
        "scales": np.asarray(points[["scale_0", "scale_1", "scale_2"]][indices].tolist(), dtype=np.float32),
        "quats": np.asarray(points[["rot_0", "rot_1", "rot_2", "rot_3"]][indices].tolist(), dtype=np.float32),
        "f_dc": np.asarray(points[["f_dc_0", "f_dc_1", "f_dc_2"]][indices].tolist(), dtype=np.float32),
        "opacities": np.asarray(points["opacity"][indices], dtype=np.float32),
    }


def make_asset(gaussians: dict[str, np.ndarray], output: Path, translation_scale: float) -> None:
    """Write native 3DGS attributes in the portable MTGS checkpoint format."""
    # The source stores standard SH DC coefficients. MTGS degree zero stores
    # RGB logits, so convert the DC term to RGB before writing the checkpoint.
    sh_c0 = 0.28209479177387814
    colors = np.clip(gaussians["f_dc"] * sh_c0 + 0.5, 1e-4, 1.0 - 1e-4)
    features_dc = np.log(colors / (1.0 - colors))
    scales = gaussians["scales"] + np.log(translation_scale)
    num_points = len(gaussians["means"])
    state_dict = {
        "gauss_params.means": torch.from_numpy(gaussians["means"] * translation_scale),
        "gauss_params.scales": torch.from_numpy(scales),
        "gauss_params.quats": torch.from_numpy(gaussians["quats"]),
        "gauss_params.features_dc": torch.from_numpy(features_dc.astype(np.float32)),
        "gauss_params.features_rest": torch.empty((num_points, 0, 3), dtype=torch.float32),
        "gauss_params.opacities": torch.from_numpy(gaussians["opacities"][:, None]),
    }
    asset = {
        "background": {
            "config": {
                "type": "VanillaGaussianSplattingModel",
                "sh_degree": 0,
                "scale_dim": 3,
                "recon2world_translation": np.zeros(3, dtype=np.float64),
            },
            "state_dict": state_dict,
        }
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asset, output)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return a normalized wxyz quaternion without requiring scipy."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array([0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                         (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale])
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array([(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                             (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale])
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array([(matrix[0, 2] - matrix[2, 0]) / scale,
                             (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                             (matrix[1, 2] + matrix[2, 1]) / scale])
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array([(matrix[1, 0] - matrix[0, 1]) / scale,
                             (matrix[0, 2] + matrix[2, 0]) / scale,
                             (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale])
    return quat / np.linalg.norm(quat)


def rotation_matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix for a wxyz quaternion."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("Gaussian rotation quaternion has zero norm")
    w, x, y, z = np.asarray(quaternion, dtype=np.float64) / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_camera_parameters(path: Path, frame_count: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load and validate the documented OpenCV camera-to-world matrices."""
    with path.open(encoding="utf-8") as file:
        params = json.load(file)
    convention = params.get("coordinate_convention", {})
    if "OpenCV" not in convention.get("camera_axes", ""):
        raise ValueError(f"{path} does not declare OpenCV camera axes")
    frames = sorted(params["frames"], key=lambda frame: frame["frame_index"])
    if len(frames) != frame_count or [frame["frame_index"] for frame in frames] != list(range(frame_count)):
        raise ValueError(f"camera parameter frames do not match the {frame_count} source images")
    intrinsics = np.asarray([frame["intrinsic"] for frame in frames], dtype=np.float64)
    camera_to_world = np.asarray([frame["camera_to_world"] for frame in frames], dtype=np.float64)
    if intrinsics.shape != (frame_count, 3, 3) or camera_to_world.shape != (frame_count, 4, 4):
        raise ValueError("unexpected intrinsic or camera_to_world shape")
    inverses = np.asarray([frame["world_to_camera"] for frame in frames], dtype=np.float64)
    if not np.allclose(camera_to_world @ inverses, np.eye(4), atol=2e-4):
        raise ValueError("camera_to_world and world_to_camera are not inverses")
    size = params["image_size"]
    return intrinsics, camera_to_world, int(size["width"]), int(size["height"])


def align_reconstruction(
    gaussians: dict[str, np.ndarray], camera_to_world: np.ndarray, translation_scale: float
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Align source world coordinates to SimEngine forward/left/up coordinates."""
    first_pose = camera_to_world[0]
    world_rotation = CAMERA_TO_EGO_ROTATION @ first_pose[:3, :3].T
    source_origin = first_pose[:3, 3]

    aligned = dict(gaussians)
    aligned["means"] = (gaussians["means"].astype(np.float64) - source_origin) @ world_rotation.T
    source_rotations = np.asarray([rotation_matrix_from_quaternion(q) for q in gaussians["quats"]])
    aligned_rotations = np.einsum("ij,njk->nik", world_rotation, source_rotations)
    aligned["quats"] = np.asarray([rotation_matrix_to_quaternion(r) for r in aligned_rotations], dtype=np.float32)
    aligned["scales"] = gaussians["scales"].copy()
    aligned_poses = np.tile(np.eye(4, dtype=np.float64), (len(camera_to_world), 1, 1))
    aligned_camera_rotations = world_rotation @ camera_to_world[:, :3, :3]
    aligned_poses[:, :3, :3] = aligned_camera_rotations @ CAMERA_TO_EGO_ROTATION.T
    aligned_poses[:, :3, 3] = (
        (camera_to_world[:, :3, 3] - source_origin) @ world_rotation.T * translation_scale
    )
    aligned["means"] = aligned["means"].astype(np.float32)
    return aligned, aligned_poses


def make_cameras(width: int, height: int, intrinsic: np.ndarray) -> dict[str, Any]:
    """Build the duplicated forward camera channels required by NavFormer."""
    sensor_quaternion = rotation_matrix_to_quaternion(CAMERA_TO_EGO_ROTATION)
    return {
        name: {
            "channel": name,
            "sensor2ego_rotation": sensor_quaternion.copy(),
            "sensor2ego_translation": np.zeros(3),
            "intrinsic": intrinsic.copy(),
            "distortion": np.zeros(5),
            "height": height,
            "width": width,
        }
        for name in CAMERA_NAMES
    }


def make_frame(
    token: str, scene_id: str, frame_idx: int, cameras: dict[str, Any],
    intrinsic: np.ndarray, ego2global: np.ndarray, frame_period: float,
) -> dict[str, Any]:
    cams = {}
    for name in CAMERA_NAMES:
        cam = cameras[name]
        cams[name] = {
            "data_path": f"placeholder/{name}/{token}.jpg",
            "sensor2lidar_rotation": CAMERA_TO_EGO_ROTATION.copy(),
            "sensor2lidar_translation": np.zeros(3, dtype=np.float64),
            "cam_intrinsic": np.asarray(intrinsic, dtype=np.float64),
            "distortion": np.asarray(cam["distortion"], dtype=np.float64),
        }
    return {
        "token": token,
        "frame_idx": frame_idx,
        "timestamp": int(round(frame_idx * frame_period * 1e6)),
        "log_name": scene_id,
        "log_token": scene_id,
        "scene_name": scene_id,
        "scene_token": scene_id,
        "map_location": "synthetic",
        "vehicle_name": "synthetic_ego",
        "lidar_path": None,
        "lidar2ego": np.eye(4, dtype=np.float64),
        "lidar2ego_rotation": np.array([1., 0., 0., 0.]),
        "lidar2ego_translation": np.zeros(3),
        "ego2global": ego2global.copy(),
        "ego2global_rotation": rotation_matrix_to_quaternion(ego2global[:3, :3]),
        "ego2global_translation": ego2global[:3, 3].copy(),
        "lidar2global": ego2global.copy(),
        "can_bus": np.zeros(18, dtype=np.float64),
        "ego_dynamic_state": [0., 0., 0., 0.],
        "driving_command": np.array([0, 0, 1, 0], dtype=np.int64),
        "traffic_lights": [],
        "roadblock_ids": [],
        "anns": {"gt_boxes": np.zeros((0, 7), dtype=np.float32), "gt_names": np.array([], dtype=str),
                 "gt_velocity_3d": np.zeros((0, 3), dtype=np.float32), "instance_tokens": [], "track_tokens": []},
        "cams": cams,
    }


def camera_positions_to_vehicle_centers(camera_positions: np.ndarray, headings: np.ndarray) -> np.ndarray:
    """Return Pacifica center positions whose rear axle is each camera position."""
    centers = np.asarray(camera_positions, dtype=np.float64).copy()
    centers[:, 0] += PACIFICA_REAR_AXLE_TO_CENTER * np.cos(headings)
    centers[:, 1] += PACIFICA_REAR_AXLE_TO_CENTER * np.sin(headings)
    return centers.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/highway_nowater_seq80_3dgs_camera_20260812/20260812_154009"))
    parser.add_argument("--output-root", type=Path, default=Path("data/sim_engine/custom/highway_nowater_seq80_3dgs_camera_20260812"))
    parser.add_argument("--ply", type=Path, help="native 3DGS PLY; defaults to <source>/point_clouds/combined_001.ply")
    parser.add_argument("--scene-id", default="highway_nowater_seq80_3dgs_camera_20260812-000")
    parser.add_argument("--max-points", type=int, default=100_000, help="maximum number of native Gaussians retained")
    parser.add_argument(
        "--translation-scale", type=float, default=350.0,
        help="metres per GSRecon reconstruction unit (350 is the initial closed-loop calibration for the default source)",
    )
    parser.add_argument(
        "--frame-period", type=float, default=0.5,
        help="seconds between source frames; SimEngine's sample_rate is frame_period / 0.05",
    )
    args = parser.parse_args()
    if min(args.max_points, args.translation_scale, args.frame_period) <= 0:
        raise ValueError("point count, translation scale, and frame period must be positive")
    sample_rate = args.frame_period / 0.05
    if not np.isclose(sample_rate, round(sample_rate)):
        raise ValueError("--frame-period must be an integer multiple of SimEngine's 0.05 s tick")
    sample_rate = int(round(sample_rate))

    image_paths = sorted((args.source / "001").glob("view_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    if not image_paths:
        raise FileNotFoundError(f"no view_*.png below {args.source / '001'}")
    ply_path = args.ply or args.source / "point_clouds" / "combined_001.ply"
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    with image_paths[0].open("rb") as file:
        file.seek(16)  # PNG IHDR width/height are big-endian uint32 at byte 16.
        image_width, image_height = struct.unpack(">II", file.read(8))

    gaussians = read_3dgs_ply(ply_path, args.max_points)
    timesteps = len(image_paths)
    intrinsic_per_frame, source_poses, width, height = load_camera_parameters(
        args.source / "001" / "camera_parameters_ply.json", timesteps
    )
    if (width, height) != (image_width, image_height):
        raise ValueError(f"camera parameter size {width}x{height} != PNG size {image_width}x{image_height}")
    gaussians, ego2globals = align_reconstruction(gaussians, source_poses, args.translation_scale)
    camera_positions = ego2globals[:, :3, 3].astype(np.float32)
    headings = np.unwrap(np.arctan2(ego2globals[:, 1, 0], ego2globals[:, 0, 0])).astype(np.float32)
    positions = camera_positions_to_vehicle_centers(camera_positions, headings)
    velocity = np.gradient(positions[:, :2], args.frame_period, axis=0).astype(np.float32)
    angular_velocity = np.gradient(headings, args.frame_period).astype(np.float32)

    # SimEngine stores one static calibration. Intrinsic drift in the source is
    # small; the median is robust and the exact per-frame values remain in the
    # OpenScene metadata for downstream consumers.
    render_intrinsic = np.median(intrinsic_per_frame, axis=0)
    cameras = make_cameras(width, height, render_intrinsic)

    scene_id = args.scene_id
    if len(scene_id) < 5 or scene_id[-4] != "-" or not scene_id[-3:].isdigit():
        raise ValueError("--scene-id must end with a reproduction suffix such as -000")
    frames = {
        f"camera-{idx:03d}": make_frame(
            f"camera-{idx:03d}", scene_id, idx, cameras, intrinsic_per_frame[idx],
            ego2globals[idx], args.frame_period,
        )
        for idx in range(timesteps)
    }
    ego_state = {
        "position": positions,
        "heading": headings,
        "velocity": velocity,
        "angular_velocity": angular_velocity,
        "valid": np.ones(timesteps, dtype=np.int8),
        "length": np.full((timesteps, 1), 5.176, dtype=np.float32),
        "width": np.full((timesteps, 1), 2.297, dtype=np.float32),
        "height": np.full((timesteps, 1), 1.777, dtype=np.float32),
    }
    scene = {
        "id": scene_id, "name": scene_id, "token": scene_id, "dataset": "custom.3dgs_camera",
        "map": "synthetic", "log_length": timesteps, "sample_rate": sample_rate, "base_timestamp": 0,
        "sdc_id": "ego", "cameras": cameras,
        "lidar": {"channel": "LIDAR_TOP", "sensor2ego_rotation": np.array([1., 0., 0., 0.]), "sensor2ego_translation": np.zeros(3)},
        "object_track": {"ego": {"type": "VEHICLE", "metadata": {"type": "VEHICLE", "track_length": timesteps, "object_id": "ego"}, "state": ego_state}},
        "dynamic_map_states": {},
        "map_features": {"synthetic_lane": {"type": "LANE_SURFACE_STREET", "polyline": positions[:, :2].astype(np.float64), "roadblock_id": 0, "entry_lanes": [], "exit_lanes": [], "left_neighbor": [], "right_neighbor": []}},
        "metadata": {
            "old_origin_in_current_coordinate": np.zeros(2),
            "digitaltwin_asset_id": scene_id[:-4],
            "digitaltwin_ego2globals": list(ego2globals),
            "openscene_data_infos_dict": frames,
            "rear_axle_to_center_dist": PACIFICA_REAR_AXLE_TO_CENTER,
            "source_translation_scale": args.translation_scale,
            "source_translation_unit": "GSRecon reconstruction unit",
            "ego_reference": "Pacifica vehicle center; rear axle is aligned to the source camera optical center",
            "gaussian_asset_semantics": "static asset; source PLY has no per-frame instance poses",
        },
    }
    scenario_dir = args.output_root / "scenarios"
    asset_path = args.output_root / "assets" / scene_id[:-4] / "background" / f"{scene_id[:-4]}.ckpt"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    with (scenario_dir / "all_scenarios.pkl").open("wb") as file:
        pickle.dump({scene_id: scene}, file, protocol=pickle.HIGHEST_PROTOCOL)
    make_asset(gaussians, asset_path, args.translation_scale)
    print(f"wrote {scenario_dir / 'all_scenarios.pkl'}")
    print(f"wrote {asset_path} ({len(gaussians['means']):,} native Gaussians; source had {len(image_paths)} frames at {width}x{height})")
    print(
        f"camera path: {np.linalg.norm(np.diff(camera_positions[:, :2], axis=0), axis=1).sum():.6f} "
        f"metres, camera_start={camera_positions[0, :2]}, camera_end={camera_positions[-1, :2]}, "
        f"translation_scale={args.translation_scale:g}"
    )
    print(
        f"ego center is offset {PACIFICA_REAR_AXLE_TO_CENTER:.3f} m forward so SimEngine's rear axle "
        "matches the source camera"
    )


if __name__ == "__main__":
    main()

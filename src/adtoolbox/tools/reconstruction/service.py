import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ...backends.process import invoke, root_settings
from ...core.errors import BackendError, DependencyError, InputError
from ...core.io import file_path, read_json, write_json
from ..edit_config_generator.compiler import compile_config, to_rigid


def _run(data, config, edit_config=None):
    root, settings = root_settings(
        config, "reconstruction", ["root", "python", "checkpoint", "preprocess_python", "segmentation_model"]
    )
    video = file_path(data["video_path"], "video_path")
    checkpoint = file_path(settings["checkpoint"], "checkpoint")
    inference = file_path(str(root / "inference.py"), "inference.py")
    requested_features = ["--export_camera_params", "-export_ply"]
    if edit_config:
        requested_features += ["--edit_config"]
    if data.get("export_render_state"):
        if edit_config:
            raise InputError("Native render-state export is supported only for unedited reconstruction")
        requested_features += ["--export_render_state"]
    if data.get("ego_trajectory_offsets_path"):
        requested_features += ["--ego_trajectory_offsets"]
    text = inference.read_text(encoding="utf-8")
    missing = [flag for flag in requested_features if flag not in text]
    if missing:
        raise DependencyError("Backend lacks required extensions: " + ", ".join(missing) + "; see extensions/README.md")
    run = Path(settings.get("output_dir", "runs/reconstruction")).expanduser().resolve() / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True)
    inputs, outputs = run / "input", run / "output"
    extension = Path(__file__).with_name("preprocess.py").resolve()
    count = data.get("num_frames", 40)
    pre = invoke(
        [
            str(settings["preprocess_python"]),
            str(extension),
            str(video),
            str(inputs),
            "--num-frames",
            str(count),
            "--scene-name",
            "000",
            "--segformer-model",
            str(settings["segmentation_model"]),
            "--device",
            str(settings.get("device", "cuda:0")),
        ],
        cwd=root,
        timeout=settings.get("timeout", 3600),
        log_path=run / "preprocess.log",
    )
    (run / "preprocess.log").write_text(pre.stdout + pre.stderr, encoding="utf-8")
    metadata = read_json(inputs / "source.json")
    command = [
        str(settings["python"]),
        str(inference),
        "--image_dir",
        str(inputs),
        "--scene_names",
        "000",
        "--input_views",
        "1",
        "--sequence_length",
        str(count),
        "--start_idx",
        "0",
        "--mode",
        "2",
        "--ckpt_path",
        str(checkpoint),
        "--output_path",
        str(outputs),
        "--output_fps",
        str(metadata["fps"]),
        "-images",
        "-export_ply",
        "--export_camera_params",
    ]
    if edit_config:
        command += ["--edit_config", str(edit_config), "--export_dynamic_frame_ply"]
    if data.get("ego_trajectory_offsets_path"):
        command += [
            "--ego_trajectory_offsets",
            str(file_path(data["ego_trajectory_offsets_path"], "ego offsets")),
        ]
    if data.get("export_render_state"):
        command += ["--export_render_state"]
    process = invoke(command, cwd=root, timeout=settings.get("timeout", 3600), log_path=run / "inference.log")
    (run / "inference.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    combined = sorted((outputs / "point_clouds").glob("combined_*.ply"))
    if len(combined) != 1:
        raise BackendError("Backend did not produce exactly one combined scene PLY")
    scene_id = combined[0].stem.removeprefix("combined_")
    scene_dir = outputs / scene_id
    result = {
        "scene_ply": str(combined[0]),
        "camera_params": str(scene_dir / "ply_camera_params.json"),
        "rendered_video": str(scene_dir / "rendered_video.mp4"),
        "output_dir": str(outputs),
        "input_dir": str(inputs / "000" / "images"),
        "frame_count": count,
        "source_fps": metadata["fps"],
        "coordinate_system": "backend_world",
        "metric_scale_known": False,
    }
    for key in ("camera_params", "rendered_video"):
        file_path(result[key], key)
    if data.get("export_render_state"):
        result["render_state"] = str(file_path(str(scene_dir / "render_state.pt"), "render_state"))
    result["_artifacts"] = [
        {"kind": key, "path": result[key]} for key in ("scene_ply", "camera_params", "rendered_video")
    ]
    return result


def reconstruct(data, config):
    return _run(data, config)


def edit(data, config):
    if not data.get("edit_config_path") and not data.get("ego_trajectory_offsets_path"):
        raise InputError("Provide edit_config_path or ego_trajectory_offsets_path")
    if not data.get("edit_config_path"):
        return _run(data, config)
    source = file_path(data["edit_config_path"], "edit_config_path")
    portable = read_json(source)
    if "Nodes" in portable:
        rigid = portable
        portable = {
            "coordinate_system": data.get("coordinate_system", "camera_local_meters"),
            "operations": [{"operation": name} for name in rigid.get("Nodes", {}).get("RigidNodes", {})],
        }
    else:
        compile_config(
            {"operations": portable.get("operations"), "coordinate_system": portable.get("coordinate_system")}
        )
        rigid = to_rigid(portable)
    if portable["coordinate_system"] == "camera_local_meters" and "trajectory_edit" in rigid["Nodes"]["RigidNodes"]:
        from ...backends.coordinates import convert_edit_config

        if not data.get("camera_params_path") or not data.get("meters_to_scene_units"):
            raise InputError("Camera-local trajectory editing requires camera_params_path and meters_to_scene_units")
        rigid, _ = convert_edit_config(rigid, data["camera_params_path"], data["meters_to_scene_units"])
    # The renderer needs the same detector boxes that defined instance IDs.
    instance_info = (
        data.get("instance_info_path")
        or rigid.get("instance_info_path")
        or config.get("scene_editing", {}).get("instance_info_path")
    )
    if any(op["operation"] != "add" for op in portable["operations"]):
        if not instance_info:
            raise InputError("Configure scene_editing.instance_info_path for existing-instance edits")
        rigid["instance_info_path"] = str(file_path(instance_info, "instance_info_path"))
        discovered = read_json(instance_info)
        if discovered.get("pixel_space") != "reconstruction_input":
            raise InputError("Renderer edits require instances tracked with pixel_space=reconstruction_input")
        available = {item["instance_id"] for item in discovered.get("instances", [])}
        if "Nodes" not in read_json(source) and any(
            op.get("instance_id") not in available for op in portable["operations"] if op["operation"] != "add"
        ):
            raise InputError("Edit references an instance absent from the discovery result")
    with tempfile.TemporaryDirectory(prefix="adtoolbox-edit-") as directory:
        path = write_json(Path(directory) / "edit.json", rigid)
        result = _run(data, config, path)
    result["scene_config"] = str(source)
    return result

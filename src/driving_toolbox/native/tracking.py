import math
from pathlib import Path

from ..core.errors import DependencyError, InputError
from ..core.io import file_path, write_json


def track(data):
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError:
        raise DependencyError(
            "Install driving-toolbox[vision] or configure an instance_discovery backend"
        ) from None
    if not data.get("weights"):
        raise InputError("Provide weights explicitly; model weights are not downloaded automatically")
    weights = file_path(data["weights"], "weights")
    model = YOLO(str(weights))
    capture = None
    if data.get("video_path"):
        video = file_path(data["video_path"], "video_path")
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise InputError("Video could not be opened")
        source_fps = capture.get(cv2.CAP_PROP_FPS)

        def frames():
            for _ in range(data.get("num_frames", 40)):
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
    elif data.get("image_dir"):
        paths = sorted(
            p
            for p in Path(data["image_dir"]).expanduser().glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        source_fps = None

        def frames():
            for path in paths[: data.get("num_frames", 40)]:
                image = cv2.imread(str(path))
                if image is None:
                    raise InputError(f"Unreadable frame: {path.name}")
                yield image
    else:
        raise InputError("Provide video_path or image_dir")
    tracks = {}
    count = 0
    try:
        for i, frame in enumerate(frames()):
            count += 1
            if data.get("pixel_space") == "reconstruction_input":
                import numpy as np
                from PIL import Image

                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                height = max(14, round(image.height * (518 / image.width) / 14) * 14)
                image = image.resize((518, height), Image.Resampling.BICUBIC)
                if height > 518:
                    top = (height - 518) // 2
                    image = image.crop((0, top, 518, top + 518))
                frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            prediction = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=data.get("classes", [2, 5, 7]),
                conf=data.get("confidence", 0.25),
                verbose=False,
            )[0]
            boxes = prediction.boxes
            if boxes is None or boxes.id is None:
                continue
            for track_id, category, bbox in zip(
                boxes.id.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.xyxy.cpu().tolist(), strict=True
            ):
                item = tracks.setdefault(
                    int(track_id), {"track_id": int(track_id), "class_id": int(category), "frames": []}
                )
                height, width = frame.shape[:2]
                x1, y1, x2, y2 = bbox
                box = [
                    max(0, math.floor(x1)),
                    max(0, math.floor(y1)),
                    min(width, math.ceil(x2)),
                    min(height, math.ceil(y2)),
                ]
                if box[0] < box[2] and box[1] < box[3]:
                    item["frames"].append({"frame_index": i, "bbox": box})
    finally:
        if capture is not None:
            capture.release()
    if not count:
        raise InputError("No readable frames")
    instances = []
    for key in sorted(tracks):
        if tracks[key]["frames"]:
            instances.append({"instance_id": len(instances), **tracks[key]})
    result = {
        "instances": instances,
        "frame_count": count,
        "source_fps": source_fps,
        "pixel_space": data.get("pixel_space", "source"),
        "_warnings": ["Bounding-box tracks do not include segmentation masks or metric 3D poses."],
    }
    if data.get("output_path"):
        path = write_json(data["output_path"], {k: v for k, v in result.items() if not k.startswith("_")})
        result.update(instance_info_path=path, _artifacts=[{"kind": "instances", "path": path}])
    return result

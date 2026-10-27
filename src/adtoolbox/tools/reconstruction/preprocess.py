# Adapted from the ADToolbox authors, 2026. Modified: explicit model path, frame bounds and metadata. Apache-2.0.
"""Convert one video into scene backend single-view images and sky masks."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_DIR = Path("models/sky_segmentation")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert one MP4 into scene backend images and sky masks.")
    parser.add_argument("input_video", type=Path, help="Source video path")
    parser.add_argument("output_root", type=Path, help="scene backend input root directory")
    parser.add_argument("--scene-name", default="video_000", help="Output scene name")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to extract. Omit to extract every source frame.",
    )
    parser.add_argument("--duration-seconds", type=float, default=None, help="Use only the first N seconds.")
    parser.add_argument(
        "--crop-bottom", type=int, default=0, help="Crop this many pixels from the bottom of every frame."
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--segformer-model", default=str(MODEL_DIR))
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {args.input_video}")
    if args.num_frames is not None and args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive")
    if args.crop_bottom < 0:
        raise ValueError("--crop-bottom must be non-negative")
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 0 and 100")

    capture = cv2.VideoCapture(str(args.input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input_video}")

    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_frame_count <= 0:
        raise RuntimeError("The video reports no readable frames")

    if source_fps <= 0:
        raise ValueError("Source video has invalid FPS")
    if args.num_frames and args.num_frames > source_frame_count:
        raise ValueError("Source video has fewer frames than requested")
    usable_frame_count = source_frame_count
    if args.duration_seconds is not None:
        usable_frame_count = min(source_frame_count, round(args.duration_seconds * source_fps))
    requested_count = args.num_frames or usable_frame_count
    if requested_count > usable_frame_count:
        print(f"Resampling {usable_frame_count} source frames to {requested_count} output frames.")
    indices = (
        np.arange(requested_count, dtype=np.int64)
        if args.num_frames is not None
        else np.arange(usable_frame_count, dtype=np.int64)
    )

    scene_dir = args.output_root / args.scene_name
    # 与 Waymo 预处理相同，Cityscapes 类别 10 是天空。
    device = torch.device(args.device)
    processor = SegformerImageProcessor.from_pretrained(args.segformer_model)
    segformer = SegformerForSemanticSegmentation.from_pretrained(args.segformer_model).to(device).eval()
    sky_masks_dir = scene_dir / "sky_masks"
    sky_masks_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = scene_dir / "images"
    frames_dir.mkdir(parents=True, exist_ok=True)
    image_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    for output_index, source_index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read source frame {source_index}")
        if args.crop_bottom:
            # 去除车内或设备造成的底部固定遮挡。
            if args.crop_bottom >= frame.shape[0]:
                raise ValueError("--crop-bottom must be smaller than the frame height")
            frame = frame[: -args.crop_bottom]
        # scene backend 单视角文件名：帧编号_视角编号，0 表示唯一视角。
        output_path = frames_dir / f"{output_index:03d}_0.jpg"
        if not cv2.imwrite(str(output_path), frame, image_params):
            raise RuntimeError(f"Cannot write frame: {output_path}")

        inputs = processor(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), return_tensors="pt")
        with torch.no_grad():
            logits = segformer(pixel_values=inputs["pixel_values"].to(device)).logits
        labels = (
            torch.nn.functional.interpolate(logits, size=frame.shape[:2], mode="bilinear", align_corners=False)
            .argmax(1)[0]
            .cpu()
            .numpy()
        )
        sky_mask = labels == 10
        cv2.imwrite(str(sky_masks_dir / f"{output_index:03d}_0.png"), sky_mask.astype(np.uint8) * 255)
    capture.release()

    (args.output_root / "source.json").write_text(
        json.dumps({"fps": source_fps, "frame_count": len(indices)}), encoding="utf-8"
    )
    print(f"Extracted {len(indices)} images and sky masks to {scene_dir}")


if __name__ == "__main__":
    main()

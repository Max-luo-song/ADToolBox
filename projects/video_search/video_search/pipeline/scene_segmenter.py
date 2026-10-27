"""Stage 6: scene split accepted source videos."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


PTS_RE = re.compile(r"pts_time:(?P<time>\d+(?:\.\d+)?)")
BV_RE = re.compile(r"(BV[0-9A-Za-z]+)")


@dataclass
class SceneConfig:
    output_dir: Path = Path("data/internet_videos")
    threshold: float = 0.12
    min_duration: float = 4.0
    edge_margin: float = 1.0
    reencode: bool = False


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def get_duration(video: Path) -> float:
    proc = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ]
    )
    return float(proc.stdout.strip())


def detect_cut_points(video: Path, threshold: float) -> list[float]:
    proc = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(video),
            "-filter:v",
            f"select='gt(scene,{threshold})',showinfo",
            "-f",
            "null",
            "-",
        ],
        check=False,
    )
    if proc.returncode not in (0, 255):
        raise RuntimeError(proc.stderr.strip() or f"Scene detection failed: {video}")

    return sorted({float(match.group("time")) for match in PTS_RE.finditer(proc.stderr)})


def build_segments(
    duration: float,
    cut_points: list[float],
    *,
    min_duration: float,
    edge_margin: float,
) -> list[tuple[float, float]]:
    boundaries = [0.0]
    boundaries.extend(t for t in cut_points if edge_margin < t < duration - edge_margin)
    boundaries.append(duration)
    boundaries = sorted(set(boundaries))

    segments: list[tuple[float, float]] = []
    current_start = boundaries[0]
    for boundary in boundaries[1:]:
        if boundary - current_start < min_duration:
            continue
        segments.append((current_start, boundary))
        current_start = boundary

    if current_start < duration:
        if segments and duration - current_start < min_duration:
            prev_start, _ = segments[-1]
            segments[-1] = (prev_start, duration)
        elif duration - current_start >= min_duration:
            segments.append((current_start, duration))

    return segments or [(0.0, duration)]


def export_segment(video: Path, output: Path, start: float, end: float, *, reencode: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(end - start, 0.001)
    if reencode:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(video),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(video),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            str(output),
        ]
    proc = run(cmd, check=False)
    if proc.returncode != 0:
        if reencode:
            raise RuntimeError(proc.stderr.strip() or f"Failed to export segment: {output}")
        export_segment(video, output, start, end, reencode=True)


def source_group_name(video: Path) -> str:
    for value in (video.parent.name, video.stem):
        match = BV_RE.search(value)
        if match:
            return match.group(1)
    return video.stem


def split_video(video: Path, config: SceneConfig) -> list[dict]:
    video = video.resolve()
    duration = get_duration(video)
    cuts = detect_cut_points(video, threshold=config.threshold)
    segments = build_segments(
        duration,
        cuts,
        min_duration=config.min_duration,
        edge_margin=config.edge_margin,
    )

    video_dir = config.output_dir / source_group_name(video) / "scenes"
    rows: list[dict] = []
    for idx, (start, end) in enumerate(segments, 1):
        out = video_dir / f"{video.stem}_scene_{idx:04d}_{start:.2f}-{end:.2f}.mp4"
        export_segment(video, out, start, end, reencode=config.reencode)
        rows.append(
            {
                "source_video": str(video),
                "segment_video": str(out.resolve()),
                "scene_index": idx,
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "duration": round(end - start, 3),
            }
        )
    return rows


def split_scenes(video: Path, config: SceneConfig) -> list[dict]:
    return split_video(video, config)

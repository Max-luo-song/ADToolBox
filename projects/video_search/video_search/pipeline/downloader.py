"""Stage 4: download accepted full videos."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import common_headers, real_user_home_env, safe_name, yt_dlp_cmd


@dataclass
class DownloadConfig:
    output_dir: Path = Path("data/internet_videos")
    limit: int = 0
    cookies_browser: str = ""
    max_height: int = 360
    timeout: int = 900
    accepted_only: bool = True
    skip_existing: bool = True


def load_precheck_rows(path: Path, *, accepted_only: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            if accepted_only and not row.get("accepted"):
                continue
            if row.get("url"):
                rows.append(row)
    return rows


def candidate_dir(row: dict[str, Any], output_dir: Path) -> Path:
    video_id = safe_name(str(row.get("video_id") or row.get("url") or "unknown"))
    return output_dir / video_id


def output_template(row: dict[str, Any], output_dir: Path) -> str:
    title = safe_name(str(row.get("title") or "untitled"))
    return str(candidate_dir(row, output_dir) / f"{title}.%(ext)s")


def find_downloaded_file(row: dict[str, Any], output_dir: Path) -> Path | None:
    search_dir = candidate_dir(row, output_dir)
    files = sorted(path for path in search_dir.glob("*.mp4") if path.is_file() and path.stat().st_size > 0)
    if not files:
        files = sorted(path for path in search_dir.glob("*") if path.is_file() and path.stat().st_size > 0)
    return files[0] if files else None


def cleanup_candidate_files(row: dict[str, Any], output_dir: Path) -> None:
    search_dir = candidate_dir(row, output_dir)
    if not search_dir.exists():
        return
    for path in search_dir.glob("*"):
        if path.is_file():
            path.unlink(missing_ok=True)


def is_decodable_video(path: Path) -> tuple[bool, str]:
    duration_proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        duration = max(float(duration_proc.stdout.strip()), 0.0)
    except ValueError:
        duration = 0.0
    if duration <= 0:
        return False, duration_proc.stderr.strip() or "could not determine video duration"

    timestamps = [duration * ratio for ratio in (0.2, 0.4, 0.6, 0.8)]
    with tempfile.TemporaryDirectory(prefix="video_validate_") as tmp:
        for idx, timestamp in enumerate(timestamps):
            frame = Path(tmp) / f"frame_{idx}.jpg"
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    str(path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    str(frame),
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0 or not frame.exists() or frame.stat().st_size == 0:
                return False, proc.stderr.strip() or f"video decode validation failed at {timestamp:.3f}s"
    return True, ""


def format_strategies(max_height: int) -> list[tuple[str, str]]:
    return [
        ("仅视频流 MP4", f"bestvideo[height<={max_height}][ext=mp4]/bestvideo[height<={max_height}]"),
        ("单文件 MP4", f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best"),
        (
            "音视频合并 MP4",
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
            f"bestvideo[height<={max_height}]+bestaudio/"
            f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best",
        ),
    ]


def download_one(row: dict[str, Any], config: DownloadConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir(row, config.output_dir).mkdir(parents=True, exist_ok=True)
    existing = find_downloaded_file(row, config.output_dir)
    if config.skip_existing and existing:
        ok, error = is_decodable_video(existing)
        if not ok:
            print(f"已有文件不可解码，删除后重新下载：{existing.name}")
            existing.unlink(missing_ok=True)
        else:
            return {**row, "downloaded": True, "skipped_existing": True, "video_path": str(existing.resolve()), "error": ""}

    last_error = ""
    for strategy_name, fmt in format_strategies(config.max_height):
        cleanup_candidate_files(row, config.output_dir)
        result = download_one_with_format(row, config, fmt=fmt, strategy_name=strategy_name)
        if not result.get("downloaded"):
            last_error = str(result.get("error") or "")
            continue
        downloaded = Path(str(result["video_path"]))
        ok, error = is_decodable_video(downloaded)
        if ok:
            result["download_strategy"] = strategy_name
            return result
        print(f"下载文件不可解码，准备换策略重试：{strategy_name}")
        downloaded.unlink(missing_ok=True)
        last_error = error
    return {**row, "downloaded": False, "video_path": "", "error": last_error or "all download strategies failed"}


def download_one_with_format(row: dict[str, Any], config: DownloadConfig, *, fmt: str, strategy_name: str) -> dict[str, Any]:
    existing = find_downloaded_file(row, config.output_dir)
    if config.skip_existing and existing:
        return {**row, "downloaded": True, "skipped_existing": True, "video_path": str(existing.resolve()), "error": ""}

    platform = str(row.get("platform") or "")
    cmd = yt_dlp_cmd() + [
        "--no-playlist",
        "--format",
        fmt,
        "--merge-output-format",
        "mp4",
        "--newline",
        "--progress",
        "-o",
        output_template(row, config.output_dir),
    ]
    cmd.extend(common_headers(platform))
    if config.cookies_browser and platform == "bilibili":
        cmd += ["--cookies-from-browser", config.cookies_browser]
    cmd.append(str(row["url"]))

    print(f"下载策略：{strategy_name}")
    output_lines: list[str] = []
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=real_user_home_env(),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            output_lines.append(line)
            print(line, flush=True)
        returncode = proc.wait(timeout=config.timeout)
    except subprocess.TimeoutExpired:
        if proc:
            proc.kill()
        return {**row, "downloaded": False, "video_path": "", "error": f"timeout after {config.timeout}s"}
    except KeyboardInterrupt:
        if proc:
            proc.kill()
        raise
    except Exception as exc:
        print(f"下载失败：{exc}", file=sys.stderr)
        return {**row, "downloaded": False, "video_path": "", "error": str(exc)}

    downloaded = find_downloaded_file(row, config.output_dir)
    if returncode != 0 or not downloaded:
        return {
            **row,
            "downloaded": False,
            "video_path": str(downloaded.resolve()) if downloaded else "",
            "error": "\n".join(output_lines[-20:]) or "download failed",
        }
    return {**row, "downloaded": True, "skipped_existing": False, "video_path": str(downloaded.resolve()), "error": ""}


def download_candidates(precheck_jsonl: Path, config: DownloadConfig) -> list[dict[str, Any]]:
    rows = load_precheck_rows(precheck_jsonl, accepted_only=config.accepted_only)
    if config.limit > 0:
        rows = rows[: config.limit]
    if not rows:
        raise SystemExit("No candidates to download.")

    results: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        print(f"[{idx}/{len(rows)}] 正在下载：{row.get('title') or row.get('url')}")
        results.append(download_one(row, config))
    return results

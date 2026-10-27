"""Stage 3: cheap candidate prefilter before full download."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .common import common_headers, dashscope_chat_completion, extract_json, image_data_url, real_user_home_env, run, safe_name, yt_dlp_cmd
from .search_query import DEFAULT_BASE_URL


DEFAULT_MODEL = "qwen3-vl-plus"
HARD_NEGATIVE_KEYWORDS = {
    "游戏", "模拟器", "摩托", "骑行", "步行", "航拍", "车评", "开箱", "教学", "事故合集", "停车", "倒车", "vlog",
    "game", "simulator", "motorcycle", "bike", "bicycle", "walking", "drone", "review",
}


@dataclass
class PrecheckConfig:
    platform: str = "bilibili"
    limit: int = 10
    sample_clips: int = 1
    clip_duration: float = 3.0
    frames_per_clip: int = 1
    min_duration: float = 10.0
    max_duration: float = 1200.0
    min_confidence: float = 0.65
    cookies_browser: str = ""
    preview_clip_dir: Path = Path("logs/precheck_clips")
    preview_frame_dir: Path = Path("logs/precheck_frames")
    max_height: int = 640
    download_timeout: int = 180
    ffmpeg_timeout: int = 60
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL


def load_candidates(path: Path, *, platform: str) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requirement = str(payload.get("requirement") or "")
    search_plan = payload.get("search_plan") or {}
    results = payload.get("results") or {}
    platforms = [platform] if platform != "both" else list(results.keys())
    candidates = [
        item
        for plat in platforms
        for item in results.get(plat, [])
        if item.get("type") != "error" and item.get("url")
    ]
    return requirement, search_plan, candidates


def metadata_filter(item: dict[str, Any], *, min_duration: float, max_duration: float) -> tuple[bool, list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    duration = item.get("duration_sec")
    if duration is not None:
        try:
            duration_f = float(duration)
            if duration_f < min_duration:
                reasons.append(f"duration {duration_f:.1f}s < min_duration {min_duration:.1f}s")
            if duration_f > max_duration:
                reasons.append(f"duration {duration_f:.1f}s > max_duration {max_duration:.1f}s")
        except (TypeError, ValueError):
            warnings.append("duration is not numeric")
    else:
        warnings.append("duration missing")

    blob = " ".join([str(item.get("title") or ""), str(item.get("description") or ""), " ".join(str(tag) for tag in item.get("tags") or [])]).lower()
    matched = [kw for kw in sorted(HARD_NEGATIVE_KEYWORDS) if kw.lower() in blob]
    if matched:
        reasons.append("hard negative keywords: " + ", ".join(matched[:8]))
    return not reasons, reasons, warnings


def sample_starts(duration: float | None, *, sample_clips: int, clip_duration: float) -> list[float]:
    if duration is None or duration <= clip_duration:
        return [0.0]
    starts = []
    for idx in range(max(1, sample_clips)):
        center = duration * (idx + 1) / (sample_clips + 1)
        starts.append(round(min(max(center - clip_duration / 2, 0.0), max(duration - clip_duration, 0.0)), 3))
    return sorted(set(starts))


def download_preview_clip(item: dict[str, Any], *, start: float, config: PrecheckConfig) -> Path:
    platform = str(item.get("platform") or "")
    video_id = safe_name(str(item.get("video_id") or item.get("display_id") or item.get("url") or "video"), limit=80)
    clip_dir = config.preview_clip_dir / f"{platform}_{video_id}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    output_tmpl = str(clip_dir / f"preview_{start:.2f}.%(ext)s")
    cmd = yt_dlp_cmd() + [
        "--no-playlist",
        "--download-sections",
        f"*{start:.2f}-{start + config.clip_duration:.2f}",
        "--format",
        "bestvideo[height<=360][ext=mp4]/best[height<=360][ext=mp4]/best[height<=360]/best",
        "--force-keyframes-at-cuts",
        "-o",
        output_tmpl,
    ]
    cmd.extend(common_headers(platform))
    if config.cookies_browser and platform == "bilibili":
        cmd += ["--cookies-from-browser", config.cookies_browser]
    cmd.append(str(item["url"]))

    proc = run(cmd, timeout=config.download_timeout, env=real_user_home_env())
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"yt-dlp preview download failed: {item['url']}")
    candidates = [p for p in sorted(clip_dir.glob(f"preview_{start:.2f}.*")) if p.is_file() and p.stat().st_size > 0]
    if not candidates:
        raise RuntimeError("preview clip was not created")
    return candidates[0]


def extract_frames(clip: Path, *, config: PrecheckConfig) -> list[Path]:
    frame_dir = config.preview_frame_dir / clip.parent.name / clip.stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = frame_dir / "frame_%03d.jpg"
    vf = f"fps={max(1, config.frames_per_clip)}/3,scale='min({config.max_height},iw)':-2"
    proc = run(["ffmpeg", "-y", "-i", str(clip), "-vf", vf, "-frames:v", str(config.frames_per_clip), "-q:v", "3", str(pattern)], timeout=config.ffmpeg_timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ffmpeg frame extraction failed: {clip}")
    return [path for path in sorted(frame_dir.glob("frame_*.jpg")) if path.stat().st_size > 0]


def visual_precheck(*, frames: list[Path], requirement: str, search_plan: dict[str, Any], item: dict[str, Any], config: PrecheckConfig) -> tuple[bool, float, str]:
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY or OPENAI_API_KEY.")
    if not frames:
        return False, 0.0, "no preview frames"

    prompt = f"""你是自动驾驶视频数据集的预筛审核员。下面是候选视频少量预览帧。

用户需求：{requirement}
结构化需求：{json.dumps(search_plan.get("structured") or {}, ensure_ascii=False)}
视频标题：{item.get("title") or ""}
视频简介：{str(item.get("description") or "")[:500]}

请判断这个候选视频是否值得完整下载。预筛要相对宽松，但必须拒绝明显错误类型。
accepted=true 应同时满足：汽车第一视角/行车记录仪视角；包含道路/车辆/交通环境；画面清晰；不是封面/PPT/游戏/骑行/航拍/车评教学。

只输出 JSON：
{{"accepted": true, "confidence": 0.0, "reason": "一句话说明"}}
"""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": image_data_url(frame)}} for frame in frames]
    body = dashscope_chat_completion(
        base_url=config.base_url,
        api_key=api_key,
        payload={"model": config.model, "messages": [{"role": "user", "content": content}], "temperature": 0},
    )
    parsed = extract_json(body["choices"][0]["message"]["content"])
    return bool(parsed.get("accepted")), float(parsed.get("confidence", 0.0)), str(parsed.get("reason", ""))


def precheck_one(item: dict[str, Any], *, requirement: str, search_plan: dict[str, Any], config: PrecheckConfig) -> dict[str, Any]:
    row = {
        "platform": item.get("platform"),
        "video_id": item.get("video_id"),
        "title": item.get("title"),
        "url": item.get("url"),
        "duration_sec": item.get("duration_sec"),
        "metadata_pass": False,
        "metadata_reasons": [],
        "metadata_warnings": [],
        "visual_pass": False,
        "accepted": False,
        "confidence": 0.0,
        "reason": "",
        "preview_clips": [],
        "preview_frames": [],
    }
    metadata_pass, reasons, warnings = metadata_filter(item, min_duration=config.min_duration, max_duration=config.max_duration)
    row.update({"metadata_pass": metadata_pass, "metadata_reasons": reasons, "metadata_warnings": warnings})
    if not metadata_pass:
        row["reason"] = "; ".join(reasons)
        return row

    try:
        duration = float(item["duration_sec"]) if item.get("duration_sec") is not None else None
    except (TypeError, ValueError):
        duration = None

    frames: list[Path] = []
    try:
        for start in sample_starts(duration, sample_clips=config.sample_clips, clip_duration=config.clip_duration):
            clip = download_preview_clip(item, start=start, config=config)
            row["preview_clips"].append(str(clip.resolve()))
            clip_frames = extract_frames(clip, config=config)
            frames.extend(clip_frames)
            row["preview_frames"].extend(str(frame.resolve()) for frame in clip_frames)
        visual_pass, confidence, reason = visual_precheck(frames=frames, requirement=requirement, search_plan=search_plan, item=item, config=config)
        row.update({"visual_pass": visual_pass, "confidence": round(confidence, 4), "reason": reason})
        row["accepted"] = bool(metadata_pass and visual_pass and confidence >= config.min_confidence)
    except Exception as exc:
        row["reason"] = f"precheck failed: {exc}"
    return row


def precheck_candidates(input_json: Path, config: PrecheckConfig) -> list[dict[str, Any]]:
    requirement, search_plan, candidates = load_candidates(input_json, platform=config.platform)
    if config.limit > 0:
        candidates = candidates[: config.limit]
    if not candidates:
        raise SystemExit("No candidates to precheck.")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(candidates, 1):
        print(f"[{idx}/{len(candidates)}] 正在预筛选：{item.get('title') or item.get('url')}")
        rows.append(precheck_one(item, requirement=requirement, search_plan=search_plan, config=config))
    return rows

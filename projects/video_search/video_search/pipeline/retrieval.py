"""Stage 2: structured search plan -> platform search results."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .common import common_headers, real_user_home_env, yt_dlp_cmd
from .search_query import DEFAULT_BASE_URL, DEFAULT_MODEL, generate_search_plan


@dataclass
class RetrievalConfig:
    platform: str = "bilibili"
    per_query: int = 5
    max_results: int = 10
    cookies_browser: str = ""
    enrich_metadata: bool = True
    metadata_timeout: int = 45
    pinned_bv: str = ""
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL


def require_yt_dlp() -> None:
    if shutil.which("yt-dlp"):
        return
    try:
        subprocess.run([sys.executable, "-m", "yt_dlp", "--version"], capture_output=True, text=True, check=True)
    except Exception as exc:
        raise RuntimeError("yt-dlp is required. Install it in the active environment.") from exc


def normalize_result(item: dict[str, Any], *, platform: str, query: str) -> dict[str, Any]:
    url = str(item.get("webpage_url") or item.get("url") or item.get("original_url") or "")
    video_id = str(item.get("display_id") or item.get("id") or "")
    if platform == "youtube" and not url.startswith("http") and video_id:
        url = f"https://www.youtube.com/watch?v={video_id}"
    if platform == "bilibili" and not url.startswith("http") and video_id.startswith("BV"):
        url = f"https://www.bilibili.com/video/{video_id}"

    return {
        "platform": platform,
        "type": "video",
        "video_id": video_id,
        "display_id": item.get("display_id"),
        "title": str(item.get("title") or item.get("fulltitle") or video_id),
        "url": url,
        "duration_sec": item.get("duration"),
        "view_count": item.get("view_count"),
        "uploader": item.get("uploader") or item.get("channel"),
        "upload_date": item.get("upload_date"),
        "thumbnail": item.get("thumbnail"),
        "matched_query": query,
        "metadata_enriched": False,
    }


def pinned_bilibili_result(bv: str) -> dict[str, Any]:
    return {
        "platform": "bilibili", "type": "video", "video_id": bv, "display_id": bv,
        "title": f"固定候选 {bv}", "url": f"https://www.bilibili.com/video/{bv}/",
        "duration_sec": None, "view_count": None, "uploader": None, "upload_date": None,
        "thumbnail": None, "matched_query": "高级设置：固定候选", "metadata_enriched": False,
        "pinned": True,
    }


def search_platform(platform: str, query: str, *, limit: int, cookies_browser: str) -> list[dict[str, Any]]:
    require_yt_dlp()
    target = f"bilisearch{limit}:{query}" if platform == "bilibili" else f"ytsearch{limit}:{query}"
    cmd = yt_dlp_cmd() + [target, "--flat-playlist", "--dump-json", "--no-warnings", "--ignore-errors"]
    cmd.extend(common_headers(platform))
    if cookies_browser and platform == "bilibili":
        cmd += ["--cookies-from-browser", cookies_browser]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=real_user_home_env())
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(proc.stderr.strip() or f"yt-dlp search failed: {query}")

    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(normalize_result(json.loads(line), platform=platform, query=query))
        except json.JSONDecodeError:
            continue
    return rows


def dedupe_results(results: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for item in results:
        if item.get("type") == "error":
            out.append(item)
            continue
        key = (str(item.get("platform")), str(item.get("video_id") or item.get("url")))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def bilibili_fallback_queries(requirement: str, plan: dict[str, Any]) -> list[str]:
    structured = plan.get("structured") if isinstance(plan.get("structured"), dict) else {}
    weather = str(structured.get("weather") or "")
    scene = str(structured.get("scene") or "")
    candidates = ["".join(requirement.split())]
    if weather == "night" or any(word in requirement for word in ("夜间", "夜晚", "晚上", "夜景")):
        candidates += ["夜间行车记录仪", "夜晚行车记录仪", "晚上行车记录仪", "行车记录仪 夜间", "夜间开车", "夜间驾驶"]
        if scene == "urban" or "城市" in requirement:
            candidates += ["城市夜景 开车", "夜间城市 行车记录仪"]
    if weather == "sunny" or any(word in requirement for word in ("白天", "晴天", "日间")):
        candidates += ["白天行车记录仪", "行车记录仪 白天", "白天开车"]
    if "跟车" in requirement:
        candidates += ["行车记录仪 跟车", "城市跟车 行车记录仪", "跟车 第一视角"]
    candidates += ["行车记录仪 第一视角", "前挡风玻璃 开车"]

    out: list[str] = []
    seen: set[str] = set()
    for query in candidates:
        query = query.strip()
        if query and query not in seen:
            seen.add(query)
            out.append(query)
    return out


def enrich_metadata(item: dict[str, Any], *, cookies_browser: str, timeout: int) -> dict[str, Any]:
    if item.get("type") == "error":
        return item
    platform = str(item.get("platform") or "")
    url = str(item.get("url") or "")
    if not url:
        return {**item, "metadata_error": "missing url"}

    cmd = yt_dlp_cmd() + ["--dump-json", "--skip-download", "--no-warnings", "--no-playlist", "--ignore-errors"]
    cmd.extend(common_headers(platform))
    if cookies_browser and platform == "bilibili":
        cmd += ["--cookies-from-browser", cookies_browser]
    cmd.append(url)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=real_user_home_env())
    except subprocess.TimeoutExpired:
        return {**item, "metadata_error": f"metadata timeout after {timeout}s"}
    if proc.returncode != 0 and not proc.stdout.strip():
        return {**item, "metadata_error": proc.stderr.strip() or "metadata fetch failed"}

    line = next((line for line in proc.stdout.splitlines() if line.strip()), "")
    if not line:
        return {**item, "metadata_error": "metadata response empty"}
    try:
        meta = json.loads(line)
    except json.JSONDecodeError:
        return {**item, "metadata_error": "metadata JSON parse failed"}

    item.update(
        {
            "video_id": str(meta.get("display_id") or meta.get("id") or item.get("video_id") or ""),
            "numeric_id": str(meta.get("id") or ""),
            "display_id": meta.get("display_id") or item.get("display_id"),
            "title": str(meta.get("title") or meta.get("fulltitle") or item.get("title") or ""),
            "url": str(meta.get("webpage_url") or item.get("url") or ""),
            "duration_sec": meta.get("duration") if meta.get("duration") is not None else item.get("duration_sec"),
            "view_count": meta.get("view_count") if meta.get("view_count") is not None else item.get("view_count"),
            "uploader": meta.get("uploader") or meta.get("channel") or item.get("uploader"),
            "upload_date": meta.get("upload_date") or item.get("upload_date"),
            "thumbnail": meta.get("thumbnail") or item.get("thumbnail"),
            "description": meta.get("description") or "",
            "tags": meta.get("tags") if isinstance(meta.get("tags"), list) else [],
            "metadata_enriched": True,
        }
    )
    item.pop("metadata_error", None)
    return item


def retrieve_from_plan(requirement: str, plan: dict[str, Any], config: RetrievalConfig) -> dict[str, Any]:
    platforms = ["bilibili", "youtube"] if config.platform == "both" else [config.platform]
    results: dict[str, list[dict[str, Any]]] = {}

    for platform in platforms:
        queries = plan["bilibili_queries"] if platform == "bilibili" else plan["youtube_queries"]
        all_items: list[dict[str, Any]] = []
        for query in queries:
            print(f"正在检索 {platform}：{query}", file=sys.stderr)
            try:
                all_items.extend(search_platform(platform, str(query), limit=config.per_query, cookies_browser=config.cookies_browser))
            except Exception as exc:
                all_items.append({"platform": platform, "type": "error", "matched_query": str(query), "error": str(exc)})

        if platform == "bilibili" and not any(item.get("type") != "error" for item in all_items):
            fallback = bilibili_fallback_queries(requirement, plan)
            if fallback:
                plan["bilibili_fallback_queries"] = fallback
            for query in fallback:
                print(f"正在检索 {platform} 备用搜索串：{query}", file=sys.stderr)
                try:
                    all_items.extend(search_platform(platform, query, limit=config.per_query, cookies_browser=config.cookies_browser))
                    if len(dedupe_results(all_items, limit=config.max_results)) >= config.max_results:
                        break
                except Exception as exc:
                    all_items.append({"platform": platform, "type": "error", "matched_query": query, "error": str(exc), "fallback_query": True})

        deduped = dedupe_results(all_items, limit=config.max_results)
        if platform == "bilibili" and config.pinned_bv:
            print(f"已将固定候选置顶：{config.pinned_bv}", file=sys.stderr)
            deduped = dedupe_results([pinned_bilibili_result(config.pinned_bv), *deduped], limit=config.max_results)
        if config.enrich_metadata:
            enriched: list[dict[str, Any]] = []
            for item in deduped:
                if item.get("type") != "error":
                    print(f"正在补全 {platform} 元数据：{item.get('url')}", file=sys.stderr)
                enriched.append(enrich_metadata(item, cookies_browser=config.cookies_browser, timeout=config.metadata_timeout))
            deduped = enriched
        results[platform] = deduped

    return {
        "requirement": requirement,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "search_plan": plan,
        "results": results,
    }


def retrieve_videos(requirement: str, config: RetrievalConfig) -> dict[str, Any]:
    plan = generate_search_plan(requirement, model=config.model, base_url=config.base_url)
    return retrieve_from_plan(requirement, plan, config)

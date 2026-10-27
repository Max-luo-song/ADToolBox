"""Core retrieval service.

The public entry point is ``retrieve_video(requirement, config=None)``.
CLI and agent callers should use this module instead of orchestrating pipeline
stages themselves.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import SearchConfig
from .pipeline.common import safe_name
from .pipeline.downloader import DownloadConfig, download_candidates
from .pipeline.indexer import IndexConfig, index_segments
from .pipeline.postfilter import PostFilterConfig, filter_one, import_source_video
from .pipeline.prefilter import PrecheckConfig, precheck_candidates
from .pipeline.retrieval import RetrievalConfig, retrieve_from_plan
from .pipeline.scene_segmenter import SceneConfig, split_scenes
from .pipeline.search_query import generate_search_plan
from .first_person_detector import iter_videos


SEARCH_PLAN_LOG = "01_search_query.json"
RETRIEVAL_LOG = "02_retrieval.json"
PREFILTER_LOG = "03_prefilter.jsonl"
DOWNLOAD_LOG = "04_download.jsonl"
POSTFILTER_LOG = "05_postfilter.jsonl"
SCENE_LOG = "06_scene_segmenter.jsonl"
INDEX_LOG = "07_indexer.jsonl"
SEARCH_RESULTS_LOG = "08_search_results.json"
RERANK_LOG = "09_rerank.json"
RESULT_LOG = "result.json"


@dataclass
class VideoSearchResult:
    requirement: str
    video_path: str
    run_dir: str
    mode: str
    match_score: float
    timings: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _new_run_dir(config: SearchConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = config.logs_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir.resolve()


def _candidate_count(retrieval: dict[str, Any]) -> int:
    results = retrieval.get("results") or {}
    return sum(
        1
        for rows in results.values()
        for row in rows
        if isinstance(row, dict) and row.get("type") != "error" and row.get("url")
    )


def _search_index(
    query: str,
    config: SearchConfig,
    *,
    source_files: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search the Qwen SentrySearch index, optionally restricted to source files."""
    db_path = config.runtime_home / ".sentrysearch" / "db"
    if not db_path.exists():
        return []

    from .vendor.sentrysearch.embedder import get_embedder, reset_embedder
    from .vendor.sentrysearch.store import SentryStore

    store = SentryStore(db_path=db_path, backend="qwen_api")
    total = store.collection.count()
    if total <= 0:
        return []

    allowed: list[str] | None = None
    if source_files:
        allowed = sorted({str(Path(path).expanduser().resolve()) for path in source_files})

    try:
        embedding = get_embedder("qwen_api").embed_query(query)
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": min(config.search_results, total),
            "include": ["metadatas", "distances"],
        }
        if allowed:
            if len(allowed) == 1:
                kwargs["where"] = {"source_file": allowed[0]}
            else:
                kwargs["where"] = {"source_file": {"$in": allowed}}

        raw = store.collection.query(**kwargs)
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        hits: list[dict[str, Any]] = []
        for metadata, distance in zip(metadatas, distances):
            hits.append(
                {
                    "source_file": str(metadata["source_file"]),
                    "start_time": float(metadata["start_time"]),
                    "end_time": float(metadata["end_time"]),
                    "similarity_score": 1.0 - float(distance),
                }
            )
        hits.sort(key=lambda item: item["similarity_score"], reverse=True)
        return hits
    finally:
        reset_embedder()


def _trim_hit(
    hit: dict[str, Any],
    requirement: str,
    config: SearchConfig,
    *,
    label: str = "",
    crf: int = 0,
) -> Path:
    from .vendor.sentrysearch.chunker import _get_video_duration
    from .vendor.sentrysearch.trimmer import trim_clip

    duration = _get_video_duration(str(hit["source_file"]))
    clip_length = min(10.0, duration)
    start = min(max(0.0, float(hit["start_time"])), duration - clip_length)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = f"_{label}" if label else ""
    output = config.output_dir / f"{safe_name(requirement, limit=80)}_{stamp}{suffix}.mp4"
    trim_clip(
        source_file=str(hit["source_file"]),
        start_time=start,
        end_time=start + clip_length,
        output_path=str(output),
        padding=0.0,
        accurate=True,
        crf=crf,
    )
    return output.resolve()


def _export_reranked_hits(
    details: dict[str, Any], requirement: str, config: SearchConfig, run_dir: Path,
) -> Path:
    output_config = replace(config, output_dir=config.output_dir / run_dir.name)
    candidates = {item["candidate_index"]: item for item in details["candidates"]}
    ranked = sorted(details["ranked_results"], key=lambda item: item["relevance_score"], reverse=True)
    for rank, result in enumerate(ranked, start=1):
        score = float(result["relevance_score"])
        label = f"rank{rank:02d}_score{score:.6f}"
        result["video_path"] = str(_trim_hit(
            candidates[result["candidate_index"]], requirement, output_config,
            label=f"{label}_compressed", crf=28,
        ))
        if rank == 1:
            result["lossless_video_path"] = str(_trim_hit(
                candidates[result["candidate_index"]], requirement, output_config,
                label=f"{label}_lossless",
            ))
    details["ranked_results"] = ranked
    return Path(ranked[0]["lossless_video_path"])


def _finish(
    *,
    requirement: str,
    run_dir: Path,
    mode: str,
    hit: dict[str, Any],
    video_path: Path,
    timings: dict[str, float],
    total_start: float,
) -> VideoSearchResult:
    timings["total_seconds"] = round(time.perf_counter() - total_start, 3)
    result = VideoSearchResult(
        requirement=requirement,
        video_path=str(video_path),
        run_dir=str(run_dir),
        mode=mode,
        match_score=round(float(hit["similarity_score"]), 6),
        timings=timings,
    )
    _write_json(run_dir / RESULT_LOG, {"status": "success", **result.to_dict()})
    return result



def _ensure_dashscope_key() -> None:
    """Reuse the root Agent key when it is also a DashScope key."""
    if os.environ.get("DASHSCOPE_API_KEY", "").strip():
        return
    agent_key = os.environ.get("LLM_API_KEY", "").strip()
    if agent_key:
        os.environ["DASHSCOPE_API_KEY"] = agent_key
        return
    raise RuntimeError("DASHSCOPE_API_KEY or LLM_API_KEY is required")


def index_local(
    path: str | Path,
    config: SearchConfig | None = None,
) -> dict[str, Any]:
    """Import local videos into this component and add them to the local index."""
    _ensure_dashscope_key()
    config = config or SearchConfig()
    config.validate()

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Path not found: {source}")

    videos = iter_videos(source)
    if not videos:
        raise RuntimeError(f"No videos found: {source}")

    processed: list[str] = []
    newly_indexed: list[str] = []
    total_seconds = 0.0

    for video in videos:
        local_video = video.resolve()
        if config.import_source_videos:
            local_video, _ = import_source_video(video, library_dir=config.library_dir)
        indexed, seconds = index_segments(
            [{"segment_video": str(local_video)}],
            IndexConfig(
                runtime_home=config.runtime_home,
                chunk_duration=config.chunk_duration,
                preprocess_chunks=config.preprocess_index_chunks,
            ),
        )
        processed.append(str(local_video))
        newly_indexed.extend(indexed)
        total_seconds += seconds

    return {
        "source": str(source),
        "processed_files": processed,
        "newly_indexed_files": newly_indexed,
        "indexed_file_count": len(newly_indexed),
        "db_path": str(config.runtime_home / ".sentrysearch" / "db"),
        "seconds": round(total_seconds, 3),
    }


def search_local(
    requirement: str,
    config: SearchConfig | None = None,
) -> VideoSearchResult:
    """Search only the existing local index. Never fall back to the internet."""
    requirement = requirement.strip()
    if not requirement:
        raise ValueError("requirement must not be empty")
    _ensure_dashscope_key()

    config = config or SearchConfig()
    config.validate()
    run_dir = _new_run_dir(config)
    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    try:
        stage_start = time.perf_counter()
        hits = _search_index(requirement, config)
        timings["local_search_seconds"] = round(time.perf_counter() - stage_start, 3)
        _write_json(
            run_dir / "local_search.json",
            {
                "threshold": config.match_threshold,
                "hits": hits,
            },
        )

        if not hits:
            raise RuntimeError("Local index is empty or contains no matching video")

        if float(hits[0]["similarity_score"]) < config.match_threshold:
            raise RuntimeError(
                "Best local match score "
                f"{float(hits[0]['similarity_score']):.3f} is below "
                f"threshold {config.match_threshold:.3f}"
            )

        from .reranker import rerank_hits

        best, rerank_details = rerank_hits(requirement, hits, run_dir=run_dir)
        video_path = _export_reranked_hits(rerank_details, requirement, config, run_dir)
        _write_json(run_dir / RERANK_LOG, rerank_details)
        return _finish(
            requirement=requirement,
            run_dir=run_dir,
            mode="local_hit_reranked",
            hit=best,
            video_path=video_path,
            timings=timings,
            total_start=total_start,
        )
    except Exception as exc:
        timings["total_seconds"] = round(time.perf_counter() - total_start, 3)
        _write_json(
            run_dir / RESULT_LOG,
            {
                "status": "error",
                "requirement": requirement,
                "run_dir": str(run_dir),
                "error": {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                },
                "timings": timings,
            },
        )
        raise


def search_internet(
    requirement: str,
    config: SearchConfig | None = None,
) -> VideoSearchResult:
    """Run internet retrieval explicitly, without searching the local index first."""
    config = config or SearchConfig()
    return retrieve_video(requirement, replace(config, local_first=False))

def retrieve_video(
    requirement: str,
    config: SearchConfig | None = None,
) -> VideoSearchResult:
    """Retrieve one video clip matching a natural-language requirement.

    The service first searches the existing local index when ``local_first`` is
    enabled. If there is no confident local match, it performs internet
    retrieval, filtering, scene splitting and indexing, then searches only the
    scenes produced by the current run.
    """
    requirement = requirement.strip()
    if not requirement:
        raise ValueError("requirement must not be empty")
    _ensure_dashscope_key()

    config = config or SearchConfig()
    config.validate()
    run_dir = _new_run_dir(config)
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    try:
        if config.local_first:
            stage_start = time.perf_counter()
            local_hits = _search_index(requirement, config)
            timings["00_local_search_seconds"] = round(
                time.perf_counter() - stage_start, 3
            )
            _write_json(
                run_dir / "00_local_search.json",
                {
                    "threshold": config.match_threshold,
                    "hits": local_hits,
                },
            )
            if (
                local_hits
                and float(local_hits[0]["similarity_score"]) >= config.match_threshold
            ):
                video_path = _trim_hit(local_hits[0], requirement, config)
                return _finish(
                    requirement=requirement,
                    run_dir=run_dir,
                    mode="local_hit",
                    hit=local_hits[0],
                    video_path=video_path,
                    timings=timings,
                    total_start=total_start,
                )

        stage_start = time.perf_counter()
        plan = generate_search_plan(requirement)
        timings["01_search_query_seconds"] = round(
            time.perf_counter() - stage_start, 3
        )
        _write_json(
            run_dir / SEARCH_PLAN_LOG,
            {"requirement": requirement, "search_plan": plan},
        )

        stage_start = time.perf_counter()
        retrieval = retrieve_from_plan(
            requirement,
            plan,
            RetrievalConfig(
                platform=config.platform,
                per_query=config.per_query,
                max_results=config.max_results,
                cookies_browser=config.cookies_browser,
            ),
        )
        timings["02_retrieval_seconds"] = round(
            time.perf_counter() - stage_start, 3
        )
        _write_json(
            run_dir / RETRIEVAL_LOG,
            {
                "requirement": requirement,
                "created_at": retrieval.get("created_at"),
                "search_plan": plan,
                "results": retrieval.get("results", {}),
            },
        )
        if _candidate_count(retrieval) == 0:
            raise RuntimeError("No video candidates were retrieved")

        stage_start = time.perf_counter()
        prefilter_rows = precheck_candidates(
            run_dir / RETRIEVAL_LOG,
            PrecheckConfig(
                platform=config.platform,
                limit=config.precheck_limit,
                sample_clips=config.sample_clips,
                cookies_browser=config.cookies_browser,
                preview_clip_dir=run_dir / "03_prefilter_clips",
                preview_frame_dir=run_dir / "03_prefilter_frames",
            ),
        )
        timings["03_prefilter_seconds"] = round(
            time.perf_counter() - stage_start, 3
        )
        _write_jsonl(run_dir / PREFILTER_LOG, prefilter_rows)
        if not any(row.get("accepted") for row in prefilter_rows):
            raise RuntimeError("No candidate passed the preview filter")

        stage_start = time.perf_counter()
        download_rows = download_candidates(
            run_dir / PREFILTER_LOG,
            DownloadConfig(
                output_dir=config.library_dir,
                limit=config.download_limit,
                cookies_browser=config.cookies_browser,
            ),
        )
        timings["04_download_seconds"] = round(
            time.perf_counter() - stage_start, 3
        )
        _write_jsonl(run_dir / DOWNLOAD_LOG, download_rows)
        downloaded = [
            Path(str(row["video_path"])).expanduser().resolve()
            for row in download_rows
            if row.get("downloaded") and row.get("video_path")
        ]
        if not downloaded:
            raise RuntimeError("All candidate video downloads failed")

        postfilter_rows: list[dict[str, Any]] = []
        scene_rows: list[dict[str, Any]] = []
        index_rows: list[dict[str, Any]] = []
        postfilter_seconds = 0.0
        scene_seconds = 0.0
        index_seconds = 0.0

        for video in downloaded:
            stage_start = time.perf_counter()
            source_video, postfilter_row = filter_one(
                video,
                PostFilterConfig(
                    library_dir=config.library_dir,
                    frames=config.frames,
                    min_confidence=config.min_confidence,
                    import_local=True,
                ),
            )
            postfilter_seconds += time.perf_counter() - stage_start
            postfilter_rows.append(postfilter_row)
            if not postfilter_row.get("accepted"):
                continue

            stage_start = time.perf_counter()
            scenes = split_scenes(
                source_video,
                SceneConfig(
                    output_dir=config.library_dir,
                    threshold=config.scene_threshold,
                ),
            )
            scene_seconds += time.perf_counter() - stage_start
            scene_rows.extend(scenes)

            stage_start = time.perf_counter()
            newly_indexed, elapsed_seconds = index_segments(
                scenes,
                IndexConfig(
                    runtime_home=config.runtime_home,
                    chunk_duration=config.chunk_duration,
                ),
            )
            index_seconds += time.perf_counter() - stage_start
            index_rows.append(
                {
                    "source_video": str(source_video),
                    "scene_count": len(scenes),
                    "newly_indexed_segments": newly_indexed,
                    "reported_index_seconds": elapsed_seconds,
                }
            )

        timings["05_postfilter_seconds"] = round(postfilter_seconds, 3)
        timings["06_scene_segmenter_seconds"] = round(scene_seconds, 3)
        timings["07_indexer_seconds"] = round(index_seconds, 3)
        _write_jsonl(run_dir / POSTFILTER_LOG, postfilter_rows)
        _write_jsonl(run_dir / SCENE_LOG, scene_rows)
        _write_jsonl(run_dir / INDEX_LOG, index_rows)

        current_scene_files = [
            str(Path(str(row["segment_video"])).expanduser().resolve())
            for row in scene_rows
            if row.get("segment_video")
        ]
        if not current_scene_files:
            raise RuntimeError("No downloaded video passed the full-video filter")

        stage_start = time.perf_counter()
        hits = _search_index(
            requirement,
            config,
            source_files=current_scene_files,
        )
        timings["08_search_results_seconds"] = round(
            time.perf_counter() - stage_start, 3
        )
        _write_json(
            run_dir / SEARCH_RESULTS_LOG,
            {
                "threshold": config.match_threshold,
                "source_files": current_scene_files,
                "hits": hits,
            },
        )
        if not hits:
            raise RuntimeError("No indexed scene matched the requirement")
        if float(hits[0]["similarity_score"]) < config.match_threshold:
            raise RuntimeError(
                "Best scene match score "
                f"{float(hits[0]['similarity_score']):.3f} is below "
                f"threshold {config.match_threshold:.3f}"
            )

        video_path = _trim_hit(hits[0], requirement, config)
        return _finish(
            requirement=requirement,
            run_dir=run_dir,
            mode="internet_retrieval",
            hit=hits[0],
            video_path=video_path,
            timings=timings,
            total_start=total_start,
        )
    except Exception as exc:
        timings["total_seconds"] = round(time.perf_counter() - total_start, 3)
        _write_json(
            run_dir / RESULT_LOG,
            {
                "status": "error",
                "requirement": requirement,
                "run_dir": str(run_dir),
                "error": {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                },
                "timings": timings,
            },
        )
        raise

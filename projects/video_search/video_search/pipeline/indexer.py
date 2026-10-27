"""Stage 7: index scene clips with the vendored Qwen SentrySearch runtime."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from ..vendor.sentrysearch.chunker import _get_video_duration, chunk_video, expected_chunk_spans, preprocess_chunk
from ..vendor.sentrysearch.embedder import get_embedder, reset_embedder
from ..vendor.sentrysearch.store import SentryStore


@dataclass
class IndexConfig:
    runtime_home: Path
    chunk_duration: int = 12
    overlap: int = 0
    preprocess_chunks: bool = False


def elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 3)


def _store(config: IndexConfig) -> SentryStore:
    db_path = config.runtime_home / ".sentrysearch" / "db"
    return SentryStore(db_path=db_path, backend="qwen_api")


def is_fully_indexed(video: Path, config: IndexConfig) -> bool:
    abs_path = str(video.resolve())
    spans = expected_chunk_spans(
        _get_video_duration(abs_path),
        chunk_duration=config.chunk_duration,
        overlap=config.overlap,
    )
    if not spans:
        return False
    store = _store(config)
    return all(store.has_chunk(store.make_chunk_id(abs_path, start)) for start, _ in spans)


def index_video(video: Path, config: IndexConfig) -> tuple[bool, float]:
    start = time.perf_counter()
    config.runtime_home.mkdir(parents=True, exist_ok=True)
    if is_fully_indexed(video, config):
        return False, elapsed(start)

    abs_path = str(video.resolve())
    store = _store(config)
    chunks = chunk_video(
        abs_path,
        chunk_duration=config.chunk_duration,
        overlap=config.overlap,
    )
    if not chunks:
        return False, elapsed(start)

    did_index = False
    temp_dir = Path(chunks[0]["chunk_path"]).parent
    embedder = get_embedder("qwen_api")
    try:
        for chunk in chunks:
            chunk_id = store.make_chunk_id(abs_path, float(chunk["start_time"]))
            if store.has_chunk(chunk_id):
                continue
            embedding_path = chunk["chunk_path"]
            if config.preprocess_chunks:
                embedding_path = preprocess_chunk(embedding_path, target_resolution=None, target_fps=10)
            embedding = embedder.embed_video_chunk(str(embedding_path))
            store.add_chunk(
                chunk_id,
                embedding,
                {
                    "source_file": abs_path,
                    "start_time": float(chunk["start_time"]),
                    "end_time": float(chunk["end_time"]),
                },
            )
            did_index = True
    finally:
        reset_embedder()
        shutil.rmtree(temp_dir, ignore_errors=True)

    return did_index, elapsed(start)


def index_segments(segments: list[dict], config: IndexConfig) -> tuple[list[str], float]:
    indexed: list[str] = []
    total = 0.0
    for segment in segments:
        segment_video = Path(segment["segment_video"])
        did_index, seconds = index_video(segment_video, config)
        total = round(total + seconds, 3)
        if did_index:
            indexed.append(str(segment_video.resolve()))
    return indexed, total

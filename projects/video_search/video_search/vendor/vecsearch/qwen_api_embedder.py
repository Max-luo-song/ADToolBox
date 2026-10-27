"""DashScope Qwen multimodal embedding backend.

This backend uses qwen3-vl-embedding through DashScope's dedicated
multimodal-embedding API. Local video chunks are passed directly as base64
video data URLs. There is intentionally no frame-sampling fallback.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import time
from collections import deque
from pathlib import Path

import requests

from .base_embedder import BaseEmbedder


API_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
MODEL = "qwen3-vl-embedding"
DIMENSIONS = 768
DEFAULT_RPM = 55
MIN_VIDEO_BYTES = 24 * 1024
MAX_VIDEO_BYTES = 50_700 * 1024


class QwenAPIKeyError(RuntimeError):
    """Raised when DASHSCOPE_API_KEY is missing."""


class QwenAPIError(RuntimeError):
    """Raised when DashScope returns an API error."""


class _RateLimiter:
    def __init__(self, max_per_minute: int = DEFAULT_RPM):
        self._max = max_per_minute
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= 60:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max:
            sleep_for = 60.0 - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


def _retry(fn, *, max_retries: int = 2, initial_delay: float = 2.0, max_delay: float = 2.0):
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc).lower()
            retryable = any(
                code in msg
                for code in (
                    "429",
                    "503",
                    "rate",
                    "timeout",
                    "ssl",
                    "connection",
                    "connectionerror",
                    "unexpected_eof",
                    "eof occurred",
                    "failed to resolve",
                    "nameresolutionerror",
                )
            )
            if not retryable or attempt == max_retries:
                raise
            wait = min(delay, max_delay)
            print(
                f"  DashScope embedding error (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {wait:.0f}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay *= 2


def _file_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


class QwenAPIEmbedder(BaseEmbedder):
    """DashScope qwen3-vl-embedding backend."""

    def __init__(
        self,
        dimensions: int = DIMENSIONS,
        fps: float = 1.0,
    ):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise QwenAPIKeyError(
                "DASHSCOPE_API_KEY is not set.\n\n"
                "Set it manually:\n"
                "  export DASHSCOPE_API_KEY=your-key\n"
            )
        self._api_key = api_key
        self._dimensions = int(os.environ.get("DASHSCOPE_EMBEDDING_DIM", dimensions))
        self._fps = float(os.environ.get("DASHSCOPE_EMBEDDING_FPS", fps))
        self._limiter = _RateLimiter()

    def _call(self, contents: list[dict], *, enable_fusion: bool, verbose: bool = False) -> list[float]:
        payload = {
            "model": MODEL,
            "input": {"contents": contents},
            "parameters": {
                "dimension": self._dimensions,
                "output_type": "dense",
                "enable_fusion": enable_fusion,
            },
        }
        if self._fps is not None:
            payload["parameters"]["fps"] = self._fps

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        self._limiter.wait()
        t0 = time.monotonic()

        def send():
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=120)
            try:
                body = response.json()
            except Exception:
                body = {"message": response.text}
            if response.status_code != 200 or body.get("code"):
                raise QwenAPIError(
                    f"DashScope embedding error: status={response.status_code}, "
                    f"code={body.get('code')}, message={body.get('message')}"
                )
            return body

        body = _retry(send)
        elapsed = time.monotonic() - t0
        embeddings = body.get("output", {}).get("embeddings", [])
        if not embeddings:
            raise QwenAPIError(f"DashScope response contains no embeddings: {body}")

        embedding = embeddings[0]["embedding"]
        if verbose:
            print(
                f"    [verbose] qwen_api dims={len(embedding)}, "
                f"fusion={enable_fusion}, api_time={elapsed:.2f}s",
                file=sys.stderr,
            )
        return embedding

    def embed_video_chunk(self, chunk_path: str, verbose: bool = False) -> list[float]:
        path = Path(chunk_path)
        size = path.stat().st_size
        if size < MIN_VIDEO_BYTES or size > MAX_VIDEO_BYTES:
            raise QwenAPIError(
                "DashScope qwen3-vl-embedding video input requires chunk size "
                f"between 24KB and 50,700KB. Got {size} bytes for {chunk_path}. "
                "Adjust --chunk-duration, --target-resolution, --target-fps, "
                "or --no-preprocess. No frame fallback is enabled."
            )

        if verbose:
            print(
                f"    [verbose] qwen_api sending video data URL "
                f"({size / 1024:.0f}KB)",
                file=sys.stderr,
            )
        return self._call([{"video": _file_data_url(path)}], enable_fusion=False, verbose=verbose)

    def embed_query(self, query_text: str, verbose: bool = False) -> list[float]:
        return self._call([{"text": query_text}], enable_fusion=False, verbose=verbose)

    def embed_image(self, image_path: str, verbose: bool = False) -> list[float]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return self._call([{"image": _file_data_url(path)}], enable_fusion=False, verbose=verbose)

    def dimensions(self) -> int:
        return self._dimensions

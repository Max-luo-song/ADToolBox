"""OSS-backed qwen3-vl-rerank for local video candidates."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import tempfile
import time
import urllib.parse
from email.utils import formatdate
from pathlib import Path
from typing import Any

import requests

from .vendor.sentrysearch.chunker import preprocess_chunk
from .vendor.sentrysearch.trimmer import trim_clip

from .config import SearchConfig


RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
MAX_VIDEO_CANDIDATES = 4
MAX_RERANK_VIDEO_BYTES = 50_000 * 1024


def _settings() -> tuple[str, str, str, str, str, str]:
    config = SearchConfig()
    values = {
        key: os.environ.get(key, "").strip()
        for key in (
            "OSS_ACCESS_KEY_ID",
            "OSS_ACCESS_KEY_SECRET",
        )
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise RuntimeError("OSS rerank requires: " + ", ".join(missing))
    bucket = os.environ.get("OSS_BUCKET", "").strip() or config.oss_bucket
    endpoint = (
        os.environ.get("OSS_ENDPOINT", "").strip() or config.oss_endpoint
    ).replace("https://", "").replace("http://", "").rstrip("/")
    if not bucket or not endpoint:
        raise RuntimeError("Set OSS_BUCKET and OSS_ENDPOINT for video reranking")
    return (
        values["OSS_ACCESS_KEY_ID"],
        values["OSS_ACCESS_KEY_SECRET"],
        bucket,
        endpoint,
        (os.environ.get("OSS_PREFIX", "").strip() or config.oss_prefix).strip("/") or "toolbox-rerank",
        os.environ.get("OSS_SECURITY_TOKEN", "").strip(),
    )


def _authorization(method: str, access: str, secret: str, resource: str, *, content_type: str = "", date: str = "") -> str:
    string_to_sign = f"{method}\n\n{content_type}\n{date}\n{resource}"
    signature = base64.b64encode(
        hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1).digest()
    ).decode("ascii")
    return f"OSS {access}:{signature}"


def _upload(local_path: Path, key: str, settings: tuple[str, str, str, str, str, str]) -> str:
    access, secret, bucket, endpoint, _, security_token = settings
    resource = f"/{bucket}/{key}"
    url = f"https://{bucket}.{endpoint}/{urllib.parse.quote(key, safe='/')}"
    date = formatdate(usegmt=True)
    headers = {
        "Date": date,
        "Content-Type": "video/mp4",
        "Authorization": _authorization("PUT", access, secret, resource, content_type="video/mp4", date=date),
    }
    if security_token:
        headers["x-oss-security-token"] = security_token
    with local_path.open("rb") as video_file:
        response = requests.put(url, data=video_file, headers=headers, timeout=180)
    if response.status_code // 100 != 2:
        raise RuntimeError(f"OSS upload failed ({response.status_code}): {response.text[:300]}")

    expires = str(int(time.time()) + 600)
    string_to_sign = f"GET\n\n\n{expires}\n{resource}"
    signature = base64.b64encode(
        hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1).digest()
    ).decode("ascii")
    query = {
        "OSSAccessKeyId": access,
        "Expires": expires,
        "Signature": signature,
    }
    if security_token:
        query["security-token"] = security_token
    return url + "?" + urllib.parse.urlencode(query)


def _delete(url: str, settings: tuple[str, str, str, str, str, str]) -> None:
    access, secret, bucket, endpoint, _, security_token = settings
    parsed = urllib.parse.urlsplit(url)
    key = urllib.parse.unquote(parsed.path.lstrip("/"))
    resource = f"/{bucket}/{key}"
    date = formatdate(usegmt=True)
    headers = {"Date": date, "Authorization": _authorization("DELETE", access, secret, resource, date=date)}
    if security_token:
        headers["x-oss-security-token"] = security_token
    response = requests.delete(f"https://{bucket}.{endpoint}/{urllib.parse.quote(key, safe='/')}", headers=headers, timeout=60)
    if response.status_code // 100 != 2 and response.status_code != 404:
        raise RuntimeError(f"OSS cleanup failed ({response.status_code}): {response.text[:300]}")


def rerank_hits(requirement: str, hits: list[dict[str, Any]], *, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rerank up to four recalled local video chunks using direct video input."""
    candidates = hits[:MAX_VIDEO_CANDIDATES]
    if not candidates:
        raise ValueError("Cannot rerank an empty candidate list")
    settings = _settings()
    if not os.environ.get("DASHSCOPE_API_KEY", "").strip():
        raise RuntimeError("DASHSCOPE_API_KEY is required for video rerank")

    uploaded: list[str] = []
    details: dict[str, Any] = {"model": "qwen3-vl-rerank", "candidate_count": len(candidates), "candidates": []}
    try:
        with tempfile.TemporaryDirectory(prefix="sentry-rerank-") as temp_dir:
            documents = []
            for index, hit in enumerate(candidates):
                clip = Path(temp_dir) / f"candidate_{index:02d}.mp4"
                trim_clip(str(hit["source_file"]), float(hit["start_time"]), float(hit["end_time"]), str(clip))
                upload_clip = Path(preprocess_chunk(str(clip), target_resolution=None, target_fps=10))
                if not 24 * 1024 <= upload_clip.stat().st_size <= MAX_RERANK_VIDEO_BYTES:
                    raise RuntimeError(f"Rerank video exceeds API size limits: {upload_clip}")
                key = f"{settings[4]}/run-{run_dir.name}-{index:02d}-{int(time.time())}.mp4"
                signed_url = _upload(upload_clip, key, settings)
                uploaded.append(signed_url)
                documents.append({"video": signed_url})
                details["candidates"].append({**hit, "candidate_index": index})

            payload = {
                "model": "qwen3-vl-rerank",
                "input": {"query": {"text": requirement}, "documents": documents},
                "parameters": {
                    "top_n": len(documents),
                    "return_documents": True,
                    "fps": 1.0,
                    "instruct": "按照查询判断视频整体场景和关键交通行为的相关性，并对候选视频排序。",
                },
            }
            response = requests.post(
                RERANK_URL,
                headers={"Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}", "Content-Type": "application/json"},
                json=payload,
                timeout=240,
            )
            body = response.json()
            if response.status_code != 200 or body.get("code"):
                raise RuntimeError(f"qwen3-vl-rerank failed ({response.status_code}): {body.get('message') or body.get('code')}")
            results = body.get("output", {}).get("results") or []
            if not results:
                raise RuntimeError("qwen3-vl-rerank returned no results")
            ranked = []
            for result in results:
                index = int(result["index"])
                ranked.append({"candidate_index": index, "relevance_score": float(result["relevance_score"])})
            details["ranked_results"] = ranked
            details["request_id"] = body.get("request_id")
            winner = max(ranked, key=lambda item: item["relevance_score"])
            selected = dict(candidates[winner["candidate_index"]])
            selected["retrieval_score"] = selected["similarity_score"]
            selected["similarity_score"] = winner["relevance_score"]
            return selected, details
    finally:
        for signed_url in uploaded:
            try:
                _delete(signed_url, settings)
            except Exception:
                pass

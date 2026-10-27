"""Shared helpers for pipeline stages."""

from __future__ import annotations

import base64
import json
import os
import pwd
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests


def run(
    cmd: list[str],
    *,
    timeout: int = 120,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check, env=env)


def real_user_home_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = pwd.getpwuid(os.getuid()).pw_dir
    return env


def yt_dlp_cmd() -> list[str]:
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return [sys.executable, "-m", "yt_dlp"]


def safe_name(value: str, *, limit: int = 100) -> str:
    value = value.strip() or "unknown"
    value = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", value)
    return value[:limit].strip("_") or "unknown"


def common_headers(platform: str) -> list[str]:
    headers = [
        "--add-header",
        "User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
        "--add-header",
        "Accept-Language:zh-CN,zh;q=0.9,en;q=0.8",
    ]
    if platform == "bilibili":
        headers.extend(
            [
                "--add-header",
                "Referer:https://www.bilibili.com/",
                "--add-header",
                "Origin:https://www.bilibili.com",
            ]
        )
    return headers


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines[1:] if not line.strip().startswith("```")).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def dashscope_chat_completion(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int = 120,
    retries: int = 2,
    retry_delays: tuple[float, ...] = (2.0, 2.0),
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                data=json.dumps(payload, ensure_ascii=False),
                timeout=timeout,
            )
            body = response.json()
            if response.status_code != 200 or body.get("error"):
                raise RuntimeError(f"DashScope API error: status={response.status_code}, body={body}")
            return body
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.SSLError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            delay = retry_delays[min(attempt, len(retry_delays) - 1)]
            print(f"DashScope 请求失败（{exc.__class__.__name__}），{delay:g}s 后重试...", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError(f"DashScope request failed after {retries + 1} attempts: {last_error}") from last_error

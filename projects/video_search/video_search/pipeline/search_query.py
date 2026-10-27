"""Stage 1: requirement -> structured search plan."""

from __future__ import annotations

import json
import os
from typing import Any

from .common import dashscope_chat_completion, extract_json


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-vl-flash"

SYSTEM_PROMPT = """你是自动驾驶视频数据集检索专家。
用户会描述想找的视频内容。请把需求解析为结构化检索计划，并分别生成适合 YouTube 和 Bilibili 的搜索串。

要求：
- 默认目标是汽车第一视角/行车记录仪/前挡风玻璃外道路视角。
- YouTube 搜索串使用英文。
- Bilibili 搜索串使用中文。
- 搜索串要适合直接输入平台搜索框。
- 只输出 JSON，不要 Markdown。

JSON 格式：
{
  "original_input": "用户原始输入",
  "structured": {
    "scene": "highway | urban | rural | parking | tunnel | unknown",
    "weather": "sunny | rainy | snowy | foggy | night | unknown",
    "events": ["事件关键词"],
    "objects": ["目标对象关键词"],
    "viewpoint": "first_person_dashcam | other | unknown",
    "extra_context": "补充说明"
  },
  "youtube_queries": ["英文搜索串，3到5条"],
  "bilibili_queries": ["中文搜索串，3到5条"],
  "negative_keywords": ["应排除的内容关键词"]
}
"""


def generate_search_plan(
    requirement: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY or OPENAI_API_KEY.")

    body = dashscope_chat_completion(
        base_url=base_url,
        api_key=api_key,
        payload={
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": requirement},
            ],
        },
    )

    plan = extract_json(body["choices"][0]["message"]["content"])
    plan.setdefault("original_input", requirement)
    plan.setdefault("youtube_queries", [])
    plan.setdefault("bilibili_queries", [])
    plan.setdefault("negative_keywords", [])
    return plan

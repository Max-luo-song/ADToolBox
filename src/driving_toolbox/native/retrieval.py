import math
import re
import shutil
from collections import Counter
from pathlib import Path

from ..core.errors import InputError
from ..core.io import file_path, read_json


def _tokens(text):
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())
    return words


def search(data):
    catalog_path = file_path(data["catalog_path"], "catalog_path")
    records = read_json(catalog_path)
    if not isinstance(records, list) or not records:
        raise InputError("Video catalog must be a non-empty array")
    query = Counter(_tokens(data["requirement"]))
    docs = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("caption"), str):
            raise InputError("Each video record needs caption and video_path")
        path = Path(record.get("video_path", "")).expanduser()
        if not path.is_absolute():
            path = catalog_path.parent / path
        path = file_path(str(path), "video_path")
        docs.append(
            (record, path, Counter(_tokens(record["caption"] + " " + " ".join(record.get("tags", [])))))
        )
    avg_len = sum(sum(tokens.values()) for _, _, tokens in docs) / len(docs) or 1
    ranked = []
    for record, path, tokens in docs:
        score = 0.0
        for term in query:
            freq = tokens.get(term, 0)
            df = sum(term in counts for _, _, counts in docs)
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            score += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * sum(tokens.values()) / avg_len))
        if score > data.get("min_score", 0):
            ranked.append({"video_path": str(path), "caption": record["caption"], "score": round(score, 6)})
    ranked.sort(key=lambda x: (-x["score"], x["video_path"]))
    ranked = ranked[: data.get("top_k", 5)]
    if not ranked:
        return {"matched": False, "candidates": [], "method": "caption_bm25"}
    selected = ranked[0]["video_path"]
    if data.get("export_path"):
        destination = Path(data["export_path"]).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected, destination)
        selected = str(destination)
    return {
        "matched": True,
        "video_path": selected,
        "candidates": ranked,
        "method": "caption_bm25",
        "_warnings": [
            "Caption retrieval is lexical; visual-semantic retrieval requires an optional backend."
        ],
    }

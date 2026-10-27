"""OpenAI-compatible HTTP transport; no agent state or tool-selection logic."""

import base64
import json
import mimetypes
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import BackendError, DependencyError, InputError
from .schema import validate


def ask_json(system, payload, schema, config, images=()):
    settings = config.get("llm", {})
    key = os.environ.get(settings.get("api_key_env", "ADTOOLBOX_API_KEY"))
    base = settings.get("base_url") or os.environ.get("ADTOOLBOX_BASE_URL")
    model = settings.get("model") or os.environ.get("ADTOOLBOX_MODEL")
    if not key or not base or not model:
        raise DependencyError(
            "Configure ADTOOLBOX_API_KEY, ADTOOLBOX_BASE_URL and ADTOOLBOX_MODEL; see docs/configuration.md"
        )
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    content = [{"type": "text", "text": text}]
    for path in images:
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise InputError("Model image inputs must be JPEG, PNG or WebP")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64," + encoded}})
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system + " Return only a JSON object matching: " + json.dumps(schema),
            },
            {"role": "user", "content": content if images else text},
        ],
        "max_tokens": int(settings.get("max_tokens", 2048)),
    }
    request = Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=float(settings.get("timeout", 120))) as response:
            raw = json.load(response)["choices"][0]["message"]["content"]
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:-1])
        result = json.loads(raw)
        validate(result, schema, "model_output")
        return result
    except HTTPError as exc:
        raise BackendError(f"Model endpoint returned HTTP {exc.code}") from None
    except (URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
        raise BackendError(f"Invalid or unavailable model response ({type(exc).__name__})") from None

"""JSON subprocess entry point used by the Agent local-search tool adapter."""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from dotenv import load_dotenv

from .config import SearchConfig
from .service import search_local


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    load_dotenv(override=False)
    try:
        payload = json.load(sys.stdin)
        requirement = payload.get("requirement")
        if not isinstance(requirement, str) or not requirement.strip():
            raise ValueError("requirement must be a non-empty string")
        library = payload.get("library", "default")
        if not isinstance(library, str):
            raise ValueError("library must be a string")

        with redirect_stdout(sys.stderr):
            result = search_local(requirement.strip(), SearchConfig.for_library(library))

        print(json.dumps({"status": "success", "result": result.to_dict()}, ensure_ascii=False))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

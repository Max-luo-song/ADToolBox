"""Command-line entry point for Driving Video Search."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from .config import LIBRARIES, SearchConfig
from .service import index_local, search_internet, search_local


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Driving Video Search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Add local video files to the local index.")
    index_parser.add_argument("path", help="Local video file or directory.")
    index_parser.add_argument("--chunk-duration", type=int, default=None)
    index_parser.add_argument("--library", choices=LIBRARIES, default="default")

    local_parser = subparsers.add_parser("search-local", help="Search only the local video index.")
    local_parser.add_argument("requirement", help="Natural-language video requirement.")
    local_parser.add_argument("--match-threshold", type=float, default=0.41)
    local_parser.add_argument("--library", choices=LIBRARIES, default="default")

    internet_parser = subparsers.add_parser("search-internet", help="Search the internet explicitly.")
    internet_parser.add_argument("requirement", help="Natural-language video requirement.")
    internet_parser.add_argument(
        "--platform",
        choices=["bilibili", "youtube", "both"],
        default="bilibili",
    )
    internet_parser.add_argument("--match-threshold", type=float, default=0.41)
    internet_parser.add_argument(
        "--cookies-browser",
        default="",
        help="Optional browser name for Bilibili cookies.",
    )

    return parser


def main() -> None:
    load_dotenv(override=False)
    args = build_parser().parse_args()

    try:
        if args.command == "index":
            config = SearchConfig.for_library(args.library)
            if args.chunk_duration is not None:
                config = replace(config, chunk_duration=args.chunk_duration)
            result = index_local(
                args.path,
                config,
            )
        elif args.command == "search-local":
            result = search_local(
                args.requirement,
                replace(SearchConfig.for_library(args.library), match_threshold=args.match_threshold),
            ).to_dict()
        else:
            result = search_internet(
                args.requirement,
                SearchConfig(
                    platform=args.platform,
                    match_threshold=args.match_threshold,
                    cookies_browser=args.cookies_browser,
                ),
            ).to_dict()
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

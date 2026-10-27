"""Fail the release check if private paths, agent modules or runtime assets are included."""

import argparse
import re
import tarfile
import zipfile
from pathlib import Path

FORBIDDEN_PARTS = {"agent", "environments", "logs", ".secrets", ".git", "__pycache__", "runs", "outputs"}
PATTERNS = [re.compile(pattern) for pattern in (
    r"/inspire/", r"/Users/", r"/root/", r"from agent\.", r"import agent\b",
    r"sk-[A-Za-z0-9_-]{24,}", r"AKIA[A-Z0-9]{16}",
)]
# Origin names must stay inside legally required attribution files only.
ORIGIN_PATTERN = re.compile(r"dggt|worldengine|algengine|smart_edit|sentrysearch|adagent", re.IGNORECASE)
ATTRIBUTION_FILES = {"LICENSE", "NOTICE", "COPYING", "THIRD_PARTY_NOTICES.md", "CONTRIBUTORS.md"}
ENGINE_PREFIXES = ("projects/reconstruction/", "projects/simulation/")


def check(name, content):
    parts = Path(name).parts
    if any(part in FORBIDDEN_PARTS for part in parts):
        return f"Forbidden private/runtime directory: {name}"
    if Path(name).name.startswith(".env") and Path(name).name != ".env.example":
        return f"Environment secrets file: {name}"
    if Path(name).suffix in {".pth", ".pt", ".pkl", ".ply", ".pyc"}:
        return f"Runtime/model asset: {name}"
    if name.endswith("scripts/check_distribution.py"):
        return None
    limit = 8_000_000 if name.startswith(ENGINE_PREFIXES) else 2_000_000
    if len(content) > limit:
        return f"Unexpected large file: {name}"
    text = content.decode("utf-8", errors="ignore")
    if any(pattern.search(text) for pattern in PATTERNS):
        return f"Private path, framework import or credential pattern: {name}"
    if Path(name).name not in ATTRIBUTION_FILES and ORIGIN_PATTERN.search(text):
        return f"Upstream origin name outside attribution files: {name}"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    problems = []
    count = 0
    if args.archive:
        if zipfile.is_zipfile(args.archive):
            with zipfile.ZipFile(args.archive) as archive:
                entries = [(name, archive.read(name)) for name in archive.namelist() if not name.endswith("/")]
        else:
            with tarfile.open(args.archive) as archive:
                entries = [(member.name, archive.extractfile(member).read()) for member in archive.getmembers() if member.isfile()]
    else:
        entries = []
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if any(part in {".git", ".venv", "__pycache__", ".ruff_cache", "build", "dist", "runs", "outputs"} or part.endswith(".egg-info") for part in relative.parts):
                continue
            if path.is_symlink():
                problems.append(f"Symlink must not be released: {relative}")
            elif path.is_file():
                entries.append((str(relative), path.read_bytes()))
    for name, content in entries:
        count += 1
        problem = check(name, content)
        if problem:
            problems.append(problem)
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"Distribution boundary check passed ({count} files)")


if __name__ == "__main__":
    main()

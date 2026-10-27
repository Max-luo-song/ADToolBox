"""Service layer: failure evidence analysis for a single scene token.

Wraps the run.py CLI for a single scene token and returns the parsed
VLM forensic result. VLM credentials live in .secrets/vlm.json (0600),
which is intentionally NOT vendored into the open-source repo.
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ANALYSIS_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ANALYSIS_ROOT / "configs" / "complete.json"
DEFAULT_CONFIG_EXAMPLE = ANALYSIS_ROOT / "configs" / "complete.json.example"
DEFAULT_RUN_NAME = "failure_evidence_analysis"


@dataclass
class SceneAnalysisResult:
    token: str
    status: str
    failure_reason: str
    reason_category: str
    observations: list
    candidate_assessment: list
    uncertainty: str
    official_metrics: dict
    report_path: str
    result_path: str
    raw: dict


class AnalysisError(RuntimeError):
    """Raised when the scene analysis fails."""


def _resolve_python() -> str:
    """Pick a Python that can run the analysis (needs numpy/opencv/Pillow)."""
    env = os.environ.get("FA_PYTHON")
    if env:
        return env
    return sys.executable



def _check_secret() -> Path:
    secret = ANALYSIS_ROOT / ".secrets" / "vlm.json"
    if not secret.is_file():
        raise AnalysisError(
            f"VLM secret not found: {secret}. "
            "Create it with keys endpoint/model/api_key and chmod 0600. See README."
        )
    if secret.stat().st_mode & 0o077:
        raise AnalysisError(f"VLM secret must have mode 0600: {secret}")
    return secret


def analyze_scene(
    token: str,
    *,
    config_path: str | None = None,
    run_name: str = DEFAULT_RUN_NAME,
    resume: bool = True,
    timeout: int = 1800,
) -> SceneAnalysisResult:
    """Run forensic analysis for one scene token and return the parsed result."""
    if not token or not token.strip():
        raise ValueError("failure_analyzer requires a non-empty scene 'token'")
    _check_secret()

    config = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG
    if not config.is_file():
        if config == DEFAULT_CONFIG and DEFAULT_CONFIG_EXAMPLE.is_file():
            raise AnalysisError(
                f"analysis config not found: {config}. "
                f"Copy {DEFAULT_CONFIG_EXAMPLE} to {config} and fill in your paths."
            )
        raise AnalysisError(f"analysis config not found: {config}")

    run_py = ANALYSIS_ROOT / "run.py"
    cmd = [
        _resolve_python(), str(run_py), "scene",
        "--token", token.strip(),
        "--config", str(config),
        "--run-name", run_name,
    ]
    if resume:
        cmd.append("--resume")

    proc = subprocess.run(
        cmd, cwd=str(ANALYSIS_ROOT), capture_output=True, text=True, timeout=timeout,
    )

    # the pipeline writes per-scene outputs under runs/<run_name>/scenes/<token>/
    scene_dir = ANALYSIS_ROOT / "runs" / run_name / "scenes" / token.strip()
    result_path = scene_dir / "result.json"
    status_path = scene_dir / "status.json"
    report_path = scene_dir / "report.md"

    # Resolve the real token dir (the pipeline may suffix-match the token)
    if not result_path.is_file():
        scenes_root = ANALYSIS_ROOT / "runs" / run_name / "scenes"
        if scenes_root.is_dir():
            matches = [d for d in scenes_root.iterdir()
                       if d.is_dir() and (d.name == token.strip() or d.name.endswith("-" + token.strip()))]
            if len(matches) == 1:
                scene_dir = matches[0]
                result_path = scene_dir / "result.json"
                status_path = scene_dir / "status.json"
                report_path = scene_dir / "report.md"

    if not result_path.is_file():
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise AnalysisError(
            f"scene analysis produced no result.json for token {token!r} "
            f"(exit={proc.returncode}). {tail}"
        )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    status = {}
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = {}
    if status.get("state") == "failed":
        raise AnalysisError(
            f"scene analysis failed for {token!r}: "
            f"{status.get('error_type')} {status.get('message')}"
        )

    return SceneAnalysisResult(
        token=result.get("token", token.strip()),
        status=result.get("status", status.get("state", "unknown")),
        failure_reason=result.get("failure_reason", ""),
        reason_category=result.get("reason_category", ""),
        observations=result.get("observations", []),
        candidate_assessment=result.get("candidate_assessment", []),
        uncertainty=result.get("uncertainty", ""),
        official_metrics=result.get("official_metrics", {}),
        report_path=str(report_path) if report_path.is_file() else "",
        result_path=str(result_path),
        raw=result,
    )


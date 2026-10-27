"""资产目录扫描。"""

import glob
import os


def scan_assets(base_dir: str) -> dict[str, list[str]]:
    """扫描资产目录,返回 {category: [path, ...]}。目录不存在时返回空 dict。"""
    if not os.path.isdir(base_dir):
        return {}

    assets: dict[str, list[str]] = {}
    for category in sorted(os.listdir(base_dir)):
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue
        files = sorted(
            f.replace("\\", "/")
            for f in glob.glob(os.path.join(category_path, "*.ply"))
        )
        if files:
            assets[category] = files
    return assets


def format_asset_catalog(assets: dict[str, list[str]]) -> str:
    if not assets:
        return "(无可用资产 — 请检查 ASSETS_DIR 配置)"
    lines = []
    for category, files in assets.items():
        lines.append(f"  {category}:")
        for f in files:
            lines.append(f"    - {f}")
    return "\n".join(lines)


def scan_smpl_assets(base_dir: str) -> dict[str, dict[str, list[str]]]:
    """扫描SMPL资产,返回 {category: {models: [...], motions: [...]}}。

    目录不存在时返回空 dict。
    """
    if not os.path.isdir(base_dir):
        return {}

    assets: dict[str, dict[str, list[str]]] = {}
    for category in sorted(os.listdir(base_dir)):
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue

        models = sorted(
            f.replace("\\", "/")
            for f in glob.glob(os.path.join(category_path, "*.ply"))
        )
        motions = sorted(
            f.replace("\\", "/")
            for f in glob.glob(os.path.join(category_path, "*_motion.pt"))
        )

        if models or motions:
            assets[category] = {
                "models": models,
                "motions": motions,
            }

    return assets


def format_smpl_asset_catalog(assets: dict[str, dict[str, list[str]]]) -> str:
    """格式化SMPL资产目录为字符串。"""
    if not assets:
        return "(无可用SMPL资产 — 请检查 SMPL_ASSETS_DIR 配置)"
    lines = []
    for category, data in assets.items():
        lines.append(f"  {category}:")
        if data.get("models"):
            lines.append("    models:")
            for f in data["models"]:
                lines.append(f"      - {f}")
        if data.get("motions"):
            lines.append("    motions:")
            for f in data["motions"]:
                lines.append(f"      - {f}")
    return "\n".join(lines)

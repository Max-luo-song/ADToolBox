"""配置文件 I/O:根据扩展名选择 JSON 或 YAML。"""

import json
from pathlib import Path
from typing import Any

from .yaml_format import dump_yaml


def save_config(config: dict[str, Any] | list[Any], path: Path) -> None:
    """保存配置到文件;扩展名为 .yaml/.yml 时写 YAML,否则写 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            dump_yaml(config, f)
        else:
            json.dump(config, f, indent=2, ensure_ascii=False)

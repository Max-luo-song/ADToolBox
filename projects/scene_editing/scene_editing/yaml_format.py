"""自定义 YAML dumper:纯数字内嵌列表用 flow 样式行内显示。"""

from typing import Any

import yaml


class ConfigDumper(yaml.SafeDumper):
    """外层列表 block 样式,纯数字内嵌列表用 flow 样式行内显示。"""


class QuotedString(str):
    """需要强制双引号输出的字符串。"""


def _represent_list(dumper: yaml.SafeDumper, data: list) -> yaml.nodes.SequenceNode:
    flow = bool(data) and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in data
    )
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


def _represent_quoted_string(
    dumper: yaml.SafeDumper, data: QuotedString
) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


def _quote_method_values(value: Any, key: str | None = None) -> Any:
    if key == "method":
        return _quote_strings(value)
    if isinstance(value, dict):
        return {
            item_key: _quote_method_values(item_value, str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_quote_method_values(item) for item in value]
    return value


def _quote_strings(value: Any) -> Any:
    if isinstance(value, str):
        return QuotedString(value)
    if isinstance(value, list):
        return [_quote_strings(item) for item in value]
    if isinstance(value, dict):
        return {
            item_key: _quote_strings(item_value)
            for item_key, item_value in value.items()
        }
    return value


ConfigDumper.add_representer(list, _represent_list)
ConfigDumper.add_representer(QuotedString, _represent_quoted_string)


def dump_yaml(config: dict[str, Any], stream: Any) -> None:
    yaml.dump(
        _quote_method_values(config),
        stream,
        Dumper=ConfigDumper,
        allow_unicode=True,
        sort_keys=False,
        indent=2,
    )

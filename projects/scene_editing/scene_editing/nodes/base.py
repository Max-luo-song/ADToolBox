"""节点构建器基类。"""

from abc import ABC, abstractmethod
from typing import Any


class NodeBuilder(ABC):
    """节点构建器抽象基类。"""

    @abstractmethod
    def build(self, params: dict[str, Any]) -> dict[str, Any]:
        """根据参数构建节点配置。

        Args:
            params: 节点配置参数

        Returns:
            节点配置字典
        """
        pass

    def validate(self, params: dict[str, Any]) -> None:
        """验证参数。

        Args:
            params: 节点配置参数

        Raises:
            ValueError: 参数验证失败时抛出
        """
        pass

    def _require_fields(
        self, params: dict[str, Any], fields: list[str], operation: str = ""
    ) -> None:
        """检查必需字段是否存在。

        Args:
            params: 参数字典
            fields: 必需字段列表
            operation: 操作名称（用于错误信息）

        Raises:
            ValueError: 缺少必需字段时抛出
        """
        for field in fields:
            if field not in params:
                prefix = f"操作 '{operation}' " if operation else ""
                raise ValueError(f"{prefix}缺少必需字段: {field}")

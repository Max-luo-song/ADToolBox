"""Background节点构建器。"""

from typing import Any

from .base import NodeBuilder


class BackgroundNodeBuilder(NodeBuilder):
    """Background节点配置构建器。

    用于添加背景点云（道路、建筑等静态场景）。
    """

    def build(self, params: dict[str, Any]) -> dict[str, Any]:
        """构建Background节点配置。

        Args:
            params: 包含以下字段:
                - add_obj: 资产文件路径列表
                - offset: 位置偏移列表
                - time_window: 时间窗口列表（可选，默认[0, -1, 10]）

        Returns:
            Background节点配置字典
        """
        add_obj = params["add_obj"]
        offset = params["offset"]
        time_window = params.get("time_window")

        # 如果未指定time_window，使用默认值
        if time_window is None:
            time_window = [[0, -1, 10]] * len(add_obj)

        # 验证长度一致性
        if not (len(add_obj) == len(offset) == len(time_window)):
            raise ValueError(
                f"add_obj/offset/time_window 长度不一致: "
                f"{len(add_obj)}/{len(offset)}/{len(time_window)}"
            )

        return {
            "add": {
                "add_obj": add_obj,
                "offset": offset,
                "time_window": time_window,
            }
        }

    def validate(self, params: dict[str, Any]) -> None:
        """验证Background节点参数。

        Args:
            params: 参数字典

        Raises:
            ValueError: 参数验证失败时抛出
        """
        self._require_fields(params, ["add_obj", "offset"], "Background.add")

        add_obj = params["add_obj"]
        offset = params["offset"]
        time_window = params.get("time_window")

        if not isinstance(add_obj, list) or len(add_obj) == 0:
            raise ValueError("add_obj 必须是非空列表")

        if not isinstance(offset, list) or len(offset) == 0:
            raise ValueError("offset 必须是非空列表")

        if time_window is not None:
            if not isinstance(time_window, list):
                raise ValueError("time_window 必须是列表")
            if len(time_window) != len(add_obj):
                raise ValueError(
                    f"time_window 长度({len(time_window)}) 与 add_obj 长度({len(add_obj)}) 不匹配"
                )

        # 验证offset格式
        for i, off in enumerate(offset):
            if not isinstance(off, list) or len(off) != 3:
                raise ValueError(f"offset[{i}] 必须是 [x, y, z] 格式")

        # 验证time_window格式
        if time_window:
            for i, tw in enumerate(time_window):
                if not isinstance(tw, list) or len(tw) != 3:
                    raise ValueError(f"time_window[{i}] 必须是 [start, end, interval] 格式")

"""SMPLNodes节点构建器。"""

from typing import Any

from .base import NodeBuilder


class SMPLNodeBuilder(NodeBuilder):
    """SMPLNodes配置构建器。

    支持以下操作:
    - trajectory: 修改SMPL实例轨迹
    - add: 添加SMPL人体模型
    - remove: 删除SMPL实例
    - replace: 替换SMPL模型或运动数据
    """

    def build(self, params: dict[str, Any]) -> dict[str, Any]:
        """构建SMPLNodes配置。

        Args:
            params: 包含以下可能的操作:
                - trajectory: 修改轨迹参数
                - add: 添加SMPL模型参数
                - remove: 删除对象参数
                - replace: 替换对象参数

        Returns:
            SMPLNodes配置字典
        """
        config: dict[str, Any] = {}

        # 处理 trajectory
        if "trajectory" in params:
            traj_params = params["trajectory"]
            config["trajectory"] = {
                "instance_id": traj_params["instance_id"],
                "offset": traj_params["offset"],
            }

        # 处理 add
        if "add" in params:
            add_params = params["add"]
            add_config: dict[str, Any] = {
                "ref_id": add_params.get("ref_id"),
                "add_obj": add_params["add_obj"],
                "add_motion": add_params["add_motion"],
                "offset": add_params["offset"],
            }
            config["add"] = add_config

        # 处理 remove
        if "remove" in params:
            remove_params = params["remove"]
            config["remove"] = {
                "instance_id": remove_params["instance_id"],
            }

        # 处理 replace
        if "replace" in params:
            replace_params = params["replace"]
            replace_config: dict[str, Any] = {
                "target_id": replace_params["target_id"],
                "replace_obj": replace_params["replace_obj"],
            }
            if "replace_motion" in replace_params:
                replace_config["replace_motion"] = replace_params["replace_motion"]
            config["replace"] = replace_config

        return config

    def validate(self, params: dict[str, Any]) -> None:
        """验证SMPLNodes参数。

        Args:
            params: 参数字典

        Raises:
            ValueError: 参数验证失败时抛出
        """
        # 验证 trajectory
        if "trajectory" in params:
            traj_params = params["trajectory"]
            self._require_fields(
                traj_params, ["instance_id", "offset"], "trajectory"
            )

        # 验证 add
        if "add" in params:
            add_params = params["add"]
            self._require_fields(
                add_params,
                ["add_obj", "add_motion", "offset"],
                "add",
            )
            self._validate_array_lengths(
                add_params["add_obj"],
                add_params["add_motion"],
                add_params["offset"],
                "add",
            )

        # 验证 remove
        if "remove" in params:
            remove_params = params["remove"]
            self._require_fields(
                remove_params, ["instance_id"], "remove"
            )

        # 验证 replace
        if "replace" in params:
            replace_params = params["replace"]
            self._require_fields(
                replace_params, ["target_id", "replace_obj"], "replace"
            )

    def _validate_array_lengths(
        self, *arrays: Any, context: str = ""
    ) -> None:
        """验证多个数组长度一致。

        Args:
            *arrays: 要验证的数组
            context: 上下文信息（用于错误信息）

        Raises:
            ValueError: 长度不一致时抛出
        """
        lengths = [len(arr) for arr in arrays if isinstance(arr, list)]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"{context} 数组长度不一致: {lengths}"
            )

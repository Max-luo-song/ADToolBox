"""RigidNodes节点构建器。"""

from typing import Any

from .base import NodeBuilder


class RigidNodeBuilder(NodeBuilder):
    """RigidNodes配置构建器。

    支持以下操作:
    - trajectory_edit: 编辑现有实例轨迹
    - rotate: 旋转实例
    - scale: 缩放实例
    - add: 添加刚体对象
    - remove: 删除实例
    - replace: 替换对象
    - trajectory: 修改现有实例轨迹
    """

    def build(self, params: dict[str, Any]) -> dict[str, Any]:
        """构建RigidNodes配置。

        Args:
            params: 包含以下可能的操作:
                - trajectory_edit: 编辑现有实例轨迹参数
                - rotate: 旋转实例参数
                - scale: 缩放实例参数
                - add: 添加对象参数
                - remove: 删除对象参数
                - replace: 替换对象参数
                - trajectory: 修改轨迹参数

        Returns:
            RigidNodes配置字典
        """
        config: dict[str, Any] = {}

        # 处理 trajectory_edit
        if "trajectory_edit" in params:
            ti_params = params["trajectory_edit"]
            config["trajectory_edit"] = {
                "instance_id": ti_params["instance_id"],
                "waypoints": ti_params["waypoints"],
                "method": ti_params["method"],
            }

        # 处理 rotate
        if "rotate" in params:
            rotate_params = params["rotate"]
            config["rotate"] = {
                "instance_id": rotate_params["instance_id"],
                "rotation": rotate_params["rotation"],
            }

        # 处理 scale
        if "scale" in params:
            scale_params = params["scale"]
            config["scale"] = {
                "instance_id": scale_params["instance_id"],
                "scale": scale_params["scale"],
            }

        # 处理 add
        if "add" in params:
            add_params = params["add"]
            config["add"] = {
                "add_obj": add_params["add_obj"],
                "anchor_pixel": add_params["anchor_pixel"],
                "scale": add_params["scale"],
            }

        # 处理 remove
        if "remove" in params:
            remove_params = params["remove"]
            config["remove"] = {
                "instance_id": remove_params["instance_id"],
            }

        # 处理 replace
        if "replace" in params:
            replace_params = params["replace"]
            config["replace"] = {
                "target_id": replace_params["target_id"],
                "replace_obj": replace_params["replace_obj"],
                "scale": replace_params["scale"],
            }

        # 处理 trajectory
        if "trajectory" in params:
            traj_params = params["trajectory"]
            config["trajectory"] = {
                "instance_id": traj_params["instance_id"],
                "offset": traj_params["offset"],
            }

        return config

    def validate(self, params: dict[str, Any]) -> None:
        """验证RigidNodes参数。

        Args:
            params: 参数字典

        Raises:
            ValueError: 参数验证失败时抛出
        """
        allowed_operations = {
            "trajectory_edit",
            "rotate",
            "scale",
            "add",
            "remove",
            "replace",
            "trajectory",
        }
        unknown_operations = sorted(set(params) - allowed_operations)
        if unknown_operations:
            raise ValueError(f"RigidNodes 不支持的操作: {unknown_operations}")

        # 验证 trajectory_edit
        if "trajectory_edit" in params:
            ti_params = params["trajectory_edit"]
            self._require_fields(
                ti_params,
                ["instance_id", "waypoints", "method"],
                "trajectory_edit",
            )
            # instance_id、waypoints、method 分别按实例对齐
            self._validate_array_lengths(
                ti_params["instance_id"],
                ti_params["waypoints"],
                ti_params["method"],
                context="trajectory_edit",
            )
            self._validate_waypoint_groups(ti_params["waypoints"], "trajectory_edit")

        # 验证 rotate
        if "rotate" in params:
            rotate_params = params["rotate"]
            self._require_fields(
                rotate_params, ["instance_id", "rotation"], "rotate"
            )
            self._validate_array_lengths(
                rotate_params["instance_id"],
                rotate_params["rotation"],
                context="rotate",
            )

        # 验证 scale
        if "scale" in params:
            scale_params = params["scale"]
            self._require_fields(scale_params, ["instance_id", "scale"], "scale")
            self._validate_array_lengths(
                scale_params["instance_id"],
                scale_params["scale"],
                context="scale",
            )

        # 验证 add
        if "add" in params:
            add_params = params["add"]
            self._require_fields(
                add_params, ["add_obj", "anchor_pixel", "scale"], "add"
            )
            self._validate_array_lengths(
                add_params["add_obj"],
                add_params["anchor_pixel"],
                add_params["scale"],
                context="add",
            )
            if len(add_params["add_obj"]) != 1:
                raise ValueError("重建引擎 add 只支持一个资产")
            pixel = add_params["anchor_pixel"][0]
            if not (
                isinstance(pixel, list)
                and len(pixel) == 2
                and all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in pixel
                )
            ):
                raise ValueError("add.anchor_pixel 必须是 [[x, y]]")
            scale = add_params["scale"][0]
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0:
                raise ValueError("add.scale 必须包含一个正数")

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
                replace_params, ["target_id", "replace_obj", "scale"], "replace"
            )
            self._validate_array_lengths(
                replace_params["target_id"],
                replace_params["replace_obj"],
                replace_params["scale"],
                context="replace",
            )
            if len(replace_params["target_id"]) != 1:
                raise ValueError("重建引擎 replace 只支持一个实例和一个资产")
            scale = replace_params["scale"][0]
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0:
                raise ValueError("replace.scale 必须包含一个正数")

        # 验证 trajectory
        if "trajectory" in params:
            traj_params = params["trajectory"]
            self._require_fields(
                traj_params, ["instance_id", "offset"], "trajectory"
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

    def _validate_waypoint_groups(self, value: Any, context: str) -> None:
        """验证按对象分组的 waypoints: [[[x, y, z], ...], ...]。"""
        if not isinstance(value, list) or len(value) == 0:
            raise ValueError(f"{context}.waypoints 必须是非空列表")
        for group_index, group in enumerate(value):
            if not isinstance(group, list) or len(group) == 0:
                raise ValueError(
                    f"{context}.waypoints[{group_index}] 必须是非空路径点列表"
                )
            for point_index, point in enumerate(group):
                if not (
                    isinstance(point, list)
                    and len(point) == 3
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in point)
                ):
                    raise ValueError(
                        f"{context}.waypoints[{group_index}][{point_index}] 必须是 [x, y, z] 格式"
                    )

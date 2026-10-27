"""根据结构化参数组装 3DGS 场景编辑配置。"""

from typing import Any

from .nodes import BackgroundNodeBuilder, RigidNodeBuilder, SMPLNodeBuilder


def _base_config(args: dict[str, Any] | None = None) -> dict[str, Any]:
    """创建带项目默认值的完整编辑配置外壳。"""
    args = args or {}
    config = {
        "debug": bool(args.get("debug", False)),
        "weather_type": str(args.get("weather_type", "")),
    }
    if "legend" in args:
        config["legend"] = args["legend"]
    else:
        config["is_legend"] = bool(args.get("is_legend", False))
    config["Nodes"] = {}
    return config


def build_config(
    args: dict[str, Any],
    valid_paths: set[str] | None = None,
) -> dict[str, Any]:
    """
    将结构化参数组装为 Background 节点配置 JSON。

    必需字段: add_obj, offset
    可选字段: time_window, weather_type, debug, is_legend

    若提供 valid_paths,会校验 add_obj 中的每个路径都在其中。
    """
    add_obj = args["add_obj"]
    offset = args["offset"]
    time_window = args.get("time_window") or [[0, -1, 10]] * len(add_obj)

    if not (len(add_obj) == len(offset) == len(time_window)):
        raise ValueError(
            f"add_obj/offset/time_window 长度不一致: "
            f"{len(add_obj)}/{len(offset)}/{len(time_window)}"
        )

    if valid_paths:
        invalid = [p for p in add_obj if p not in valid_paths]
        if invalid:
            raise ValueError(
                f"以下资产路径不存在: {invalid}。可用路径见 list_assets。"
            )

    # 使用新的 BackgroundNodeBuilder 构建配置
    bg_builder = BackgroundNodeBuilder()
    bg_config = bg_builder.build({
        "add_obj": add_obj,
        "offset": offset,
        "time_window": time_window,
    })

    config = _base_config(args)
    config["Nodes"]["Background"] = bg_config
    return config


def build_rigid_node_config(
    args: dict[str, Any],
    valid_paths: set[str] | None = None,
) -> dict[str, Any]:
    """
    将结构化参数组装为 RigidNodes 配置 JSON。

    支持的操作:
    - trajectory_edit: 编辑原场景刚体轨迹
    - add: 添加刚体对象
    - remove: 删除实例
    - replace: 替换对象
    - trajectory: 修改现有实例轨迹

    若提供 valid_paths,会校验 add_obj/replace_obj 中的每个路径都在其中。
    """
    builder = RigidNodeBuilder()

    # 验证参数
    builder.validate(args)

    # 验证路径
    if valid_paths:
        for key in ["add_obj", "replace_obj"]:
            for section in args.values():
                if isinstance(section, dict) and key in section:
                    paths = section[key]
                    if isinstance(paths, list):
                        invalid = [p for p in paths if p not in valid_paths]
                        if invalid:
                            raise ValueError(
                                f"以下资产路径不存在: {invalid}。可用路径见 list_assets。"
                            )

    return builder.build(args)


def build_rigid_config(
    args: dict[str, Any],
    valid_paths: set[str] | None = None,
) -> dict[str, Any]:
    """将结构化参数组装为完整 RigidNodes 配置 JSON。"""
    config = _base_config(args)
    config["Nodes"]["RigidNodes"] = build_rigid_node_config(args, valid_paths)
    return config


def build_smpl_node_config(
    args: dict[str, Any],
    valid_paths: set[str] | None = None,
    valid_motion_paths: set[str] | None = None,
) -> dict[str, Any]:
    """
    将结构化参数组装为 SMPLNodes 配置 JSON。

    支持的操作:
    - trajectory: 修改SMPL实例轨迹
    - add: 添加SMPL人体模型
    - remove: 删除SMPL实例
    - replace: 替换SMPL模型或运动数据

    若提供 valid_paths,会校验 add_obj/replace_obj 中的每个路径都在其中。
    若提供 valid_motion_paths,会校验 add_motion/replace_motion 中的每个路径都在其中。
    """
    builder = SMPLNodeBuilder()

    # 验证参数
    builder.validate(args)

    # 验证模型路径
    if valid_paths:
        for key in ["add_obj", "replace_obj"]:
            for section in args.values():
                if isinstance(section, dict) and key in section:
                    paths = section[key]
                    if isinstance(paths, list):
                        invalid = [p for p in paths if p not in valid_paths]
                        if invalid:
                            raise ValueError(
                                f"以下SMPL模型路径不存在: {invalid}。可用路径见 list_smpl_assets。"
                            )

    # 验证运动路径
    if valid_motion_paths:
        for key in ["add_motion", "replace_motion"]:
            for section in args.values():
                if isinstance(section, dict) and key in section:
                    paths = section[key]
                    if isinstance(paths, list):
                        invalid = [p for p in paths if p not in valid_motion_paths]
                        if invalid:
                            raise ValueError(
                                f"以下运动资产路径不存在: {invalid}。可用路径见 list_smpl_assets。"
                            )

    return builder.build(args)


def build_smpl_config(
    args: dict[str, Any],
    valid_paths: set[str] | None = None,
    valid_motion_paths: set[str] | None = None,
) -> dict[str, Any]:
    """将结构化参数组装为完整 SMPLNodes 配置 JSON。"""
    config = _base_config(args)
    config["Nodes"]["SMPLNodes"] = build_smpl_node_config(
        args, valid_paths, valid_motion_paths
    )
    return config


def build_full_scene_config(
    args: dict[str, Any],
    valid_paths: set[str] | None = None,
    valid_motion_paths: set[str] | None = None,
    valid_smpl_paths: set[str] | None = None,
) -> dict[str, Any]:
    """
    将结构化参数组装为完整场景配置 JSON。

    整合所有节点类型:
    - RigidNodes: 刚体对象
    - SMPLNodes: 人体模型
    - Background: 背景点云
    """
    config = _base_config(args)

    # 构建 Background
    if "background" in args and args["background"]:
        bg_args = args["background"]
        if "add_obj" in bg_args and "offset" in bg_args:
            config["Nodes"]["Background"] = build_config(
                bg_args, valid_paths
            )["Nodes"]["Background"]

    # 构建 RigidNodes
    if "rigid_nodes" in args and args["rigid_nodes"]:
        config["Nodes"]["RigidNodes"] = build_rigid_node_config(
            args["rigid_nodes"], valid_paths
        )

    # 构建 SMPLNodes
    if "smpl_nodes" in args and args["smpl_nodes"]:
        config["Nodes"]["SMPLNodes"] = build_smpl_node_config(
            args["smpl_nodes"], valid_smpl_paths, valid_motion_paths
        )

    return config

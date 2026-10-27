"""MCP 工具处理函数。"""

import copy
import json
import random
from pathlib import Path
from typing import Any, Awaitable, Callable

from mcp.types import TextContent

from ..assets import scan_assets, scan_smpl_assets
from ..io import save_config as write_config

VEHICLE_ASSET_CATEGORIES = {"car", "vehicle", "vehicles"}
PEDESTRIAN_ASSET_CATEGORIES = {"smpl", "pedestrian", "person", "people"}


def register_tool_handlers(
    assets_dir: str,
    output_dir: str,
) -> dict[str, Callable[[dict[str, Any]], Awaitable[list[TextContent]]]]:
    """注册所有工具处理函数。"""
    draft = _new_config({})
    ego_offsets: dict[str, Any] = {}
    randomize: dict[str, Any] = {"enabled": False}

    return {
        "list_instances": _list_instances,
        "list_assets": _make_list_assets(assets_dir),
        "start_config": _make_start_config(draft),
        "edit_rigid_trajectory": _make_edit_rigid_trajectory(draft),
        "rotate_rigid_instance": _make_rotate_rigid_instance(draft),
        "scale_rigid_instance": _make_scale_rigid_instance(draft),
        "add_rigid_asset": _make_add_rigid_asset(draft, assets_dir),
        "remove_rigid_instance": _make_remove_rigid_instance(draft),
        "replace_rigid_instance": _make_replace_rigid_instance(draft, assets_dir),
        "remove_added_rigid_asset": _make_remove_added_rigid_asset(draft),
        "set_ego_trajectory_offsets": _make_set_ego_trajectory_offsets(ego_offsets),
        "preview_config": _make_preview_config(draft, randomize),
        "save_config": _make_save_config(draft, randomize, output_dir),
        "save_ego_trajectory_offsets": _make_save_ego_trajectory_offsets(ego_offsets, output_dir),
        "reset_config": _make_reset_config(draft, randomize),
    }


def _make_list_assets(
    assets_dir: str,
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        assets = scan_assets(assets_dir)
        category = args.get("category")
        if category:
            if category not in assets:
                return [_text(f"未找到类别 '{category}'。可用类别: {', '.join(assets.keys()) or '(空)'}")]
            result: dict[str, list[str]] = {category: assets[category]}
        else:
            result = assets
        return [_json(result)]

    return handler


def _make_list_smpl_assets(
    smpl_assets_dir: str,
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        smpl_assets = scan_smpl_assets(smpl_assets_dir)
        category = args.get("category")
        if category:
            if category not in smpl_assets:
                return [_text(f"未找到SMPL类别 '{category}'。可用类别: {', '.join(smpl_assets.keys()) or '(空)'}")]
            result: dict[str, dict[str, list[str]]] = {category: smpl_assets[category]}
        else:
            result = smpl_assets
        return [_json(result)]

    return handler


def _make_start_config(
    draft: dict[str, Any],
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        draft.clear()
        draft.update(_new_config(args))
        return [_result("已开始新的配置草稿。", draft)]

    return handler


def _make_set_globals(
    draft: dict[str, Any],
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        _ensure_draft(draft)
        _apply_globals(draft, args)
        return [_result("已更新全局字段。", draft)]

    return handler


def _make_set_randomize(
    draft: dict[str, Any],
    randomize: dict[str, Any],
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        try:
            _ensure_draft(draft)
            randomize.clear()
            randomize.update(_normalize_randomize(args))
        except (TypeError, ValueError) as exc:
            return [_text(f"参数错误: {exc}")]
        return [
            _result(
                "已更新随机批量生成设置。",
                draft,
                {"randomize": copy.deepcopy(randomize)},
            )
        ]

    return handler


def _make_add_background_asset(
    draft: dict[str, Any],
    assets_dir: str,
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        try:
            _ensure_draft(draft)
            asset = _require_str(args, "asset")
            _validate_background_asset(asset, assets_dir)
            position = _require_vector(args, "position")
            start_frame = int(args.get("start_frame", 0))
            end_frame = int(args.get("end_frame", -1))
            interval = int(args.get("interval", 10))

            add = _node_section(draft, "Background").setdefault("add", {})
            _append(add, "add_obj", asset)
            _append(add, "offset", position)
            _append(add, "time_window", [start_frame, end_frame, interval])
        except (KeyError, TypeError, ValueError) as exc:
            return [_text(f"参数错误: {exc}")]
        return [_result("已追加背景资产。", draft)]

    return handler


def _make_edit_rigid_trajectory(
    draft: dict[str, Any],
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        try:
            _ensure_draft(draft)
            instance_id = int(args["instance_id"])
            removed = draft["Nodes"].get("RigidNodes", {}).get("remove", {})
            if instance_id in removed.get("instance_id", []):
                raise ValueError("该实例已标记删除,不能在同一配置中编辑其轨迹。")
            waypoints = _require_waypoints(args, "waypoints")
            method = _method(args.get("method"), waypoints)

            edit = _node_section(draft, "RigidNodes").setdefault("trajectory_edit", {})
            _append(edit, "instance_id", instance_id)
            _append(edit, "waypoints", waypoints)
            _append(edit, "method", method)
        except (KeyError, TypeError, ValueError) as exc:
            return [_text(f"参数错误: {exc}")]
        return [_result("已追加刚体轨迹编辑。", draft)]

    return handler


def _make_rotate_rigid_instance(
    draft: dict[str, Any],
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        try:
            _ensure_draft(draft)
            instance_id = _require_nonnegative_int(args, "instance_id")
            rotation = _require_vector(args, "rotation")
            rotate = _node_section(draft, "RigidNodes").setdefault("rotate", {})
            _append(rotate, "instance_id", instance_id)
            _append(rotate, "rotation", rotation)
        except (KeyError, TypeError, ValueError) as exc:
            return [_text(f"参数错误: {exc}")]
        return [_result("已追加刚体旋转。", draft)]

    return handler


def _make_scale_rigid_instance(
    draft: dict[str, Any],
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        try:
            _ensure_draft(draft)
            instance_id = _require_nonnegative_int(args, "instance_id")
            scale = _require_positive_number(args, "scale")
            scale_edit = _node_section(draft, "RigidNodes").setdefault("scale", {})
            _append(scale_edit, "instance_id", instance_id)
            _append(scale_edit, "scale", scale)
        except (KeyError, TypeError, ValueError) as exc:
            return [_text(f"参数错误: {exc}")]
        return [_result("已追加刚体缩放。", draft)]

    return handler


def _make_replace_rigid_instance(
    draft: dict[str, Any],
    assets_dir: str,
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        try:
            _ensure_draft(draft)
            instance_id = _require_nonnegative_int(args, "instance_id")
            asset = _require_str(args, "asset")
            _validate_rigid_asset(asset, assets_dir)
            scale = _require_positive_number(args, "scale")
            _node_section(draft, "RigidNodes")["replace"] = {
                "target_id": [instance_id],
                "replace_obj": [asset],
                "scale": [scale],
            }
        except (KeyError, TypeError, ValueError) as exc:
            return [_text(f"参数错误: {exc}")]
        return [_result("已设置 重建引擎 刚体实例替换。", draft)]

    return handler


def _make_add_rigid_asset(
    draft: dict[str, Any],
    assets_dir: str,
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        try:
            _ensure_draft(draft)
            asset = _require_str(args, "asset")
            _validate_rigid_asset(asset, assets_dir)
            anchor_pixel = args["anchor_pixel"]
            scale = args["scale"]
            if not (
                isinstance(anchor_pixel, list)
                and len(anchor_pixel) == 2
                and all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in anchor_pixel
                )
            ):
                raise ValueError("anchor_pixel 必须是两个非负整数 [x, y]")
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0:
                raise ValueError("scale 必须是正数")

            _node_section(draft, "RigidNodes")["add"] = {
                "add_obj": [asset],
                "anchor_pixel": [anchor_pixel],
                "scale": [scale],
            }
        except (KeyError, TypeError, ValueError) as exc:
            return [_text(f"参数错误: {exc}")]
        return [_result("已设置 重建引擎 刚体资产插入。", draft)]

    return handler


def _make_remove_rigid_instance(
    draft: dict[str, Any],
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        try:
            instance_id = _require_nonnegative_int(args, "instance_id")
        except (KeyError, TypeError, ValueError) as exc:
            return [_text(f"参数错误: {exc}")]

        _ensure_draft(draft)
        node = _node_section(draft, "RigidNodes")
        removed = node.setdefault("remove", {}).setdefault("instance_id", [])
        if instance_id not in removed:
            removed.append(instance_id)

        edit = node.get("trajectory_edit")
        if edit:
            keep = [
                index for index, value in enumerate(edit["instance_id"])
                if value != instance_id
            ]
            if keep:
                node["trajectory_edit"] = {
                    key: [values[index] for index in keep]
                    for key, values in edit.items()
                }
            else:
                node.pop("trajectory_edit")
        return [_result(
            f"已在草稿中标记删除刚体实例 {instance_id},并清除该实例的轨迹编辑。"
            "保存并由场景编辑器执行后生效。",
            draft,
        )]

    return handler


def _make_remove_added_rigid_asset(
    draft: dict[str, Any],
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        try:
            index = _require_nonnegative_int(args, "index")
            _ensure_draft(draft)
            node = draft["Nodes"].get("RigidNodes", {})
            add = node.get("add", {})
            assets = add.get("add_obj", [])
            if index >= len(assets):
                raise ValueError(
                    f"新增资产序号越界: {index},当前有 {len(assets)} 项。"
                    "请先调用 preview_config,序号从 0 开始。"
                )
        except (KeyError, TypeError, ValueError) as exc:
            return [_text(f"参数错误: {exc}")]

        asset = assets[index]
        for values in add.values():
            del values[index]
        if not assets:
            node.pop("add")
            if not node:
                draft["Nodes"].pop("RigidNodes")
        return [_result(
            f"已从草稿撤销第 {index + 1} 个新增刚体资产。后续序号已前移,需保存配置。",
            draft,
            {"removed_asset": asset, "removed_index": index},
        )]

    return handler


def _make_add_smpl_person(
    draft: dict[str, Any],
    smpl_assets_dir: str,
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        try:
            _ensure_draft(draft)
            model = _require_str(args, "model")
            motion = _require_str(args, "motion")
            _validate_smpl_asset(model, motion, smpl_assets_dir)
            position = _require_vector(args, "position")

            add = _node_section(draft, "SMPLNodes").setdefault("add", {})
            if "ref_id" in args:
                _append(add, "ref_id", int(args["ref_id"]))
            _append(add, "add_obj", model)
            _append(add, "add_motion", motion)
            _append(add, "offset", position)
        except (KeyError, TypeError, ValueError) as exc:
            return [_text(f"参数错误: {exc}")]
        return [_result("已追加SMPL行人。", draft)]

    return handler


def _make_preview_config(
    draft: dict[str, Any],
    randomize: dict[str, Any],
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        _ensure_draft(draft)
        if not randomize.get("enabled"):
            return [_json(draft)]
        payload: dict[str, Any] = {"config": copy.deepcopy(draft)}
        payload["randomize"] = copy.deepcopy(randomize)
        return [_json(payload)]

    return handler


def _make_save_config(
    draft: dict[str, Any],
    randomize: dict[str, Any],
    output_dir: str,
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        _ensure_draft(draft)
        rigid = draft.get("Nodes", {}).get("RigidNodes")
        if set(draft.get("Nodes", {})) != {"RigidNodes"} or not isinstance(rigid, dict) or not rigid:
            return [_text("参数错误: 重建引擎 配置必须包含 RigidNodes 编辑操作")]
        if set(rigid) - {"trajectory_edit", "rotate", "scale", "remove", "add", "replace"}:
            return [_text("参数错误: 包含 重建引擎 不支持的刚体编辑操作")]
        if any(name in rigid for name in ("trajectory_edit", "rotate", "scale", "remove", "replace")) and not draft.get("instance_info_path"):
            return [_text("参数错误: 编辑已有实例必须提供 instance_info_path")]
        output_path = args.get("output_path")
        if not isinstance(output_path, str) or not output_path.endswith(".json"):
            return [_text("参数错误: 重建引擎 output_path 必须是 .json 文件")]
        path = Path(output_path)
        if not path.is_absolute():
            path = Path(output_dir) / path
        try:
            write_config(draft, path)
        except OSError as exc:
            return [_text(f"保存失败: {exc}")]
        return [_result(f"配置已保存到: {path}", draft, {"saved_path": str(path)})]

    return handler


def _make_set_ego_trajectory_offsets(
    offsets: dict[str, Any],
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        try:
            value = args["offsets"]
            if not isinstance(value, list) or not value or not all(_is_vector(item) for item in value):
                raise ValueError("offsets 必须是非空的 [[前, 左, 上], ...]")
            offsets.clear()
            offsets.update({"coordinate_system": "camera_local_forward_left_up", "offsets": copy.deepcopy(value)})
        except (KeyError, TypeError, ValueError) as exc:
            return [_text(f"参数错误: {exc}")]
        return [_json(offsets)]

    return handler


def _make_save_ego_trajectory_offsets(
    offsets: dict[str, Any], output_dir: str,
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        if not offsets:
            return [_text("参数错误: 请先调用 set_ego_trajectory_offsets")]
        output_path = args.get("output_path")
        if not isinstance(output_path, str) or not output_path.endswith(".json"):
            return [_text("参数错误: output_path 必须是 .json 文件")]
        path = Path(output_path)
        if not path.is_absolute():
            path = Path(output_dir) / path
        try:
            write_config(offsets, path)
        except OSError as exc:
            return [_text(f"保存失败: {exc}")]
        return [_json({"saved_path": str(path), **offsets})]

    return handler


def _make_reset_config(
    draft: dict[str, Any],
    randomize: dict[str, Any],
) -> Callable[[dict[str, Any]], Awaitable[list[TextContent]]]:
    async def handler(args: dict[str, Any]) -> list[TextContent]:
        draft.clear()
        draft.update(_new_config({}))
        randomize.clear()
        randomize.update({"enabled": False})
        return [_result("已重置配置草稿。", draft)]

    return handler


def _new_config(args: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if "instance_info_path" in args:
        config["instance_info_path"] = str(args["instance_info_path"])
    config["Nodes"] = {}
    return config


def _apply_globals(draft: dict[str, Any], args: dict[str, Any]) -> None:
    if "debug" in args:
        draft["debug"] = bool(args["debug"])
    if "weather_type" in args:
        draft["weather_type"] = str(args["weather_type"])
    if "legend" in args:
        draft.pop("is_legend", None)
        draft["legend"] = args["legend"]
    if "is_legend" in args:
        draft.pop("legend", None)
        draft["is_legend"] = bool(args["is_legend"])


def _ensure_draft(draft: dict[str, Any]) -> None:
    if "Nodes" not in draft:
        draft.clear()
        draft.update(_new_config({}))


def _node_section(draft: dict[str, Any], name: str) -> dict[str, Any]:
    nodes = draft.setdefault("Nodes", {})
    return nodes.setdefault(name, {})


def _append(section: dict[str, Any], key: str, value: Any) -> None:
    section.setdefault(key, []).append(value)


def _asset_category(asset: str, assets_dir: str) -> str:
    assets = scan_assets(assets_dir)
    for category, paths in assets.items():
        if asset in paths:
            return category
    raise ValueError(f"资产路径不存在: {asset}。请先调用 list_assets。")


def _validate_background_asset(asset: str, assets_dir: str) -> None:
    category = _asset_category(asset, assets_dir)
    if category in VEHICLE_ASSET_CATEGORIES:
        raise ValueError(
            f"资产 {asset} 属于车辆类别 '{category}',应使用 add_rigid_asset。"
        )
    if category in PEDESTRIAN_ASSET_CATEGORIES:
        raise ValueError(
            f"资产 {asset} 属于行人/人体类别 '{category}',应使用 add_smpl_person。"
        )


def _validate_rigid_asset(asset: str, assets_dir: str) -> None:
    category = _asset_category(asset, assets_dir)
    if category not in VEHICLE_ASSET_CATEGORIES:
        raise ValueError(
            f"资产 {asset} 属于 '{category}' 类别,不是车辆资产;普通/其它资产应使用 add_background_asset。"
        )


def _validate_smpl_asset(model: str, motion: str, smpl_assets_dir: str) -> None:
    smpl_assets = scan_smpl_assets(smpl_assets_dir)
    valid_models = {p for data in smpl_assets.values() for p in data.get("models", [])}
    valid_motions = {p for data in smpl_assets.values() for p in data.get("motions", [])}
    if model not in valid_models:
        raise ValueError(f"SMPL模型路径不存在: {model}。请先调用 list_smpl_assets。")
    if motion not in valid_motions:
        raise ValueError(f"SMPL运动路径不存在: {motion}。请先调用 list_smpl_assets。")


def _require_nonnegative_int(args: dict[str, Any], key: str) -> int:
    value = args[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} 必须是非负整数")
    return value


def _require_str(args: dict[str, Any], key: str) -> str:
    value = args[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} 必须是非空字符串")
    return value


def _require_positive_number(args: dict[str, Any], key: str) -> float | int:
    value = args[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{key} 必须是正数")
    return value


def _require_vector(args: dict[str, Any], key: str) -> list[float | int]:
    value = args[key]
    if not _is_vector(value):
        raise ValueError(f"{key} 必须是 [x, y, z]")
    return value


def _require_waypoints(args: dict[str, Any], key: str) -> list[list[float | int]]:
    value = args[key]
    if _is_vector(value):
        return [value]
    if isinstance(value, list) and value and all(_is_vector(point) for point in value):
        return value
    raise ValueError(f"{key} 必须是非空路径点列表 [[x,y,z], ...]")


def _method(value: Any, waypoints: list[list[float | int]]) -> str:
    if value is None:
        return "cubic" if len(waypoints) >= 3 else "linear"
    if value not in {"linear", "cubic"}:
        raise ValueError("method 必须是 linear 或 cubic")
    return str(value)


def _is_vector(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    )


def _normalize_randomize(args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ValueError("randomize 必须是对象")
    enabled = bool(args.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    count = int(args.get("count", 10))
    if count < 1 or count > 1000:
        raise ValueError("randomize.count 必须在 1 到 1000 之间")

    normalized: dict[str, Any] = {
        "enabled": True,
        "count": count,
        "offset_jitter": _jitter_vector(
            args.get("offset_jitter", [1, 1, 0]), "offset_jitter"
        ),
        "waypoint_jitter": _jitter_vector(
            args.get("waypoint_jitter", [1, 1, 0]), "waypoint_jitter"
        ),
        "time_window_jitter": _jitter_vector(
            args.get("time_window_jitter", [5, 5, 2]), "time_window_jitter"
        ),
    }
    if "seed" in args and args["seed"] is not None:
        normalized["seed"] = int(args["seed"])
    return normalized


def _jitter_vector(value: Any, name: str) -> list[float | int]:
    if not _is_vector(value):
        raise ValueError(f"randomize.{name} 必须是 [x, y, z]")
    if any(item < 0 for item in value):
        raise ValueError(f"randomize.{name} 不能包含负数")
    return value


def _randomized_payload(
    config: dict[str, Any],
    randomize: dict[str, Any],
) -> dict[str, Any]:
    rng = random.Random(randomize.get("seed"))
    configs = [
        _randomize_config(copy.deepcopy(config), randomize, rng)
        for _ in range(int(randomize["count"]))
    ]
    return {
        "configs": configs,
        "randomize": {
            key: copy.deepcopy(value)
            for key, value in randomize.items()
            if key in {"enabled", "count", "seed"}
        },
    }


def _randomize_config(
    value: Any,
    randomize: dict[str, Any],
    rng: random.Random,
    key: str | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _randomize_config(item_value, randomize, rng, item_key)
            for item_key, item_value in value.items()
    }
    if isinstance(value, list):
        if key == "offset":
            return [
                _jitter_position(item, randomize["offset_jitter"], rng)
                for item in value
            ]
        if key == "waypoints":
            return [
                _jitter_waypoints(item, randomize["waypoint_jitter"], rng)
                for item in value
            ]
        if key == "time_window":
            return [
                _jitter_time_window(item, randomize["time_window_jitter"], rng)
                for item in value
            ]
        return [_randomize_config(item, randomize, rng) for item in value]
    return value


def _jitter_position(
    vector: Any,
    jitter: list[float | int],
    rng: random.Random,
) -> Any:
    if not _is_vector(vector):
        return vector
    return [
        coord if delta == 0 else coord + rng.uniform(-float(delta), float(delta))
        for coord, delta in zip(vector, jitter, strict=True)
    ]


def _jitter_waypoints(
    waypoints: Any,
    jitter: list[float | int],
    rng: random.Random,
) -> Any:
    if _is_vector(waypoints):
        return _jitter_position(waypoints, jitter, rng)
    if isinstance(waypoints, list):
        return [_jitter_position(point, jitter, rng) for point in waypoints]
    return waypoints


def _jitter_time_window(
    time_window: Any,
    jitter: list[float | int],
    rng: random.Random,
) -> Any:
    if not _is_vector(time_window):
        return time_window
    start, end, interval = [int(item) for item in time_window]
    start += _randint_delta(jitter[0], rng)
    if end != -1:
        end += _randint_delta(jitter[1], rng)
    interval = max(1, interval + _randint_delta(jitter[2], rng))
    return [start, end, interval]


def _randint_delta(delta: float | int, rng: random.Random) -> int:
    width = int(delta)
    if width <= 0:
        return 0
    return rng.randint(-width, width)


def _text(text: str) -> TextContent:
    return TextContent(type="text", text=text)


def _json(data: Any) -> TextContent:
    return TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))


def _result(
    message: str,
    draft: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> TextContent:
    payload = {"message": message, "config": copy.deepcopy(draft)}
    if extra:
        payload.update(extra)
    return _json(payload)


async def _list_instances(args: dict[str, Any]) -> list[TextContent]:
    path = Path(args["instance_info_path"]).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        info = json.load(handle)
    return [_json({**info, "instance_info_path": str(path)})]

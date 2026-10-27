"""MCP 工具定义。"""

from typing import Any

from mcp.types import Tool


def _vector_property(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
        "description": f"{description},格式 [x, y, z] = [前, 左, 上]",
    }


def _waypoints_property(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": _vector_property("路径点"),
        "minItems": 1,
        "description": description,
    }


def _randomize_property() -> dict[str, Any]:
    return {
        "type": "object",
        "description": "随机批量生成设置。enabled=false 或省略时保持单份配置行为。",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description": "是否开启随机批量生成",
                "default": False,
            },
            "count": {
                "type": "integer",
                "description": "生成数量,范围 1-1000,默认 10",
                "default": 10,
                "minimum": 1,
                "maximum": 1000,
            },
            "seed": {
                "type": "integer",
                "description": "随机种子。指定后同一草稿和参数可复现同一批结果",
            },
            "offset_jitter": {
                **_vector_property("offset 随机扰动范围"),
                "default": [1, 1, 0],
            },
            "waypoint_jitter": {
                **_vector_property("waypoints 随机扰动范围"),
                "default": [1, 1, 0],
            },
            "time_window_jitter": {
                **_vector_property("time_window 随机扰动范围"),
                "default": [5, 5, 2],
            },
        },
    }


def get_tool_definitions(
    assets_catalog: str,
    categories: str,
    output_dir: str,
) -> list[Any]:
    """获取所有工具定义。"""
    return [
        _list_instances_tool(),
        _list_assets_tool(categories),
        _start_config_tool(),
        _edit_rigid_trajectory_tool(),
        _rotate_rigid_instance_tool(),
        _scale_rigid_instance_tool(),
        _add_rigid_asset_tool(assets_catalog),
        _remove_rigid_instance_tool(),
        _replace_rigid_instance_tool(assets_catalog),
        _remove_added_rigid_asset_tool(),
        _set_ego_trajectory_offsets_tool(),
        _preview_config_tool(),
        _save_config_tool(output_dir),
        _save_ego_trajectory_offsets_tool(output_dir),
        _reset_config_tool(),
    ]


def _list_assets_tool(categories: str) -> Tool:
    return Tool(
        name="list_assets",
        description=(
            "列出当前可用的3D资产文件,按类别分组。需要资产路径时先调用本工具。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": f"可选:按类别筛选。当前类别: {categories}",
                },
            },
        },
    )


def _list_smpl_assets_tool(smpl_categories: str) -> Tool:
    return Tool(
        name="list_smpl_assets",
        description=(
            "列出当前可用的SMPL资产文件(人体模型和运动数据)。需要人体模型或运动路径时先调用本工具。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": f"可选:按类别筛选。当前类别: {smpl_categories}",
                },
            },
        },
    )


def _start_config_tool() -> Tool:
    return Tool(
        name="start_config",
        description=(
            "开始一份新的编辑配置草稿。会清空当前草稿并创建标准结构。"
            "编辑 DGGT 已有实例时,传入逐帧 bbox 实例清单的 instance_info_path 并写入配置顶层。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "instance_info_path": {
                    "type": "string",
                    "description": "DGGT instance_info.json 路径",
                },
            },
        },
    )


def _set_globals_tool() -> Tool:
    return Tool(
        name="set_globals",
        description="修改当前配置草稿的全局字段,不会清空 Nodes。",
        inputSchema={
            "type": "object",
            "properties": _global_properties(),
        },
    )


def _set_randomize_tool() -> Tool:
    return Tool(
        name="set_randomize",
        description=(
            "设置当前草稿保存时使用的随机批量生成参数。"
            "开启后 save_config 会基于当前草稿生成多份随机扰动配置。"
        ),
        inputSchema={
            "type": "object",
            "properties": _randomize_property()["properties"],
        },
    )


def _add_background_asset_tool(catalog: str) -> Tool:
    return Tool(
        name="add_background_asset",
        description=f"""向当前草稿的 Nodes.Background.add 追加一个普通背景资产。

资产类别规则:
- 普通/其它静态资产(如交通锥、路障、标志牌等)属于 Background
- 车辆资产属于 RigidNodes,使用 add_rigid_asset
- 行人/人体资产属于 SMPLNodes,使用 add_smpl_person

当前可用资产:
{catalog}
""",
        inputSchema={
            "type": "object",
            "properties": {
                "asset": {
                    "type": "string",
                    "description": "背景资产路径,必须来自 list_assets 返回结果",
                },
                "position": _vector_property("资产位置"),
                "start_frame": {
                    "type": "integer",
                    "description": "开始帧,默认 0",
                    "default": 0,
                },
                "end_frame": {
                    "type": "integer",
                    "description": "结束帧,-1 表示持续到结束,默认 -1",
                    "default": -1,
                },
                "interval": {
                    "type": "integer",
                    "description": "每隔多少帧新增一次,默认 10",
                    "default": 10,
                },
            },
            "required": ["asset", "position"],
        },
    )


def _edit_rigid_trajectory_tool() -> Tool:
    return Tool(
        name="edit_rigid_trajectory",
        description=(
            "向当前草稿的 Nodes.RigidNodes.trajectory_edit 追加一个已有刚体实例轨迹编辑。"
            "车辆属于刚体;如果用户给出 instance_id,表示编辑已有刚体实例。"
            "waypoints 给 1 个点表示整体偏移,2 个点通常 linear,3 个及以上可用 cubic。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "integer",
                    "description": "要编辑的原场景刚体实例 ID",
                },
                "waypoints": _waypoints_property("路径点数组,每项为 [x, y, z]"),
                "method": {
                    "type": "string",
                    "enum": ["linear", "cubic"],
                    "description": "插值方法。3 个及以上路径点通常用 cubic",
                    "default": "linear",
                },
            },
            "required": ["instance_id", "waypoints"],
        },
    )


def _rotate_rigid_instance_tool() -> Tool:
    return Tool(
        name="rotate_rigid_instance",
        description="绕实例中心旋转已有刚体。rotation 为世界固定 XYZ 轴外旋角度，单位为度。",
        inputSchema={
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "要旋转的原场景刚体实例 ID",
                },
                "rotation": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "旋转角度 [rx, ry, rz]，单位为度",
                },
            },
            "required": ["instance_id", "rotation"],
        },
    )


def _scale_rigid_instance_tool() -> Tool:
    return Tool(
        name="scale_rigid_instance",
        description="绕实例中心等比例缩放已有刚体。",
        inputSchema={
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "要缩放的原场景刚体实例 ID",
                },
                "scale": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "等比例缩放倍数",
                },
            },
            "required": ["instance_id", "scale"],
        },
    )


def _replace_rigid_instance_tool(catalog: str) -> Tool:
    return Tool(
        name="replace_rigid_instance",
        description=f"""设置当前草稿中唯一的 DGGT Mode2 刚体实例替换。

替换位置来自 instance_info_path 中该实例的逐帧 bbox。
scale 是替换资产 Gaussian 的统一缩放倍数。再次调用会覆盖前一次替换。

当前可用资产:
{catalog}
""",
        inputSchema={
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "要替换的原场景刚体实例 ID",
                },
                "asset": {
                    "type": "string",
                    "description": "替换车辆的 3DGS PLY 路径，必须来自 list_assets 返回结果",
                },
                "scale": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "替换资产 Gaussian 的统一缩放倍数",
                },
            },
            "required": ["instance_id", "asset", "scale"],
        },
    )


def _add_rigid_asset_tool(catalog: str) -> Tool:
    return Tool(
        name="add_rigid_asset",
        description=f"""设置当前草稿中唯一的 DGGT Mode2 车辆 Gaussian PLY 插入。

anchor_pixel 是 DGGT 处理后第一帧图像中的 [x, y] 道路落点。
scale 是资产 Gaussian 的统一缩放倍数。再次调用会覆盖前一次插入。

当前可用资产:
{catalog}
""",
        inputSchema={
            "type": "object",
            "properties": {
                "asset": {
                    "type": "string",
                    "description": "车辆 3DGS PLY 路径,必须来自 list_assets 返回结果",
                },
                "anchor_pixel": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "DGGT 处理后第一帧图像中的道路落点 [x, y]",
                },
                "scale": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Gaussian 资产统一缩放倍数",
                },
            },
            "required": ["asset", "anchor_pixel", "scale"],
        },
    )


def _remove_rigid_instance_tool() -> Tool:
    return Tool(
        name="remove_rigid_instance",
        description=(
            "在当前草稿的 Nodes.RigidNodes.remove 中标记删除一个原场景刚体实例。"
            "instance_id 必须由用户提供,不可猜测;本服务不读取原场景实例清单。"
            "同一实例重复删除不重复记录,该实例已有的轨迹编辑会被撤销。"
            "这里只生成删除配置,保存并由场景编辑器执行后生效。"
            "如果用户要撤销草稿中刚新增的车辆,使用 remove_added_rigid_asset。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "原场景中要删除的刚体实例 ID",
                },
            },
            "required": ["instance_id"],
        },
    )


def _remove_added_rigid_asset_tool() -> Tool:
    return Tool(
        name="remove_added_rigid_asset",
        description=(
            "从当前草稿的 Nodes.RigidNodes.add 中撤销 DGGT 车辆插入,同步移除 anchor_pixel 和 scale。"
            "先调用 preview_config 查看 add_obj 顺序,index 从 0 开始,不是场景 instance_id。"
            "当前 DGGT 插入只有一项,index 固定为 0。"
            "只修改当前草稿,保留磁盘资产文件;需要调用 save_config 保存结果。"
            "原场景已有对象请使用 remove_rigid_instance。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "preview_config 中 RigidNodes.add.add_obj 的序号,从 0 开始",
                },
            },
            "required": ["index"],
        },
    )


def _add_smpl_person_tool(smpl_catalog: str) -> Tool:
    return Tool(
        name="add_smpl_person",
        description=f"""向当前草稿的 Nodes.SMPLNodes.add 追加一个 SMPL 行人。

资产类别规则:
- 行人/人体资产属于 SMPLNodes
- 普通/其它静态资产属于 Background
- 车辆资产属于 RigidNodes

当前可用SMPL资产:
{smpl_catalog}
""",
        inputSchema={
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "SMPL模型路径(.ply),必须来自 list_smpl_assets 返回结果",
                },
                "motion": {
                    "type": "string",
                    "description": "SMPL运动路径(.pt),必须来自 list_smpl_assets 返回结果",
                },
                "position": _vector_property("行人位置"),
                "ref_id": {
                    "type": "integer",
                    "description": "参考实例ID,可选",
                },
            },
            "required": ["model", "motion", "position"],
        },
    )


def _preview_config_tool() -> Tool:
    return Tool(
        name="preview_config",
        description="查看当前配置草稿。不会保存文件。",
        inputSchema={"type": "object", "properties": {}},
    )


def _save_config_tool(output_dir: str) -> Tool:
    return Tool(
        name="save_config",
        description=f"保存 DGGT RigidNodes 配置为 JSON。相对路径基于 OUTPUT_DIR ({output_dir})。",
        inputSchema={
            "type": "object",
            "properties": {
                "output_path": {
                    "type": "string",
                    "description": "DGGT JSON 输出文件路径",
                },
            },
            "required": ["output_path"],
        },
    )


def _set_ego_trajectory_offsets_tool() -> Tool:
    return Tool(
        name="set_ego_trajectory_offsets",
        description=(
            "设置 DGGT Mode 2 自车渲染轨迹相对偏移。每项是相对于原相机轨迹"
            "局部坐标的 [前, 左, 上] 偏移，使用 DGGT 原生单位，不要换算米。"
            "传一个向量作用于所有帧，传多个向量则按帧使用。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "offsets": {
                    "type": "array",
                    "items": _vector_property("自车相对偏移"),
                    "minItems": 1,
                    "description": "一个全局偏移或逐帧偏移数组 [[前, 左, 上], ...]",
                },
            },
            "required": ["offsets"],
        },
    )


def _save_ego_trajectory_offsets_tool(output_dir: str) -> Tool:
    return Tool(
        name="save_ego_trajectory_offsets",
        description=f"保存自车相对轨迹偏移 JSON。相对路径基于 OUTPUT_DIR ({output_dir})。",
        inputSchema={
            "type": "object",
            "properties": {
                "output_path": {"type": "string", "description": "必须以 .json 结尾"},
            },
            "required": ["output_path"],
        },
    )


def _reset_config_tool() -> Tool:
    return Tool(
        name="reset_config",
        description="清空当前 DGGT 配置草稿。",
        inputSchema={"type": "object", "properties": {}},
    )


def _global_properties() -> dict[str, Any]:
    return {
        "debug": {
            "type": "boolean",
            "description": "调试模式",
            "default": False,
        },
        "weather_type": {
            "type": "string",
            "description": "天气类型,默认空字符串",
            "default": "",
        },
        "is_legend": {
            "type": "boolean",
            "description": "是否渲染图例",
            "default": False,
        },
        "legend": {
            "type": "object",
            "description": "可选:按节点类型控制图例渲染,如 {\"rigid\": false, \"smpl\": false}",
            "properties": {
                "rigid": {"type": "boolean"},
                "smpl": {"type": "boolean"},
            },
        },
    }


def _list_instances_tool() -> Tool:
    return Tool(
        name="list_instances",
        description=(
            "读取 DGGT 的 bbox 实例清单。编辑已有对象前先确认 instance_id，"
            "以及 frames 中各已标注帧的 frame_index 和 bbox。"
            "bbox 使用模型输入图坐标 [x1,y1,x2,y2]，不需要 mask_path 或 ply_path。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "instance_info_path": {
                    "type": "string",
                    "description": "逐帧 bbox instance_info.json 的绝对路径",
                },
            },
            "required": ["instance_info_path"],
        },
    )

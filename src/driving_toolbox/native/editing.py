from ..catalog import get_tool
from ..core.errors import InputError
from ..core.io import file_path, write_json
from ..core.schema import validate


def compile_config(data):
    validate(data, get_tool("edit_config_generator")["input_schema"])
    operations = data.get("operations")
    if not operations:
        raise InputError("Provide operations or edit_plan")
    validate(operations, get_tool("edit_config_generator")["input_schema"]["properties"]["operations"])
    ids = None
    if "instances" in data:
        ids = {item["instance_id"] for item in data["instances"]}
    for operation in operations:
        name = operation["operation"]
        if name != "add":
            if "instance_id" not in operation:
                raise InputError(f"{name} requires instance_id")
            if ids is not None and operation["instance_id"] not in ids:
                raise InputError("Edit references an instance absent from supplied instances")
        required = {
            "trajectory_edit": ["waypoints"],
            "rotate": ["rotation"],
            "scale": ["scale"],
            "add": ["asset_path", "scale", "anchor_pixel"],
            "replace": ["asset_path", "scale"],
            "remove": [],
        }[name]
        for key in required:
            if key not in operation:
                raise InputError(f"{name} requires {key}")
        if name in {"add", "replace"}:
            file_path(operation["asset_path"], "asset_path")
    config = {
        "schema_version": "1.0",
        "coordinate_system": data.get("coordinate_system", "camera_local_meters"),
        "operations": operations,
    }
    result = {"edit_config": config}
    if data.get("output_path"):
        path = write_json(data["output_path"], config)
        result.update(saved_path=path, _artifacts=[{"kind": "edit_config", "path": path}])
    return result


def to_rigid(config):
    """Translate a portable edit document to the optional renderer's wire format."""
    rigid = {}
    for item in config["operations"]:
        name = item["operation"]
        if name in rigid and name in {"add", "replace"}:
            raise InputError("This renderer supports only one add and one replace operation per invocation")
        section = rigid.setdefault(name, {})

        def append(key, value, section=section):
            section.setdefault(key, []).append(value)

        if name == "replace":
            append("target_id", item["instance_id"])
            append("replace_obj", str(file_path(item["asset_path"], "asset_path")))
            append("scale", item["scale"])
        elif name == "add":
            append("add_obj", str(file_path(item["asset_path"], "asset_path")))
            append("anchor_pixel", item["anchor_pixel"])
            append("scale", item["scale"])
        else:
            append("instance_id", item["instance_id"])
            if name == "trajectory_edit":
                append("waypoints", item["waypoints"])
                append("method", "cubic" if item.get("method") == "catmull_rom" else "linear")
            elif name in {"scale", "rotate"}:
                key = "rotation" if name == "rotate" else "scale"
                append(key, item[key])
    return {"Nodes": {"RigidNodes": rigid}}

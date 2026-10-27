from .errors import InputError, ToolError
from .schema import validate


def execute(name, handler, input_data, config=None):
    from ..catalog import get_tool

    envelope = {
        "schema_version": "1.0",
        "tool": name,
        "status": "error",
        "result": {},
        "artifacts": [],
        "warnings": [],
        "error": None,
    }
    try:
        if config is not None and not isinstance(config, dict):
            raise InputError("Runtime config must be an object")
        validate(input_data, get_tool(name)["input_schema"])
        value = handler(input_data, config or {})
        if not isinstance(value, dict):
            raise TypeError("Tool implementation must return an object")
        envelope.update(status="success", result=value)
        envelope["warnings"] = value.pop("_warnings", [])
        envelope["artifacts"] = value.pop("_artifacts", [])
    except (ToolError, ValueError, TypeError, OSError, KeyError) as exc:
        envelope["error"] = {"code": getattr(exc, "code", "execution_failed"), "message": str(exc)}
    return envelope

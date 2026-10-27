from ...core.runtime import execute


def _schema(properties, required=None):
    return {"type": "object", "properties": properties, "required": required or list(properties)}


_S = {"type": "string", "minLength": 1}
_A = {"type": "array"}


def _execute(data, config):
    from .compiler import compile_config

    if data.get("edit_plan") and not data.get("operations"):
        from .mcp_client import generate

        return generate(data, config)
    return compile_config(data)


def run(input_data, *, config=None):
    return execute("edit_config_generator", _execute, input_data, config)

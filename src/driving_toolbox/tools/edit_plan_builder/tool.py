from ...core.llm import ask_json
from ...core.runtime import execute


def _schema(properties, required=None):
    return {"type": "object", "properties": properties, "required": required or list(properties)}


_S = {"type": "string", "minLength": 1}
_A = {"type": "array"}


def _execute(data, config):
    schema = _schema({"edit_plan": _S, "target_instances": _A, "operations": _A, "constraints": _A})
    return ask_json(
        "Create a precise scene edit plan. Never invent instance IDs, camera calibration or metric scale. "
        "State missing information as constraints.",
        data,
        schema,
        config,
    )


def run(input_data, *, config=None):
    return execute("edit_plan_builder", _execute, input_data, config)

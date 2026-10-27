from ...core.llm import ask_json
from ...core.runtime import execute


def _schema(properties, required=None):
    return {"type": "object", "properties": properties, "required": required or list(properties)}


_S = {"type": "string", "minLength": 1}
_A = {"type": "array"}


def _execute(data, config):
    schema = _schema(
        {
            "scene_requirement": _S,
            "failure_type": _S,
            "required_conditions": _A,
            "required_actors": _A,
            "risk_factors": _A,
            "target_effect": _S,
        }
    )
    return ask_json(
        "Generate testable driving scenario requirements grounded only in the failure analysis.",
        data,
        schema,
        config,
    )


def run(input_data, *, config=None):
    return execute("scene_requirement_generator", _execute, input_data, config)

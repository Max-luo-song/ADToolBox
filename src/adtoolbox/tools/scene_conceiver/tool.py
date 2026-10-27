from ...core.llm import ask_json
from ...core.runtime import execute


def _schema(properties, required=None):
    return {"type": "object", "properties": properties, "required": required or list(properties)}


_S = {"type": "string", "minLength": 1}
_A = {"type": "array"}


def _execute(data, config):
    schema = _schema(
        {
            "scene_plan": _S,
            "base_video_requirement": _S,
            "edit_ideas": _A,
            "retrieval_query": _S,
            "expected_effect": _S,
        }
    )
    return ask_json(
        "Design a plausible driving scene with minimal edits. Distinguish edited actors from the ego "
        "policy response to be tested; never promise a policy will brake.",
        data,
        schema,
        config,
    )


def run(input_data, *, config=None):
    return execute("scene_conceiver", _execute, input_data, config)

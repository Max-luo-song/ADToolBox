from ...backends.process import configured
from ...core.errors import InputError
from ...core.io import file_path
from ...core.llm import ask_json
from ...core.runtime import execute


def _schema(properties, required=None):
    return {"type": "object", "properties": properties, "required": required or list(properties)}


_S = {"type": "string", "minLength": 1}
_A = {"type": "array"}


def _execute(data, config):
    external = configured("failure_analyzer", data, config)
    if external is not None:
        return external
    if data.get("token"):
        from ...core.projects import run_project

        return run_project("failure_analysis", data, config)
    if not data.get("failure_case"):
        raise InputError("Provide failure_case evidence, or configure a token-forensics backend")
    images = [file_path(path, "image_paths") for path in data.get("image_paths", [])]
    schema = _schema(
        {
            "failure_type": _A,
            "failure_reason": _S,
            "key_scene_factors": _A,
            "risk_factors": _A,
            "observations": _A,
            "uncertainty": _S,
        }
    )
    result = ask_json(
        "Analyze the supplied driving failure evidence. Distinguish observations from hypotheses. "
        "Do not claim collision responsibility without official evidence. A token alone is not evidence.",
        data,
        schema,
        config,
        images=images,
    )
    return {"failure_analysis": result}


def run(input_data, *, config=None):
    return execute("failure_analyzer", _execute, input_data, config)

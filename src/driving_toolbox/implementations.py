"""Tool implementations. All heavy libraries are imported only when requested."""

from .backends.process import configured
from .core.errors import InputError
from .core.io import file_path
from .core.llm import ask_json


def _schema(properties, required=None):
    return {"type": "object", "properties": properties, "required": required or list(properties)}


_S = {"type": "string", "minLength": 1}
_A = {"type": "array"}


def failure_analyzer(data, config):
    external = configured("failure_analyzer", data, config)
    if external is not None:
        return external
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


def scene_requirement_generator(data, config):
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


def scene_conceiver(data, config):
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


def edit_plan_builder(data, config):
    schema = _schema({"edit_plan": _S, "target_instances": _A, "operations": _A, "constraints": _A})
    return ask_json(
        "Create a precise scene edit plan. Never invent instance IDs, camera calibration or metric scale. "
        "State missing information as constraints.",
        data,
        schema,
        config,
    )


def video_retriever(data, config):
    external = configured("video_retriever", data, config)
    if external is not None:
        return external
    if not data.get("catalog_path"):
        raise InputError("Provide catalog_path, or configure a semantic retrieval backend")
    from .native.retrieval import search

    return search(data)


def instance_discovery(data, config):
    external = configured("instance_discovery", data, config)
    if external is not None:
        return external
    from .native.tracking import track

    return track(data)


def reconstruction(data, config):
    external = configured("reconstruction", data, config)
    if external is not None:
        return external
    from .backends.gaussian import reconstruct

    return reconstruct(data, config)


def edit_config_generator(data, config):
    from .native.editing import compile_config

    if data.get("edit_plan") and not data.get("operations"):
        from .catalog import get_tool

        op_schema = get_tool("edit_config_generator")["input_schema"]["properties"]["operations"]
        value = ask_json(
            "Translate the edit plan to typed operations using only supplied instance IDs. "
            "All positions are offsets in the specified coordinate system. Never guess missing scales.",
            data,
            _schema({"operations": op_schema}),
            config,
        )
        data = {**data, "operations": value["operations"]}
    return compile_config(data)


def scene_editor(data, config):
    external = configured("scene_editor", data, config)
    if external is not None:
        return external
    from .backends.gaussian import edit

    return edit(data, config)


def scene_validator(data, config):
    from .native.validation import check

    return check(data, config)


def recovery_advisor(data, config):
    evaluation = data["evaluation"]
    if evaluation.get("passed") is True:
        return {
            "action": "continue",
            "reason": "The supplied evaluation passed",
            "suggestions": [],
            "executes_actions": False,
        }
    issues = evaluation.get("issues") or ["Evaluation did not explicitly pass"]
    return {
        "action": "inspect",
        "reason": "Inspect the failed or incomplete evaluation before another attempt",
        "suggestions": [str(item) for item in issues],
        "executes_actions": False,
    }


def policy_runner(data, config):
    external = configured("policy_runner", data, config)
    if external is not None:
        return external
    from .backends.policy import evaluate

    return evaluate(data, config)


def failure_case_filter(data, config):
    from .native.curation import filter_case

    return filter_case(data)


def policy_trainer(data, config):
    external = configured("policy_trainer", data, config)
    if external is not None:
        return external
    from .backends.policy import train

    return train(data, config)

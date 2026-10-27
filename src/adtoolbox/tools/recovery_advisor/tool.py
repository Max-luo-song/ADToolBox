from ...core.runtime import execute


def _execute(data, config):
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


def run(input_data, *, config=None):
    return execute("recovery_advisor", _execute, input_data, config)

from ...backends.process import configured
from ...core.runtime import execute


def _execute(data, config):
    external = configured("policy_runner", data, config)
    if external is not None:
        return external
    from .launcher import evaluate

    return evaluate(data, config)


def run(input_data, *, config=None):
    return execute("policy_runner", _execute, input_data, config)

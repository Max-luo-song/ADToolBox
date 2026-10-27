from ...backends.process import configured
from ...core.runtime import execute


def _execute(data, config):
    external = configured("policy_trainer", data, config)
    if external is not None:
        return external
    from .trainer import train

    return train(data, config)


def run(input_data, *, config=None):
    return execute("policy_trainer", _execute, input_data, config)

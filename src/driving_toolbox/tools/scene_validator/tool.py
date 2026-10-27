from ...core.runtime import execute


def _execute(data, config):
    from .validator import check

    return check(data, config)


def run(input_data, *, config=None):
    return execute("scene_validator", _execute, input_data, config)

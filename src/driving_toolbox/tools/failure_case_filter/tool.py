from ...core.runtime import execute


def _execute(data, config):
    from .evaluator import filter_case

    return filter_case(data)


def run(input_data, *, config=None):
    return execute("failure_case_filter", _execute, input_data, config)

from ...backends.process import configured
from ...core.runtime import execute


def _execute(data, config):
    external = configured("reconstruction", data, config)
    if external is not None:
        return external
    from .service import reconstruct

    return reconstruct(data, config)


def run(input_data, *, config=None):
    return execute("reconstruction", _execute, input_data, config)

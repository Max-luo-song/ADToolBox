from ...backends.process import configured
from ...core.runtime import execute


def _execute(data, config):
    external = configured("instance_discovery", data, config)
    if external is not None:
        return external
    from .tracker import track

    return track(data)


def run(input_data, *, config=None):
    return execute("instance_discovery", _execute, input_data, config)

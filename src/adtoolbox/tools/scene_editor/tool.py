from ...backends.process import configured
from ...core.runtime import execute


def _execute(data, config):
    external = configured("scene_editor", data, config)
    if external is not None:
        return external
    from ..reconstruction.service import edit

    return edit(data, config)


def run(input_data, *, config=None):
    return execute("scene_editor", _execute, input_data, config)

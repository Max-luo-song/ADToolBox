"""Apply instance edits, coordinate conversion and optional camera offsets."""

from ..core.runtime import execute
from ..implementations import scene_editor as _execute


def run(input_data, *, config=None):
    return execute("scene_editor", _execute, input_data, config)

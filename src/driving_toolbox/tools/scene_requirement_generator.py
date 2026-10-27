"""Translate a failure analysis into grounded scenario requirements."""

from ..core.runtime import execute
from ..implementations import scene_requirement_generator as _execute


def run(input_data, *, config=None):
    return execute("scene_requirement_generator", _execute, input_data, config)

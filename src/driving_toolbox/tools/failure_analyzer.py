"""Analyze observations and uncertainty without inventing missing evidence."""

from ..core.runtime import execute
from ..implementations import failure_analyzer as _execute


def run(input_data, *, config=None):
    return execute("failure_analyzer", _execute, input_data, config)

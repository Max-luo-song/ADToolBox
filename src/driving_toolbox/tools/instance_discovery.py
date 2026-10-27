"""Detect and track visible traffic participants with an optional tracking backend."""

from ..core.runtime import execute
from ..implementations import instance_discovery as _execute


def run(input_data, *, config=None):
    return execute("instance_discovery", _execute, input_data, config)

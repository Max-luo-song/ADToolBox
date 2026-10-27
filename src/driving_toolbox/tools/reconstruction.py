"""Reconstruct video into dynamic Gaussian assets and camera parameter artifacts."""

from ..core.runtime import execute
from ..implementations import reconstruction as _execute


def run(input_data, *, config=None):
    return execute("reconstruction", _execute, input_data, config)

"""Check artifacts and optionally inspect sampled video frames; report unverified physical properties."""

from ..core.runtime import execute
from ..implementations import scene_validator as _execute


def run(input_data, *, config=None):
    return execute("scene_validator", _execute, input_data, config)

"""Compile typed edits into a portable, validated scene-edit document."""

from ..core.runtime import execute
from ..implementations import edit_config_generator as _execute


def run(input_data, *, config=None):
    return execute("edit_config_generator", _execute, input_data, config)

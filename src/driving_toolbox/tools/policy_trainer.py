"""Launch supervised ILFT with explicit data, configuration and initial weights."""

from ..core.runtime import execute
from ..implementations import policy_trainer as _execute


def run(input_data, *, config=None):
    return execute("policy_trainer", _execute, input_data, config)

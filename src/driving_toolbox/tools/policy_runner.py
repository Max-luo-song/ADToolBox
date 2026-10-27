"""Run an explicitly configured closed-loop policy and collect runner reports."""

from ..core.runtime import execute
from ..implementations import policy_runner as _execute


def run(input_data, *, config=None):
    return execute("policy_runner", _execute, input_data, config)

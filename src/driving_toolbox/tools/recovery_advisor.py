"""Return diagnostic advice without running retries or dispatching other tools."""

from ..core.runtime import execute
from ..implementations import recovery_advisor as _execute


def run(input_data, *, config=None):
    return execute("recovery_advisor", _execute, input_data, config)

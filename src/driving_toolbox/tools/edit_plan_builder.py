"""Ground an edit plan in supplied instances and coordinate information."""

from ..core.runtime import execute
from ..implementations import edit_plan_builder as _execute


def run(input_data, *, config=None):
    return execute("edit_plan_builder", _execute, input_data, config)

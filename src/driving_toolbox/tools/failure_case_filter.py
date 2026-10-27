"""Screen heading-aware dynamic vehicle box overlaps and optionally append cases to JSONL."""

from ..core.runtime import execute
from ..implementations import failure_case_filter as _execute


def run(input_data, *, config=None):
    return execute("failure_case_filter", _execute, input_data, config)

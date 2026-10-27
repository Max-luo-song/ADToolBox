"""Rank captioned local video records; use a configured backend for semantic retrieval."""

from ..core.runtime import execute
from ..implementations import video_retriever as _execute


def run(input_data, *, config=None):
    return execute("video_retriever", _execute, input_data, config)

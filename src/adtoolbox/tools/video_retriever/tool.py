from ...backends.process import configured
from ...core.runtime import execute


def _execute(data, config):
    external = configured("video_retriever", data, config)
    if external is not None:
        return external
    if data.get("catalog_path"):
        from .caption_search import search

        return search(data)
    from ...core.projects import run_project

    return run_project("video_search", data, config)


def run(input_data, *, config=None):
    return execute("video_retriever", _execute, input_data, config)

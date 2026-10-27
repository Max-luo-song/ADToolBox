"""Independent tools for driving scenario engineering."""

__version__ = "0.1.0"


def run_tool(name, input_data, *, config=None):
    from .registry import run_tool as dispatch

    return dispatch(name, input_data, config=config)


def list_tools():
    from .catalog import list_tools as listing

    return listing()

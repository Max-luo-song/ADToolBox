from importlib import import_module

from .catalog import get_tool


def run_tool(name, input_data, *, config=None):
    get_tool(name)
    module = import_module(f"adtoolbox.tools.{name}")
    return module.run(input_data, config=config)

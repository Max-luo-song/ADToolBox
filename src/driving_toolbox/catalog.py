import json
from importlib.resources import files


def list_tools():
    return json.loads(files("driving_toolbox").joinpath("catalog.json").read_text(encoding="utf-8"))


def get_tool(name):
    for item in list_tools():
        if item["name"] == name:
            return item
    raise ValueError(f"Unknown tool: {name}")

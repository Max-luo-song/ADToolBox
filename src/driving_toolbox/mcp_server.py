"""Expose exact tool schemas through the optional official MCP runtime."""

import argparse
import asyncio
import json

from .catalog import list_tools
from .core.io import read_json
from .registry import run_tool


def create_server(config=None):
    try:
        from mcp import types
        from mcp.server.lowlevel import Server
    except ImportError:
        raise SystemExit("Install driving-toolbox[mcp] to start the MCP server") from None
    server = Server("TOOLBOX")
    read_only = {
        "failure_analyzer",
        "scene_requirement_generator",
        "scene_conceiver",
        "edit_plan_builder",
        "scene_validator",
        "recovery_advisor",
    }

    @server.list_tools()
    async def discover():
        return [
            types.Tool(
                name=spec["name"],
                description=spec["description"],
                inputSchema=spec["input_schema"],
                annotations=types.ToolAnnotations(
                    readOnlyHint=spec["name"] in read_only,
                    destructiveHint=spec["name"] not in read_only,
                    openWorldHint=True,
                ),
            )
            for spec in list_tools()
        ]

    @server.call_tool()
    async def call(name, arguments):
        result = await asyncio.to_thread(run_tool, name, arguments, config=config)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))],
            structuredContent=result,
            isError=result["status"] == "error",
        )

    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    config = read_json(args.config) if args.config else {}

    async def serve():
        from mcp.server.stdio import stdio_server

        server = create_server(config)
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(serve())


if __name__ == "__main__":
    main()

"""Scene Editing MCP Server。

由客户端 LLM(Claude/Cursor 等)负责自然语言解析,
本服务器只做确定性的配置组装与文件 I/O。
"""

import asyncio
import json
import os
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool
from pydantic import AnyUrl

from .assets import format_asset_catalog, scan_assets
from .tools import get_tool_definitions, register_tool_handlers

ASSETS_DIR = os.environ.get("ASSETS_DIR", "assets")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.getcwd())

app: Server = Server("scene-editing-mcp")


def _load_assets() -> dict[str, list[str]]:
    return scan_assets(ASSETS_DIR)


@app.list_tools()
async def list_tools() -> list[Tool]:
    assets = _load_assets()
    catalog = format_asset_catalog(assets)
    categories = ", ".join(assets.keys()) if assets else "(空)"

    return get_tool_definitions(
        assets_catalog=catalog,
        categories=categories,
        output_dir=OUTPUT_DIR,
    )


# 注册工具处理函数
_tool_handlers = register_tool_handlers(
    assets_dir=ASSETS_DIR,
    output_dir=OUTPUT_DIR,
)


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    args = arguments if isinstance(arguments, dict) else {}

    handler = _tool_handlers.get(name)
    if handler:
        return await handler(args)

    raise ValueError(f"未知工具: {name}")


@app.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri=AnyUrl("assets://catalog"),
            name="资产清单",
            description="当前可用的3D资产文件列表(JSON)",
            mimeType="application/json",
        ),
    ]


@app.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    if str(uri) == "assets://catalog":
        return json.dumps(_load_assets(), ensure_ascii=False, indent=2)
    raise ValueError(f"未知资源: {uri}")


async def main() -> None:
    async def wake_stdio_reader() -> None:
        # This environment does not reliably wake an idle loop for AnyIO's stdin worker.
        while True:
            await asyncio.sleep(0.1)

    wake_task = asyncio.create_task(wake_stdio_reader())
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    finally:
        wake_task.cancel()
        try:
            await wake_task
        except asyncio.CancelledError:
            pass


def main_sync() -> None:
    """同步入口,供 console_scripts / shim 使用。"""
    import asyncio

    asyncio.run(main())


if __name__ == "__main__":
    main_sync()

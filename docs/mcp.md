# MCP integration

Install `python -m pip install -e '.[mcp]'`, then configure an MCP client:

```json
{
  "mcpServers": {
    "toolbox": {
      "command": "/path/to/venv/bin/tool-box-mcp",
      "args": ["--config", "/path/to/runtime.local.json"],
      "env": {"TOOLBOX_API_KEY": "set-this-locally"}
    }
  }
}
```

The server exposes all 14 tools over stdio using the official MCP Python SDK. Each tool exposes its exact input JSON Schema in MCP metadata and `tool-box schema NAME`. Pass tool arguments directly. Invalid tool input becomes an MCP tool error.

The server only exposes explicit calls. It does not select tools, run a workflow or maintain agent history. Use absolute input paths when your client starts the server from another working directory. Tools may write artifacts and model-backed tools may transmit supplied data to your configured endpoint.

中文：MCP 仅负责工具发现与显式调用，不包含 Agent 编排。调用参数直接填写对应工具输入。密钥在客户端本地配置，勿提交到仓库。

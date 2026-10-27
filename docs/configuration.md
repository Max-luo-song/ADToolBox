# Runtime configuration

The core package has no third-party runtime dependencies. Install from the repository; a public package-index release is not assumed:

```bash
python -m pip install -e .
python -m pip install -e '.[mcp]'   # optional MCP server
python -m pip install -e '.[vision]' # optional local detector/tracker
cp configs/runtime.example.json configs/runtime.local.json
```

Only configure the sections used by your chosen tool. Each backend can live in a different Python environment. `adtoolbox doctor --config configs/runtime.local.json` checks availability, not model correctness.

For model-backed tools, export `ADTOOLBOX_API_KEY`, `ADTOOLBOX_BASE_URL` and `ADTOOLBOX_MODEL`. Alternatively put `base_url` and `model` in the `llm` section and select a credential variable with `api_key_env`. The CLI does not automatically load `.env`; export variables in your shell. API keys are never accepted in tool inputs or committed configuration. Text evidence and explicitly attached images are transmitted to the endpoint you configure.

## External JSON backend

Heavy tools and retrieval/forensics can use a standalone subprocess instead of their built-in adapter:

```json
{
  "backends": {
    "video_retriever": {
      "command": ["/path/to/python", "/path/to/your_retrieval_bridge.py"],
      "cwd": "/path/to/retrieval-project",
      "timeout": 300
    }
  }
}
```

The command reads one tool-input JSON object from stdin and writes exactly one response:

```json
{"status": "success", "result": {"video_path": "/path/to/selected.mp4"}}
```

Non-zero exits, timeouts and malformed responses fail explicitly. This is an integration interface, not a bundled semantic retrieval model. `examples/backends/json_bridge.py` is an executable reference bridge for caption retrieval. Commands are trusted runtime configuration, never model-generated strings; subprocesses run without a shell.

## 中文

先安装基础库，再按需安装 MCP、检测跟踪等依赖。模型密钥通过环境变量传入；`.env` 不会自动加载。每个重型后端可以使用自己的 Python 环境。配置中的 command 是可信的本地命令数组，工具输入不能修改它。doctor 只能检查环境，不证明 GPU 推理已成功。

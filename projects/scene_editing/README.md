# Scene editing configuration project / 场景编辑配置项目

Migrated from the existing working editing component. This directory includes the complete deterministic MCP service, rigid/background/human node builders, asset scanner, JSON/YAML I/O and the natural-language model client. It does not include the global Agent scheduler or third-party asset files.

```bash
python -m venv environments/scene_editing
environments/scene_editing/bin/python -m pip install -e projects/scene_editing
```

Set `scene_editing.python` in runtime configuration to that interpreter. Configure ADToolbox model credentials as usual. For natural-language editing, provide `edit_plan`, `output_path`, matching `instance_info_path`, and explicit `asset_path` / `asset_scale` for addition or replacement. The tool launches `bridge.py`, connects to the real MCP service, calls its edit tools and saves a renderer-executable config. It returns `backend_config_path`, `saved_path` and the recorded `tool_calls`.

The resulting config can go directly into `scene_editor.edit_config_path`. Camera-local trajectory offsets still require camera poses and calibrated scale. Typed portable operations remain available without a model.

中文：`scene_editing/` 是完整编辑服务，`clients/` 是独立交互客户端，`bridge.py` 是原先工具使用的模型到 MCP 调用逻辑。自然语言路径会真正调用编辑工具并保存配置。
